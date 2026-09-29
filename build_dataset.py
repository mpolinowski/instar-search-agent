#!/usr/bin/env python3
"""
build_dataset.py -- Stage-1 converter (SPEC_Dataset_Creation.md).

Turns ./data (a tree of article dirs, each with one index.mdx + 0..N local
images) into a self-contained Hugging Face dataset at ./hf_dataset
(Parquet shards, `train` split, load_dataset-loadable).

One row per article (spec 2.0). NOT one-row-per-image, NOT ImageFolder.
  text       = byte-faithful original MDX         (spec 2.1)
  content    = cleaned, fence-aware Markdown      (spec 2.3)
  image_refs = relative-local ![alt](path) only   (spec 2.4)
  images     = RGBA PNG, aligned to refs          (spec 2.2, C3/C4)
C1..C5 (spec 1) enforced by the hard gate (spec 3.2). Determinism (spec 6).
Logs use [DATASET]/[WARN]/[ERROR] (AGENTS.md 5.2). Exit 0 only if the
hard gate + handoff smoke test (spec 4) pass.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from PIL import Image

import datasets as ds  # Features/Value/Sequence/Image/Dataset/load_dataset

# Refuse absurdly large rasters (decompression-bomb guard). The spec 2.2 cap
# handles legit-but-large images; this only trips on corrupt/hostile files.
Image.MAX_IMAGE_PIXELS = 100_000_000

# Deterministic image encoding (spec 2.2 "fixed quality", spec 6 idempotence).
PNG_COMPRESSLEVEL = 9          # fixed zlib level -> stable bytes across runs

# Decodable local image extensions (spec 2.4 resolution rules).
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[%(levelname)s] %(message)s")
log = logging.getLogger("build_dataset")


def di(msg: str) -> None:
    log.info("[DATASET] %s", msg)


def dw(msg: str) -> None:
    log.warning("[DATASET/WARN] %s", msg)


def de(msg: str) -> None:
    log.error("[DATASET/ERROR] %s", msg)


@dataclass(frozen=True)
class Config:
    data: Path
    out: Path
    max_image_edge: int = 4096     # 0 = uncapped (spec default 4096)
    keep_frames: bool = False      # debug only; multi-frame breaks the embedder
    shard_rows: int = 256
    strict: bool = False           # fail on any dropped image
    limit: int = 0                 # 0 = all; else first N (dry-run)

    @property
    def hf_dir(self) -> Path:
        # --out IS the HF dataset dir per spec 5. Shards + README go directly
        # here (not out/hf_dataset/...).
        return self.out

    @property
    def report_path(self) -> Path:
        return self.out / "REPORT.md"

    @property
    def summary_path(self) -> Path:
        return self.out / "REPORT.md.json"


def discover_articles(data_root: Path) -> List[Path]:
    """Every index.mdx under data_root, sorted by relative path (codepoint)."""
    found = [p for p in data_root.rglob("index.mdx") if p.is_file()]
    found.sort(key=lambda p: str(p.relative_to(data_root)).encode("utf-8"))
    return found


def article_id_for(rel: str, seen: set) -> str:
    """Stable, collision-safe, path-derived ID (spec 6, C5).

    relpath with the trailing index.mdx stripped. Unique within the corpus by
    construction; a collision (impossible from the FS) appends sha1[:8] of the
    full relpath -- never random.
    """
    parts = [p for p in rel.replace(os.sep, "/").split("/") if p]
    if parts and parts[-1] == "index.mdx":
        parts = parts[:-1]
    base = "/".join(parts) or "root"
    if base in seen:
        h = hashlib.sha1(rel.replace(os.sep, "/").encode("utf-8")).hexdigest()[:8]
        return f"{base}#{h}"
    return base


def parse_frontmatter(mdx: str) -> Tuple[Dict[str, Any], str]:
    """Split a leading YAML frontmatter block out of the MDX.

    Returns (metadata, remainder). If no frontmatter, ({}, mdx as-is).
    Frontmatter format: `---\\n ... \\n ---` at the very start of the file.
    """
    if not mdx.startswith("---"):
        return {}, mdx

    # Find the closing `---` on its own line.
    m = re.search(r"^(?:---)\s*$", mdx[3:], re.MULTILINE)
    if not m:
        # No close found -- not valid frontmatter; return as-is.
        return {}, mdx

    body = mdx[3 : 3 + m.start()]           # YAML text between the `---`s
    tail = mdx[3 + m.end():]                # everything after (lstrip newlines)
    tail = tail.lstrip("\n")

    data: Dict[str, Any] = {}
    try:
        parsed = yaml.safe_load(body)
        if isinstance(parsed, dict):
            data = parsed
    except yaml.YAMLError:
        # Frontmatter YAML was malformed. We still drop it from `content`
        # (it is not prose) but we lose the metadata fields. The caller
        # reports this as a warning.
        log.warning("parse_frontmatter: YAML parse failed; dropping frontmatter")
    return data, tail


# ==========================================================================
# TOKENIZER + CLEANER  (spec 2.3 — fence-aware, MDX → clean Markdown)
# ==========================================================================
#
# We do NOT regex-substitute the whole article. Instead we tokenize into
# protected chunks and run targeted rules only on the prose.
#
#   CHUNK KIND    | HOW WE TREAT IT
#   --------------+----------------
#   fence         | preserved byte-for-byte (spec 2.3 "sacred")
#   comment       | removed entirely (implementation detail / disabled block)
#   text          | cleaned: imports, JSX, inline HTML -> Markdown, braces
#
# This is state-aware: fenced regions are tracked across lines, comment
# boundaries are matched to their closing `-->`, and nested cases (a fence
# marked up inside a comment) resolve correctly because we emit a chunk for
# the outermost boundary we encounter first (comment), which contains the
# whole inner content.
#
# --------------------------------------------------------------------------
FENCE_OPEN_RE  = re.compile(r"^(`{3,}|~{3,})\s*")
COMMENT_OPEN   = "<!--"
COMMENT_CLOSE  = "-->"


def tokenize_mdx(body: str) -> List[Tuple[str, str]]:
    """Split an article body into ordered (kind, text) chunks.

    Kinds: "text" (prose, to clean), "fence" (kept verbatim), "comment"
    (dropped by the cleaner). Order is preserved so that a text chunk always
    appears *before* any chunk that follows it in the source.

    A fence is a run starting at a line whose first non-space chars are ```
    or ~~~, ending at the next line that is (modulo trailing spaces) the same
    marker alone. A comment is a run from `<!--` up to the first `-->`.
    A line can be both text and the start of a comment (e.g.
    `foo <!-- bar -->`); the part before `<!--` is emitted as text first.
    """
    lines = body.split("\n")
    n = len(lines)
    chunks: List[Tuple[str, str]] = []
    out_text: List[str] = []
    i = 0

    def flush_text() -> None:
        if out_text:
            chunks.append(("text", "".join(out_text)))
            out_text.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # --- fenced code block -------------------------------------------
        m = FENCE_OPEN_RE.match(stripped)
        if m:
            flush_text()
            marker = m.group(1)
            fence_buf = [line + "\n"]
            i += 1
            while i < n:
                fence_buf.append(lines[i] + "\n")
                if lines[i].strip() == marker:
                    i += 1
                    break
                i += 1
            chunks.append(("fence", "".join(fence_buf)))
            continue

        # --- html comment ------------------------------------------------
        if COMMENT_OPEN in line:
            start = line.index(COMMENT_OPEN)
            before = line[:start]
            if before.strip():
                out_text.append(before + "\n")
            cbuf: List[str] = []
            j = i
            closed = False
            tail_after_close: Optional[str] = None
            while j < n:
                seg = lines[j]
                if COMMENT_CLOSE in seg:
                    close_idx = seg.index(COMMENT_CLOSE)
                    keep_end = close_idx + len(COMMENT_CLOSE)
                    if j == i:
                        cbuf.append(seg[start:keep_end])
                    else:
                        cbuf.append(seg[:keep_end])
                    tail_after_close = seg[keep_end:]
                    i = j + 1
                    closed = True
                    break
                cbuf.append(seg)
                j += 1
            if not closed:            # unterminated comment: consume to EOF
                i = n
            # Emit the pending text (if any) BEFORE the comment chunk.
            flush_text()
            chunks.append(("comment", "\n".join(cbuf)))
            if tail_after_close and tail_after_close.strip():
                # Text after a same-line comment close: treat as prose.
                out_text.append(tail_after_close + "\n")
            continue

        # --- plain text ---------------------------------------------------
        out_text.append(line + "\n")
        i += 1

    flush_text()
    return chunks




# ---- PascalCase + prose-tag tables ---------------------------------------
PASCAL = r"[A-Z][A-Za-z0-9]*"
# Inline HTML we convert to markdown (keep meaning). The rest of the common
# prose tags we strip.
_LINK_PAIR_RE     = re.compile(r"<(Link|OutboundLink)\b[^>]*>(.*?)</\1\s*>", re.S)
_SELF_CLOSE_RE    = re.compile(r"<" + PASCAL + r"\b[^<>]*?/>")
_PAIR_RE          = re.compile(r"<(" + PASCAL + r")\b[^>]*>.*?</\1\s*>", re.S)
_JAVASCRIPT_PROPS_RE = re.compile(r"[^>]{props\.[^>}]*\}>")
_IMPORT_EXPORT_RE  = re.compile(r"^[ \t]*(?:import|export)\b.*$", re.M)
_BRACE_PROSE_RE   = re.compile(r"^[ \t]*\{[^}]*\}[ \t]*$")


def _clean_inline_html(text: str) -> str:
    """Convert inline HTML we know to markdown; strip the rest of common prose tags."""
    # <strong>/<b>
    text = re.sub(r"<\s*/?\s*b\b[^>]*>",  "**", text, flags=re.I)
    text = re.sub(r"<\s*/?\s*strong\b[^>]*>", "**", text, flags=re.I)
    # <em>/<i>
    text = re.sub(r"<\s*/?\s*em\b[^>]*>", "*", text, flags=re.I)
    text = re.sub(r"<\s*/?\s*i\b[^>]*>",  "*", text, flags=re.I)
    # <u>
    text = re.sub(r"<\s*/?\s*u\b[^>]*>",  "",  text, flags=re.I)
    # <br/>
    text = re.sub(r"<\s*/?\s*br\b\s*/?\s*>", "\n", text, flags=re.I)
    # <a href=x ...>text</a>  ->  [text](x)
    def _a(m: "re.Match") -> str:
        href = m.group(1)
        inner = m.group(2).strip()
        # strip any nested tags inside
        inner = re.sub(r"<[^>]+>", "", inner)
        return f"[{inner}]({href})"
    text = re.sub(
        r"<\s*a\b[^>]*?href\s*=\s*['\"]([^'\"]+)['\"][^>]*>(.*?)</\s*a\s*>",
        _a, text, flags=re.I | re.S)
    # remaining inline / block tags: drop tags, keep children
    for tag in ("div", "p", "span", "section", "article", "figure",
                "ul", "ol", "li", "sup", "sub", "code", "pre",
                "table", "thead", "tbody", "tr", "td", "th"):
        text = re.sub(rf"<\s*/?\s*{tag}\b[^>]*>", "", text, flags=re.I)
    return text


def _clean_text_chunk(text: str) -> str:
    """Clean one prose chunk. Fence chunks never reach here."""
    # 1) imports / exports (line-level)
    text = _IMPORT_EXPORT_RE.sub("", text)

    # 2) <Link> / <OutboundLink> pairs -> keep the text, drop the tag (href
    #    points into the web backend, keep the readable label).
    text = _LINK_PAIR_RE.sub(lambda m: m.group(2).strip(), text)

    # 3) Other paired PascalCase (rare in this corpus) -> keep text.
    def _keep_children(m: "re.Match") -> str:
        # grab everything after the first > and before the closing tag
        s = m.group(0)
        inner = s[s.index(">") + 1 : s.rindex("</")]
        return inner
    text = _PAIR_RE.sub(_keep_children, text)

    # 4) Self-closing PascalCase components: drop entirely.
    text = _SELF_CLOSE_RE.sub("", text)

    # 5) Prose-level JSX expression lines (e.g. `props.location.pathname`)
    #    left over after the above (should be rare).
    text = _JAVASCRIPT_PROPS_RE.sub("", text)

    # 6) Inline HTML -> markdown / stripped.
    text = _clean_inline_html(text)

    # 7) Line-level `{...}` prose (a brace-group that is its own line): drop.
    text = "\n".join(
        ln for ln in text.split("\n") if not _BRACE_PROSE_RE.match(ln))

    # Whitespace tidy (don't collapse legitimate newlines more than 2).
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip("\n")


def clean_mdx(body: str) -> str:
    """Body -> clean markdown content. Fences keep content, comments drop."""
    out: List[str] = []
    for kind, text in tokenize_mdx(body):
        if kind == "text":
            cleaned = _clean_text_chunk(text)
            if cleaned:
                out.append(cleaned)
        elif kind == "fence":
            # Preserve the fence verbatim (minus trailing newlines).
            out.append(text.rstrip("\n"))
        # "comment" is dropped
    # Join with a blank line between blocks for readable markdown.
    return "\n\n".join(out).strip()


# ==========================================================================
# IMAGE EXTRACTOR  (spec 2.4 — narrow, relative local, effective-body)
# ==========================================================================


# A Markdown image ref: ![alt](dest). `dest` may contain **literal parentheses**
# (real filenames like `IP-Cam-Viewer_Android_(1).png` or
# `1,8mm_with_IRfilter_(940nm).png`) and an optional "title", so a naive
# `[^)]+` regex truncates it. We match the `![alt](` opener with a regex and
# then scan forward counting paren depth to find the *markdown* close-paren
# (the one whose depth returns to the opener's level), which is the balanced
# approach and is immune to literal `(...)` inside the path.


def _iter_markdown_image_dests(text: str):
    """Yield the raw `dest` string of every `![alt](...)` in text.

    Handles literal parens inside the path and an optional trailing `title`.
    A ref never spans a newline (the corpus is single-line).
    """
    i = 0
    while True:
        m = re.search(r"!\[[^\]]*\]\(", text[i:])
        if not m:
            break
        start = i + m.end()          # index just after the opening '('
        depth = 0
        j = start
        while j < len(text):
            c = text[j]
            if c == "\n":
                break                # not a valid (single-line) ref
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == -1:      # this ')' closes the opener we matched
                    break
            j += 1
        if depth != -1:              # unterminated; skip to avoid infinite loop
            i = start
            continue
        yield text[start:j].strip()
        i = j + 1


def _split_dest_title(dest: str) -> str:
    """Strip a trailing ` "title"` / `'title'` from a dest, if present."""
    m = re.match(r"^(\S.*?)\s+[\"']([^\"']*)[\"']$", dest)
    if m:
        return m.group(1).strip()
    return dest.strip()


# Paths we do NOT treat as local relative:
_PREFIX_DENY = (
    "http://", "https://", "data:", "mailto:", "ftp://", "ftps://",
    "//", "/", "mailto:",
)
_TEMPLATE_RE = re.compile(r"\{\{.*?\}\}")


def _is_in_scope(ref: str) -> bool:
    """Per spec 2.4: reject absolute, remote, anchor, and template refs."""
    if not ref:
        return False
    for p in _PREFIX_DENY:
        if ref.startswith(p):
            return False
    if ref.startswith("#"):  # anchor-only
        return False
    if _TEMPLATE_RE.search(ref):
        return False
    return True


def extract_image_refs(
    body: str,
    article_dir: Path,
    data_root: Path,
) -> List[Tuple[str, Optional[Path], Optional[str]]]:
    """Scan the effective body for in-scope image refs.

    Returns a list of (original_ref, resolved_abs_path_or_None, reason).
    - original_ref: the dest as it appears in the MDX (e.g. "./x.png")
    - resolved_abs_path: absolute path to the file, or None if in-scope but missing
    - reason: None when the ref was in-scope & valid; otherwise a short code:
        "rejected" (absolute/remote/anchor/template) -- NOT reported,
        "missing"  (in-scope but file not found),
        "not-file" (resolved, exists, but not a regular file),
        "unsupported-ext" (local file, not a decodable image type).

    The *order* of in-scope refs is what determines `image_refs`; refs dropped
    by rejection are not reported (spec 2.4: "Ignore ... do not count as an
    error"). Only in-scope-but-defective refs are reported (per 3.0).

    Note: the dest is parsed with a *balanced-paren* scanner (_iter_markdown_
    image_dests), not a `[^)]+` regex, so filenames that contain literal
    `(...)` are not truncated and mis-reported as missing.
    """
    # effective body: strip fences + comments
    body_eff = _strip_code(body)
    out: List[Tuple[str, Optional[Path], Optional[str]]] = []
    for raw_dest in _iter_markdown_image_dests(body_eff):
        original_ref = _split_dest_title(raw_dest)
        if not _is_in_scope(original_ref):
            continue
        resolved = (article_dir / original_ref).resolve()
        if not resolved.exists():
            out.append((original_ref, None, "missing"))
            continue
        if not resolved.is_file():
            out.append((original_ref, resolved, "not-file"))
            continue
        ext = resolved.suffix.lower()
        if ext not in IMAGE_EXTS:
            out.append((original_ref, resolved, "unsupported-ext"))
            continue
        out.append((original_ref, resolved, None))
    return out


def _strip_code(text: str) -> str:
    """Remove fenced blocks and HTML comments, preserving the rest."""
    out: List[str] = []
    for kind, chunk in tokenize_mdx(text):
        if kind == "text":
            out.append(chunk)
    return "".join(out)


# ==========================================================================
# IMAGE NORMALIZER (spec 2.2 -- C3) and ROW BUILDER (spec 5)
# ==========================================================================


def normalize_image(path: Path, max_edge: int, keep_frames: bool) -> Tuple[bytes, bool, bool]:
    """Decode -> single color frame -> optional downscale -> deterministic PNG.

    Returns (png_bytes, was_multi_frame, was_downscaled).
    Raises on undecodable images (spec 2.2: caller drops that entry only).
    Pure function of (source bytes, max_edge) for determinism (spec 6 C5):
    fixed mode (RGBA), fixed resampler (LANCZOS), fixed PNG compresslevel.
    """
    obj = Image.open(path)
    try:
        nframes = getattr(obj, "n_frames", 1) or 1
        was_multi = nframes > 1
        if was_multi and not keep_frames:
            # PIL opens animated images at frame 0; .copy() detaches it into a
            # single 2D image (drops the frame list + disposal metadata).
            try:
                obj.seek(0)
            except (ValueError, EOFError):
                pass
            obj = obj.copy()
        if obj.mode not in ("RGB", "RGBA"):
            obj = obj.convert("RGBA")          # consistent mode (C3)

        w, h = obj.size
        long_edge = max(w, h)
        down = False
        if max_edge and long_edge > max_edge:
            scale = max_edge / float(long_edge)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            obj = obj.resize((nw, nh), Image.LANCZOS)
            down = True

        buf = io.BytesIO()
        obj.save(buf, format="PNG", compresslevel=PNG_COMPRESSLEVEL)
        return buf.getvalue(), was_multi, down
    finally:
        try:
            obj.close()
        except Exception:
            pass


def _first_h1(content: str) -> Optional[str]:
    for line in content.splitlines():
        m = re.match(r"^#\s+(.*\S)\s*$", line.strip())
        if m:
            return m.group(1)
    return None


def _metadata_column(meta: Dict[str, Any]) -> str:
    """All frontmatter keys other than title/path (spec 2.5), as a JSON string.

    Stored as a JSON string (not a nested struct) for clean, unambiguous
    Parquet round-trip -- core columns are unaffected. Documented in the
    report per spec 2.1 ("report whichever representation you chose").
    """
    cleaned = {k: v for k, v in meta.items() if k not in ("title", "path")}
    return json.dumps(cleaned, sort_keys=True, ensure_ascii=False, default=str)


def build_all_rows(
    cfg: Config,
    articles: List[Path],
    image_cache: Dict[str, Tuple[bytes, bool, bool]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Process every article in stable order; return (rows, stats, bad_items, failed).

    `image_cache` is a shared {canonical_path: (png, was_multi, was_down)} map
    so the same file is decoded once and reused per alignment slot (spec 2.4).
    """
    rows: List[Dict[str, Any]] = []
    seen_ids: set = set()
    stats: Dict[str, Any] = {
        "articles_total": len(articles),
        "converted": 0,
        "failed": 0,
        "rows_with_images": 0,
        "text_only_rows": 0,
        "resolved_local_image_refs": 0,
        "unique_image_files_embedded": 0,
        "missing_refs_dropped": 0,
        "corrupt_images_dropped": 0,
        "not_file_dropped": 0,
        "unsupported_ext_dropped": 0,
        "gif_flattened": 0,
        "gif_flattened_failed": 0,
        "images_downscaled": 0,
        "warn_empty_content": {},  # dict: rid -> human-readable reason (spec 2.3 C1 guard)
        "histogram": collections.Counter(),
        "max_images_in_one_article": 0,
    }
    bad_items: List[Dict[str, Any]] = []   # per spec 3.1 detail lines
    failed: List[Tuple[str, str]] = []

    for apath in articles:
        rel = apath.relative_to(cfg.data)
        rel_str = str(rel).replace(os.sep, "/")
        rid = article_id_for(rel_str, seen_ids)
        seen_ids.add(rid)
        try:
            text = apath.read_text(encoding="utf-8")
        except Exception as e:  # spec 3.0: record + drop item only
            stats["failed"] += 1
            failed.append((rel_str, f"read: {e}"))
            continue

        meta, body = parse_frontmatter(text)
        content = clean_mdx(body)
        # C1 guard (spec 2.3): cleaned content is final — never resurrect the
        # raw body, which is exactly the implementation detail the cleaner must
        # remove (imports, JSX components). If cleaning leaves no/near-no prose
        # while the source had a body, NAME the row so the report lists it for
        # a human to inspect (spec 2.3 C1 guard, 2nd sentence; gate C1 escape
        # hatch per spec 3.2).
        if body.strip() and (not content.strip() or len(content.strip()) < 40):
            clen = len(content.strip())
            if clen == 0:
                stats["warn_empty_content"][rid] = (
                    "source body had no prose beyond imports/UI components; cleaned "
                    "content is empty — embedder falls back to `text` "
                    "(SPEC_Model_Embedding.md §5.2). Awaiting human review.")
            else:
                stats["warn_empty_content"][rid] = (
                    f"near-empty cleaned content ({clen} chars < 40) — flagged per "
                    "spec 2.3 C1 guard for human review.")

        refs = extract_image_refs(body, apath.parent, cfg.data)
        image_refs: List[str] = []
        images: List[bytes] = []
        for oref, rpath, reason in refs:
            if reason == "missing":
                stats["missing_refs_dropped"] += 1
                bad_items.append({"id": rid, "article": rel_str,
                                  "ref": oref, "resolved": None, "error": "missing"})
                continue
            if reason in ("not-file", "unsupported-ext"):
                if reason == "not-file":
                    stats["not_file_dropped"] += 1
                else:
                    stats["unsupported_ext_dropped"] += 1
                bad_items.append({"id": rid, "article": rel_str,
                                  "ref": oref, "resolved": str(rpath), "error": reason})
                continue

            key = str(rpath)
            if key in image_cache:
                png_bytes = image_cache[key][0]
            else:
                try:
                    png_bytes, was_multi, was_down = normalize_image(
                        rpath, cfg.max_image_edge, cfg.keep_frames)
                except Exception as e:  # corrupt/undecodable -> drop only (C4 kept)
                    stats["corrupt_images_dropped"] += 1
                    bad_items.append({"id": rid, "article": rel_str,
                                      "ref": oref, "resolved": str(rpath),
                                      "error": f"corrupt: {e}"})
                    continue
                image_cache[key] = (png_bytes, was_multi, was_down)
                if was_multi:
                    stats["gif_flattened"] += 1
                if was_down:
                    stats["images_downscaled"] += 1
            image_refs.append(oref)
            images.append(png_bytes)

        title = (meta.get("title") or "").strip()
        if not title:
            title = _first_h1(content) or apath.parent.name
        source_path = (meta.get("path") or "").strip() or rel_str

        row = {
            "id": rid,
            "title": title,
            "source_path": source_path,
            "text": text,
            "content": content,
            "image_refs": image_refs,
            "images": images,
            "metadata": _metadata_column(meta),
        }
        rows.append(row)
        stats["converted"] += 1
        n = len(images)
        stats["histogram"][n] += 1
        stats["max_images_in_one_article"] = max(stats["max_images_in_one_article"], n)
        if n > 0:
            stats["rows_with_images"] += 1
            stats["resolved_local_image_refs"] += n
        else:
            stats["text_only_rows"] += 1

    stats["unique_image_files_embedded"] = len(image_cache)
    stats["histogram"] = dict(sorted(stats["histogram"].items()))
    return rows, stats, bad_items, failed


# ==========================================================================
# SCHEMA (spec 2.0) + SHARD WRITER (spec 5)
# ==========================================================================

FEATURES = ds.Features({
    "id":          ds.Value("string"),
    "title":       ds.Value("string"),
    "source_path": ds.Value("string"),
    "text":        ds.Value("string"),
    "content":     ds.Value("string"),
    "image_refs":  ds.Sequence(ds.Value("string")),
    "images":      ds.Sequence(ds.Image()),
    "metadata":    ds.Value("string"),   # JSON string of non-title/path frontmatter (spec 2.5)
})

_README_CARD = """\
---
license: other
language:
- en
size_categories:
- 1k<n<10k
task_categories:
- text-retrieval
tags:
- instar
- multimodal
- documents
---

# Instar Docs Corpus -- Stage-1 dataset

Self-contained Hugging Face dataset converted from `./data` by
`build_dataset.py` per `SPEC_Dataset_Creation.md`. **One row per article**
(not per image, not ImageFolder). Load it with:

```python
from datasets import load_dataset
ds = load_dataset("./hf_dataset")["train"]
```

## Columns
| column       | type                 | meaning |
|--------------|----------------------|---------|
| `id`         | string               | stable id derived from the article relative path (spec 6) |
| `title`      | string               | frontmatter title -> first H1 -> dir name |
| `source_path`| string               | frontmatter `path` -> relative path |
| `text`       | string               | pristine original MDX (byte-faithful) |
| `content`    | string               | cleaned fence-aware Markdown |
| `image_refs` | list[string]         | local relative `![]()` refs, document order; `[]` when none |
| `images`     | list[Image]          | RGBA PNG aligned to `image_refs`; `[]` when none (never None) |
| `metadata`   | string (JSON)        | remaining frontmatter keys (author, excerpt, tags, ...) |

## Encoding
- Images: normalized to RGBA before re-encode; animated GIFs flattened to the
  first frame; long edge capped at the build-time `--max-image-edge`;
  re-encoded as PNG with a fixed compresslevel for byte-stable rebuilds.
- Text-only articles store `images = []` and `image_refs = []` (never `None`).
"""


def write_shards(rows: List[Dict[str, Any]], out_dir: Path, shard_rows: int):
    """Write rows as `train` Parquet shards directly under out_dir + README.md.

    Returns (n_shards, [shard paths]). out_dir IS the HF dataset dir (spec 5);
    load_dataset(out_dir) reads these (verified: Sequence(Image) round-trips,
    empty rows read back as []).
    """
    if not rows:
        raise RuntimeError("build produced zero rows; refusing to write an empty dataset")
    n = -(-len(rows) // shard_rows)     # ceil division
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: List[Path] = []
    for k in range(n):
        chunk = rows[k * shard_rows:(k + 1) * shard_rows]
        d = ds.Dataset.from_list(chunk, features=FEATURES)
        fname = f"train-{k:05d}-of-{n:05d}.parquet"
        d.to_parquet(str(out_dir / fname))
        shard_paths.append(out_dir / fname)
    (out_dir / "README.md").write_text(_README_CARD, encoding="utf-8")
    return n, shard_paths


# ==========================================================================
# HARD GATE  (spec 3.2 -- C1/C2/C3/C4)  and HANDOFF SMOKE TEST  (spec 4)
# ==========================================================================


def hard_gate(dataset_dir: Path, expected_total: int,
              warn_empty_ids: Optional[frozenset] = None) -> Tuple[bool, List[str]]:
    """Load `train` from Parquet shards and assert spec 3.2 C1..C4.

    Returns (passed, violations). Any violation -> run exits non-zero.
    C1: content non-empty str on every row, OR the row is explicitly flagged
        warn_empty_content (spec 3.2 escape hatch; human review per spec 2.3)
    C2: images is list (never None) on every row
    C3: every image is a 2D color image, mode in {RGB,RGBA}, n_frames == 1
    C4: len(images) == len(image_refs) on every row
    Extra: row count equals expected (we refuse to ship a partial rebuild).
    """
    violations: List[str] = []
    try:
        dataset = ds.load_dataset(str(dataset_dir), split="train")
    except Exception as e:
        return False, [f"load_dataset failed: {e}"]

    if len(dataset) != expected_total:
        violations.append(
            f"row count mismatch: expected {expected_total} loaded {len(dataset)}")

    warn_empty_ids = warn_empty_ids or frozenset()
    n_img_cells = 0
    i = 0
    for r in dataset:
        # C1 (empty content permitted only for explicitly flagged rows — spec 3.2)
        c = r.get("content")
        if not isinstance(c, str):
            violations.append(f"C1 row={i} id={r.get('id')}: content not str")
        elif not c.strip() and r.get("id") not in warn_empty_ids:
            violations.append(f"C1 row={i} id={r.get('id')}: empty content without warn_empty_content flag")
        # C2
        imgs = r.get("images")
        if not isinstance(imgs, (list, tuple)):
            violations.append(f"C2 row={i} id={r.get('id')}: images is {type(imgs).__name__}, not list/tuple")
            imgs = []
        # C4
        refs = r.get("image_refs")
        if not isinstance(refs, (list, tuple)):
            violations.append(f"C4 row={i} id={r.get('id')}: image_refs is not list/tuple")
            refs = []
        if len(imgs) != len(refs):
            violations.append(f"C4 row={i} id={r.get('id')}: len(images)={len(imgs)} != len(image_refs)={len(refs)}")
        # C3 - decode + mode + n_frames
        for j, im in enumerate(imgs):
            n_img_cells += 1
            try:
                # datasets decodes Image() features to PIL objects on access
                if isinstance(im, Image.Image):
                    pil = im
                else:
                    raw = im.get("bytes") if isinstance(im, dict) else im
                    if not raw:
                        raise ValueError("no image bytes")
                    pil = Image.open(io.BytesIO(raw))
                mode = pil.mode
                nframes = getattr(pil, "n_frames", 1) or 1
                if mode not in ("RGB", "RGBA"):
                    violations.append(f"C3 row={i} id={r.get('id')} img={j}: mode={mode}")
                if nframes != 1:
                    violations.append(f"C3 row={i} id={r.get('id')} img={j}: n_frames={nframes}")
                if not (pil.width > 0 and pil.height > 0):
                    violations.append(f"C3 row={i} id={r.get('id')} img={j}: size zero")
            except Exception as e:
                violations.append(f"C3 row={i} id={r.get('id')} img={j}: {e}")
        i += 1
        if i % 200 == 0:
            di(f"hard gate: checked {i}/{len(dataset)} rows")

    di(f"hard gate: scanned {len(dataset)} rows, {n_img_cells} image cells")
    return (len(violations) == 0), violations


def spec4_smoke_test(dataset_dir: Path,
                     warn_empty_ids: Optional[frozenset] = None) -> Tuple[bool, str]:
    """Spec-4 handoff cross-check: exactly what the embedder will consume."""
    try:
        d = ds.load_dataset(str(dataset_dir), split="train")
    except Exception as e:
        return False, f"load fail: {e}"
    warn_empty_ids = warn_empty_ids or frozenset()
    sample = d.select(range(min(8, len(d))))
    for i, r in enumerate(sample):
        c = r["content"]
        if not isinstance(c, str):
            return False, f"row {i}: content not str"
        # C1 escape hatch (spec 3.2 / 2.3): empty only when explicitly flagged
        if not c.strip() and r["id"] not in warn_empty_ids:
            return False, f"row {i}: content empty / not flagged"
        if r["images"] is None or not isinstance(r["images"], list):
            return False, f"row {i}: images wrong type"
        if len(r["images"]) != len(r["image_refs"]):
            return False, f"row {i}: alignment mismatch"
    # Also assert the specific empty/non-empty round-trip that spec §2.1 flagged
    # as the #1 silent-failure mode:
    first_empty = None
    first_nonempty = None
    for r in sample:
        L = len(r["images"])
        if L == 0 and first_empty is None:
            first_empty = r
        if L > 0 and first_nonempty is None:
            first_nonempty = r
    if first_empty is not None:
        assert isinstance(first_empty["images"], list) and len(first_empty["images"]) == 0
        assert first_empty["images"] is not None
    if first_nonempty is not None:
        assert first_nonempty["images"] is not None and isinstance(first_nonempty["images"], list)
        assert len(first_nonempty["images"]) == len(first_nonempty["image_refs"])
    return True, "ok"


# ==========================================================================
# REPORT  (spec 3.1, line-set is the contract; per-bad-item detail below)
# ==========================================================================


def write_report(cfg: Config, stats: Dict[str, Any], bad_items: List[Dict[str, Any]],
                 failed: List[Tuple[str, str]], gate_ok: bool, gate_violations: List[str],
                 smoke_ok: bool, smoke_msg: str, wall_seconds: float):
    lines: List[str] = []
    L = lines.append
    L("# Stage-1 dataset report (build_dataset.py)")
    L(f"- data: `{cfg.data}`")
    L(f"- out : `{cfg.out}`")
    L(f"- wall: {wall_seconds:.1f}s")
    L(f"- max-image-edge: {cfg.max_image_edge}   keep-frames: {cfg.keep_frames}   "
      f"shard-rows: {cfg.shard_rows}   strict: {cfg.strict}")
    L(f"- encoding: RGBA -> PNG, compresslevel={PNG_COMPRESSLEVEL}, GIF first-frame, "
      f"LANCZOS downscale")
    L("")
    L("## Key numbers")
    L(f"- [DATASET] articles = {stats['articles_total']}   "
      f"(converted {stats['converted']}, failed {stats['failed']})")
    hist = stats["histogram"]
    compact = ", ".join(f"{k}: {v}" for k, v in list(hist.items())[:4])
    if len(hist) > 4:
        compact += f", ... {max(hist)}: {hist[max(hist)]}"
    L(f"- [DATASET] image-count histogram: {{{compact}}}")
    L(f"- [DATASET] max images in one article: {stats['max_images_in_one_article']}")
    L(f"- [DATASET] rows with images: {stats['rows_with_images']}   "
      f"text-only rows (images==[]): {stats['text_only_rows']}")
    L(f"- [DATASET] resolved local image refs: {stats['resolved_local_image_refs']}")
    L(f"  (effective-body, in-scope: relative-local `![]()` refs that resolve to an")
    L(f"  on-disk image file. Abs/remote/`data:`/anchor/`{{{{..}}}}` refs are excluded")
    L(f"  per spec 2.4 and are not counted here. `missing` below = in-scope refs whose")
    L(f"  file is absent -- genuine source defects, enumerated per item below.)")
    L(f"- [DATASET] unique image files embedded: {stats['unique_image_files_embedded']}")
    L(f"- [DATASET] missing image refs dropped: {stats['missing_refs_dropped']}   "
      f"corrupt images dropped: {stats['corrupt_images_dropped']}")
    L(f"- [DATASET] not-file / unsupported-ext dropped: "
      f"{stats['not_file_dropped']} / {stats['unsupported_ext_dropped']}")
    L(f"- [DATASET] gif images flattened to first frame: {stats['gif_flattened']}   "
      f"(failed: {stats['gif_flattened_failed']})")
    L(f"- [DATASET] images downscaled for size: {stats['images_downscaled']}   "
      f"(long-edge cap: {cfg.max_image_edge})")
    wc = stats["warn_empty_content"]
    L(f"- [DATASET] empty/near-empty content rows (named per spec 2.3 C1 guard): {len(wc)}")
    for rid in sorted(wc):
        L(f"  - {rid}: {wc[rid]}")
    L("")
    L("## Hard gate (spec 3.2)")
    L(f"- C1/C2/C3/C4: {'PASS' if gate_ok else 'FAIL'}")
    if not gate_ok:
        for v in gate_violations:
            L(f"  - {v}")
    L(f"- handoff smoke test (spec 4): {'PASS' if smoke_ok else 'FAIL'} ({smoke_msg})")
    L("")
    L("## Missing / dropped image refs (detail)")
    if not bad_items:
        L("- (none)")
    else:
        grouped_missing = [b for b in bad_items if b.get("error") == "missing"]
        other = [b for b in bad_items if b.get("error") != "missing"]
        for b in grouped_missing:
            L(f"- {b['id']}   ref={b['ref']}   resolved={b.get('resolved')}   "
              f"error=missing")
        for b in other:
            L(f"- {b['id']}   ref={b['ref']}   resolved={b.get('resolved')}   "
              f"error={b.get('error')}")
    if failed:
        L("")
        L("## Fatal per-article read failures")
        for rel, err in failed:
            L(f"- {rel}: {err}")
    report = "\n".join(lines) + "\n"
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "articles_total": stats["articles_total"],
        "converted": stats["converted"],
        "failed": stats["failed"],
        "rows_with_images": stats["rows_with_images"],
        "text_only_rows": stats["text_only_rows"],
        "resolved_local_image_refs": stats["resolved_local_image_refs"],
        "unique_image_files_embedded": stats["unique_image_files_embedded"],
        "missing_refs_dropped": stats["missing_refs_dropped"],
        "corrupt_images_dropped": stats["corrupt_images_dropped"],
        "not_file_dropped": stats["not_file_dropped"],
        "unsupported_ext_dropped": stats["unsupported_ext_dropped"],
        "gif_flattened": stats["gif_flattened"],
        "images_downscaled": stats["images_downscaled"],
        "max_images_in_one_article": stats["max_images_in_one_article"],
        "warn_empty_content": stats["warn_empty_content"],
        "histogram": stats["histogram"],
        "hard_gate": gate_ok,
        "hard_gate_violations": gate_violations,
        "smoke_test": smoke_ok,
        "smoke_msg": smoke_msg,
        "max_image_edge": cfg.max_image_edge,
        "png_compresslevel": PNG_COMPRESSLEVEL,
        "wall_seconds": round(wall_seconds, 2),
        "out": str(cfg.out),
    }
    (cfg.out / "REPORT.md.json").write_text(json.dumps(summary, indent=2),
                                           encoding="utf-8")
    return report


# ==========================================================================
# MAIN
# ==========================================================================


def parse_args(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser(
        description="Stage-1 dataset builder (SPEC_Dataset_Creation.md).")
    p.add_argument("--data", type=Path, default=Path("./data"),
                   help="corpus root (default ./data)")
    p.add_argument("--out", type=Path, default=Path("./hf_dataset"),
                   help="HF dataset output dir (default ./hf_dataset)")
    p.add_argument("--max-image-edge", type=int, default=4096, dest="max_image_edge",
                   help="long-edge cap before re-encode; 0=uncapped (default 4096)")
    p.add_argument("--keep-frames", action="store_true", dest="keep_frames",
                   help="debug only; keep multi-frame (breaks embedder; unsupported by C3)")
    p.add_argument("--shard-rows", type=int, default=256, dest="shard_rows",
                   help="rows per Parquet shard (default 256)")
    p.add_argument("--strict", action="store_true",
                   help="fail the run on *any* dropped image (default: record + continue)")
    p.add_argument("--seed", type=int, default=None,
                   help="reserved for future random-sampling determinism; unused today")
    p.add_argument("--limit", type=int, default=0, dest="limit",
                   help="extra flag (dry-run convenience): build only the first N "
                        "articles; 0 = all (default). Not a spec-required flag.")
    a = p.parse_args(argv)

    if a.keep_frames:
        dw("--keep-frames set: multi-frame image output is expected to break the "
           "embedder and is unsupported per C3; this flag is for debugging only.")
    if a.seed is not None:
        dw(f"--seed={a.seed} reserved for future random sampling; ignored in this build.")
    return Config(
        data=a.data, out=a.out,
        max_image_edge=a.max_image_edge, keep_frames=a.keep_frames,
        shard_rows=a.shard_rows, strict=a.strict, limit=a.limit,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    if not cfg.data.is_dir():
        de(f"--data path is not a directory: {cfg.data}")
        return 2
    t0 = time.time()
    di(f"starting (data={cfg.data}, out={cfg.out})")

    articles = discover_articles(cfg.data)
    if not articles:
        de("no index.mdx found under --data")
        return 2
    if cfg.limit and cfg.limit > 0:
        articles = articles[:cfg.limit]
        dw(f"--limit={cfg.limit}: building only the first {len(articles)} articles "
           "(dry-run affordance; run without --limit for the full dataset)")
    di(f"articles to build (sorted by rel path): {len(articles)}")

    image_cache: Dict[str, Tuple[bytes, bool, bool]] = {}
    rows, stats, bad_items, failed = build_all_rows(cfg, articles, image_cache)

    if cfg.strict and (stats["missing_refs_dropped"] or stats["corrupt_images_dropped"]
                       or stats["not_file_dropped"] or stats["unsupported_ext_dropped"]
                       or failed):
        de("--strict: dropped a defective item; aborting")
        write_report(cfg, stats, bad_items, failed,
                     gate_ok=False, gate_violations=["--strict violation"],
                     smoke_ok=False, smoke_msg="aborted", wall_seconds=time.time() - t0)
        return 1

    # Ensure a clean output dir (remove prior shards/README so load_dataset
    # reads exactly this build, not leftover files).
    if cfg.out.exists():
        for name in os.listdir(cfg.out):
            pth = cfg.out / name
            if name.endswith(".parquet") or name in ("dataset_info.json",):
                if pth.is_file():
                    pth.unlink()
    n_shards, shard_paths = write_shards(rows, cfg.out, cfg.shard_rows)
    di(f"wrote {n_shards} Parquet shard(s); {sum(1 for _ in shard_paths)} shard file(s) under {cfg.out}")

    warn_empty_ids = frozenset(stats["warn_empty_content"])
    gate_ok, gate_violations = hard_gate(cfg.out, expected_total=len(rows), warn_empty_ids=warn_empty_ids)
    if gate_ok:
        di("hard gate (C1..C4): PASS")
    else:
        di(f"hard gate (C1..C4): FAIL ({len(gate_violations)} violation(s))")
        for v in gate_violations[:50]:
            de(f"gate: {v}")

    smoke_ok, smoke_msg = spec4_smoke_test(cfg.out, warn_empty_ids=warn_empty_ids)
    if smoke_ok:
        di("handoff smoke test (spec 4): PASS")
    else:
        de(f"handoff smoke test (spec 4): FAIL ({smoke_msg})")

    wall = time.time() - t0
    write_report(cfg, stats, bad_items, failed, gate_ok, gate_violations,
                 smoke_ok, smoke_msg, wall)
    di(f"report -> {cfg.out / 'REPORT.md'} and {cfg.out / 'REPORT.md.json'}")
    di(f"done in {wall:.1f}s; exit={'0' if (gate_ok and smoke_ok) else '1'}")
    return 0 if (gate_ok and smoke_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
