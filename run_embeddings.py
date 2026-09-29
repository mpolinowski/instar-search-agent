#!/usr/bin/env python3
"""run_embeddings.py — Stage 2: frozen Nemotron-VL embedding index over ./hf_dataset.

Implements SPEC_Model_Embedding.md (the authoritative stage-2 spec):

  §1  environment / device report
  §2  exact model+processor (nvidia/llama-nemotron-embed-vl-1b-v2,
      revision 5b5ca69c35bf6ec1484d2d5ff238626e67a745e2, bfloat16, sdpa,
      trust_remote_code) with an EXPLICIT multi-image strategy (§2 caveat)
  §3  dataset via load_dataset("./hf_dataset")
  §4  verification gate — hard-fail BEFORE any model loading
  §5  batched, resumable, OOM-guarded embedding loop -> .safetensors
  §6  retrieval smoke test (in-domain, out-of-domain, image query)
  §7  final report (markdown + machine-readable JSON)

This is a frozen-model vector-cache build. There is NO training: no optimizer,
no loss, no .backward(), no model.train(). All inference under
torch.inference_mode().

Multi-image strategy (§2, decided after reading the model source):
  NATIVE MULTI-IMAGE IS NOT SUPPORTED by this model. The processor's
  load_image() accepts a single PIL image / path string / dict — a list of
  images per document raises `ValueError: Invalid image` (verified at runtime).
  Each document supports exactly one <image> slot tiled to max_input_tiles
  (6) + 1 thumbnail, 256 vision tokens per patch. Therefore strategies:
    primary — embed images[0] (first in document order) + full text
    concat  — resize the row's images into a grid canvas (fixed cell px),
              then embed that single canvas + full text
  Default: primary (rationale: the model can only see ~6-7 tiles per document,
  so a 53-image article would be indistinguishable mush in one canvas; the
  first in-document image + the full article text is the stronger signal for
  a wiki-style corpus where C1 makes text the primary channel). Both lose
  images[1:]; that limitation is logged and reported.

Exit codes: 0 = green; 1 = verification gate failed; 2 = embedding failure
(unrecoverable OOM / row mismatch / stale-file refusal); 3 = retrieval FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Spec constants (SPEC_Model_Embedding.md §2 — take them from there, don't
# paraphrase a different model)
# ---------------------------------------------------------------------------
EMBED_MODEL_PATH = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBED_COMMIT_HASH = "5b5ca69c35bf6ec1484d2d5ff238626e67a745e2"
EXPECTED_DIM = 2048  # notebook-observed embedding dim; assert it (spec §2)
MODALITY_TOKENS: Dict[str, int] = {"image": 2048, "image_text": 10240, "text": 8192}
EMBEDDING_KEY = "image_text_embeddings"  # final-file key per spec §5.4 example

DEFAULT_DATASET = "./hf_dataset"
DEFAULT_MODEL_DIR = "models/llama-nemotron-embed-vl-1b-v2-5b5ca69"  # local snapshot of EMBED_COMMIT_HASH
DEFAULT_TAG = "instar_docs_v1"

# Model download logic (user requirement): skip when the local snapshot already
# exists; otherwise try huggingface.co FIRST, then fall back to a mirror when
# the primary is unreachable (this host cannot reach huggingface.co directly).
# Both endpoints are env-overridable so a different proxy can be plugged in.
HF_PRIMARY_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HF_FALLBACK_ENDPOINT = os.environ.get("HF_FALLBACK_ENDPOINT", "https://hf-mirror.com")


class StageFailure(Exception):
    """Raised for hard, named failures; carries the intended exit code."""

    def __init__(self, stage: str, message: str, code: int):
        super().__init__(message)
        self.stage = stage
        self.code = code


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def error(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)


def log(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}", flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file_head(path: str, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# §1 environment / device report
# ---------------------------------------------------------------------------
def env_report() -> Dict[str, Any]:
    cuda_ok = torch.cuda.is_available()
    out: Dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_visible": cuda_ok,
        "device": "cuda" if cuda_ok else "cpu",
        "torch_threads": torch.get_num_threads(),
    }
    if cuda_ok:
        free, total = torch.cuda.mem_get_info()
        out.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_free_bytes": int(free),
                "vram_total_bytes": int(total),
            }
        )
    try:
        import transformers

        out["transformers"] = transformers.__version__
    except Exception as ex:  # pragma: no cover - reporting only
        out["transformers_error"] = str(ex)
    try:
        import datasets

        out["datasets"] = datasets.__version__
    except Exception as ex:  # pragma: no cover - reporting only
        out["datasets_error"] = str(ex)
    log("EMBED", f"env: {json.dumps(out, sort_keys=True)}")
    return out


def pick_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}", flush=True)  # spec §1 exact line
    return device


# ---------------------------------------------------------------------------
# §4 verification gate — MUST pass before any model loading (§4: a hard check
# failure prints [VERIFY] FAIL and exits non-zero BEFORE any model loading)
# ---------------------------------------------------------------------------
MIN_CONTENT_CHARS = 40
IMAGE_SAMPLE_CAP = 400  # §4.3: bounded image-integrity sample


def first_heading(sample_text: str) -> str:
    for line in sample_text.splitlines():
        s = line.strip()
        if s.startswith("#") or s:
            return s[:96]
    return ""


def _rederive_article_order(data_root: str) -> List[str]:
    """Re-derive the expected article order + IDs INDEPENDENTLY (SPEC_Dataset_Creation.md §6).

    Mirrors build_dataset.py: sort index.mdx files by UTF-8/codepoint relative
    path; id = relpath minus trailing index.mdx (parts joined with '/');
    collision-safe `base#sha1[:8]` suffix.
    """
    from pathlib import Path

    root = Path(data_root).resolve()
    found = [p for p in root.rglob("index.mdx") if p.is_file()]
    found.sort(key=lambda p: p.relative_to(root).as_posix().encode("utf-8"))
    seen: set = set()
    ids: List[str] = []
    for p in found:
        rel = p.relative_to(root).as_posix()
        parts = [x for x in rel.split("/") if x]
        if parts and parts[-1] == "index.mdx":
            parts = parts[:-1]
        base = "/".join(parts) or "root"
        rid = f"{base}#{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:8]}" if base in seen else base
        seen.add(rid)
        ids.append(rid)
    return ids


def verify_dataset(train, data_root: Optional[str] = None) -> Dict[str, Any]:
    """Spec §4 checks. Returns the report block; raises StageFailure(1) on FAIL."""
    n = len(train)
    failures: List[str] = []
    warnings: List[str] = []

    if n == 0:
        raise StageFailure("VERIFY", "dataset has zero rows", 1)

    # --- §4.1 shape & counts -------------------------------------------------
    img_counts: Dict[int, int] = {}
    max_img = 0
    total_imgs = 0
    for i in range(n):
        imgs = train[i]["images"]
        if imgs is None:  # C2: never None
            failures.append(f"row {i}: images is None")
            continue
        k = len(imgs)
        img_counts[k] = img_counts.get(k, 0) + 1
        max_img = max(max_img, k)
        total_imgs += k

    if max_img > 30:
        big = [i for i in range(n) if len(train[i]["images"] or []) >= 30]
        log("VERIFY", f"anomalously large image rows (>=30 imgs): {len(big)} — longest {max_img} (expected long tail per SPEC_Dataset_Creation.md)")

    hist = {
        "0": img_counts.get(0, 0),
        "1": img_counts.get(1, 0),
        "2": img_counts.get(2, 0),
        "3": img_counts.get(3, 0),
    }
    hist["4+"] = sum(v for k, v in img_counts.items() if k >= 4)
    log("VERIFY", f"rows={n}  images_total={total_imgs}")
    log("VERIFY", f"image-count histogram {hist}")
    log("VERIFY", f"max images in one row: {max_img}")

    # --- §4.2 text integrity --------------------------------------------------
    # Spec §4.2: the hard check is "content (or text) non-empty"; rows with
    # empty or <40-char `content` are WARNINGS, named (not failures). The
    # embed loop falls back to `text` for such rows (spec §5.2 skeleton:
    # `r["content"] or r["text"]`).
    empty_rows: List[Tuple[int, str]] = []
    short_rows: List[Tuple[int, str]] = []
    for i in range(n):
        row = train[i]
        content = row["content"]
        if not isinstance(content, str) or len(content) == 0:
            if isinstance(row.get("text"), str) and row["text"].strip():
                empty_rows.append((i, row["id"]))
            else:
                failures.append(f"row {i} (id={row['id']}): content AND text empty (spec §4.2)")
        elif len(content) < MIN_CONTENT_CHARS:
            short_rows.append((i, row["id"]))
    if empty_rows:
        shown = ", ".join(f"{i}={rid}" for i, rid in empty_rows[:5])
        warnings.append(f"empty-content rows (embedded from `text`, spec §5.2): {len(empty_rows)} (first: {shown})")
    if short_rows:
        shown = ", ".join(f"{i}={rid}" for i, rid in short_rows[:5])
        warnings.append(f"short-content rows (<{MIN_CONTENT_CHARS} chars): {len(short_rows)} (first: {shown})")
    n_ok = n - len(empty_rows)
    log("VERIFY", f"empty-content rows: {len(empty_rows)}   rows with content: {n_ok}")
    for i in sorted({0, n // 2, n - 1}):
        r = train[i]
        log(
            "VERIFY",
            f"sample row {i} -> id={r['id']}, title={r['title']!r}, "
            f"n_images={len(r['images'] or [])}, content_len={len(r['content'])}, "
            f"first_line={first_heading(r['content'])!r}",
        )

    # --- §4.3 image integrity (bounded sample) + §4.3 alignment (ALL rows) ----
    corrupt = 0
    bad_mode = 0
    checked = 0
    stride = max(1, (total_imgs + IMAGE_SAMPLE_CAP - 1) // max(1, IMAGE_SAMPLE_CAP))
    seen_img = 0
    for i in range(n):
        r = train[i]
        imgs = r["images"]
        refs = r["image_refs"]
        if imgs is None or refs is None:
            failures.append(f"row {i}: images/image_refs is None")
            continue
        if len(imgs) != len(refs):
            failures.append(f"row {i} (id={r['id']}): len(images)={len(imgs)} != len(image_refs)={len(refs)} (C4)")
            continue
        for t, im in enumerate(imgs):
            if seen_img % stride != 0:
                seen_img += 1
                continue
            seen_img += 1
            checked += 1
            if im is None:
                corrupt += 1
                warnings.append(f"row {i} img {t} (ref={refs[t]!r}): image is None in sample")
                continue
            try:
                if im.width <= 0 or im.height <= 0:
                    corrupt += 1
                    warnings.append(f"row {i} img {t} (ref={refs[t]!r}): zero dimensions")
                    continue
                if im.mode not in ("RGB", "RGBA"):
                    bad_mode += 1
                    warnings.append(f"row {i} img {t} (ref={refs[t]!r}): mode={im.mode} (expected RGB/RGBA per C3)")
            except Exception as ex:
                corrupt += 1
                failures.append(f"row {i} img {t} (ref={refs[t]!r}): {type(ex).__name__}: {ex}")
    log("VERIFY", f"images sample-checked: {checked}   corrupt/missing: {corrupt}   non-color modes: {bad_mode}")

    # --- §4.4 determinism (C5) — re-derive the order per SPEC_Dataset_Creation.md §6
    ids = list(train["id"])
    if len(set(ids)) != len(ids):
        failures.append(f"duplicate ids: {len(ids) - len(set(ids))} duplicates (C5 collision-safety violation)")
    check44 = "rederived"
    if data_root is None:
        warnings.append("determinism re-derivation skipped: --data not provided (spec §4.4)")
        check44 = "skipped(no --data)"
    elif not os.path.isdir(data_root):
        warnings.append(f"determinism re-derivation skipped: --data dir {data_root!r} not found (spec §4.4)")
        check44 = "skipped(no data dir)"
        log("VERIFY", f"note: data dir {data_root!r} not found; C5 check limited to id uniqueness")
    else:
        expected = _rederive_article_order(data_root)
        if len(expected) < len(ids):
            failures.append(f"re-derived article count {len(expected)} < dataset rows {len(ids)} (C5/§4.4)")
        else:
            # the dataset must be EXACTLY the first len(ids) articles of the
            # re-derived order (full corpus when --limit is off) — spec §6 order
            prefix = expected[: len(ids)]
            if ids == prefix:
                mode = "full order" if len(expected) == len(ids) else f"prefix of {len(expected)}-article order (--limit subset)"
                log("VERIFY", f"determinism check: id order matches independent re-derivation — {mode} (SPEC_Dataset_Creation.md §6)")
            else:
                first_diff = next(((i, e, x) for i, (e, x) in enumerate(zip(prefix, ids)) if e != x), "none-in-common-prefix")
                failures.append(f"row order does not match re-derivation (C5/§4.4); first divergence at row {first_diff[0]}: re-derived={first_diff[1]!r} vs dataset={first_diff[2]!r}")

    report: Dict[str, Any] = {
        "rows": n,
        "images_total": total_imgs,
        "histogram": hist,
        "max_images_in_row": max_img,
        "empty_content_rows": len(empty_rows),
        "short_content_rows": len(short_rows),
        "images_sample_checked": checked,
        "corrupt_or_missing_images": corrupt,
        "non_color_modes": bad_mode,
        "align_failures": [f for f in failures if "C4" in f],
        "determinism_check": check44,
        "warnings": warnings,
        "failures": failures,
    }
    if failures:
        log("VERIFY", f"FAIL ({'; '.join(failures[:5])})")
        raise StageFailure("VERIFY", "dataset verification failed", 1)
    log("VERIFY", "PASS")
    return report


# ---------------------------------------------------------------------------
# §2 model + processor load (exact path/revision from SPEC_Model_Embedding.md)
# ---------------------------------------------------------------------------
def _local_model_files(model_path: str) -> Tuple[Optional[str], Optional[str]]:
    """If model_path is a local snapshot dir, return (dir, model.safetensors)."""
    if os.path.isdir(model_path):
        mt = os.path.join(model_path, "model.safetensors")
        if os.path.isfile(mt):
            return model_path, mt
    return None, None


def _snapshot_present(model_dir: str) -> bool:
    """True when the local snapshot dir holds the weight + config files needed to load."""
    if not os.path.isdir(model_dir):
        return False
    return all(os.path.isfile(os.path.join(model_dir, n)) for n in ("model.safetensors", "config.json"))


def ensure_model_snapshot() -> str:
    """Guarantee the local model snapshot (DEFAULT_MODEL_DIR) exists; return its path.

    Spec: SPEC_Model_Embedding.md §2 pins the exact model + revision. Download
    policy: SKIP the download when the snapshot already exists on disk;
    otherwise try huggingface.co first, then the mirror
    (https://hf-mirror.com, override $HF_FALLBACK_ENDPOINT) if the primary is
    unreachable. `snapshot_download(local_dir=...)` is idempotent: existing
    files are verified/reused and only missing files are fetched, so a partial
    snapshot is completed rather than re-downloaded.
    """
    if _snapshot_present(DEFAULT_MODEL_DIR):
        log("EMBED", f"model snapshot present at {DEFAULT_MODEL_DIR} — skipping download (reuse local)")
        return DEFAULT_MODEL_DIR
    from huggingface_hub import snapshot_download

    candidates = [HF_PRIMARY_ENDPOINT]
    primary = HF_PRIMARY_ENDPOINT.rstrip("/")
    fallback = HF_FALLBACK_ENDPOINT.rstrip("/")
    if HF_FALLBACK_ENDPOINT and fallback != primary:
        candidates.append(HF_FALLBACK_ENDPOINT)
    if os.path.exists(DEFAULT_MODEL_DIR):
        warn(f"existing dir {DEFAULT_MODEL_DIR} is an incomplete snapshot — will fill missing files")
    last_err: Optional[BaseException] = None
    for endpoint in candidates:
        t0 = time.perf_counter()
        try:
            snapshot_download(
                EMBED_MODEL_PATH,
                revision=EMBED_COMMIT_HASH,
                local_dir=DEFAULT_MODEL_DIR,
                endpoint=endpoint,
            )
        except BaseException as ex:  # noqa: BLE001 — try the next endpoint, then fail loudly
            last_err = ex
            warn(f"model download from {endpoint} failed ({type(ex).__name__}: {str(ex)[:300]}) — trying next endpoint")
            continue
        log("EMBED", f"downloaded {EMBED_MODEL_PATH} @ {EMBED_COMMIT_HASH} from {endpoint} in "
                      f"{time.perf_counter() - t0:.0f}s -> {DEFAULT_MODEL_DIR}")
        return DEFAULT_MODEL_DIR
    detail = f"{type(last_err).__name__}: {str(last_err)[:300]}" if last_err else "unknown error"
    raise StageFailure(
        "EMBED",
        f"could not download model snapshot {EMBED_MODEL_PATH} @ {EMBED_COMMIT_HASH} "
        f"from any endpoint: {'; '.join(candidates)} — last error: {detail}",
        2,
    )


def load_embed_model(model_path: str, revision: str, device: str, modality: str) -> torch.nn.Module:
    from transformers import AutoModel

    log_path, model_tf = _local_model_files(model_path)
    hub_style = log_path is None
    effective_revision = revision if hub_style else None
    log(
        "EMBED",
        f"loading model path={model_path} revision={effective_revision or '(local snapshot; revision baked into files)'} "
        f"dtype=bfloat16 attn=sdpa trust_remote_code=True",
    )
    t0 = time.perf_counter()
    kwargs: Dict[str, Any] = dict(
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if effective_revision:
        kwargs["revision"] = effective_revision
    model = AutoModel.from_pretrained(model_path, **kwargs)
    # device_map="accelerate" unavailable here (no accelerate on this host); a
    # single 3.4 GB bf16 model fits on one GPU or CPU — move it manually.
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    log("EMBED", f"model loaded in {time.perf_counter() - t0:.1f}s | params={n_params/1e6:.1f}M | device={model.device if hasattr(model, 'device') else device} | dtype={next(p.dtype for p in model.parameters())}")

    # The model builds its OWN processor from config.name_or_path inside
    # modeling_llama_nemotron_vl.py __init__ — the separately created
    # AutoProcessor (as in the notebook) is NOT the one encode_* uses. The
    # notebook wires the modality setting via `embed_model.processor.p_max_length
    # = modality_to_tokens[modality]`; we do exactly that (SPEC_Model_Embedding.md §2, notebook cell 13).
    p = model.processor
    p.p_max_length = MODALITY_TOKENS[modality]
    p.max_input_tiles = 6
    p.use_thumbnail = True
    p.q_max_length = getattr(p, "q_max_length", 512)
    if hub_style or effective_revision:
        log("EMBED", f"model source: hub '{model_path}' @ {effective_revision}")
    else:
        log("EMBED", f"model source: local snapshot of {EMBED_MODEL_PATH} @ {EMBED_COMMIT_HASH} ({model_path})")
    log(
        "EMBED",
        f"processor config: p_max_length={p.p_max_length} (set from modality '{modality}') "
        f"max_input_tiles={p.max_input_tiles} use_thumbnail={p.use_thumbnail} "
        f"num_image_token={getattr(p, 'num_image_token', 'n/a')} pooling={model.config.pooling} "
        f"img_context_token_id={model.config.img_context_token_id}",
    )
    return model


# ---------------------------------------------------------------------------
# §5.3 multi-image handling (explicit strategy, §2 caveat) — logged once per run
# ---------------------------------------------------------------------------
STRATEGY_RATIONALE = (
    "primary: model processor accepts exactly ONE image per document "
    "(processing_llama_nemotron_vl.py load_image() rejects lists; one <image> "
    "slot, max_input_tiles=6 + thumbnail, 256 vision tokens/patch). Images are "
    "the secondary channel for this wiki corpus (SPEC_Dataset_Creation.md C1: "
    "text carries the primary signal), and a canvas of up to 53 images would be "
    "squashed into the same 6-tile budget and lose everything. Strategy is "
    "documented, logged, and configurable per SPEC_Model_Embedding.md §2/§7."
)


def concat_images(images: Sequence[Image.Image], cell: int) -> Image.Image:
    """Grid-canvas fallback strategy: pack N images into one canvas (spec §2(B))."""
    n = len(images)
    cols = max(1, math.isqrt(n) + (0 if math.isqrt(n) ** 2 == n else 1))
    rows = math.ceil(n / cols)
    canvas = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for k, im in enumerate(images):
        tile = im.convert("RGB").resize((cell, cell), Image.LANCZOS)
        canvas.paste(tile, ((k % cols) * cell, (k // cols) * cell))
    return canvas


def row_image_input(
    images: Sequence[Image.Image], strategy: str, cell: int, row_id: str
) -> Any:
    """Reduce a row's 0..N images to the model's one-image-per-document slot.

    Returns a PIL image, or "" (the model's text-only marker) for 0-image rows.
    """
    if len(images) == 0:
        return ""
    if strategy == "primary":
        return images[0]
    if strategy == "concat":
        return concat_images(images, cell)
    if strategy == "native":
        return list(images)  # deliberately passed through to surface the model's own error
    raise ValueError(f"unknown --multi-image strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# §5 batched, resumable embedding loop (NO training — inference only)
# ---------------------------------------------------------------------------
def _row_text(row: Dict[str, Any]) -> str:
    t = row.get("content") or row.get("text") or ""
    if not t:
        raise StageFailure("EMBED", f"row {row.get('id')}: empty text after gate — impossible", 2)
    return t


def _is_oom(exc: BaseException) -> bool:
    types: Tuple[type, ...] = (MemoryError,)
    oc = getattr(torch.cuda, "OutOfMemoryError", None)
    if oc is not None:
        types = types + (oc,)
    if isinstance(exc, types):
        return True
    s = str(exc).lower()
    return "out of memory" in s or "oom" in s


def _atomic_save(payload: Dict[str, torch.Tensor], final_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(final_path)) or ".", exist_ok=True)
    tmp = final_path + ".tmp"
    save_file(payload, tmp)
    os.replace(tmp, final_path)


def encode_range(
    model: torch.nn.Module,
    train,
    start: int,
    end: int,
    modality: str,
    strategy: str,
    cell: int,
    batch: int,
    device: str,
) -> Tuple[torch.Tensor, int]:
    """Embed rows [start, end). Returns (tensor [end-start, dim], min_batch_seen)."""
    chunks: List[torch.Tensor] = []
    i = start
    eff = max(1, batch)
    min_seen = eff
    while i < end:
        j = min(i + eff, end)
        idxs = list(range(i, j))
        if modality == "image":
            texts: Optional[List[str]] = None
            images = [row_image_input(train[k]["images"] or [], strategy, cell, train[k]["id"]) for k in idxs]
        elif modality == "text":
            images = None
            texts = [_row_text(train[k]) for k in idxs]
        else:
            images = [row_image_input(train[k]["images"] or [], strategy, cell, train[k]["id"]) for k in idxs]
            texts = [_row_text(train[k]) for k in idxs]
        try:
            with torch.inference_mode():
                e = model.encode_documents(images=images, texts=texts)
        except Exception as ex:
            if _is_oom(ex) and eff > 1:
                eff = max(1, eff // 2)
                warn(f"OOM during batch of {j - i} rows -> halving effective batch to {eff} for subsequent chunks (spec §5.2)")
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise
        if e.shape[0] != len(idxs):
            raise StageFailure("EMBED", f"row mismatch in encode: got {e.shape[0]} for {len(idxs)} rows", 2)
        if e.shape[-1] != EXPECTED_DIM:
            raise StageFailure("EMBED", f"unexpected dim {e.shape[-1]} (expected {EXPECTED_DIM})", 2)
        chunks.append(e.detach())
        min_seen = min(min_seen, eff)
        i = j
    out = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    return out.to(torch.bfloat16), min_seen


def _parts_dir(embed_file: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(embed_file)), ".parts." + os.path.basename(embed_file))


def _sidecar_path(embed_file: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(embed_file)) or ".", ".progress.json")


def _part_path(embed_file: str, k: int) -> str:
    return os.path.join(_parts_dir(embed_file), f"part_{k:05d}.safetensors")


def _part_rows_for(embed_file: str, k: int, part_rows: int, num_rows: int) -> int:
    start = k * part_rows
    return max(1, min(part_rows, num_rows - start))


def _config_key(args: Dict[str, Any], num_rows: int, dataset_path: str, dataset_head_sha: str) -> str:
    """Stable identity of what a finished file must represent (spec §5.5/§5.1)."""
    body = json.dumps(
        {
            "embed_file": os.path.abspath(args["embed_file"]),
            "modality": args["modality"],
            "multi_image": args["multi_image"],
            "batch_size": args["batch_size"],
            "part_rows": args["part_rows"],
            "image_cell": args["image_cell"],
            "limit": args["limit"],
            "model_path": args["model_path"],
            "model_revision": args["model_revision"],
            "dataset": os.path.abspath(dataset_path),
            "dataset_head_sha": dataset_head_sha,
            "num_rows": num_rows,
            "expected_dim": EXPECTED_DIM,
        },
        sort_keys=True,
    )
    return sha256_bytes(body.encode("utf-8"))


def _write_sidecar(embed_file: str, doc: Dict[str, Any]) -> None:
    p = _sidecar_path(embed_file)
    ptmp = p + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    with open(ptmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(ptmp, p)


def _read_sidecar(embed_file: str) -> Optional[Dict[str, Any]]:
    p = _sidecar_path(embed_file)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as ex:
        warn(f"sidecar unreadable ({ex}); treating run as fresh")
        return None


def _dataset_head_sha(train, n: int) -> str:
    r0, rl = train[0], train[n - 1]
    h = hashlib.sha256()
    for val in (str(n), str(r0["id"]), str(r0["content"]), str(rl["id"]), str(rl["content"])):
        b = val.encode("utf-8", "strict")
        h.update(str(len(b)).encode("ascii"))  # length-prefix each field
        h.update(b)
    return h.hexdigest()[:32]


def build_or_reuse_embeddings(
    model: torch.nn.Module,
    train,
    device: str,
    a: argparse.Namespace,
    ck: str,
) -> Dict[str, Any]:
    """Spec §5.1/§5.2/§5.4: reuse a valid file, else embed parts, resume, merge.

    Returns a stats dict for the report.
    """
    embed_file = a.embed_file
    num_rows = len(train)
    t_start = time.perf_counter()
    reused = False
    resumed_from = 0

    if os.path.exists(embed_file) and not a.force_regenerate:
        emb = load_file(embed_file)[EMBEDDING_KEY]
        if emb.shape != (num_rows, EXPECTED_DIM):
            raise StageFailure(
                "EMBED",
                f"stale embedding file {embed_file}: shape {tuple(emb.shape)} != expected ({num_rows}, {EXPECTED_DIM}). "
                "Re-run with --force-regenerate (spec §5.1: refuse to reuse a mismatched file).",
                2,
            )
        log("EMBED", f"existing file valid: {embed_file} shape={tuple(emb.shape)} dtype={emb.dtype} — reusing (spec §5.1)")
        reused = True
    elif os.path.exists(embed_file) and a.force_regenerate:
        os.remove(embed_file)
        shutil.rmtree(_parts_dir(embed_file), ignore_errors=True)
        sc = _sidecar_path(embed_file)
        if os.path.exists(sc):
            os.remove(sc)
        log("EMBED", f"--force-regenerate: removed {embed_file} + partials + sidecar; re-embedding from scratch")

    if not reused:
        sidecar = _read_sidecar(embed_file)
        if sidecar and sidecar.get("config_key") == ck and sidecar.get("status") == "partial":
            done = int(sidecar.get("completed_parts", 0))
            total_parts = math.ceil(num_rows / a.part_rows)
            if done > total_parts:
                done = total_parts
            resumed_from = done
            log("EMBED", f"resuming: {done}/{total_parts} parts already on disk (spec §5.2)")
        else:
            if sidecar is not None and sidecar.get("config_key") != ck:
                warn("sidecar config mismatch — starting a fresh run (old partials will be replaced)")
            shutil.rmtree(_parts_dir(embed_file), ignore_errors=True)
            log("EMBED", f"fresh run: {num_rows} rows, batch={a.batch_size}, part_rows={a.part_rows}, part-size target={a.part_rows * EXPECTED_DIM * 2 / 1048576:.1f} MiB")

        total_parts = math.ceil(num_rows / a.part_rows)
        doc: Dict[str, Any] = {
            "config_key": ck,
            "embed_file": os.path.abspath(embed_file),
            "modality": a.modality,
            "multi_image": a.multi_image,
            "batch_size": a.batch_size,
            "part_rows": a.part_rows,
            "num_rows": num_rows,
            "completed_parts": 0,
            "status": "partial",
            "model_path": a.model_path,
            "model_revision": a.model_revision,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_sidecar(embed_file, doc)

        oom_halvings = 0
        for k in range(total_parts):
            pp = _part_path(embed_file, k)
            s, e = k * a.part_rows, min((k + 1) * a.part_rows, num_rows)
            if k < resumed_from and os.path.exists(pp):
                existing = load_file(pp)[EMBEDDING_KEY]
                if existing.shape != (e - s, EXPECTED_DIM):
                    warn(f"part {k} on disk has shape {tuple(existing.shape)} != ({e - s}, {EXPECTED_DIM}); re-embedding")
                else:
                    log("EMBED", f"part {k + 1}/{total_parts}: already embedded ({e - s} rows) — skipping")
                    continue
            row0 = train[s] if e - s > 0 else None
            log("EMBED", f"part {k + 1}/{total_parts}: rows {s}..{e - 1} (first id={row0['id'] if row0 else 'n/a'})")
            t0 = time.perf_counter()
            part_emb, min_batch = encode_range(model, train, s, e, a.modality, a.multi_image, a.image_cell, a.batch_size, device)
            if min_batch < a.batch_size:
                oom_halvings += 1
            _atomic_save({EMBEDDING_KEY: part_emb.to(torch.bfloat16)}, pp)
            doc["completed_parts"] = k + 1
            _write_sidecar(embed_file, doc)
            log("EMBED", f"part {k + 1}/{total_parts}: done in {time.perf_counter() - t0:.1f}s (min eff batch {min_batch}) cum={time.perf_counter() - t_start:.1f}s")

        # --- §5.4 assembly & save ---------------------------------------------
        log("EMBED", "assembling final file ...")
        parts_list = []
        for k in range(total_parts):
            parts_list.append(load_file(_part_path(embed_file, k))[EMBEDDING_KEY])
        emb = torch.cat(parts_list, dim=0)
        assert emb.shape[0] == num_rows, f"row mismatch: {emb.shape[0]} vs {num_rows}"
        if emb.shape[-1] != EXPECTED_DIM:
            raise StageFailure("EMBED", f"final dim {emb.shape[-1]} != {EXPECTED_DIM}", 2)
        _atomic_save({EMBEDDING_KEY: emb.to(torch.bfloat16)}, embed_file)
        size_b = os.path.getsize(embed_file)
        log("EMBED", f"final: shape={tuple(emb.shape)} dtype={emb.dtype} bytes={size_b} path={embed_file} wall={time.perf_counter() - t_start:.1f}s")
        shutil.rmtree(_parts_dir(embed_file), ignore_errors=True)
        doc["status"] = "finalized"
        doc["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        doc["wall_seconds"] = round(time.perf_counter() - t_start, 1)
        _write_sidecar(embed_file, doc)

    # --- reload check (spec §8: "reloads cleanly") ----------------------------
    chk = load_file(embed_file)[EMBEDDING_KEY]
    if tuple(chk.shape) != (num_rows, EXPECTED_DIM):
        raise StageFailure("EMBED", f"reload check failed: {tuple(chk.shape)}", 2)
    if not torch.isfinite(chk.float()).all():
        raise StageFailure("EMBED", "reload check: non-finite values in embedding file", 2)
    log("EMBED", f"reload check OK: shape={tuple(chk.shape)} dtype={chk.dtype} finite=True")
    return {
        "reused_existing": reused,
        "resumed_parts": resumed_from,
        "num_rows": num_rows,
        "dim": EXPECTED_DIM,
        "dtype": "bfloat16",
        "wall_seconds": None if reused else round(time.perf_counter() - t_start, 1),
        "embed_file": embed_file,
        "bytes": os.path.getsize(embed_file),
    }


# ---------------------------------------------------------------------------
# §6 retrieval smoke test (acceptance test for the whole stage)
# ---------------------------------------------------------------------------
def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # fp32 cosine: model output is bfloat16 and a stored file may be fp32 —
    # unify dtype so the matmul operands always match; fp32 is the numerically
    # sound choice for a similarity score.
    x = x.to(torch.float32)
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def match_query_to_embeddings(
    query: Any, model: torch.nn.Module, target_embeddings: torch.Tensor, top_k: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Notebook `match_query_to_embeddings`, verbatim semantics (spec §6)."""
    with torch.inference_mode():
        if isinstance(query, Image.Image):
            q = model.encode_documents(images=[query])
        else:
            q = model.encode_queries([query])
    # cosine on the model's device (spec §6's notebook computes on DEVICE; the
    # stored file loads on cpu while the live query embedding is on the model device)
    dev = q.device
    cos = _l2_normalize(q) @ _l2_normalize(target_embeddings.to(dev)).T
    cos = cos.flatten()
    if not torch.isfinite(cos).all():
        raise StageFailure("RETRIEVAL", "non-finite cosine scores", 3)
    top_k = min(top_k, cos.numel())
    sorted_idx = torch.argsort(cos, descending=True)[:top_k]
    return cos[sorted_idx][:top_k], sorted_idx


def retrieval_smoke(
    model: torch.nn.Module,
    train,
    target: torch.Tensor,
    a: argparse.Namespace,
) -> Dict[str, Any]:
    n = len(train)
    top5 = 5
    probes: List[Dict[str, Any]] = [
        {
            "kind": "in-domain",
            "query": "How do I configure the MQTT broker on my Instar camera for Home Assistant",
            "keywords": ["mqtt", "home assistant", "hivemq", "mosquitto"],
        },
        {
            "kind": "in-domain",
            "query": "Set up an FTP server for motion detection recordings on a router",
            "keywords": ["ftp", "motion detection", "filezilla", "mycloud", "cuteftp"],
        },
        {"kind": "out-of-domain", "query": "banana bread recipe with sour cream", "keywords": []},
    ]

    results: List[Dict[str, Any]] = []
    in_top1: List[float] = []
    out_top1: List[float] = []
    fail_reasons: List[str] = []
    for p in probes:
        scores, idx = match_query_to_embeddings(p["query"], model, target, top_k=a.top_k)
        top = [int(i) for i in idx.tolist()[:top5]]
        rows = [
            {
                "rank": r + 1,
                "index": int(i),
                "id": train[int(i)]["id"],
                "title": train[int(i)]["title"],
                "score": round(float(s), 4),
            }
            for r, (i, s) in enumerate(zip(idx.tolist()[:top5], scores.tolist()))
        ]
        t1 = float(scores[0])
        kw_hit = None
        if p["keywords"]:
            for r in rows:
                hay = f"{r['id']} {r['title']}".lower()
                hit = next((kw for kw in p["keywords"] if kw in hay), None)
                if hit:
                    kw_hit = hit
                    break
        entry = {**p, "top": top, "top1_score": t1, "top5": rows, "keyword_hit": kw_hit}
        results.append(entry)
        log("RETRIEVAL", f"{p['kind']}: {p['query']!r} -> top5={rows}")
        if p["kind"] == "in-domain":
            in_top1.append(t1)
            if kw_hit is None:
                warn(f"in-domain query keyword plausibility: no known article keyword in top5 (eyeball this in the report)")
        else:
            out_top1.append(t1)

    # image query: first row (in row order) with at least one image
    img_row = next((i for i in range(n) if len(train[i]["images"] or []) > 0), None)
    image_result: Optional[Dict[str, Any]] = None
    if img_row is None:
        fail_reasons.append("no image row available for the image query")
    else:
        q_img = train[img_row]["images"][0]
        scores, idx = match_query_to_embeddings(q_img, model, target, top_k=a.top_k)
        rank_of_source = int([i for i in idx.tolist()].index(img_row)) + 1 if img_row in [int(i) for i in idx.tolist()] else -1
        top = [int(i) for i in idx.tolist()[:top5]]
        image_result = {
            "kind": "image",
            "source_row": img_row,
            "source_id": train[img_row]["id"],
            "top": top,
            "top1_score": round(float(scores[0]), 4),
            "source_rank_top_k": rank_of_source,
            "ok": True,
        }
        log(
            "RETRIEVAL",
            f"image (from row {img_row}, id={train[img_row]['id']}): top5={top} top1={image_result['top1_score']}",
        )

    out_ok = bool(out_top1) and bool(in_top1) and max(out_top1) < min(in_top1)
    if not out_ok:
        fail_reasons.append(
            f"out-of-domain top1 {max(out_top1) if out_top1 else 'n/a'} not lower than in-domain top1 {min(in_top1) if in_top1 else 'n/a'}"
        )
    passed = not fail_reasons
    summary = {
        "passed": passed,
        "fail_reasons": fail_reasons,
        "out_of_domain_top1": max(out_top1) if out_top1 else None,
        "in_domain_top1": min(in_top1) if in_top1 else None,
        "probes": results,
        "image_query": image_result,
    }
    log("RETRIEVAL", "PASS" if passed else f"FAIL ({'; '.join(fail_reasons)})")
    if not passed:
        raise StageFailure("RETRIEVAL", "; ".join(fail_reasons), 3)
    return summary


# ---------------------------------------------------------------------------
# §7 final summary report (markdown + machine-readable JSON)
# ---------------------------------------------------------------------------
def write_report(
    report_path: str,
    ctx: Dict[str, Any],
    verification: Dict[str, Any],
    loop_stats: Dict[str, Any],
    retrieval: Dict[str, Any],
) -> None:
    a: argparse.Namespace = ctx["args_ns"]
    md: List[str] = []
    md.append(f"# Stage-2 embedding report — {os.path.basename(a.embed_file)}")
    md.append("")
    md.append("## 1. Environment")
    for k in ("torch", "transformers", "datasets", "cuda_visible", "device", "vram_free_bytes", "vram_total_bytes", "gpu_name", "torch_threads"):
        if k in ctx["env"]:
            md.append(f"- {k}: {ctx['env'][k]}")
    md.append("")
    md.append("## 2. Model & strategy")
    md.append(f"- model path: `{a.model_path}`" + (f" @ `{a.model_revision}`" if not os.path.isdir(a.model_path) else " (local snapshot of " + EMBED_MODEL_PATH + " @ " + EMBED_COMMIT_HASH + ")"))
    md.append(f"- modality: `{a.modality}` (p_max_length={MODALITY_TOKENS[a.modality]})")
    md.append(f"- multi-image strategy: **{a.multi_image}** — {STRATEGY_RATIONALE}")
    md.append("- no training: frozen model, `torch.inference_mode()` only (SPEC_Model_Embedding.md §0/§5.6)")
    md.append("")
    md.append("## 3. Dataset verification (spec §4)")
    v = verification
    md.append(f"- rows: {v['rows']} | images total: {v['images_total']} | max images in one row: {v['max_images_in_row']}")
    md.append(f"- histogram: {json.dumps(v['histogram'])}")
    md.append(f"- empty-content rows: {v['empty_content_rows']} | short-content (<{MIN_CONTENT_CHARS}): {v['short_content_rows']}")
    md.append(f"- images sample-checked: {v['images_sample_checked']} | corrupt/missing: {v['corrupt_or_missing_images']} | non-color modes: {v['non_color_modes']}")
    md.append(f"- alignment (len(images)==len(image_refs)) failures: {len(v['align_failures'])}")
    md.append(f"- gate: PASS")
    for w in v["warnings"][:10]:
        md.append(f"  - WARN {w}")
    md.append("")
    md.append("## 4. Embedding loop (spec §5)")
    md.append(f"- rows: {loop_stats['num_rows']} | dim: {loop_stats['dim']} | dtype: {loop_stats['dtype']}")
    md.append(f"- reused existing file: {loop_stats['reused_existing']} | resumed parts: {loop_stats['resumed_parts']}")
    md.append(f"- wall time: {loop_stats['wall_seconds']} s | file: `{loop_stats['embed_file']}` ({loop_stats['bytes']} bytes)")
    md.append(f"- batch size: {a.batch_size} (auto-halves on OOM) | part rows: {a.part_rows}")
    md.append("")
    md.append("## 5. Retrieval (spec §6)")
    r = retrieval
    md.append(f"- in-domain top-1 score: {r['in_domain_top1']} | out-of-domain top-1: {r['out_of_domain_top1']} (out < in required)")
    if r.get("image_query"):
        iq = r["image_query"]
        md.append(f"- image query: source row {iq['source_row']} (id={iq['source_id']}) top1={iq['top1_score']} source_rank_in_top_k={iq['source_rank_top_k']}")
    for p in r["probes"]:
        md.append(f"- **{p['kind']}**: {p['query']!r} -> " + ", ".join(f"#{x['rank']} {x['id']} ({x['score']})" for x in p["top5"]) + (f" | keyword hit: {p['keyword_hit']}" if p.get("keyword_hit") else ""))
    md.append("")
    md.append(f"## Result: [RETRIEVAL] {'PASS' if r['passed'] else 'FAIL'}")
    if r["fail_reasons"]:
        md.append("reasons: " + "; ".join(r["fail_reasons"]))
    md.append("")
    md.append("## 6. Limitations & assumptions")
    md.append(
        "- Multi-image: only `images[0]` (primary) or the grid canvas (concat) reaches the model — "
        "images[1:] of each row are NOT directly encoded; their surrounding text remains in `content`. "
        "This is a documented, explicit decision per SPEC_Model_Embedding.md §2 ("
        "the model processor rejects per-document image lists; one <image> slot, ≤6 tiles + thumbnail)."
    )
    if a.limit:
        md.append(f"- **--limit {a.limit} active: this run embeds only the first {a.limit} rows — a dry-run file, NOT the full index.** Re-run without --limit for the durable index.")
    if ctx["env"].get("device") == "cpu":
        md.append("- CPU-only run (no visible CUDA device): expected to be slower than GPU per SPEC_Model_Embedding.md §1.")
    md.append(f"- Command: `{json.dumps(ctx['command'])}`")
    md.append("")

    jpath, mpath = ctx["report_json"], report_path
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": a.dataset,
        "embed_file": a.embed_file,
        "num_rows": loop_stats["num_rows"],
        "dim": loop_stats["dim"],
        "dtype": loop_stats["dtype"],
        "modality": a.modality,
        "multi_image": a.multi_image,
        "verification": verification,
        "loop_stats": loop_stats,
        "retrieval": retrieval,
        "env": ctx["env"],
        "command": ctx["command"],
        "result": "PASS" if r["passed"] else "FAIL",
    }
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False, default=str)
    log("EMBED", f"report: {mpath}")
    log("EMBED", f"report (json): {jpath}")


# ---------------------------------------------------------------------------
# entry point — stage flow per SPEC_Model_Embedding.md §7
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage 2: frozen Nemotron-VL embedding index over ./hf_dataset (no training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset dir (spec §3)")
    ap.add_argument("--data", default="./data", help="stage-1 raw source dir, used ONLY to re-derive the C5 row order (spec §4.4)")
    ap.add_argument("--modality", choices=["image_text", "text", "image"], default="image_text")
    ap.add_argument("--multi-image", dest="multi_image", choices=["primary", "concat", "native"], default="primary",
                    help="per-row 0..N image handling (spec §2/§7); 'native' will fail against this model (documented)")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=8, help="auto-halves on OOM (spec §5.2)")
    ap.add_argument("--part-rows", dest="part_rows", type=int, default=128, help="rows per resumable safetensors part (spec §5.2)")
    ap.add_argument("--image-cell", dest="image_cell", type=int, default=512, help="cell px for the 'concat' strategy")
    ap.add_argument("--embed-file", dest="embed_file", default=None,
                    help="default: ./embeddings/<tag>_<modality>.safetensors (spec §5.1)")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="stable dataset tag for the default embed file name")
    ap.add_argument("--force-regenerate", action="store_true", help="ignore existing file (spec §7)")
    ap.add_argument("--top-k", dest="top_k", type=int, default=100, help="retrieval top-k (spec §6)")
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N rows — fast dry-run (spec §7); never the durable index")
    ap.add_argument("--model-path", dest="model_path", default=DEFAULT_MODEL_DIR,
                    help="local snapshot dir (default; auto-downloaded if missing: HF first, then "
                         "$HF_FALLBACK_ENDPOINT mirror — see ensure_model_snapshot) or explicit model/hub id (used as-is)")
    ap.add_argument("--model-revision", dest="model_revision", default=EMBED_COMMIT_HASH, help="hub revision; ignored for a local dir")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    command = [sys.executable, os.path.abspath("run_embeddings.py")] + (list(argv) if argv is not None else sys.argv[1:])
    print(f"[EMBED] command: {' '.join(command)}", flush=True)
    print(f"[EMBED] stage-2 build starts: dataset={a.dataset} modality={a.modality} multi_image={a.multi_image} "
          f"batch_size={a.batch_size} top_k={a.top_k} limit={a.limit}", flush=True)

    if a.multi_image == "native":
        warn("strategy 'native' requests per-document image LISTS; the model processor is KNOWN to reject these "
             "(processing_llama_nemotron_vl.py load_image: 'Invalid image'). The first failing batch will abort the run — "
             "use 'primary' (default) or 'concat' per SPEC_Model_Embedding.md §2.")

    # §1 env/device
    env = env_report()
    device = pick_device()
    if a.multi_image != "native":
        log("EMBED", f"multi-image strategy: {a.multi_image} — {STRATEGY_RATIONALE.split(';')[0]} (spec §2; logged once per spec §5.3)")

    # §3 load dataset (before §4 gate; §4 must hard-fail BEFORE any model loading)
    from datasets import load_dataset

    t0 = time.perf_counter()
    ds = load_dataset(a.dataset)
    train = ds["train"]
    n_full = len(train)
    print(f"[INFO] Number of samples: {n_full}", flush=True)  # spec §3 exact line
    print(f"[INFO] Features: {train.column_names}", flush=True)
    if a.limit is not None:
        if a.limit <= 0:
            raise StageFailure("EMBED", "--limit must be > 0", 2)
        train = train.select(range(min(a.limit, n_full)))
        if a.embed_file is None:
            a.embed_file = f"./embeddings/{a.tag}_{a.modality}_limit{a.limit}.safetensors"
        warn(f"--limit {a.limit}: embedding only the first {len(train)} rows (dry-run; the durable index needs the full {n_full} rows)")
    if a.embed_file is None:
        a.embed_file = f"./embeddings/{a.tag}_{a.modality}.safetensors"

    # §4 gate (hard-fails before model loading)
    verification = verify_dataset(train, data_root=a.data)

    # model — guarantee the default local snapshot exists before loading
    # (skip-if-present / HF-primary / mirror-fallback; an explicit --model-path
    # is the user's own responsibility and is used as-is, no auto-download)
    if a.model_path == DEFAULT_MODEL_DIR:
        ensure_model_snapshot()
    model = load_embed_model(a.model_path, a.model_revision, device, a.modality)

    ck = _config_key(
        {
            "embed_file": a.embed_file,
            "modality": a.modality,
            "multi_image": a.multi_image,
            "batch_size": a.batch_size,
            "part_rows": a.part_rows,
            "image_cell": a.image_cell,
            "limit": a.limit,
            "model_path": a.model_path,
            "model_revision": a.model_revision,
        },
        len(train),
        a.dataset,
        _dataset_head_sha(train, len(train)),
    )

    # §5 loop (load-or-create, resume, OOM guard)
    loop_stats = build_or_reuse_embeddings(model, train, device, a, ck)

    # §6 retrieval smoke test
    target = load_file(a.embed_file)[EMBEDDING_KEY].float()
    if target.shape != (len(train), EXPECTED_DIM):
        raise StageFailure("RETRIEVAL", f"target embeddings shape {tuple(target.shape)} != ({len(train)}, {EXPECTED_DIM})", 3)
    retrieval = retrieval_smoke(model, train, target, a)

    # §7 report
    os.makedirs(os.path.dirname(os.path.abspath(a.embed_file)), exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.embed_file))[0]
    report_md = os.path.join(os.path.dirname(os.path.abspath(a.embed_file)), f"REPORT_{stem}.md")
    report_json = os.path.join(os.path.dirname(os.path.abspath(a.embed_file)), f"REPORT_{stem}.json")
    write_report(
        report_md,
        {
            "env": env,
            "args_ns": a,
            "command": command,
            "report_json": report_json,
        },
        verification,
        loop_stats,
        retrieval,
    )
    log("EMBED", f"stage-2 complete (limit={a.limit} rows): [RETRIEVAL] PASS — file={a.embed_file}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StageFailure as f:
        error(f"{f.stage}: {f}")
        sys.exit(f.code)
    except Exception as ex:  # loud, named failure (no silent shrink — spec §8)
        error(f"UNEXPECTED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        sys.exit(2)
