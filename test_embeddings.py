#!/usr/bin/env python3
"""test_embeddings.py — interactive TUI to query the Stage-2 embedding index.

Stage-3 (optional) companion to run_embeddings.py, per AGENTS.md §3:

    [./embeddings/*.safetensors] ── Stage 3 (optional) ──▶ manual retrieval session

Extends the SPEC_Model_Embedding.md §6 PoC (`_l2_normalize` +
`match_query_to_embeddings`) into an interactive terminal UI:

  * text queries   — encoded with the model's `encode_queries([...])`,
  * image queries  — `image <path>` encoded with `encode_documents(images=[...])`,
  * ranked top-k   — cosine top-k over the full index (default 10 results),
  * `show <rank>`  — read the article behind any ranked row (markdown in place),
  * `stats` / `history` / `topk` / `repeat` — inspect index and session.

Strictly READ-ONLY over ./hf_dataset + ./embeddings: no writes, no
re-embedding, no re-download unless the default model snapshot is missing
(then it is completed, same as run_embeddings.py), and — as in
run_embeddings.py — NO training: no optimizer, no loss, no `model.train()`;
all inference under torch.inference_mode.

Model + processor wiring and the cosine top-k loop are IMPORTED VERBATIM
from run_embeddings.py (`ensure_model_snapshot`, `load_embed_model`,
`match_query_to_embeddings`, and the spec constants), so query encoding is
identical to the §6 retrieval smoke test — same processor config
(`p_max_length` from modality, `max_input_tiles=6`, thumbnail on), same
asymmetry (text → `encode_queries`, image → `encode_documents(images=...)`).

Device: auto by default — CUDA when the torch build sees a GPU, else CPU.
Force with `--device cuda|cpu`. On CPU every query is still correct, just
slower (single-digit seconds per query for a typical 512-token query).

Exit codes (mirror run_embeddings.py):
  0 = clean quit from the TUI
  1 = setup failure (missing dataset / embedding index / model snapshot)
  2 = index shape / row-count mismatch (stale or mis-shaped file refused)
  3 = query/retrieval failure (model rejected the query)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import deque
from typing import Any, Deque, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# --- Stage-2 plumbing, reused VERBATIM (identical query encoding, spec §6) ---
from run_embeddings import (
    DEFAULT_MODEL_DIR,
    EMBED_COMMIT_HASH,
    EMBEDDING_KEY,
    EMBED_MODEL_PATH,
    EXPECTED_DIM,
    StageFailure,
    ensure_model_snapshot,
    load_embed_model,
    match_query_to_embeddings,
)

DEFAULT_EMBED_FILE = "./embeddings/instar_docs_v1_image_text.safetensors"
DEFAULT_DATASET = "./hf_dataset"
PROMPT = "instar> "


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def fatal(console, code: int, msg: str) -> int:
    console.print(f"[bold red][TUI] FATAL:[/bold red] {msg}")
    return code


# ---------------------------------------------------------------------------
# Corpus — read-only view over the HF dataset (string columns only).
# Never touches train["images"] (PIL decode); image COUNTS come from the
# aligned image_refs column (SPEC_Dataset_Creation.md C4 guarantees they
# match `images`), so rows with 53 images stay cheap to inspect.
# ---------------------------------------------------------------------------
def first_heading(content: str) -> str:
    for line in (content or "").splitlines():
        s = line.strip()
        if s.startswith("#"):
            h = s.lstrip("#").strip()
            if h:
                return h
    return ""


def snippet_of(text: str, n: int) -> str:
    s = " ".join((text or "").split())
    if not s:
        return "(no text)"
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


class Corpus:
    """String-column projection of the train split (zero image decoding)."""

    def __init__(self, train):
        names = list(train.column_names)
        n = len(train)
        self.n = n

        def col(name: str) -> Optional[List[Any]]:
            return list(train[name]) if name in names else None

        self.ids: List[str] = col("id") or [f"row-{i}" for i in range(n)]
        self.titles: Optional[List[Optional[str]]] = col("title")
        contents = col("content") if "content" in names else col("text")
        self.contents: List[str] = contents or [""] * n
        self.source_paths: List[str] = col("source_path") or list(self.ids)
        refs = col("image_refs") or [[] for _ in range(n)]
        self.image_counts: List[int] = [len(r) for r in refs]
        self.image_refs: List[List[str]] = refs

    def title(self, i: int) -> str:
        t = (self.titles[i] if self.titles else None)
        if t and str(t).strip():
            return str(t).strip()
        return first_heading(self.contents[i]) or self.ids[i]

    def content(self, i: int) -> str:
        return self.contents[i] or ""


# ---------------------------------------------------------------------------
# CLI + device
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Interactive TUI over the frozen embedding index (Stage 3, "
                    "read-only — no writes, no training). "
                    "Query plumbing is identical to run_embeddings.py §6.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--embed-file", dest="embed_file", default=DEFAULT_EMBED_FILE,
                    help="Stage-2 .safetensors index to query (row count must match the dataset)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset dir for id/title/content (spec §3)")
    ap.add_argument("--model-path", dest="model_path", default=DEFAULT_MODEL_DIR,
                    help="local snapshot dir (auto-completed if missing, like run_embeddings.py) or explicit model/hub id (used as-is)")
    ap.add_argument("--model-revision", dest="model_revision", default=EMBED_COMMIT_HASH,
                    help="hub revision; ignored for a local dir")
    ap.add_argument("--modality", choices=["image_text", "text", "image"], default="image_text",
                    help="processor p_max_length (spec §2); must match the index build")
    ap.add_argument("--top-k", dest="top_k", type=int, default=10, help="results shown per query")
    ap.add_argument("--snippet", dest="snippet_chars", type=int, default=110,
                    help="chars of article text shown in the results table")
    ap.add_argument("--show-chars", dest="show_chars", type=int, default=16000,
                    help="max chars rendered by `show` (0 = unlimited)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto = CUDA when the torch build sees a GPU, else CPU")
    return ap.parse_args(list(argv) if argv is not None else None)


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def print_banner(console, a: argparse.Namespace, corpus: Corpus, target: torch.Tensor, device: str) -> None:
    n = corpus.n
    n_text_only = sum(1 for c in corpus.image_counts if c == 0)
    grid = Table.grid(padding=(0, 2), expand=False)
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("index", f"{a.embed_file}  ({n} × {target.size(1)} {target.dtype}, "
                          f"{os.path.getsize(a.embed_file) / 1e6:.2f} MB)")
    grid.add_row("corpus", f"{a.dataset} — {n} articles · {n_text_only} text-only · "
                           f"{n - n_text_only} with ≥1 image (max {max(corpus.image_counts) if n else 0}/row)")
    grid.add_row("model", f"{a.model_path}  ({EMBED_MODEL_PATH} @ {EMBED_COMMIT_HASH[:7]}) · "
                          f"modality={a.modality} · device={device}")
    grid.add_row("usage", "type a text query · `image <file>` for an image query · `help` for all commands")
    console.print(Panel(grid, title="instar-search-agent — embedding TUI", border_style="cyan"))
    if device == "cpu":
        console.print("[yellow]CPU mode:[/yellow] queries encode on CPU — expect a few seconds per query.")


# ---------------------------------------------------------------------------
# Boot — fail fast (dataset → index → model), so setup errors never touch the
# multi-GB model load (same gate-before-model spirit as run_embeddings.py).
# ---------------------------------------------------------------------------
def boot(a: argparse.Namespace, console) -> "int | Tuple[Corpus, torch.Tensor, Any, str, int]":
    """Returns the exit code on setup failure, else
    (corpus, target [fp32 cpu tensor], model, device, topk)."""
    from datasets import load_dataset
    from safetensors.torch import load_file

    if not os.path.isdir(a.dataset):
        return fatal(console, 1,
                     f"dataset dir not found: `{a.dataset}` — run `build_dataset.py` first (SPEC_Dataset_Creation.md)")
    if not os.path.isfile(a.embed_file):
        return fatal(console, 1,
                     f"embedding index not found: `{a.embed_file}` — run `run_embeddings.py` first "
                     f"(SPEC_Model_Embedding.md §5)")

    with console.status("[bold cyan]loading HF dataset (string columns only, no image decode) …"):
        ds = load_dataset(a.dataset)
        train = ds["train"]
        corpus = Corpus(train)
        if corpus.n == 0:
            return fatal(console, 1, f"dataset is empty: {a.dataset}")

    with console.status("[bold cyan]loading embedding index …"):
        blob = load_file(a.embed_file)
        if EMBEDDING_KEY not in blob:
            return fatal(console, 2, f"`{a.embed_file}` has no tensor `{EMBEDDING_KEY}` (found {list(blob)})")
        target = blob[EMBEDDING_KEY].to(torch.float32).cpu()

    # load-or-reuse gate (spec §5.1 spirit): a mis-shaped/stale file is refused,
    # never silently trusted.
    if target.size(0) != corpus.n:
        return fatal(console, 2,
                     f"row mismatch: index has {target.size(0)} rows, dataset has {corpus.n} — "
                     f"stale index; regenerate with `run_embeddings.py --force-regenerate`")
    if target.size(1) != EXPECTED_DIM:
        return fatal(console, 2,
                     f"dim mismatch: index dim={target.size(1)} != {EXPECTED_DIM} (SPEC_Model_Embedding.md §2)")

    device = resolve_device(a.device)
    if a.device == "auto":
        info(f"device: {device}" + ("" if device == "cuda" else " — no CUDA visible in this torch build; CPU fallback (correct, slower)"))

    if a.model_path == DEFAULT_MODEL_DIR:
        with console.status("[bold cyan]guaranteeing local model snapshot (skip-if-present / HF / mirror) …"):
            ensure_model_snapshot()

    t0 = time.perf_counter()
    with console.status("[bold cyan]loading frozen embed model (~3.4 GB, bfloat16, sdpa) …"):
        model = load_embed_model(a.model_path, a.model_revision, device, a.modality)
    console.print(f"[dim][TUI] model ready in {time.perf_counter() - t0:.1f}s on {device}[/dim]")

    if a.top_k < 1:
        console.print("[yellow][TUI] --top-k must be ≥ 1; clamping to 1[/yellow]")
    topk = max(1, min(a.top_k, corpus.n))
    return corpus, target, model, device, topk


# ---------------------------------------------------------------------------
# Query + rendering
# ---------------------------------------------------------------------------
def open_query_image(path: str) -> Optional[Image.Image]:
    """Open an image query (PIL). GIF → first frame; any mode → RGB,
    mirroring the Stage-1 normalisation (SPEC_Dataset_Creation.md §2.2)."""
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as ex:
        return None


def do_query(
    console,
    model: torch.nn.Module,
    target: torch.Tensor,
    corpus: Corpus,
    query: Any,
    kind: str,
    topk: int,
    snippet_chars: int,
    label: Optional[str] = None,
) -> Optional[Tuple[List[int], List[float]]]:
    """One retrieval over the full index (spec §6 semantics, via run_embeddings)."""
    if label is None:
        label = query if kind == "text" else "<image>"
    t0 = time.perf_counter()
    try:
        scores, idx = match_query_to_embeddings(query, model, target, top_k=topk)
    except StageFailure as f:
        console.print(f"[bold red][RETRIEVAL][/bold red] {f}")
        return None
    except Exception as ex:  # loud, named failure (AGENTS.md §5.2) — never silently skip
        console.print(f"[bold red][RETRIEVAL][/bold red] {type(ex).__name__}: {ex}")
        return None
    elapsed = time.perf_counter() - t0
    sl: List[float] = scores.tolist()
    ii: List[int] = idx.tolist()
    print_results(console, corpus, sl, ii, label, elapsed, snippet_chars)
    top1 = sl[0] if sl else float("nan")
    console.print(
        f"[dim][RETRIEVAL] mode={kind} top-1={top1:.3f} k={len(ii)}/{corpus.n} in {elapsed:.2f}s — "
        f"`show <rank>` opens an article[/dim]"
    )
    return ii, sl


def score_bar(value: float, width: int = 14) -> str:
    """Visible score bar (rich's Bar renders the empty part as spaces, which
    makes 0.32 and 0.78 look identical — we want the empty part to show)."""
    v = 0.0 if value < 0 else (1.0 if value > 1.0 else float(value))
    filled = int(round(v * width))
    return "█" * filled + "░" * (width - filled)


def print_results(console, corpus: Corpus, scores: List[float], idx: List[int],
                  label: str, elapsed: float, snippet_chars: int) -> None:
    label = " ".join(label.split())
    if len(label) > 64:
        label = label[:63] + "…"
    table = Table(
        title=f"top {len(idx)} for {label}  ·  {elapsed:.2f}s",
        title_justify="left",
        header_style="bold dim",
        pad_edge=False,
        expand=False,
    )
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("cos", justify="right", no_wrap=True)
    table.add_column("score", no_wrap=True)
    table.add_column("article (id)", overflow="ellipsis")
    table.add_column("imgs", justify="right", style="dim", no_wrap=True)
    table.add_column("snippet", overflow="fold", ratio=2)
    for r, (s, i) in enumerate(zip(scores, idx), 1):
        s = float(s)
        bar = score_bar(s, 14)
        table.add_row(
            str(r),
            f"{s:.3f}",
            bar,
            corpus.ids[i],
            str(corpus.image_counts[i]),
            snippet_of(corpus.content(i), snippet_chars),
        )
    console.print(table)


def do_show(console, corpus: Corpus, rank: int, a: argparse.Namespace,
            last_ranking: Optional[List[int]], full: bool) -> None:
    """Render the article at `rank` of the LAST query (default rank 1)."""
    if not last_ranking:
        console.print("[yellow]no query yet — type a text query or `image <path>` first[/yellow]")
        return
    if rank < 1 or rank > len(last_ranking):
        console.print(f"[red]rank {rank} out of range (1…{len(last_ranking)} of the last query)[/red]")
        return
    i = last_ranking[rank - 1]
    cap = 0 if full else a.show_chars
    text = corpus.content(i)
    truncated = cap > 0 and len(text) > cap
    if truncated:
        text = text[:cap] + "\n\n…[truncated — run `show " + str(rank) + " full` for the whole article]"
    refs = corpus.image_refs[i]
    ref_note = f" · {len(refs)} image(s): " + " ".join(refs) if refs else ""
    header = (f"#{rank}  {corpus.ids[i]}  —  {corpus.title(i)}\n"
              f"[dim]{corpus.source_paths[i]}{ref_note}[/dim]")
    console.print(Panel(Markdown(text), title=header, border_style="cyan", expand=False))
    console.print(f"[dim]{len(corpus.content(i))} chars total[/dim]")


def image_histogram(image_counts: List[int]) -> str:
    bins = {0: 0, 1: 0, 2: 0, 3: 0, "4+": 0}
    for c in image_counts:
        bins[c if c in (0, 1, 2, 3) else "4+"] += 1
    return "  ".join(f"{k}:{v}" for k, v in bins.items())


def do_stats(console, a: argparse.Namespace, corpus: Corpus, target: torch.Tensor,
             model: torch.nn.Module, device: str, topk: int) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("index", f"{a.embed_file} — {corpus.n} × {target.size(1)} {target.dtype} "
                           f"({os.path.getsize(a.embed_file) / 1e6:.2f} MB)")
    table.add_row("corpus", f"{a.dataset} — {corpus.n} articles · image histogram {{{image_histogram(corpus.image_counts)}}}")
    table.add_row("model", f"{a.model_path} ({EMBED_MODEL_PATH} @ {EMBED_COMMIT_HASH[:7]}, bfloat16, sdpa) "
                           f"· modality={a.modality} · device={device}")
    table.add_row("runtime", f"torch {torch.__version__} · cuda available: {torch.cuda.is_available()}"
                             + (f" · {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    table.add_row("session", f"top-k={topk} · snippet={a.snippet_chars} chars · show cap={a.show_chars or 'off'} chars")
    table.add_row("usage", "read-only over dataset + index · zero image decodes · no training (frozen model)")
    console.print(Panel(table, title="stats", border_style="dim"))


HELP_TEXT = """
**commands**

| command | effect |
|---|---|
| `<text ……>` | text query — everything that isn't a command (encoded via `encode_queries`) |
| `image <path>` | image query for an existing file (encoded via `encode_documents(images=…)`); if the path does not exist the line is treated as a text query |
| `show [rank] [full]` | open the article at that rank of the last query (default 1; `full` skips the char cap) |
| `topk [n]` | show / set results per query (clamped to 1 … row count) |
| `stats` | index, corpus, model and session facts |
| `history` | last 20 input lines |
| `repeat` | re-run the last query |
| `clear` | clear the screen (TTY only) |
| `help` / `?` | this help |
| `q` / `quit` / `exit` | leave (Ctrl-D also leaves; Ctrl-C cancels the current line) |

scores are cosine similarities (≈0 … 1 for this model); the bar column scales to the full 0 … 1 range.
"""


def print_help(console) -> None:
    console.print(Markdown(HELP_TEXT))


def repl(console, a: argparse.Namespace, corpus: Corpus, target: torch.Tensor,
         model: torch.nn.Module, device: str, topk: int) -> int:
    history: Deque[str] = deque(maxlen=20)
    last_ranking: Optional[List[int]] = None
    last_query_line: Optional[str] = None

    def run_text(line: str) -> None:
        nonlocal last_ranking, last_query_line
        res = do_query(console, model, target, corpus, line, "text", topk, a.snippet_chars)
        if res is not None:
            last_ranking = res[0]
            last_query_line = line

    def run_image(line: str, path: str) -> None:
        nonlocal last_ranking, last_query_line
        im = open_query_image(path)
        if im is None:
            console.print(f"[red]could not open image query `{path}` (bad file or unsupported format)[/red]")
            return
        res = do_query(console, model, target, corpus, im, "image", topk, a.snippet_chars, label=f"image {path}")
        if res is not None:
            last_ranking = res[0]
            last_query_line = line

    while True:
        try:
            raw = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        line = raw.strip()
        if not line:
            continue
        history.append(line)
        head, sep, rest = line.partition(" ")
        head_l = head.lower()

        if head_l in ("q", "quit", "exit"):
            break
        if head_l in ("help", "?"):
            print_help(console)
            continue
        if head_l == "clear":
            if console.is_terminal:
                console.clear()
            continue

        if head_l == "topk":
            if not rest:
                console.print(f"top-k is {topk}")
            else:
                try:
                    v = int(rest)
                except ValueError:
                    console.print(f"[red]`topk` expects an integer, got `{rest}`[/red]")
                    continue
                topk = max(1, min(v, corpus.n))
                console.print(f"top-k set to {topk}" + (f" (clamped to {corpus.n} rows)" if v > corpus.n else ""))
            continue

        if head_l == "stats":
            do_stats(console, a, corpus, target, model, device, topk)
            continue

        if head_l == "history":
            for k, h in enumerate(history, 1):
                console.print(f"[dim]{k:>2}[/dim] {h}")
            continue

        if head_l == "repeat":
            if last_query_line is None:
                console.print("[yellow]nothing to repeat yet[/yellow]")
            else:
                if last_query_line.lower().startswith("image ") and os.path.isfile(last_query_line.split(" ", 1)[1].strip()):
                    run_image(last_query_line, last_query_line.split(" ", 1)[1].strip())
                else:
                    run_text(last_query_line)
            continue

        if head_l == "show":
            parts = line.split()
            rank = 1
            full = False
            if len(parts) > 1:
                try:
                    rank = int(parts[1])
                except ValueError:
                    console.print(f"[red]`show` expects `show [rank] [full]`, got `{line}`[/red]")
                    continue
            if len(parts) > 2 and parts[2].lower() == "full":
                full = True
            do_show(console, corpus, rank, a, last_ranking, full)
            continue

        if head_l == "image" and sep:
            path = rest.strip().strip("\"'`")
            if os.path.isfile(path):
                run_image(line, path)
            else:
                # not a file → treat the whole line as a text query
                run_text(line)
            continue

        # default: text query (whole line, verbatim)
        run_text(line)

    console.print("[dim]bye.[/dim]")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    from rich.console import Console

    console = Console()
    res = boot(a, console)
    if isinstance(res, int):
        return res
    corpus, target, model, device, topk = res
    print_banner(console, a, corpus, target, device)
    console.print("[dim][TUI] ready — type a query or `help`[/dim]")
    return repl(console, a, corpus, target, model, device, topk)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as ex:  # loud, named failure (AGENTS.md §5.2) — never silent
        print(f"[ERROR] UNEXPECTED: {type(ex).__name__}: {ex}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
