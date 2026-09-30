# SPEC — Embed a Local Multimodal Dataset & Run the Embedding/Retrieval Loop

> **Audience:** an AI coding agent given this spec to write and run the code described below.
> **Goal:** take a *local* multimodal dataset (already built with the Hugging Face `datasets` library) and run the heavy "training-like" step from the reference notebook — the **batched embedding-generation loop** — producing a durable `.safetensors` embedding index, then prove retrieval works against it.

---

## 0. What is (and is not) "training" here — read first

The reference notebook does **not fine-tune anything**. It has **zero gradients**. The only heavy compute is:

1. For every document, pass its **image(s) + text** through a **frozen** multimodal embedding model (`encode_documents(...)` under `torch.inference_mode`).
2. Collect one embedding vector per document.
3. Persist those vectors to a `.safetensors` file.

Everything after that (query embedding, cosine similarity, top‑k, reranking, generation) is plain vector math + inference on a *different* frozen model.

Treat "run the training loop" as **"run the batched embedding-generation loop with resume + caching + a retrieval smoke test."** Do **not** introduce a `Trainer`, a loss, backprop, or any gradient step. If you find yourself adding `model.train()` or an optimizer, you are wrong — stop.

### Reference notebook
- Local copy: `./research/hugging_face_multimodal_rag_tutorial.ipynb`
- Source: <https://github.com/mrdbourke/learn-huggingface/blob/main/notebooks/hugging_face_multimodal_rag_tutorial.ipynb>
- The cells to reproduce are the ones under **"Load models"**, **"Create/Load embeddings"**, and **"Retrieve samples based on input query"** (see "Exact model & API" below for the precise calls).

### Local data you are working with
- `./data/` — the *raw* source tree: many article dirs, each with one `index.mdx` plus 0 to N image files (`.png`/`.jpg`/`.jpeg`/`.webp`/`.gif`). Roughly **1000+ articles** and **3,500+ images**.
- `./SPEC_Dataset_Creation.md` — the authoritative spec that turns `./data` into a local HF dataset. This spec consumes *its* output; for dataset shape/schema it is the source of truth (AGENTS.md §7 precedence).
- `./research/Prompt_Training.md` — a *previous, mistaken* "fine-tune the LLM" spec. It is superseded by this file. Do **not** follow its fine-tuning framing.
- The HF dataset produced by `./SPEC_Dataset_Creation.md` is expected at `./hf_dataset` (Parquet shards, `train` split), loadable via `load_dataset("./hf_dataset")`.

---

## 1. Environment & setup

Pin or verify the runtime before writing any model code. The notebook's `requirements.txt` is the baseline:

```text
torch>=2.9            # CUDA build if a GPU is available
transformers>=4.57    # must support the Nemotron embed/rerank model + Qwen3-VL
datasets>=4.4
safetensors>=0.7
accelerate>=1.12
Pillow>=12.0
qwen-vl-utils         # only if you include the optional generation step
tqdm
```

Device selection — mirror the notebook:

```python
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")
```

Do the embedding generation on GPU if one exists; on CPU it is orders of magnitude slower but must still work.

Report at the top of the run log: torch version, whether CUDA is visible, `DEVICE`, and free VRAM. If VRAM is small, the batch size in §5 must be reduced before an OOM crash (the notebook starts at `IMAGE_TEXT_BATCH_SIZE = 8`).

---

## 2. Exact model & API — establish before writing embed code

Identify and pin the embedding model exactly as the notebook does. Do **not** substitute a generic vision‑language or text embedding model without flagging it.

From the notebook, the load + processor block is:

```python
from transformers import AutoModel, AutoProcessor

EMBED_MODEL_PATH  = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBED_COMMIT_HASH = "5b5ca69c35bf6ec1484d2d5ff238626e67a745e2"  # allows sdpa, no flash-attn dep

modality_to_tokens = {"image": 2048, "image_text": 10240, "text": 8192}

embed_model = AutoModel.from_pretrained(
    EMBED_MODEL_PATH,
    revision=EMBED_COMMIT_HASH,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    device_map="auto",
).eval()

embed_processor = AutoProcessor.from_pretrained(
    EMBED_MODEL_PATH,
    revision=EMBED_COMMIT_HASH,
    trust_remote_code=True,
    max_input_tiles=6,
    use_thumbnail=True,
    p_max_length=modality_to_tokens["image_text"],   # 10240
)
```

Key facts you must respect:
- **Modality:** the notebook uses `image_text` (combined image+text) because it reports the best performance. Use the same default. Keep `text`-only and `image`-only as selectable fallbacks but do not make them the default.
- **Embedding call for a document** uses `encode_documents(images=[...], texts=[...])`. A **query** uses `encode_queries([query])` for text or `encode_documents(images=[query])` for an image. Keep that asymmetry — it is intentional.
- The model is **frozen** (`.eval()`, `torch.inference_mode`). Never train it.
- Embedding dim in the notebook is `2048`, dtype `bfloat16`. Assert the produced dim matches what you store.
- **Model availability (host constraint, measured on the build host):** this host **cannot reach `huggingface.co`** (connection unreachable). `run_embeddings.py::ensure_model_snapshot()` therefore guarantees the local snapshot before any load: it **skips the download** when `models/llama-nemotron-embed-vl-1b-v2-5b5ca69/` already holds `model.safetensors` + `config.json`; otherwise it downloads the pinned `EMBED_MODEL_PATH @ EMBED_COMMIT_HASH`, trying `huggingface.co` first and falling back to the mirror `https://hf-mirror.com` (both overridable via `$HF_ENDPOINT` / `$HF_FALLBACK_ENDPOINT`; a partial snapshot is completed, not re-downloaded). Verified 2026-09-29: primary failed, mirror delivered the full 3.36 GB snapshot (~303 s at ~11 MB/s), and the local-snapshot load path worked.

**Multi-image caveat (important for your data):** the notebook assumes **exactly one image per example** (`dataset["train"]["image"]` is a single PIL image). Your dataset has **0 to N images per row** (`images: list[Image]`). Before writing the loop, inspect the Nemotron processor/model to determine how it accepts multiple images for one document. Decide one of these and document it:
- (A) the model/processor natively accepts `images=[img1, img2, ...]` for a single text → pass the list; **or**
- (B) it does not → pick a documented strategy (e.g. concatenate/resize subimages into one canvas, or embed the primary image only + full text) and state the limitation explicitly.

Do **not** silently collapse N images into one without reporting which strategy you used and why. If you cannot confirm multi-image support from the model docs/source, default to strategy (B) with a clear log line and make it configurable.

**Decision for this corpus (implemented in `run_embeddings.py`):** strategy **(B)** — the model's processor natively rejects per-document image lists (`processing_llama_nemotron_vl.py load_image()` accepts exactly one PIL image / path / dict; a list raises `ValueError: Invalid image` — verified at runtime). **Default: `primary`** (embed `images[0]` in document order + the full cleaned text; rationale: with `max_input_tiles=6` + 1 thumbnail the model can only see ~6–7 tiles per document, so a 53-image article is indistinguishable mush in one `concat` canvas, while the first in-document image + full text is the stronger signal for this wiki-style corpus). Both strategies lose `images[1:]`; that limitation is logged once per run and reported (see §5.3 / AGENTS.md §5.1). `concat` remains an alternative; `native` is exposed for completeness and will abort the first batch on this model (documented).

---

## 3. Load the local dataset

Load with the standard `datasets` API (this is the whole point of being "packaged to be read by the hugging face datasets library" — no `load_from_disk`-only path, no raw folder access):

```python
from datasets import load_dataset

ds = load_dataset("./hf_dataset")          # -> {"train": Dataset}
train = ds["train"]

print(f"[INFO] Number of samples: {len(train)}")
print(f"[INFO] Features: {train.column_names}")
```

Expected columns (per `SPEC_Dataset_Creation.md` §2.0): `id`, `title`, `source_path`, `text`, `content`, `image_refs` (`list[string]`), `images` (`list[Image]`), plus optional `blocks`/`metadata`.

Row access pattern used throughout:
```python
row = train[42]
row["content"]        # cleaned markdown text
row["images"]         # list[PIL.Image.Image] (may be empty)
row["image_refs"]     # list[str] aligned with images
```

If `./hf_dataset` does not yet exist, **stop and surface that** — building it is governed by `./SPEC_Dataset_Creation.md`, not this spec. Do not invent a dataset on the fly.

---

## 4. Verify the dataset before embedding (mandatory gate)

Embedding is the expensive step. Do **not** run it on an unverified dataset. Run a verification pass and hard-fail (non-zero exit) if the core integrity checks fail. At minimum check and report:

1. **Shape & counts**
   - total rows; must be > 0.
   - distribution of image counts: how many rows have 0 / 1 / 2 / 3 / 4+ images. (Your data is expected to be heavily **0‑ and 1‑image**, some **N‑image**.)
   - max image count seen; flag any anomalously large row.

2. **Text integrity**
   - every row has a non-empty `content` (or `text`) string.
   - sample and print the first heading + `len(content)` for a few deterministic rows (e.g. index 0, a mid row, last row).
   - report any row where `content` is empty or extremely short (< 40 chars) as a warning.

3. **Image integrity** — for a bounded sample (e.g. every Nth row, or up to a cap), open each image:
   - decodes to a valid `PIL.Image`, `.mode` is sane (RGB/RGBA/L), and dimensions are > 0.
   - count: references vs. actually-loadable images; report missing/corrupt image count and the offending `id` + `image_ref`.
   - confirm `len(images) == len(image_refs)` for every row (alignment is a hard requirement, `SPEC_Dataset_Creation.md` C4).

4. **Determinism check**
   - re-derive the `id` ordering and confirm it is sorted/stable per `SPEC_Dataset_Creation.md` C5/§6.

Write the verification output as a compact report block:

```text
[VERIFY] rows=1067  images_total=3483
[VERIFY] image-count histogram {0: 432, 1: 207, 2: 91, 3: 52, 4+: 285}
[VERIFY] empty-content rows: 0   corrupt/missing images: 0
[VERIFY] sample row 42 -> id=..., title=..., n_images=3, content_len=4210
[VERIFY] PASS
```

If any hard check fails, print `[VERIFY] FAIL (...)` and exit non-zero **before** any model loading.

---

## 5. The embedding loop (the "training" step)

This is the one thing that actually burns compute. Port the notebook's **"Create/Load embeddings"** cell to your `list[Image]` dataset. Requirements:

### 5.1 Idempotent load-or-create
- Target file: `./embeddings/<dataset_tag>_image_text.safetensors` (choose a stable `<dataset_tag>`, e.g. `instar_docs_v1`).
- If the file **exists**, load it and go straight to the retrieval smoke test. Verify its row count == `len(train)` and dim == expected; if mismatched, refuse to reuse and ask to regenerate (do not silently trust a stale file).
- If **missing**, run the generation loop below.

### 5.2 Batched generation with resume
- Batch size: start at `BATCH_SIZE = 8`, expose it as a CLI/env flag so it can be lowered on low VRAM.
- Iterate `range(0, len(train), BATCH_SIZE)`. For each chunk, `train.select(indices)` to get the slice, then pull `texts = chunk["content"]` and `images = chunk["images"]` (a list where each element is itself a `list[PIL.Image]` — possibly empty).
- Embed **under `torch.inference_mode()`**, in **bfloat16** on the model's device.
- **Resume:** process chunks in a fixed order and write a small progress sidecar (e.g. `embeddings/.progress.json` with `{"completed_chunks": n, "batch_size": B, "sha_of_model": ..., "sha_of_dataset_head": ...}`). On a restart, skip chunks already present in an appendable/mergeable form. Simplest robust approach: embed in chunk order, save intermediate `.safetensors` per N chunks, and merge at the end. Do not re-embed work already done.
- Guard against OOM: catch `torch.cuda.OutOfMemoryError` (or CUDA OOM), lower the effective batch size by half for subsequent chunks, and log it. Do not crash the whole run over one heavy chunk.

### 5.3 Multi-image handling inside one example
Feed the full `list[PIL.Image]` for that row into the processor for `image_text` modality, preserving the document→images association. Whichever multi-image strategy you chose in §2, apply it **consistently here and at inference time**. Log the strategy name once at loop start.

### 5.4 Assembly & save
- Concatenate chunk embeddings into one tensor of shape `(num_rows, dim)`. Assert `num_rows == len(train)`.
- `save_file({"image_text_embeddings": emb.to(torch.bfloat16)}, path)`.
- Log final shape, dtype, dim, wall time, and bytes written.

### 5.5 Determinism & reproducibility
- Same dataset order (§3/§4), same model revision, same strategy ⇒ same embedding matrix (up to allowed numerical tolerance). Record the model revision + a short hash of the first row's `content` in the sidecar so a later loader can sanity-check compatibility.

### 5.6 No training, ever
- No optimizer, no loss, no `.backward()`, no `model.train()`. This is an offline vector-cache build. If the row count is large, still this is the same "cache embeddings once, reuse forever" idea the notebook states.

Example skeleton (adapt to the §2 multi-image decision):

```python
from safetensors.torch import save_file, load_file
import torch

EMBED_FILE = "./embeddings/instar_docs_v1_image_text.safetensors"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))

if os.path.exists(EMBED_FILE):
    emb = load_file(EMBED_FILE)["image_text_embeddings"].to(DEVICE)
else:
    chunks = []
    for i in tqdm(range(0, len(train), BATCH_SIZE), desc="Embedding image+text"):
        idx  = [j for j in range(i, i + BATCH_SIZE) if j < len(train)]
        chunk = train.select(idx)
        texts  = [r["content"] or r["text"] for r in chunk]
        images = [r["images"] or [None] for r in chunk]   # see §2 multi-image strategy
        with torch.inference_mode():
            e = embed_model.encode_documents(images=images, texts=texts)
        chunks.append(e)
    emb = torch.cat(chunks, dim=0)
    assert emb.shape[0] == len(train), f"row mismatch: {emb.shape[0]} vs {len(train)}"
    save_file({"image_text_embeddings": emb.to(torch.bfloat16)}, EMBED_FILE)
    print(f"[INFO] Saved embeddings {tuple(emb.shape)} -> {EMBED_FILE}")
```

> The skeleton intentionally shows the *shape* of the loop. You must reconcile `images=` with the §2 multi-image decision before running it for real.

---

## 6. Retrieval smoke test (proves the loop actually worked)

After embeddings exist, port the notebook's **`match_query_to_embeddings`** and run a few queries to confirm the index is usable. This is the acceptance test for the whole exercise.

```python
import torch
from PIL import Image

def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

def match_query_to_embeddings(query, target_embeddings, top_k=100):
    with torch.inference_mode():
        if isinstance(query, Image.Image):
            q = embed_model.encode_documents(images=[query])
        else:
            q = embed_model.encode_queries([query])
    cos = _l2_normalize(q) @ _l2_normalize(target_embeddings).T
    cos = cos.flatten()
    sorted_idx = torch.argsort(cos, descending=True)[:top_k]
    return cos[sorted_idx][:top_k], sorted_idx
```

Run at least two deterministic text queries relevant to *this* corpus (e.g. `"Docker installation for the Agent DVR"`, `"MQTT broker setup"`) and print the top‑5 `id` / `title` + cosine score. Also run one **out-of-domain** query (e.g. `"banana bread recipe"`) and confirm it returns *low* scores / unrelated docs — that's a good sign the embeddings encode *this* content, not junk.

Pass criteria (log a `[RETRIEVAL] PASS/FAIL` line):
- Embedding file loads and shape is `(num_rows, dim)`.
- Top‑k for an in-domain query returns docs whose `title`/`content` plausibly match (agent should eyeball and state this in the report).
- Out-of-domain query scores are visibly lower than in-domain.
- A single image query also runs end-to-end (use any image from `train[<i>]["images"]`) and returns a valid ranking.

> This PoC is extended by the interactive TUI — `test_embeddings.py` (§9) —
> which reuses `match_query_to_embeddings` and the model wiring *verbatim*, so
> rankings typed into the TUI are produced by exactly this math.

---

## 7. Deliverables

Produce a single runnable entry point (script or tightly-ordered cells), e.g. `run_embeddings.py`, whose flow is:

```text
env/device report                  (§1)
load local HF dataset              (§3)
verify dataset (hard gate)         (§4)   ← hard-fails BEFORE any model work (per §4)
ensure model snapshot (skip-if-present / HF-primary / mirror-fallback)   (§2 availability note)
load model + processor             (§2)
embed-loop: load-or-create         (§5)
retrieval smoke test               (§6)
final summary report
```
(i.e. the actual `run_embeddings.py` order: dataset → gate → model — the gate never wastes a multi-GB download on a failing dataset.)

CLI/env flags to expose (all present in `run_embeddings.py`, plus the extras marked `+`):
- `--dataset ./hf_dataset` (default)
- `--modality image_text|text|image` (default `image_text`)
- `--batch-size 8` (default; auto-halves on OOM)
- `--embed-file ./embeddings/<tag>_image_text.safetensors`
- `--force-regenerate` (ignore existing file)
- `--multi-image concat|primary|native` (maps to the §2 decision; default `primary` per the decision above)
- `--top-k 100` (retrieval)
- `--limit N` (optionally embed only the first N rows for a fast dry-run)
- `+ --data ./data` (raw source dir, used ONLY to independently re-derive the C5 row order in the §4 gate)
- `+ --tag instar_docs_v1` (stable dataset tag for the default `--embed-file` name)
- `+ --model-path models/llama-nemotron-embed-vl-1b-v2-5b5ca69` (local snapshot dir by default; auto-downloaded if missing per the §2 availability note; an explicit foreign path is used as-is with no auto-download)
- `+ --model-revision 5b5ca69c35bf6ec1484d2d5ff238626e67a745e2` (hub revision; ignored for a local dir)
- `+ --part-rows 128` (rows per resumable intermediate `.safetensors` part)
- `+ --image-cell 512` (cell px for the `concat` strategy)
- `+ env: `HF_ENDPOINT` (primary hub, default `https://huggingface.co`) and `HF_FALLBACK_ENDPOINT` (mirror, default `https://hf-mirror.com`)

The **final summary report** (printed, and written to `./embeddings/REPORT_*.md`) must contain:
1. Device + VRAM + torch/transformers versions.
2. Model path + revision + chosen modality + multi-image strategy (with a one-line rationale).
3. Verification block from §4 (counts, histogram, integrity).
4. Loop stats: rows, dim, dtype, wall time, final file path + size.
5. Retrieval results: top‑5 for each probe query + scores, and the PASS/FAIL line.
6. Any limitations/assumptions (esp. multi-image behavior and anything skipped).

Code quality: type hints, no hardcoded image counts, memory-conscious (no giant list of decoded images held forever), deterministic ordering, graceful per-row image error handling that still lets the row's text be embedded, and `[INFO]/[WARN]/[ERROR]` logging with a final non-zero exit on any hard failure (verification fail, row mismatch, unrecoverable OOM).

---

## 8. Hard acceptance checklist

The task is complete **only** when all of these are true:

- [ ] No gradient step / optimizer / `model.train()` anywhere.
- [ ] Model loaded from the **exact** Nemotron embed path + revision from the notebook (`bfloat16`, `sdpa`, `trust_remote_code`).
- [ ] Dataset loaded via `load_dataset("./hf_dataset")` — not raw folder reads, not `load_from_disk`-only.
- [ ] Dataset verified (counts, text non-empty, images decode, `len(images)==len(image_refs)`) **before** any embedding; hard-fail on failure.
- [ ] Multi-image-per-row behavior is an *explicit, documented* decision (native/concat/primary), not an accident.
- [ ] Embedding loop is batched, resumable, OOM-guarded, deterministic in order.
- [ ] Output `.safetensors` has shape `(num_rows, dim)` and reloads cleanly.
- [ ] At least 1 in-domain + 1 out-of-domain + 1 image retrieval all produce sane rankings.
- [ ] Final report + PASS line exist.

If any item cannot be satisfied (e.g. no GPU and the corpus is too large to embed on CPU in reasonable time), say so explicitly in the report and provide the exact command + expected runtime; do not silently shrink the task.

---

## 9. Stage 3 (optional) — interactive TUI (`test_embeddings.py`)

The §6 PoC is extended into a user-facing terminal UI: **`test_embeddings.py`**.
This is the "manual" Stage-3 row of AGENTS.md §3 — a testing tool, **not** a
build gate: it is read-only over `./hf_dataset` + `./embeddings/`, writes
nothing, re-embeds nothing, and trains nothing (same frozen-model discipline
as §5.6: `torch.inference_mode`, no optimizer/loss/`model.train()`).

**Prerequisite:** Stage 2 green (the §8 checklist holds) — the TUI needs both
`./hf_dataset` and `./embeddings/instar_docs_v1_image_text.safetensors`, and
refuses a stale/mis-shaped index with exit `2` (the §5.1 load-or-reuse gate:
row count and dim must match, never silently trusted).

**Query encoding is identical to §6 by construction.** `test_embeddings.py`
does not reimplement model wiring — it imports and reuses, verbatim, from
`run_embeddings.py`:

| Reused from `run_embeddings.py` | Purpose |
|---|---|
| `EMBED_MODEL_PATH` / `EMBED_COMMIT_HASH` / `EXPECTED_DIM` / `EMBEDDING_KEY` | the pinned §2 model, the §5.4 tensor key |
| `ensure_model_snapshot()` | skip-if-present / HF-primary / mirror-fallback (§2 availability note) — skipped when the default snapshot exists |
| `load_embed_model(...)` | `bfloat16`, `sdpa`, `trust_remote_code`, processor wiring (`p_max_length=10240` for `image_text`, `max_input_tiles=6`, thumbnail on) |
| `match_query_to_embeddings(...)` | the §6 cosine top-k loop — `encode_queries([...])` for text, `encode_documents(images=[...])` for images (the intentional asymmetry) |
| `StageFailure` | hard, named failure carrying the exit code |

**Startup order (fail fast, model last — a setup error never wastes the
multi-GB load):** dataset dir + index file exist → load the dataset
(**string columns only** — `images` is never decoded; article image *counts*
come from the aligned `image_refs` column, C4) → load the index →
`ensure_model_snapshot()` → `load_embed_model(...)` → ready banner.

**Device:** `auto` (default) — CUDA when the torch build sees a GPU, else
CPU (correct, just slower); force with `--device cuda|cpu`.

**TUI commands:**

| Input | Action |
|---|---|
| `<text …>` | text query → top-k table: rank, cosine + score bar, article id, title, image count, snippet |
| `image <path>` | image query (PIL open → first frame → RGB); if `<path>` is not an existing file, the line is treated as a text query (so prose starting with "image …" still searches) |
| `show [rank] [full]` | open the article at that rank of the last query (Markdown rendered, char-capped; `full` lifts the cap) |
| `topk [n]` | show / set results per query (clamped to 1 … row count) |
| `stats` | index / corpus / model / runtime / session facts (incl. image histogram) |
| `history` / `repeat` | last 20 input lines / re-run the last query |
| `clear` · `help`/`?` · `q`/`quit`/`exit` | misc (Ctrl-D leaves, Ctrl-C cancels the line) |

**Exit codes:** `0` clean quit · `1` setup failure (dataset/index/model
missing) · `2` index shape or row-count mismatch (stale file refused, §5.1) ·
`3` a query rejected by the model (logged as `[RETRIEVAL] …`, the REPL keeps
running).

```bash
# repo root, with the §1 venv:
.venv/bin/python test_embeddings.py                # auto: GPU if visible, else CPU
.venv/bin/python test_embeddings.py --top-k 10     # 10 results per query
```

**Logging:** `[TUI] …` lifecycle lines and one `[RETRIEVAL] mode=… top-1=… k=… in …s`
line per query (grep-able, AGENTS.md §5.2).

**Quality note (AGENTS.md §5.2):** the TUI is a manual exploration tool, not an
eval. The §6 smoke test remains the acceptance check — use the TUI to *eyeball*
in-domain / out-of-domain behaviour, not to claim accuracy.
