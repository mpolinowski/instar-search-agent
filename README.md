# instar-search-agent

A **multimodal document-search pipeline** that turns a local wiki-style corpus
(`./data` — 1,067 MDX articles, each with 0 to 53 images) into a
**retrieval-ready vector index**: text queries *and* image queries resolved
against the same frozen multimodal embedding space. No training, no fine-tuning
— a pure vector-cache build over a **frozen** NVIDIA Nemotron vision-language
embedding model.

## I. What this repository does

- **Specification-driven agentic coding.** The code in this repo is a
  byproduct of two machine-checkable contracts — `SPEC_Dataset_Creation.md`
  and `SPEC_Model_Embedding.md` — governed by `AGENTS.md`. Every build step
  has a **named hard gate**, a human report **and** a machine-readable
  `REPORT_*.json`, deterministic exit codes, and byte-identical rebuilds —
  so a second agent (or a CI job) can execute, audit, and re-run the whole
  pipeline without a human in the loop.
- **Multimodal-RAG pipeline.** Stage 1 normalizes MDX + images into a
  standard Hugging Face `datasets` (Parquet) dataset; Stage 2 embeds every
  article (text **and** image) into a 2,048-dim **vector embedding** matrix
  stored in one durable `.safetensors` file.
- **Knowledge base for agentic chat interfaces.** The final index is a
  drop-in knowledge base: any text query or an image query returns a
  ranked list of articles in milliseconds (cosine top-k — verified
  separation: in-domain 0.624 vs. out-of-domain 0.194, image query
  top-1 0.853 on its own row).
- **Nemotron, frozen, offline.** The embedder is
  `nvidia/llama-nemotron-embed-vl-1b-v2` (pinned commit
  `5b5ca69`, bfloat16, `sdpa`), used strictly as a frozen feature extractor
  under `torch.inference_mode` — no optimizer, no loss, no gradient step
  anywhere in the repo.

## II. How to use the repository

### 1. Prepare your corpus — `./data/`

An arbitrary tree of **article folders**; the pipeline never writes here:

```text
data/
└── …any nesting…
    └── Your_Article/
        ├── index.mdx        ← the article (Markdown/MDX, optional frontmatter)
        ├── photo1.png       ← 0..N local images
        └── sub/
            └── deeper_article/
                └── index.mdx + images   ← nesting is fine
```

Format contract the ingestion expects (details: `SPEC_Dataset_Creation.md` §2):

- Each article = **one `index.mdx` in its own folder**.
- `index.mdx` = Markdown/MDX: optional `---` frontmatter (`title`, `path`,
  any extra keys → column `metadata`), then headings, prose, tables, fences.
  MDX components (`<Callout/>`, `import x from …`) are stripped; the prose
  *inside* them is kept.
- Images = **local relative refs** `![alt](path.ext)` in the article body
  (`.png .jpg .jpeg .webp .gif`; GIFs use the first frame). Absolute paths,
  `http(s):`, refs inside code fences or HTML comments are ignored.
  Filenames with literal parentheses (`photo_(1).png`) are supported.
- Broken/missing image refs are **dropped and reported** (every one, with
  reason, in `REPORT.md`) — never silently lost.

### 2. Set up the environment

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch                      # CUDA build (verified 2.14.0+cu130)
pip install transformers==4.57.6 datasets==5.0.1 safetensors==0.8.0
pip install huggingface_hub==0.36.2 pillow==12.3.0 pyarrow==25.0.1 tokenizers==0.22.2
```

Verified on: Python 3.14, Linux, RTX 5090 Laptop (24 GB VRAM).

### 3. Run Stage 1 — build the dataset (must exit 0 before Stage 2)

```bash
.venv/bin/python build_dataset.py --data ./data --out ./hf_dataset
```

- ~2.5 min on the reference corpus; writes `hf_dataset/` (Parquet shards +
  `README.md` + `REPORT.md(.json)`).
- Exits `0` **only** if the hard gate (C1–C4) + handoff smoke test are green;
  non-zero + named failing check otherwise. Deterministic: same corpus ⇒
  byte-identical shards.

### 4. Run Stage 2 — build the embedding index + retrieval smoke test

```bash
.venv/bin/python run_embeddings.py \
    --dataset ./hf_dataset \
    --modality image_text \
    --multi-image primary \
    --batch-size 8 \
    --embed-file ./embeddings/instar_docs_v1_image_text.safetensors
```

- **First run auto-downloads the model** (3.36 GB) into `models/`:
  huggingface.co first, then the mirror `https://hf-mirror.com`
  (override with `$HF_ENDPOINT` / `$HF_FALLBACK_ENDPOINT`).
  Later runs **skip the download** entirely.
- Verified timing: ~5 min embed (RTX 5090) + ~5 min cold-model download,
  ends with `[RETRIEVAL] PASS`.
- Exit codes: `0` green · `1` gate failed · `2` embed/row failure ·
  `3` retrieval FAIL.

### 5. Folder structure you will have after both runs

```text
instar-search-agent/
├── data/        ← your source, unchanged
├── hf_dataset/
│   ├── train-00000-of-00005.parquet …  ← the dataset (images embedded, 5 shards)
│   ├── README.md                       ← schema + load instructions
│   └── REPORT.md(.json)                ← validation evidence (counts, all dropped refs)
├── embeddings/
│   ├── instar_docs_v1_image_text.safetensors  ← THE vector index (1,067 × 2,048 bf16)
│   ├── .progress.json                    ← resume/sidecar state
│   └── REPORT_*.md + .json               ← device, model+revision, strategy, retrieval results
└── models/llama-nemotron-embed-vl-1b-v2-5b5ca69/   ← frozen model snapshot (~3.2 GB, auto-downloaded)
```

### 6. Query the index (optional) — the knowledge base for your chat agent

```python
import torch
from safetensors.torch import load_file
from transformers import AutoModel

d = load_file("embeddings/instar_docs_v1_image_text.safetensors")["image_text_embeddings"].float()
m = AutoModel.from_pretrained(
    "models/llama-nemotron-embed-vl-1b-v2-5b5ca69",
    dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa",
)
m.processor.p_max_length = 10240          # processor wiring, as in run_embeddings.py
q = m.encode_queries(["how do I enable motion detection on the SD card?"])
top = (q.float() @ d.T).topk(5).indices.tolist()
```

Text queries: `encode_queries(...)`. Image queries: `encode_documents(images=[pil_img])`.
(Processor wiring follows `run_embeddings.py::load_embed_model` exactly.)

---

## III. Technical specifications — how the two scripts work

### Stage 1 — `build_dataset.py` (contract: `SPEC_Dataset_Creation.md`)

For every article, in **deterministic order** (relative path, codepoint-sorted
— never filesystem order):

1. **`text`** — full original MDX, byte-faithful (source of truth).
2. **Frontmatter** → `title`, `source_path`, `metadata`; `id` derived from
   the relative path (stable across runs, no volatile parts).
3. **`content`** — fence-aware MDX→Markdown cleaning: strips imports, JSX
   components (prose inside kept), `{…}` expressions; **code fences are
   sacred** (kept byte-for-byte). Rows whose cleaned body ends up empty are
   *named* in `REPORT.md` (C1 guard) — 25 on the reference corpus.
4. **Image extraction** — balanced-paren scan of `![alt](rel/path)` in the
   *effective body* (code fences + HTML comments excluded); local relative
   paths only, ordered by byte offset in `text`.
5. **Normalization** — force RGB/RGBA, GIF → first frame, long-edge cap
   4,096px, fixed PNG re-encode (pure function ⇒ reproducible bytes).
   Corrupt image → dropped from **both** `images` and `image_refs`
   (alignment C4 preserved) + reported with reason.
6. **Alignment** — `len(images) == len(image_refs)` per row (hard gate).

**Output:** Parquet shards of 256 rows (5 shards / 1,067 rows on the
reference corpus) + `README.md` + `REPORT.md(.json)` with all measured
counts and the full per-bad-item list (85 dropped refs).

**Gate** (exit non-zero on any failure, before the embedder may run):
C1 content non-empty or named · C2 `images` round-trips as `[]`, never
`None` · C3 valid 2D color images · C4 alignment — plus the handoff smoke
test that simulates exactly what Stage 2 consumes.

### Stage 2 — `run_embeddings.py` (contract: `SPEC_Model_Embedding.md`)

Execution order (dataset → gate → model, so a failing dataset never wastes
the multi-GB download):

1. **Env/device report** — torch/CUDA/VRAM dump; `cuda` if available, else `cpu`.
2. **Dataset load** — `load_dataset("./hf_dataset")`; `--limit N` dry-run mode.
3. **Verification gate** (hard-fail before any model work): row count,
   image-count histogram (ragged 0..N — never assumes one image per row),
   empty/short-content rows named, image decode sample (mode / single frame),
   C4 alignment, C5 determinism re-derivation.
4. **Model guarantee** — `ensure_model_snapshot()`: **skip if**
   `models/…-5b5ca69/` already holds `model.safetensors` + `config.json`;
   else download the pinned repo + revision from huggingface.co, falling
   back to `$HF_FALLBACK_ENDPOINT` (default `https://hf-mirror.com`); a
   partial snapshot is *completed*, not re-downloaded. An explicit
   `--model-path` is used as-is (no auto-download).
5. **Model load** — bfloat16, `sdpa`, `trust_remote_code`; processor wired
   (`p_max_length` from modality: image_text=10240, `max_input_tiles=6`,
   thumbnail on); 1,678M params.
6. **Multi-image strategy** — this model accepts exactly **one** image per
   document; default `primary` (first in-document image + full text),
   `concat` (grid canvas) as alternative, `native` documented-to-fail. The
   chosen strategy + rationale is logged once and reported.
7. **Embedding loop** — `torch.inference_mode`, batch 8 (auto-halves on OOM),
   resumable 128-row parts written first, then merged; `.progress.json`
   sidecar; **load-or-create** reuse gate — a stale/mis-shaped existing file
   is refused, never silently trusted.
8. **Assembly** — `(num_rows, 2048)` bfloat16, atomic save, independent
   reload check (shape + finite values).
9. **Retrieval smoke test** — one in-domain text query (must rank a plausible
   doc), one out-of-domain (must score *lower*), one image query (must run
   end-to-end) → `[RETRIEVAL] PASS/FAIL`.
10. **Report** — `embeddings/REPORT_*.md` + `.json`: device, model path +
    revision, modality, strategy + rationale, wall time, retrieval top-5
    with scores, all limitations.

**No training, ever:** zero optimizer / loss / `backward()` / `model.train()`
calls (spec §5.6; grep-verified on both scripts).

---

**Where to go deeper:** `AGENTS.md` (operating map for agents), then the two
`SPEC_*.md` contracts (authoritative), then `hf_dataset/REPORT.md` and
`embeddings/REPORT_*.md` for the measured evidence of the latest runs.
