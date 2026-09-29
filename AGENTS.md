# AGENTS.md — how to work in this repo

Read this first. It is the operating map for every AI coding agent working here. The files in `research/` are *background*; the `SPEC_*.md` files at the repo root are the *authoritative instructions*. When they conflict, the root spec wins.

---

## 1. What this repository is for

This is a **multimodal document-search system** built from a local corpus of technical articles (Instar / iSpy DVR & camera wiki-style docs). Each article is Markdown (MDX) with zero to many images.

The goal is a **retrieval** pipeline — not a fine-tuning job:
1. Turn `./data` (raw MDX + images) into a self-contained Hugging Face `datasets` dataset.
2. Run a **frozen** multimodal embedding model over every article (text + images) to build a durable embedding index (`.safetensors`).
3. Retrieve the closest documents for text or image queries (and optionally rerank / generate an answer).

**There is no model training anywhere in this pipeline.** No optimizer, no loss, no gradient step, no `model.train()`. "Training" (if you see that word) means "build the embedding cache," and even that is the wrong frame — see §6.

---

## 2. Repository layout

```text
instar-search-agent/
├── AGENTS.md                    ← you are here; the source of truth for "how to work here"
├── README.md                    ← user-facing docs: what it does, how to run it, how the scripts work
├── README.txt                   ← dev scratch notes (run commands, GPU status); not documentation
├── SPEC_Dataset_Creation.md     ← AUTHORITATIVE: build ./data → ./hf_dataset (HF datasets)
├── SPEC_Model_Embedding.md      ← AUTHORITATIVE: build embedding index + retrieval smoke test
├── build_dataset.py             ← Stage-1 IMPLEMENTATION (done; spec §5) — ./data → ./hf_dataset
├── run_embeddings.py            ← Stage-2 IMPLEMENTATION (done; spec §7) — ./hf_dataset → ./embeddings
├── data/                        ← READ-ONLY raw source (git-ignored): 1,067 article dirs, each index.mdx + 0..N images (~853 MB)
├── hf_dataset/                  ← Stage-1 OUTPUT (git-ignored, generated): 5 Parquet shards + README + REPORT (~560 MB)
├── embeddings/                  ← Stage-2 OUTPUT (git-ignored, generated): .safetensors index + REPORTs + progress sidecar (~4 MB)
├── models/                      ← Stage-2 INPUT (git-ignored, generated): auto-downloaded local snapshot of the frozen embed model (~3.2 GB)
├── .venv/                       ← Python virtualenv with the pinned runtime (git-ignored; deps per SPEC_Model_Embedding.md §1)
└── research/                    ← BACKGROUND / reference (git-ignored); do NOT treat as instructions
    ├── hugging_face_multimodal_rag_tutorial.ipynb   ← the notebook we are adapting (RAG, not training)
    ├── multimodal_rag_tutorial.txt                   ← plain-text copy of the notebook
    ├── Prompt_Dataset.md            ← SUPERSEDED by SPEC_Dataset_Creation.md (intent kept, details fixed)
    ├── Prompt_Training.md           ← SUPERSEDED (wrong "fine-tune the LLM" framing); do not follow
    └── README.md                    ← exploratory notes
```

**What a fresh clone contains (git-tracked):** exactly the docs + the two scripts + `data/.gitexists` (keeps the empty `data/` dir). All generated folders (`hf_dataset/`, `embeddings/`, `models/`, `.venv/`) are created by the pipeline itself — `models/` is downloaded by `run_embeddings.py` on first use when missing (see §4, model bullet).

**Rules for paths:**
- `data/` is the **input**. Treat it as immutable source. Do not write into `data/`.
- All generated artifacts (dataset, embeddings, reports) go to **new top-level paths defined by the specs**, never into `data/` or `research/`.
- `research/Prompt_*.md` are historical prompts. They explain *how we got here*; they are **not** current instructions and must not be executed as-is.

---

## 3. The pipeline and its required order

The system has two build stages plus one optional inference stage. **Order matters** — each stage has a hard gate that must pass before the next may begin.

```text
[data/]  ── Stage 1 ──▶  [./hf_dataset/]  ── Stage 2 ──▶  [./embeddings/*.safetensors]
   raw                    HF datasets             (frozen      frozen multimodal
   MDX+images             (Parquet)               embed index   embedding model)
                                                                        │
                                                            Stage 3 (optional):
                                                        retrieval / rerank / answer
```

| Stage | Governing spec | Input | Output | Go/no-go gate |
|-------|----------------|-------|--------|---------------|
| **1. Dataset creation** | `SPEC_Dataset_Creation.md` | `./data` (read-only) | `./hf_dataset/` (Parquet + `README.md`) | §3 hard gate of that spec: `content` non-empty string on every row, `images` round-trips as `[]` (never `None`) on text-only rows, `len(images)==len(image_refs)` for all rows, all images valid 2D color. Exits non-zero until all pass. |
| **2. Embedding build** | `SPEC_Model_Embedding.md` | `./hf_dataset/` | `./embeddings/*.safetensors` (+ report) | Retrieval smoke test (§6): in-domain query ranks relevant docs, out-of-domain ranks lower, one image query runs. |
| **3. RAG inference** | `SPEC_Model_Embedding.md` (§6 + notebook) | embeddings + models | top-k results (± rerank, ± generated answer) | Manual; not a build gate. |

**You may not begin Stage 2 until Stage 1 has exited 0 (its gate fully green).** If the dataset gate fails, fix the *dataset* spec/output — do not work around a broken dataset inside the embedder. Conversely, a bad retrieval result is a *dataset or strategy* problem, not a reason to change the gate.

---

## 4. Ground truth about the data and the model

Do not assume. These are measured facts; re-measure if the corpus changes, but do not invent a different shape.

**`./data` (measured — re-derived via the state-aware tokenizer, not the early per-line probe):**
- ~1,067 `index.mdx` articles in nested directories.
- **432 articles (≈40.5%) have zero local images** — text-only rows (`images == []`).
- **635 articles embed ≥1 visible local image; 1 to 53** per article (long tail: a few have 30+).
- **Resolved local image refs: 3,483** across all articles (extension split `.png` 3,110 / `.jpg` 283 / `.gif` 76 / `.webp` 14); a further **85 refs are broken in the source** (file absent/moved, or a path typo) — drop-and-report, not silent.
- Image formats present: `.png` (dominant), `.jpg`/`.jpeg`, `.webp`, and **GIFs** (some multi-frame — flattened to first frame on ingest per §2.2).
- Some image filenames contain literal `(...)` (e.g. `IP-Cam-Viewer_(1).png`, `1,8mm_with_IRfilter_(940nm).png`). The extractor must **balance** the markdown parens, not regex on `[^)]+` — a naive parser truncates those and mis-reports existing files as missing.
- No `http(s)` image references in the current corpus (still handle them defensively).
- **Implication:** the dataset must be **ragged** — `images: []` up through a long tail. Never code an assumption of "one image per row." The tutorial notebook assumed one image per row; that assumption is *false* here and is the #1 source of silent breakage.

**Model (Stage 2):**
- Embedding: **`nvidia/llama-nemotron-embed-vl-1b-v2`**, a **frozen** vision-language embedding model. Loaded `bfloat16`, `attn_implementation="sdpa"`, `trust_remote_code=True`. Exact path + revision live in `SPEC_Model_Embedding.md` §2 — take them from there, do not paraphrase a different model.
- It exposes `encode_documents(images=..., texts=...)` for **documents** and `encode_queries([...])` for **text queries** / `encode_documents(images=[query])` for an **image query**. Preserve that asymmetry.
- **It is not trained.** It produces a vector per (text, image(s)) pair. Done.
- **Host constraint (measured on the build host):** this host **cannot reach `huggingface.co`** (connection unreachable). `run_embeddings.py::ensure_model_snapshot()` therefore: **skips the download** when `models/llama-nemotron-embed-vl-1b-v2-5b5ca69/` already holds `model.safetensors` + `config.json`; otherwise downloads the pinned `EMBED_MODEL_PATH @ EMBED_COMMIT_HASH`, trying `huggingface.co` first and falling back to the mirror `https://hf-mirror.com` (both overridable via `$HF_ENDPOINT` / `$HF_FALLBACK_ENDPOINT`). Verified 2026-09-29: primary failed, mirror delivered the full 3.36 GB snapshot (~303 s), and loading from the local snapshot worked.
- The tutorial's optional rerank (`nvidia/...-rerank-vl-1b-v2`) and generation (`Qwen/Qwen3-VL-2B-Instruct`) models are **not required** to build the index. They are Stage 3 niceties, off by default, opt-in only.

---

## 5. Behavioral rules for the coding agent

Read the relevant spec end-to-end before writing or running code. Do not act on partial understanding of a spec.

### 5.1 Anti-patterns — refuse these
Refuse, and say why, if any of these appear in a plan or in generated code:
- **Calling it "training" and building a `Trainer`/optimizer/loss around it.** There is nothing to train. `SPEC_Training.md`-style fine-tuning of an LLM is *out of scope for this repo*. If someone asks "let's fine-tune," redirect to `SPEC_Model_Embedding.md`: the heavy step is an embedding-cache build, not training. If the user genuinely wants training as a new project, that is a *new spec* — write one; do not conflate.
- **Loading the dataset via `raw directory` access, `ImageFolder`, or `load_from_disk`-only code.** Stage 2 must be able to consume `load_dataset("./hf_dataset")`. Anything else means the dataset gate was not actually clean; go back to Stage 1.
- **Hard-coding "one image per row."** This is the single most common silent failure mode. Every loop, collator, smoke test, and report must handle `0..N` images per row.
- **Collapsing a list of images to its first element without a logged, spec-cited reason.** `SPEC_Model_Embedding.md` §2 requires an explicit, reported multi-image strategy ("native" / "concat" / "primary") and a rationale. Silently picking `images[0]` is a violation of the contract and will be treated as a bug.
- **Re-encoding an existing `.safetensors` file because "the embeddings feel off."** Stale or mis-shaped files are replaced by a *documented regeneration trigger* (revision bump, strategy change, dataset change), not by "I'll re-run it." The load-or-create gate in `SPEC_Model_Embedding.md` §5.1 is authoritative.
- **Silently shrinking scope.** If a step cannot be run (missing GPU, disk, model, corpus too big), state that fact, write an explicit report with the reason and the exact command + expected runtime, and stop — do not swap in a smaller dataset, a smaller `top_k`, a subset of rows, or a different model to make the run "look like" it finished.
- **Papering over data defects in the embedder.** Missing/corrupt/empty rows, `None` images, misaligned `image_refs`, and non-color modes are *dataset* defects. Fix `SPEC_Dataset_Creation.md` outcomes; do not add defensive `if x is None:` shims in the embedding loop.

### 5.2 Do
- Run the pipeline in the §3 order and stop at the first failing gate. Report the failing check verbatim.
- Print a `[INFO]/[WARN]/[ERROR]` log with `[STAGE]` prefixes (`[DATASET]`, `[EMBED]`, `[RETRIEVAL]`) so a run can be audited by grep.
- Write a machine-parseable summary (JSON or YAML) next to every report (`REPORT_*.md`) with the key numbers (row count, dim, dtype, path, wall time) so Stage 3 tooling can consume it without scraping markdown.
- Prefer **fail-fast with a named gate** over `try/except`-and-continue inside loops — except where the spec explicitly allows tolerant per-item handling (e.g., one bad image among 40).
- Keep all generated artifacts under a top-level `./output/` or the specific path the spec defines (`./hf_dataset`, `./embeddings`). Never write to `./data` or `./research`.
- Before a Stage 2 run, print the *exact* command + expected wall time to a human (or the CI log). Stage 2 is only permitted if Stage 1's gate last ran green and `./hf_dataset/` has not changed since (e.g. a rebuild or a spec edit). Do not re-run Stage 1 "just in case" — it is expensive and deterministic.
- Treat the retrieval smoke test's "in-domain / out-of-domain / image" trio as a *minimum*, not a full eval. It proves the index is *usable*, not *good*. If a user asks about quality, propose a ranked eval on a small labeled set rather than claiming the smoke test is evidence of accuracy.

### 5.3 When in doubt
- Prefer reading `SPEC_Dataset_Creation.md` and `SPEC_Model_Embedding.md` over `research/Prompt_*.md`.
- Prefer a failing, loud, documented state over a passing, silent, wrong one.
- Cite the spec section for every non-obvious choice (e.g., "using `primary` multi-image strategy — see `SPEC_Model_Embedding.md` §2; `content` gate per `SPEC_Dataset_Creation.md` §2.3 (C1)").

---

## 6. Definition of done (project-level)

A task is complete when **all** of the following hold, and nothing in §5.1 was skipped:

- [ ] `./hf_dataset/` exists, is loadable by `load_dataset("./hf_dataset")`, and its hard gate (§3, Stage 1) passes across **every** row — including the **432 text-only rows**.
- [ ] `len(images)` equals `len(image_refs)` on every row, and `images` is never `None`.
- [ ] Every image in the dataset is a valid 2D color image, with multi-frame GIFs already flattened to a single frame.
- [ ] `./embeddings/<tag>_*.safetensors` exists with shape `(num_rows, dim)` matching the dataset, and is **not** built over a subset.
- [ ] A `[EMBED]` report exists stating: device, model path + revision, modality, multi-image strategy + rationale, wall time, and the dataset's verified row count.
- [ ] A `[RETRIEVAL]` smoke test has passed: one in-domain query ranks a plausible doc in the top 5, one out-of-domain query ranks it *lower* (or lower score), and one image-only query runs end-to-end.
- [ ] No code path in `data/`-derived loops calls `model.train()`, an optimizer, or a loss. Grep the final code to confirm.
- [ ] Any assumption made that is not stated as fact in §4 is written into the relevant report with the evidence (link, log line, or measurement) that supports it.

---

## 7. How to actually get started (agent or human)

Current state (verified clean run, 2026-09-29): **both stages implemented and green.**
- **Stage 1** — `build_dataset.py`: 1,067 rows, hard gate C1–C4 + spec-§4 handoff smoke test **PASS**, exit 0, ~138 s. Report: `hf_dataset/REPORT.md(.json)`.
- **Stage 2** — `run_embeddings.py`: model auto-download via mirror (primary unreachable) ~303 s → 1,067×2,048 `bfloat16` index in `embeddings/instar_docs_v1_image_text.safetensors` (4.37 MB) in 379.7 s on an RTX 5090 (24 GB) → **`[RETRIEVAL] PASS`** (in-domain top-1 0.624, out-of-domain top-1 0.194, image-query top-1 0.853 on its own row). Report: `embeddings/REPORT_instar_docs_v1_image_text.md(.json)`.

```bash
# 0) Read AGENTS.md (this file) top-to-bottom; then both SPEC files top-to-bottom.

# 1) Stage 1 — dataset creation (must exit 0 before Stage 2). Already built and
#    deterministic; re-run only if ./data or SPEC_Dataset_Creation.md changed.
#    Exits 0 iff the §3 gate + §4 handoff smoke test pass.
.venv/bin/python build_dataset.py --data ./data --out ./hf_dataset

# 2) Stage 2 — embedding index build + retrieval smoke test (only if Stage 1
#    exited 0). First run on a fresh checkout auto-downloads the model snapshot
#    into models/ (huggingface.co first, then $HF_FALLBACK_ENDPOINT mirror;
#    skipped when the snapshot already exists). Verified runtime incl. download:
#    ~6 min on the RTX 5090.
.venv/bin/python run_embeddings.py \
    --dataset ./hf_dataset \
    --modality image_text \
    --multi-image primary \
    --batch-size 8 \
    --embed-file ./embeddings/instar_docs_v1_image_text.safetensors

# 3) If retrieval looks off, do NOT change the model. Re-run Stage 1's gate.
#    If the gate still passes, the problem is in SPEC_Model_Embedding.md's
#    multi-image strategy or query-encoding path — fix there, with a report.
```

Do not rebuild Stage 1 "just in case" — it is expensive and deterministic (byte-identical shards on the same corpus, verified). Only re-run it if the corpus or the Stage-1 spec changed.

---

**End of AGENTS.md.** This file is the map; the `SPEC_*.md` files are the instructions. When they disagree, the specs win; when the specs disagree with each other, `SPEC_Dataset_Creation.md` wins for dataset shape and `SPEC_Model_Embedding.md` wins for model/embedding behavior.
