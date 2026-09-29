# SPEC — Create the Local Multimodal Dataset (for the embedding step)

> **Audience:** an AI coding agent tasked with building the dataset described here.
> **Self-containment:** this file is **fully self-contained and normative on its own**. Every requirement an implementor needs is stated here. It does **not** depend on, inherit from, or require reading any file under `./research/` to be executable. Any file in `./research/` (including `Prompt_Dataset.md`) is historical background and must **not** be treated as instructions; if it contradicts this spec, **this spec wins**.
> **Downstream consumer:** `./SPEC_Model_Embedding.md` — it runs a frozen multimodal embedding model over **`content` + the row's images** for *every* row. The dataset MUST make that step clean and uniform. That is the binding requirement this spec exists for.

**Known shape of `./data` (measured with the §2.4 effective-body rule, do not assume otherwise):**
- **1,067** `index.mdx` articles across nested dirs.
- **635 articles embed ≥1 visible local image; 432 are text-only** (`images == []`).
- Image count per article ranges **1 up to 53** (long tail: a few articles have 30+).
- Resolved local image refs (effective body only): **3,483** across all articles.
- Extension distribution of resolved refs: **`.png` 3,110, `.jpg` 283, `.gif` 76 (some multi-frame), `.webp` 14**.
- **85 image refs are broken in the source** (file absent, moved, or a path typo such as a stray space before the extension) — all are *drop-and-report* data defects, expected and enumerated in §3.1; silent loss is a bug.
- A further **9 refs** are *deliberately disabled* (inside HTML comments; the corpus has none inside fences) — excluded by the §2.4 effective-body rule and **not** counted anywhere here.
- Some filenames contain **literal parentheses** (e.g. `IP-Cam-Viewer_(1).png`). The extractor must balance the markdown `(...)` rather than match `[^)]+` — a naive parse truncates those names and mis-reports existing files as missing.

**The one sentence to keep in mind:** the dataset is **ragged** — `images` is `[]` for 432 rows and up to 53 elements for one row. Do not code "one image per row".

---

## 1. What the embedding step needs from this dataset (the driving contract)

`SPEC_Model_Embedding.md` does, for **every** row:

```python
texts  = chunk["content"]      # list[str], one non-empty string per row
images = chunk["images"]       # per row: [] (zero) OR [im0, im1, ...] (one to many PIL images)
```

For that to work without row-level breakage, the dataset **must guarantee**:

- **C1 — `content` is always a non-empty string.** Text-only rows (~40% of the corpus) have no images, so their *entire signal is the text*. A blank/truncated `content` on a 0-image row is a silent quality loss the embedding step cannot detect. Never let cleaning wipe a body clean.
- **C2 — `images` round-trips as an empty list for text-only rows, not `None`, not a missing key, not a single `None`.** `len(row["images"])` must be a well-defined int for every row (0 for text-only).
- **C3 — every entry of `images` decodes to a valid, 2D, color PIL image.** The embedder does not want a 1-pixel placeholder, an `L`-mode 1-bit image it can't tile, or a `mode` it rejects. Normalize on ingest.
- **C4 — `image_refs[i]` aligns 1:1 with `images[i]`** (same order, same count) where present. `image_refs` may be `[]` when there are no images.
- **C5 — deterministic order.** Article order, per-row image order, and IDs are stable and reproducible across runs (§6, §2.0).

If any of C1–C5 is violated, the embedding step either crashes a whole batch or encodes garbage you won't notice until retrieval is bad. The converter is responsible for making these true **at build time**, not hoping the embedder tolerates them later.

---

## 2. Schema

### 2.0 Required columns (the core schema)

A dataset row — one per article — MUST contain these columns:

| column    | type                       | meaning |
|-----------|----------------------------|---------|
| `id`          | `string` | stable identifier derived deterministically from the article's relative path (§6) |
| `title`       | `string` | article title, in preference order: frontmatter `title` → first Markdown H1 → article directory name. Never invent one. |
| `source_path` | `string` | the `path` field from frontmatter, or the relative path if absent |
| `text`        | `string` | **the complete original MDX source, byte-faithful** — imports, JSX, comments, image refs, all original formatting and line endings preserved |
| `content`     | `string` | cleaned Markdown (§2.3) — MDX/React implementation detail removed, human-readable article + Markdown preserved |
| `image_refs`  | `list[string]` | image references **exactly as they appeared in the MDX**, in document order; `[]` when none |
| `images`      | `list[Image]` | the corresponding decoded images in the **same order** as `image_refs`; `[]` when none (NOT `None`) |

Optional, if used, keep them aligned and additive — they must never break the core columns above:

- `image_positions` / `blocks` — structured multimodal sequence (see §2.4);
- `metadata` — a dict of extra frontmatter fields (author, excerpt, dateChanged, tags, …), see §2.5.

**Constraints that hold across all columns:**
- The fundamental unit is **article + text + its images**. Do NOT model this as an image-folder dataset and do NOT emit one row per image.
- `text` MUST be the pristine original MDX (C1-adjacent: it is the source of truth). Do not normalize whitespace/line endings.
- The two columns most likely to silently break the downstream embedder — `images` (empty-list round-trip) and `content` (empty body) — get explicit correction below (§2.1, §2.3).

### 2.1 The `images` column — fix the empty-list round-trip (fixes C2)
- Feature: `features[Sequence(Image())]` (HF `datasets` `Image` per element), or a `Value("image")` sequence — whatever the current `datasets` version uses for "list of images." **Do NOT** store a bare `list[string]` of paths here (that's `image_refs`'s job) and do not collapse N images into a single image.
- **For rows with no images, `images` must serialize to a valid empty list and read back as `[]` (a list of length 0), never `None`/missing.**
- **Verify this empirically** against the installed `datasets` + Parquet path before accepting the converter. A known failure mode is that "no images" is written as a missing/absent value and comes back as `None` (or the Parquet column's non-nullability is inconsistent across a mix of empty and non-empty rows). Run this test and include its output in the report:

```python
ds = load_dataset("./hf_dataset")["train"]
for i in [0, 1, 2]:
    v = ds[i]["images"]
    assert v is not None, f"row {i}: images is None"
    assert isinstance(v, (list, tuple)), f"row {i}: got {type(v)}"
print("empty-list round-trip OK; sample lens:", [len(ds[i]["images"]) for i in range(3)])
# and across a batch where empties and non-empties mix:
batch = ds.select(range(10))
print("mixed-batch lens:", [len(x["images"]) for x in batch])
```

- If the installed stack cannot cleanly mix empty and non-empty `Sequence(Image)` rows in Parquet, **flag it explicitly** and use a supported encoding (e.g. store `images` as an always-present list, or persist PIL bytes via the `Image` feature with an empty list). **Do not** paper over it by giving empty rows a 1×1 placeholder pixel — that would feed the embedder a fake image and violate C3. Report whichever representation you chose and why.

### 2.2 The `images` content — normalize every image (fixes C3)
On ingest, run each image through a normalizer **before** storing, so the embedder always receives a sane color image:

- Open with Pillow. Force to a color mode:
  ```python
  img = Image.open(path)
  img.load()
  if img.mode not in ("RGB", "RGBA"):
      img = img.convert("RGBA")   # or "RGB"; pick one and be consistent
  ```
- **Animated/multi-frame (`.gif`):** use only the **first frame**. `Image.open(gif)[0]` after `load()`; strip disposal/animation. Do not store an `AnimatedImage`/multi-frame object — the embedder expects single 2D images. Log how many GIFs you flattened and any that failed (log only; do not fail the run over one bad GIF).
- **Truncated/oversized:** if dimensions are extreme (e.g. `w*h > 8_000_000`), downscale to a sane cap (e.g. max 4096 on the long edge) and log it — the embedder's `max_input_tiles`/tile budget makes absurdly large images both slow and a likely OOM/`max_input_tiles`-exceeded failure. Cap is configurable; default is conservative.
- Re-encode to PNG (or JPEG for large photos) at a fixed quality before writing to the `Image` feature, so Parquet does not store arbitrary in-memory PIL objects and modes are stable across readers.
- **Corrupt/unreadable image:** record a validation error for that row's reference, **drop that image entry** from both `images` and `image_refs` (keep them aligned — C4), continue with the article's other images/text. A single bad image must not zero out a 40-image article.

### 2.3 `content` — how it is produced (fence-aware MDX → clean Markdown), plus the C1 guard

`content` is the source-of-truth prose for the embedding step. It is produced from the **original `text`** by a **state-aware** cleaner — *not* a handful of naive regex substitutions. It must satisfy:

**What to remove** (MDX/React *implementation* detail, wherever it occurs):
- **Imports:** `import Foo from "./Foo"`, `import { A, B } from "../x"`, `import Image from "next/image"`, both default and named, single-line and multi-line, and `export` lines.
- **JSX/MDX components that are implementation/UI:** `<SEOHelmet ... />`, `<BreadCrumbs ... />`, `<EuiSpacer />`, `<Callout type="warning">…</Callout>`, `<Alert …>`, `<Figure …>`, `<YouTube …/>`, `<Tweet …/>`, `<CodeBlock …/>`, and any `<SomeComponent … />` (including multiline with props spanning lines, nested components).
  - If a component wraps **meaningful human-readable article prose**, keep that prose (e.g. the sentence inside `<Callout>`), drop the tag. If a component is implementation-only with no textual children, drop it entirely.
- **JSX/JS expressions:** `{someVariable}`, `{condition && <Component />}`, `props.location.pathname`, inline `{…}` interpolations — without destroying ordinary Markdown (do not touch `{` inside code spans or fenced blocks that are documentation).

**What to keep** (the article itself, as **Markdown**, never HTML and never flattened to plain text):
- All headings, paragraphs, lists, tables, links, emphasis (`**…**`, `__…__`), inline code, and fenced code blocks.
- **Fenced code blocks (` ```lang … ``` `) are sacred.** Any angle-bracket/JSX-looking text inside a fence is *content*, not a component, and MUST remain byte-for-byte. A cleaner that deletes `<MyComponent />` inside a `jsx` fence is buggy.

**Frontmatter:** if the MDX opens with `--- … ---` YAML/JSON frontmatter, **parse it** (§5) for `id`/`title`/`source_path`/`metadata`, and **remove** the raw frontmatter block from `content` (it stays in `text`).

**C1 guard (hard rule):** After cleaning, if `content` is empty or < 40 chars **while `text` has a real body**, flag that row `warn_empty_content` and report it by `id`. Do **not** silently emit an empty string for a 0-image row — for a text-only article the *entire* embedding signal is `content`. If the source genuinely has no body beyond imports/components, the report must name that article so a human can inspect it, not bury it in a count.

### 2.4 Image extraction — narrow, deterministic, relative to the article

**Scope (measured against the current `./data` corpus — 1,067 articles):**
- The *only* image reference form that matters here is **Markdown image syntax**: `![alt](relative/path/ext)`.
- Measured over the **effective body** (fences + HTML comments excluded): **3,483** Markdown image refs resolve to a local file; **85** are genuinely broken (file moved, path typo such as a stray space before the extension) and must be *dropped as validation events* (§3.0). A further 9 are commented-out (disabled) and are out of scope by definition.
- **Not in scope — do not extract, do not warn on:**
  - Image refs inside HTML comments (`<!-- … -->`) or fenced code blocks — *deliberately disabled* or literal syntax (the "effective body" rule, above).
  - Absolute paths starting with `/` or `/./` (web backend, lives in a different asset tree).
  - `http://` / `https://` / `data:` URIs (remote or inline).
  - HTML `<img src=…>` tags — the current corpus has **zero** `<img src>` tags pointing to a local relative path; the 10 that exist are all absolute web-backend paths, `data:` URIs, or Mustache templates. Do not add extractor complexity for them.
  - JSX/MDX components whose `src` is a variable or remote (`<Image src={"…"/>`, `<Image src={img}/>`, `<ImageQGallery />`, `<ImageQCards />`, `<ImageQGallery>`-family cards). These render **server-side from a remote asset store** and never point at a local file.
  - Frontmatter `image`, `social`, `toc` values (already excluded from `images` by §2.5; they are web-backend metadata).

**Rule (effective body scan):** for each row, scan the **effective body** of the article and collect every occurrence of `![alt](path)` where `path` is a **relative local path** — i.e. it does **not** start with `/`, `/./`, `http://`, `https://`, `data:`, `#`, or contain template braces (`{{…}}`, `{…}`). Resolve against the article's directory. Ignore (do not count as an error) anything else.

The **effective body** is the *visible, un-disabled* prose of the article, defined as: `text` **minus** (a) fenced code blocks and (b) HTML comments (`<!-- … -->`). Reasoning: (a) a Markdown image ref inside a JSX component that cleaning strips is still a real image — the JSX is part of the effective body, so we capture it; (b) an image ref inside an HTML comment was *deliberately disabled* by the author and must not be treated as live content (the current corpus has ~13 such cases). Fenced code blocks are also disabled (they are literal syntax documentation, not rendered content). This rule captures exactly the "visible, live" image references of the article and no others.

**Resolution:**
- `data/foo/index.mdx` + `./images/x.png` ⇒ `data/foo/images/x.png`
- `data/foo/index.mdx` + `x.png` ⇒ `data/foo/x.png`
- `data/foo/a/index.mdx` + `../x.png` ⇒ `data/foo/x.png`
- Strip an optional trailing ` "title"` or ` "title"` in the paren (Markdown allows `![alt](path "title")`); the path is the first token before whitespace or the first space.
- Accept extensions `.png .jpg .jpeg .webp .gif` and any other extension Pillow can decode. **Unknown/unsupported extensions on a local path → drop + validate as a data defect** (§3.0).
- A ref whose resolved path exists but is not a file → drop + validate.
- A ref whose resolved path does not exist → drop + validate (this is the ~99 case; expected and reported, not silently lost).

**Ordering & alignment (`image_refs[i]` ↔ `images[i]` is a hard invariant, C4):**
- `image_refs` is indexed by the **byte-offset of the `![alt](path)` occurrence in `text`**, in ascending order.
- If the *same* file is referenced twice (or the same file via two distinct strings), they are **two** `image_refs` entries aligned 1:1 with **two** `images` entries (identical bytes). This is allowed; the C4 invariant still holds (`len(images) == len(image_refs)`).
- For each resolved file, decode **once** (cache by canonical path) and re-use the same bytes for each alignment slot. Do NOT re-read the file per slot.

**What is NOT captured here (and where it goes instead):**
- The `image`/`social`/`toc` keys in frontmatter — stored only in `metadata`, never in `images`. §2.5.
- Any component name (e.g. `<EuiSpacer/>`, `<ImageQGallery/>`) that rendering would expand to an image. The article's *prose* around the component (e.g. "The 1/3 inch WDR sensor is a Panasonic CMOS chip…") remains in `content`. The card/gallery it produces is not part of this dataset — the corpus does not ship those rendered assets.

### 2.5 Frontmatter → `id`, `title`, `source_path`, `metadata`

If the file begins with a frontmatter block (`--- … ---` or `> --- … ---`), parse it into:
- `id` — **not** taken from frontmatter; derived from the article's relative path per §6 (frontmatter ids are not guaranteed unique/stable; the path is). Frontmatter `title`/`path` may inform `title`/`source_path`.
- `title` — frontmatter `title`; else first H1; else article directory name.
- `source_path` — frontmatter `path`; else the relative path.
- `metadata` — **all other** frontmatter keys (e.g. `author`, `excerpt`, `dateChanged`, `chapter`, `category`, `type`, `faq`, `image`, `social`, `toc`) preserved in a dict, verbatim. Do not hard-code a particular frontmatter schema; if a key is missing, the column is absent/`null` rather than the converter failing.
- The raw frontmatter text stays in `text` verbatim and is removed from `content`.

### 2.6 Optional structured blocks (`image_positions` / `blocks`)

*Optional* per §2.0 — if implemented, they are additive and must not disturb the core columns. Represent the article's multimodal sequence so the position of each image relative to text is recoverable:

```json
[
  {"type": "text",   "content": "# Intro\n\nSome text…"},
  {"type": "image",  "image_index": 0},
  {"type": "text",   "content": "Next section…"},
  {"type": "image",  "image_index": 1}
]
```

`image_index` indexes into `image_refs`/`images`. If it cannot be done without a large complexity cost, *omit the column* (the core `text`/`content`/`images`/`image_refs` remain sufficient for the embedding step) — do not ship a broken/partial `blocks` field.

---

## 3. Validation & report — extended to prove the embedding contract

### 3.0 Validation philosophy (self-contained)
**Tolerant but visible.** Individual bad assets must not sink the run, but they must be *visible*:
- Missing image, corrupt image, one malformed `index.mdx` → record a validation error (article, original reference, resolved path, error), **drop that item only**, continue.
- Fatal dataset-writing / Parquet / gate failure → fail the whole run (non-zero exit).
- At the end the report MUST list, per bad item: `article` (id), source MDX path, original reference, resolved path, and the specific error.
- Exit code: `0` only if the §3 hard gate (§3.2) passes; `1` (or other non-zero) otherwise.

### 3.1 What the report contains

Per the philosophy in §3.0, the report (written next to the dataset, not buried in a log-only stream) MUST include at minimum the lines below plus a per-bad-item detail section. The `N/X/Y/K` placeholders are the *actual* measured numbers from this run; fill them in, do not leave placeholders:

Values below are the **measured reality** for the current `./data` (re-measure if the corpus changes, but the *line set* is the contract):

```text
[DATASET] articles = 1067   (converted 1067, failed 0)
[DATASET] image-count histogram: {0: 432, 1: 207, 2: 91, 3: 52, 4: 46, … 53: 1}
[DATASET] max images in one article: 53
[DATASET] rows with images: 635   text-only rows (images==[]): 432
[DATASET] resolved local image refs: 3483
[DATASET] unique image files embedded: 3436
[DATASET] missing image refs dropped: 85   (see detail list below)   corrupt images dropped: 0
[DATASET] gif images flattened to first frame: 76   (failed: 0)
[DATASET] images downscaled for size: 0   (long-edge cap: 4096)
[DATASET] empty/near-empty content rows: 0
[DATASET] alignment check: len(images)==len(image_refs) for all rows: PASS
[DATASET] empty-list round-trip (empty + non-empty rows, sampled): PASS
[DATASET] MISSING IMAGE REFS (detail, one line each)
[DATASET]   <article id>  ref=<original ref>  resolved=<abs path>
…85 lines…
```

### 3.2 Hard gate

Exit **non-zero if any of these fails**, and *before the embedding step is allowed to run* (AGENTS.md §3 treats this as the Stage-1 go/no-go):
- **C1** — every row's `content` is a non-empty string, or the row is explicitly listed as `warn_empty_content` with a human-acknowledged reason.
- **C2** — every `images` value is `None`-free and list-typed; the §2.1 round-trip assert passes across a mixed (empty + non-empty) sample.
- **C4** — `len(images)==len(image_refs)` for **every** row.
- **C3** — no row's `images` contains a non-color or multi-frame object; assert `mode in {RGB,RGBA}` and `n_frames==1` on a sample.

Only when all four are green may Stage 2 be attempted.

---

## 4. Cross-check against the embedding step (do this at the end)

Before declaring done, smoke-test the *handoff* — simulate exactly what `SPEC_Model_Embedding.md` §4 verification and §5 loop consume, using a **tiny stand-in** for the model (no GPU, no download) to catch shape/emptiness failures cheaply:

```python
ds = load_dataset("./hf_dataset")["train"]
sample = ds.select(range(min(8, len(ds))))
for r in sample:
    assert isinstance(r["content"], str) and len(r["content"]) >= 1
    assert r["images"] is not None and isinstance(r["images"], list)
    assert len(r["images"]) == len(r["image_refs"])
print("handoff smoke-test OK over", len(sample), "rows")
# then, if a GPU/model is available, run the real thing from SPEC_Model_Embedding.md
```

If that fails, the dataset is **not** ready for embedding regardless of how pretty the report looks. Fix the dataset spec/output, re-run, and only then hand off.

---

## 5. Deliverable

Produce one runnable converter, **`build_dataset.py`**, that:

**Discovery:**
- Recursively find every `index.mdx` under `--data`.
- Sort articles by their **relative path** (stable, locale-independent, byte/codepoint order) — the dataset row order. Do NOT rely on filesystem traversal order.
- Derive each `id` from the relative path (§6).

**Per article (in the sorted order, incrementally — release intermediates as you go):**
1. Read the full original MDX → `text` (byte-faithful).
2. Parse frontmatter → `id`/`title`/`source_path`/`metadata` (§2.5); produce cleaned `content` (§2.3).
3. Extract image references from the **original `text`** in document order (§2.4); resolve against the article directory.
4. Normalize each image per §2.2 (color, first-frame for GIF, size cap).
5. Align `image_refs` ↔ `images` (C4) — drop a bad image from *both* and record it (§3.0).
6. Append the row to the in-memory buffer for the current Parquet shard; flush the shard at `--shard-rows`.

**Output (self-contained, local only):**
- Write to `--out` (default `./hf_dataset`) as Parquet shards under a `train` split, plus a `README.md`.
- Do **not** upload to the Hugging Face Hub; do **not** require `load_from_disk()` — the dataset MUST load via `load_dataset("./hf_dataset")`.
- Do **not** use `ImageFolder`; do **not** emit one row per image.

**Validation & gate (§3) → handoff smoke-test (§4) → `REPORT.md` → exit `0`/`1`.**

**CLI flags (all must exist):**
- `--data ./data` (default `./data`)
- `--out ./hf_dataset` (default `./hf_dataset`)
- `--max-image-edge 4096` (long-side cap before re-encode; `0` = uncapped)
- `--keep-frames` (default off; if **on**, log that multi-frame output is expected to break the embedder and is unsupported per C3 — this flag is for debugging only)
- `--shard-rows N` (default `256`; number of article rows per Parquet shard)
- `--strict` (fail the run on *any* dropped image, not just fatal write errors)
- `--seed` (optional, reserved for future random sampling determinism; not used today)

Log lines MUST use `[DATASET] …` prefixes and be greppable (AGENTS.md §5.2).

---

## 6. Determinism (C5) — the rules that must not be broken

This section is what makes "rebuild is safe" true. Re-running `build_dataset.py` against the same `./data` MUST produce:

- **Identical article order**: sorted by relative path (relative to `--data`), using codepoint order on the full relative path string.
- **Identical IDs**: a stable function of the relative path only. Proposed rule (implement this or an equivalent *collision-safe* one and document it in the report): take the article's relative directory path (with the trailing `index.mdx` stripped and the `./data/` prefix stripped), normalize path separators to `/`, replace `//` with single `/`, and use that string as `id` *if* it is unique; otherwise fall back to `sha1(relpath)[:12]` + the original string so no two IDs collide. Two articles in `a/index.mdx` and `a/b/index.mdx` MUST get distinct IDs. The ID MUST NOT include volatile parts (timestamps, absolute path, random suffix).
- **Identical image order within a row**: byte-offset order in the original `text` (§2.4).
- **Identical image bytes**: the §2.2 normalizer must be a pure function of (source file bytes, `--max-image-edge` mode choice). Re-encode at a **fixed PNG quality / compression level** (or fixed JPEG quality for JPEG outputs) so two runs over the same file produce identical Parquet-encoded bytes. Document the exact encoding choice in the report.
- **Identical `content`**: the cleaner is a pure function of `text` and the frontmatter parse — no randomness, no environment-dependent behavior.
- **No filesystem-order dependence**: every `for` over articles or their images MUST traverse a sorted structure.
