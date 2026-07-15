# Design: hybrid-site-search

## Context

- North star: `site-search/.specs/NORTH-STAR.md`.
  Its single measure of success — the half-remembered query lands the right post at rank 1 — is operationalized here as the eval harness's primary recall@1 metric (`RetrievalQualityFloor`).
  Its guiding principles anchor the decisions below: keyword-is-the-floor → FallbackPolicy; fail-loudly-at-build/degrade-silently-at-runtime → CorpusSource gating + FallbackPolicy; measured-not-assumed → Dimensionality, ChunkingPolicy; parity-by-construction → TokenizerContract; size-is-a-feature → Dimensionality, QuantizationStrategy; own-the-code → LocalImplementation.
- The site is fully static (Hugo + Hextra v0.12.0, nginx container on homelab k8s).
  There is no server-side compute at request time; anything search does must ship as static files and run in the browser.
- Keyword search is Hextra's FlexSearch integration.
  This repo already shadows two theme files with intentional divergences: `hugo/layouts/_partials/utils/fragments.html` (fixes the heading-link fragment-splitting bug that emptied 75% of the search corpus) and `hugo/assets/js/flexsearch.js` (first-`#` key-split fix).
  This change adds a second, larger divergence to `flexsearch.js`.
- The `treadmill/` section is mounted from the external Hugo module `github.com/ahgraber/ai-treadmill` — its markdown does not exist in this repo.
  Hugo's generated `en.search-data.json` is the only place blog and treadmill text co-exist as plain text.
- Corpus baseline (gate run 2026-07-11, post-fragments-fix): 3,072 sections; 20 empty (0.7%) — blog 321/0 empty, treadmill 2,749/20 empty; all sampled empties verified as legitimate title-only entries.
  Treadmill is the heaviest user of link-headings and therefore the most regression-sensitive section.
- Python tooling is a uv workspace (`experiments/*` members + shared `aiml` package); tasks run via `just`; the publish path is `.github/workflows/publish.yaml` → `docker build` of `hugo/docker/Dockerfile` (single build stage: golang/alpine + Hugo + dart-sass, no Node, no Python today).
- Reference design: bart.degoe.de "Semantic search in your browser" (working brief with full detail: `._scratch/semantic-search-plan.md`).

## Decisions

### Decision: CorpusSource

**Chosen:** Build the semantic index from Hugo's generated `en.search-data.json`, not from repo markdown.

**Rationale:** It is the only artifact where blog and module-mounted treadmill content co-exist as rendered plain text, and it is exactly what keyword search already indexes — both engines see the same corpus.
It also decouples the index builder from Hugo templating: it is a second pass over Hugo's own output.

**Alternatives considered:**

- Parse `content/*.md`: treadmill markdown isn't in this repo; would fork two corpus definitions.
- Hugo-template-side embedding: no numeric compute in Hugo templates; not viable.

### Decision: PackageLocation

**Chosen:** New top-level uv workspace member `site-search/` — dist name `site-search`, src layout `site-search/src/site_search/`, hatchling backend, conventions mirroring `experiments/aiml` (`[tool.uv] package = true`, pytest `pythonpath = "src"`, per-member `[tool.basedpyright]`).
Root `[tool.uv.workspace].members` gains `"site-search"` explicitly (the `experiments/*` glob doesn't cover it); `.ruff.toml` `known-first-party` gains `site_search`.

**Rationale:** This is build tooling with a stable CLI contract that runs unattended in CI — not a throwaway experiment.
Directory name matches the dist name and the import package derives from it per Python idiom (`site-search` ↔ `site_search`), consistent with hyphenated members like `typos-experiment`.

**Alternatives considered:**

- `experiments/<name>/`: experiments are exploratory per CLAUDE.md; this must not be swept up by `just experiments upgrade` semantics.
- `scripts/`: shell-script land; this is a multi-module package with tests and an eval harness.
- Inside `hugo/`: pollutes the Hugo tree, Docker COPY layer, and the repo's Hugo/Python separation.
- Directory named `search/` with package `site_search`: name mismatch between directory/dist and import package; also `search` is a collision-prone generic name.

### Decision: EmbeddingModel

**Chosen:** `minishlab/potion-base-8M` (model2vec) — a static 29,528-token × 256-dim lookup table.
Forward pass = tokenize → look up rows → mean → L2-normalize.
Loaded in Python via `model2vec`; reimplemented in the client as table lookups.

**Rationale:** No inference runtime in the browser (no ONNX/WASM), millisecond whole-corpus embedding at build time, ~81% of MiniLM retrieval quality at a fraction of size/latency — the entire premise of the reference design.

**Alternatives considered:**

- MiniLM via transformers.js/ONNX: multi-tens-of-MB runtime + model, cold-start cost; overkill for a blog.
- Server-side search: violates the static-site constraint.

### Decision: LocalImplementation

**Chosen:** Own pipeline code in `site-search/` (`model2vec`, `numpy`, and — only if model2vec doesn't vendor enough — `tokenizers` as libraries; no `torch`).
No dependency on the article's `static-site-search-eval` PyPI package.

**Rationale:** The author wants to read, learn from, and own this code; the pipeline is small enough that owning it is cheaper than tracking a third-party package's choices.

**Alternatives considered:**

- Depend on `static-site-search-eval`: faster start, but opaque to the learning goal and couples artifact formats to upstream churn.

### Decision: ChunkingPolicy

**Chosen:** Preserve each search-corpus heading as a section boundary.
Within a section, treat each non-empty newline-delimited body block as the paragraph-like structure available from `en.search-data.json`; greedily pack whole blocks with the page title and heading context under a 256-token potion tokenizer target.
Split only a single oversized block at sentence boundaries.
If one sentence alone exceeds the target, keep it whole rather than cut it arbitrarily.
Emit an exact partition with no overlap.

**Rationale:** Whole semantic units give retrieval and downstream synthesis cleaner evidence than arbitrary character windows.
Heading context keeps a short paragraph interpretable, while the token target bounds mean-pooling dilution in the model's own units.
Overlap duplicates evidence, increases the payload, and confounds attribution in the recall eval.
The generated corpus is flattened plain text: headings and newline-delimited blocks survive, but Markdown block types and blank-line paragraph delimiters do not.
The implementation therefore preserves only structure the corpus actually exposes and does not claim to parse Markdown.

**Alternatives considered:**

- Fixed overlapping character windows: simple, but cut sentences and duplicate evidence across chunks.
- One chunk per newline-delimited block: preserves units, but produces unnecessary manifest/vector rows when several short blocks fit under one heading-aware token target.
- Parse repository Markdown: treadmill content is module-mounted and unavailable as repository files, so this would fork the corpus definition.
- Add a general chunking framework: the required policy is a small deterministic partition over an already flattened corpus; a framework adds a second abstraction without restoring discarded Markdown structure.

### Decision: QuantizationStrategy

**Chosen:** int8 throughout.
Token table: per-row float32 scales (~118KB overhead at 29,528 rows).
Chunk vectors: one global scale.
**Fidelity tolerance: quantized-vs-float32 cosine similarity ≥ 0.999 on sampled pairs** (article measured 0.9998); this is the tolerance `SimilarityFidelity` and `ClientEmbeddingParity` refer to.

**Rationale:** The two structures have different error-propagation profiles.
Token rows get summed (mean-pooled) _before_ the final normalization safety net, so per-row scale errors could compound — per-row scales are cheap insurance.
Chunk vectors are only compared via cosine, which is scale-invariant per-vector, so one global scale cannot distort relative ranking.
Client scoring can therefore run raw int8 dot products with a float accumulator, no dequantization in the hot loop.

**Alternatives considered:**

- float32 artifacts: 4× the payload for no ranking benefit.
- Per-row scales on chunk vectors: manifest complexity for no measurable retrieval gain at this corpus size.

### Decision: Dimensionality

**Chosen:** 256 dimensions.
It is the smallest evaluated representation that meets the frozen-label retrieval floors under a genuine two-engine fusion policy.

The real-corpus comparison used the same 4,458 structure-aware chunks for both candidates.
Payload totals were 6,370,656 bytes at 128 dimensions and 10,720,865 bytes at 256 dimensions.
Measured recall@1 was:

| Dimensions | Exact keyword | Exact semantic | Exact hybrid | Paraphrase keyword | Paraphrase semantic | Paraphrase hybrid | Navigational keyword | Navigational semantic | Navigational hybrid |
| ---------: | ------------: | -------------: | -----------: | -----------------: | ------------------: | ----------------: | -------------------: | --------------------: | ------------------: |
|        128 |           40% |            30% |          20% |                20% |                 40% |               10% |                   0% |                   40% |                 30% |
|        256 |           40% |            40% |          30% |                20% |                 30% |               10% |                   0% |                   60% |                 30% |

Those initial results used unbounded, equal-weight RRF.
A follow-up sensitivity grid evaluated 108 policies per dimensionality across three `k` values, nine keyword-to-semantic weight pairs, and four symmetric input cutoffs.
No 128-dimensional policy met both floors.
At 256 dimensions, all 18 combinations in the broad region of top-5 or top-10 inputs, semantic weight 1.5–3× keyword, and `k` of 10, 30, or 60 met both floors.
The selected interior policy (top 10, keyword:semantic 1:2, `k=60`) produces 40% exact-term hybrid recall@1, matching keyword-only; 30% paraphrase hybrid recall@1, matching semantic-only; and 40%/70% navigational hybrid recall@1/@3.
Full recall, sensitivity, and per-artifact measurements are recorded in `evidence/eval-report.md`.

Verified with model2vec 0.8.2 and `minishlab/potion-base-8M`: the full raw table is 29,528 × 256, loading with `dimensionality=128` produces 29,528 × 128 exactly equal to `full.embedding[:, :128]`, and the model has direct tokenizer-ID rows with no token mapping or weights.
For plain, accented, and partially out-of-vocabulary samples, manual tokenize → row lookup → mean → L2 normalization matched `.encode()` exactly when the installed implementation's batched normalization order was reproduced.
This proves both candidates are mechanically valid; the frozen retrieval gate is what rejects 128 dimensions.

**Rationale:** Dimensionality dominates the binary payload, and payload size determines first-search transfer and initialization cost.
The 256-dimensional artifact set costs 4,350,209 additional bytes (68.3%) over 128 dimensions, so the choice needs retrieval evidence rather than a fixed size budget.
The frozen evaluation rejects the smaller candidate and shows a broad, non-point-tuned passing region for 256 dimensions.
That makes 256 the smallest quality-passing candidate, not an unconditional preference for the larger model.

**Alternatives considered:**

- 128 dimensions: 40.6% smaller total payload, but no tested general fusion policy meets both frozen retrieval floors.
- Retrained smaller model: out of scope; truncation is the reference design's measured approach.

### Decision: TokenizerContract

**Chosen:** The exported `tokenizer-config.json` carries _resolved_ behavior: an explicit boolean for accent stripping (upstream `strip_accents: null` means "inherit from `lowercase`", which is `true` — export `true`, never the tri-state), an explicit no-special-tokens flag (`[CLS]`/`[SEP]` never inserted), and OOV-drop semantics (unknown pieces dropped, never mapped to `[UNK]`).

**Rationale:** All three are documented silent-mismatch traps between HuggingFace-style tokenizers and hand-ported JS.
Resolving them at export time eliminates the whole class of client/server drift; the parity gate then verifies rather than discovers.

**Alternatives considered:**

- Export the raw HF tokenizer JSON: leaves interpretation to the JS port — exactly where the reference article got burned.

### Decision: FusionMethod

**Chosen:** Weighted Reciprocal Rank Fusion over the union of the top 10 keyword and top 10 semantic pages: `score(doc) = 1/(60 + keyword_rank) + 2/(60 + semantic_rank)`.
A page absent from one top-10 input receives no contribution from that engine.
Chunk hits roll up to pages (best chunk per page) before truncation and fusion, mirroring the existing `pageIndex`/`sectionIndex` grouping.

**Rationale:** Keyword scores and cosine similarities are on incomparable scales, while RRF needs only ranks.
The bounded inputs prevent weak long-tail ranks from diluting a decisive rank-1 signal; the 2× semantic weight compensates for the lexical baseline's measured weakness on paraphrase and navigational queries.
This policy is an interior point in a broad passing sensitivity region, while retaining conventional `k=60`.

**Alternatives considered:**

- Score normalization + linear blend: requires calibrating incomparable distributions per query; fragile.
- Semantic-only replacement: regresses exact-term queries, the current engine's strong suit.
- Full-list equal-weight RRF: fails both primary floors because middling agreement can outrank a decisive result from either engine.

### Decision: ClientIntegration

**Chosen:** New `hugo/assets/js/semantic-search.js` alongside the existing `flexsearch.js` override.
Only `flexsearch.js` is processed by Hugo Pipes `resources.ExecuteAsTemplate` (it resolves template-time URLs and params); `semantic-search.js` is template-free and loaded via plain `resources.Get`, which reinforces the pure/glue seam.
Pure logic (WordPiece tokenizer, embedder, int8 scorer, RRF) lives in template-free, side-effect-free functions loadable under Node, with browser glue (fetch, DOM, template-resolved URLs) kept separate.
`flexsearch.js` integrates at two points: the `focus → init() → preloadIndex()` lazy-load chain gains a parallel `Promise.allSettled` artifact fetch, and `search()` fuses ranks before handing results to the unchanged `displayResults()`.

**Rationale:** The Node-loadable seam is what makes client contracts (`ScoringCorrectness`, `ClientEmbeddingParity`, `KeywordFallback` logic, `PageDeduplication`, RRF ordering) testable with runnable evidence — the repo has no browser-automation harness and adding one for this is not worth it.
Rendering through `displayResults()` unchanged preserves highlighting, ARIA, and keyboard nav for free (`PresentationPreserved`).

**Alternatives considered:**

- Rewrite the search UI: churn with zero user value; loses the theme's accessibility work.
- Playwright/browser tests: heavy new infra for two browser-only observables — waived to manual QA instead (§ Verification Waivers).

### Decision: BuildIntegration

**Chosen:** Two-pass Hugo build inside the existing single Docker build stage: (1) `hugo -s ./hugo` to materialize `en.search-data.json`; (2) gate + `python -m site_search.build_index` writing into `hugo/static/search/`; (3) `hugo --minify -s ./hugo` — static files copy into `public/` and stage 2 (nginx) needs no changes.
Python arrives as uv-managed CPython 3.12.13 in the build stage; automatic Python downloads are disabled after explicit installation.
Runtime dependencies sync from wheels only, without installing the local project, while `PYTHONPATH` exposes `site-search/src`.
This makes Alpine dependency upgrades fail clearly unless they provide musllinux wheels; a dependency that only publishes glibc manylinux wheels requires either a musllinux wheel or a glibc build stage, not an implicit source build with compiler, Rust, or BLAS toolchains.
The root resolution metadata plus `site-search/pyproject.toml` and `site-search/src` are copied; unrelated workspace package sources are not.
Local parity via `just hugo search-index` (same three steps).
`hugo/static/search/` is gitignored.

**Rationale:** `en.search-data.json` is a Hugo _output_ while `hugo/static/search/` must exist as a Hugo _input_ — a cycle unless the build runs twice.
Two passes keep the Dockerfile single-stage and Node-free; the incremental cost is one extra Hugo render on `main` publishes that touch Hugo, site-search, or their shared Python resolution metadata.
Also add `site-search/**` to the publish workflow's paths filter so builder changes trigger a republish.
Local Podman builds request Docker image format so the Dockerfile `SHELL` contract matches Docker CI.
Native and multi-architecture builds both recreate the concise `:debug` tag as a manifest, avoiding Podman's image/manifest type collision and duplicate platform entries when developers switch between build modes.
Because Buildah does not reliably inject Docker's automatic `BUILDARCH`, the build script derives the build-stage architecture from `podman info` and passes it explicitly.
On Darwin, the Podman shim treats both a missing socket and a failing real `podman info` as unhealthy, restarts the machine when needed, and waits up to 30 seconds for the real API before failing with a diagnostic.

**Alternatives considered:**

- Committing artifacts to git: staleness drift between artifacts and content; repo bloat.
- Separate Docker stage / CI workflow step: more moving parts for the same three commands.
- Hugo mounts/assets pipeline for artifacts: Hugo cannot generate them; they must pre-exist a pass.

### Decision: FallbackPolicy

**Chosen:** Hybrid is always-on when artifacts are ready; every unavailability mode (fetch failure, fetch pending, fully-OOV query) degrades silently to today's keyword-only path as an explicit, tested code path.
No user-facing toggle in v1; `?searchdebug=1` side-by-side comparison is a dev-only stretch item.

**Rationale:** The keyword engine is the reliability floor (`search-never-breaks`); semantic is an enhancement layered over it, never a dependency of it.

**Alternatives considered:**

- Blocking first search on artifact load: trades guaranteed availability for marginal quality on the first keystroke.
- User-facing engine toggle: UI complexity nobody asked for; the eval harness answers "which engine is better" offline.

## Architecture

```text
                     BUILD TIME (Docker build stage / just hugo search-index)
┌──────────┐  pass 1   ┌─────────────────────┐        ┌─────────────────────────────┐
│   Hugo   │──────────▶│ en.search-data.json │───────▶│ site-search (Python)        │
└──────────┘           └─────────────────────┘  gate  │ loader → gate ─▶ FAIL loud  │
     ▲                                                │ chunker (heading + blocks,  │
     │                                                │ 256 tokens, sentence split)│
     │ pass 2 (--minify)                              │ embed (potion-base-8M)      │
     │  copies static/ → public/                      │ quantize (int8)             │
┌────┴────────────────────┐                           │ export + size report        │
│ hugo/static/search/     │◀──────────────────────────┴─────────────────────────────┘
│  token-table.bin        │
│  token-scales.bin       │        RUN TIME (browser)
│  doc-vectors.bin        │  lazy fetch on first focus (Promise.allSettled)
│  manifest.json          │─────────────────────────────────┐
│  tokenizer-config.json  │                                 ▼
│  meta.json              │   ┌──────────────────────────────────────────────┐
└─────────────────────────┘   │ flexsearch.js (override)                     │
                              │  keyword: FlexSearch (unchanged) ──┐         │
                              │  semantic-search.js:               ├─▶ RRF ─▶│─▶ displayResults()
                              │   WordPiece → embed → int8 dot ────┘ top-10, │   (unchanged UI)
                              │                                  1:2, k=60  │
                              │   rollup best-chunk-per-page                 │
                              │  any failure ──▶ keyword-only path           │
                              └──────────────────────────────────────────────┘

                     EVAL (site-search, offline)
labeled queries ──▶ keyword-only / semantic-only / hybrid (shipped JS logic via node)
                ──▶ recall@1 (primary), recall@3 (context); tokenizer parity gate (py ↔ js)
```

## Risks

- **Vocab misses on ML/arxiv jargon** (potion vocab is general-purpose): hybrid's division of labor — keyword already owns exact tokens; enforced by `RetrievalQualityFloor`'s exact-term subset.
- **Theme-upgrade drift** (`flexsearch.js`, `fragments.html` shadow v0.12.0 files with no merge signal): maintenance checklist below; re-diff both on every Hextra bump.
  Biome reformatted the committed `flexsearch.js`, so normalize upstream's copy with biome before diffing.
- **`en.search-data.json` is an undocumented internal theme contract**: the gate's structural validation is the format canary (`FormatCanary`) — rerun after any theme bump; a parse failure or suspicious-empty spike flags the contract change before the pipeline ingests garbage.
- **Silent JS/Python tokenizer mismatch** (wrong results, no crash): resolved-config export (`TokenizerContract`) plus the blocking `TokenizerParityGate` — a mismatch fails the suite, it is not advisory.
- **Silent int8 overflow in the JS dot product** (a true ~3,000,000 wraps to -64): float accumulator required; `ScoringCorrectness` has a dedicated high-magnitude scenario because casual QA cannot catch it.
- **CI build-time regression from the second Hugo pass**: publishes are gated to `main` pushes touching Hugo, site-search, or their shared lock/config files; monitor CI duration after rollout, act only if it becomes a real bottleneck.
- **Payload growth from 256 dimensions**: the selected artifact set is 68.3% larger than the rejected 128-dimensional candidate.
  Keep exact byte reporting in the build and re-evaluate smaller representations when they can be tested against the frozen retrieval floor.

## Maintenance: Hextra upgrade checklist

On every theme version bump (currently pinned v0.12.0 in `hugo/go.mod`):

1. Re-diff `hugo/assets/js/flexsearch.js` against upstream (biome-normalize upstream first).
   Divergences to preserve: first-`#` key split (droppable once upstream's fix ships in a release), semantic-search hooks (lazy fetch + RRF fusion).
2. Re-diff `hugo/layouts/_partials/utils/fragments.html` against upstream; drop the override only if upstream fixes heading-link splitting.
3. Rebuild and rerun the corpus gate (`python -m site_search.gate`) — it is the tripwire for search-data format changes, independent of visible breakage.

## Verification Waivers

- **Requirement:** DeferredSearchPayload **Reason:** The contract is browser network behavior (no artifact fetches before first search-input focus); the repo has no browser-automation harness and adding Playwright for one observable is disproportionate.
  The pure-logic side (single-initialization guard) is unit-testable, but the no-fetch-on-page-load property is only observable in a real browser.
  **Manual evidence:** Rollout QA runbook step — devtools network capture on page load and on first focus, recorded in `site-search/.specs/changes/hybrid-site-search/evidence/rollout-qa.md` (created by the rollout task group).
  **Recorded:** 2026-07-14 — executed on the local dev serve, Check 1 PASS (load only on first focus, once per page; see `evidence/rollout-qa.md`).

- **Requirement:** PresentationPreserved **Reason:** Highlighting, keyboard navigation, and screen-reader announcements are DOM/AT behaviors of the theme's existing UI; automated coverage requires browser + accessibility-tree tooling the repo does not have.
  The design keeps `displayResults()` unchanged specifically so this holds by construction — hybrid results are the same result-object shape the keyword path already renders, and the `.hextra-search-status` `aria-live` announcement is emitted by the untouched `displayResults()`, so AT behavior is preserved structurally rather than by re-implementation.
  **Manual evidence:** Rollout QA runbook step — manual keyboard-nav/highlight/rollup pass, recorded in `site-search/.specs/changes/hybrid-site-search/evidence/rollout-qa.md` (created by the rollout task group).
  **Recorded:** 2026-07-14 — keyboard navigation, highlighting, and rollup executed and PASS (see `evidence/rollout-qa.md` Check 2).
  **Scope decision (maintainer):** for v1, keyboard + visual coverage is accepted as sufficient evidence for this waiver; a dedicated screen-reader (VoiceOver) pass is not required, because hybrid results route through the unchanged `displayResults()` and its unchanged `aria-live` status element, so no AT-announcement code path is new in this change.
  The waiver is considered met on this basis.
