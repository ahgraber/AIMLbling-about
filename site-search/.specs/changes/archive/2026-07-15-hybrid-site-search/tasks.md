# Tasks: hybrid-site-search

## Workspace scaffolding

- [x] Create `site-search/pyproject.toml` (name `site-search`, hatchling, src layout, runtime deps `model2vec` + `numpy`, dev/test groups mirroring `experiments/aiml` conventions) and `site-search/src/site_search/__init__.py`
- [x] Add `"site-search"` to root `pyproject.toml` `[tool.uv.workspace].members`; add `site_search` to `.ruff.toml` `known-first-party`
- [x] Verify `uv sync --package site-search` resolves and `uv run --package site-search python -c "import site_search"` succeeds (captured output)

## Corpus gate (search-data-integrity)

- [x] Implement `site_search/loader.py`: parse `en.search-data.json` into `(route, heading_key, title, fragment_text)` records; raise a structural error on unexpected shape (single parser shared with the index builder — no fork)
- [x] Implement `site_search/gate.py`: classify each section as non-empty / legitimate-empty (title-only entry, empty intro slot, anchor-less level-1 heading) / suspicious-empty; report totals and per-source (blog vs treadmill) breakdown; exit non-zero above the suspicious-rate threshold; CLI `python -m site_search.gate <path>`
- [x] Create `site-search/tests/fixtures/search-data.sample.json` covering: plain heading with body, markdown-link heading with body, title-only entry, pre-first-heading intro, and a structurally malformed variant
- [x] Test `test_gate.py::test_section_content_completeness`: plain-heading, link-heading, title-only, and intro scenarios classify and carry content correctly (evidence: SectionContentCompleteness — all four scenarios)
- [x] Test `test_gate.py::test_gate_threshold`: below-threshold corpus passes; above-threshold corpus fails with diagnostics and no artifact output (evidence: CorpusGateBlocksSuspiciousCorpus — both scenarios)
- [x] Test `test_gate.py::test_format_canary`: malformed corpus fails with a structural diagnostic before any classification (evidence: FormatCanary)
- [x] Test `test_gate.py::test_classification_report`: report contains totals and legitimate/suspicious counts per content source (evidence: EmptyClassificationReport)
- [x] Run the gate against a fresh Hugo build; capture the measured baseline (3,298 sections / 20 legitimate empties / 3 suspicious empties, 0.091%); confirm the 1% threshold from real numbers (evidence: `evidence/index-build.md`)

## Index builder (search-index-artifacts)

- [x] Spike: confirm model2vec `StaticModel` exposes the raw token-embedding table and tokenizer (needed for `token-table.bin`); confirm potion-base-8M truncation robustness for the D=128 default; record findings in `design.md` (Dimensionality decision)
- [x] Implement `site_search/embedding.py`: load potion-base-8M, embed chunks; assert `.encode()` parity with manual lookup→mean→normalize so the client reimplementation has a fixed reference
- [x] Implement `site_search/chunking.py`: preserve heading boundaries; greedily pack whole newline-delimited blocks with page/heading context under 256 potion tokens; sentence-split only an oversized block; emit no overlap
- [x] Test `test_chunking.py`: whole-block packing, 256-token target, sentence fallback, heading isolation/context, deterministic ordering, and exact no-overlap partition (evidence: StructuralChunkIntegrity — all four scenarios)
- [x] Implement `site_search/quantize.py`: per-row int8 quantization for the token table, global-scale int8 for chunk vectors
- [x] Test `test_quantize.py::test_roundtrip_fidelity`: quantized-vs-float32 cosine ≥ 0.999 on sampled pairs (evidence: SimilarityFidelity — round-trip scenario)
- [x] Test `test_quantize.py::test_ranking_preserved`: top-k nearest neighbors of sample queries identical between quantized and float32 (evidence: SimilarityFidelity — ranking scenario)
- [x] Implement `site_search/export.py`: write `token-table.bin`, `token-scales.bin`, `doc-vectors.bin`, `manifest.json`, `tokenizer-config.json` (resolved accent boolean, no-special-tokens flag, OOV-drop semantics), `meta.json` (dims, global scale, model id, corpus stats, total payload size)
- [x] Test `test_export.py::test_artifact_set_complete`: all six artifacts produced; manifest rows ↔ vector rows 1:1 in order (evidence: SelfSufficientArtifactSet — complete-set and correspondence scenarios)
- [x] Test `test_export.py::test_correspondence_after_exclusion`: corpus with excludable chunks still yields aligned manifest/vector rows (evidence: SelfSufficientArtifactSet — exclusion write-site)
- [x] Test `test_tokenizer_config.py`: tokenizing accented, OOV-bearing, and plain strings with only the exported config reproduces reference token ids; no special tokens present (evidence: ResolvedTokenizerConfig — all three scenarios)
- [x] Test `test_export.py::test_empty_chunk_exclusion`: fully-OOV chunk excluded and reported; exported vectors contain no zero/NaN rows (evidence: EmptyChunkExclusion — both scenarios)
- [x] Wire CLI entry `[project.scripts] build-index = site_search.build_index:main` (gate → chunk → embed → quantize → export; `--search-data`/`--out` params); run against the real corpus and capture `meta.json` stats (captured output)

## Client hybrid search (client-hybrid-search)

- [x] Create `hugo/assets/js/semantic-search.js` skeleton: pure logic (tokenizer, embedder, scorer, RRF, rollup) as template-free functions loadable under Node; browser glue (artifact fetch, Hugo-template URLs) separated
- [x] Port WordPiece tokenizer (~80 lines) consuming the exported `tokenizer-config.json` (resolved accent boolean, no special tokens, OOV dropped)
- [x] Implement query embedding: token lookup with per-row dequantization → mean → normalize; return a sentinel (not a zero vector) when zero tokens are known
- [x] Implement int8 scoring loop with a plain-`Number` (float64) accumulator against raw `doc-vectors.bin` bytes
- [x] Node test `scoring.test`: ordinary vectors match reference similarity; high-magnitude vectors that would wrap a fixed-width integer accumulator score correctly (evidence: ScoringCorrectness — both scenarios)
- [x] Implement RRF fusion (k=60) over keyword + semantic rank lists, and best-chunk-per-page rollup before fusion
- [x] Node test `fusion.test`: both-signal result outranks single-signal results; unavailable/pending/fully-OOV semantic input yields keyword-only ordering unchanged (evidence: HybridRanking — agreement scenario; KeywordFallback — all three scenarios at logic level)
- [x] Node test `rollup.test`: single-chunk and multi-chunk-same-page queries each present the page once with its best section (evidence: PageDeduplication — both scenarios)
- [x] Integrate into `hugo/assets/js/flexsearch.js`: parallel artifact fetch in the `focus → init()` chain via `Promise.allSettled` with a single-initialization guard; fuse ranks in `search()` before the unchanged `displayResults()`; keyword path proceeds whenever semantic state is not ready
- [x] Build the site with artifacts absent and verify search behaves exactly as today (keyword-only, console clean) (evidence: KeywordFallback — integration-level; also exercises PresentationPreserved by construction) — verified in a real browser via `evidence/rollout-qa.md` Check 3 (blocked `/search/*` → keyword-only, no user-facing error, no uncaught exception) and automated at load-time by `hugo/assets/js/semantic-search.integration-fallback.test.js`

## Eval harness (search-quality-eval)

- [x] Author the labeled query set (~30 queries: exact-term / paraphrase / navigational) from the real corpus (real library names, arxiv titles, post topics), stored as a fixture with expected top page/section
- [x] Implement `site_search/eval.py`: run keyword-only, semantic-only, and hybrid conditions — semantic/fusion via the shipped `semantic-search.js` pure logic executed under Node so the eval measures the real system; report recall@1 (primary) and recall@3 (context) per subset
- [x] Test `test_parity.py`: fixed representative strings (accents, code identifiers/casing, deliberately-OOV; CJK if the corpus contains any, else marked N/A) through the Python reference and Node tokenizer — identical token-id sequences required, any mismatch fails (evidence: TokenizerParityGate — both scenarios; ClientEmbeddingParity — token-sequence half of all three scenarios)
- [x] Test `test_parity.py::test_embedding_vectors`: client (Node) query vectors match Python reference vectors within the design fidelity tolerance on plain/accented/partial-OOV strings (evidence: ClientEmbeddingParity — vector half)
- [x] Run the dims comparison (128 vs 256) on the eval set; record recall and per-artifact/total payload sizes in `design.md`, selecting the smallest candidate that meets the retrieval floor (Dimensionality; evidence: PayloadSizeVisibility — quality-comparison scenario)
- [x] Test `test_eval_floors.py`: hybrid recall@1 ≥ keyword-only on exact-term subset AND ≥ semantic-only on paraphrase subset; navigational subset reported for all three conditions; capture the eval report to `site-search/.specs/changes/hybrid-site-search/evidence/eval-report.md` (evidence: RetrievalQualityFloor — all three scenarios; HybridRanking — paraphrase-finds-content and exact-term-stays-on-top scenarios, sampled by the labeled set)

## Build pipeline (search-build-pipeline)

- [x] Add `just hugo search-index` recipe: `hugo -s hugo` → gate → `build-index` → artifacts in `hugo/static/search/`; document the local flow (`search-index` → `demo`) in the justfile
- [x] Run the recipe from a clean state and verify the dev server serves the artifact set (evidence: `evidence/index-build.md`; LocalIndexRecipe)
- [x] Add `hugo/static/search/` to `.gitignore`; test `test_gitignore.py` asserts `git check-ignore` matches every artifact path (evidence: GeneratedArtifactsUntracked)
- [x] Report exact per-artifact and total payload sizes on every build; test the report against filesystem byte counts (evidence: PayloadSizeVisibility — exact-size scenario)
- [x] Update `hugo/docker/Dockerfile` build stage: install uv/Python, `COPY site-search/pyproject.toml site-search/src ./site-search/`, two-pass Hugo build with gate + build-index between passes; add `site-search/**` to `publish.yaml` paths filter
- [x] Harden local Podman builds with Docker image format, positional platform validation, and bounded Darwin machine API readiness checks
- [x] Pin wheel-only uv-managed Python and select Dart Sass from the build-stage architecture in the Alpine container
- [x] Local container build: verify `/site/search/` contains the artifact set generated during that build (evidence: `evidence/published-artifact-currency.md`; PublishedArtifactCurrency — fresh-publish scenario)
- [x] Change a temporary content fixture, rebuild, and verify shipped artifacts reflect the change (evidence: `evidence/published-artifact-currency.md`; PublishedArtifactCurrency — regeneration scenario)

## Rollout QA (waiver evidence)

- [x] Create `site-search/.specs/changes/hybrid-site-search/evidence/rollout-qa.md` runbook and record: devtools network capture showing zero artifact requests on page load and exactly one fetch per artifact on first search focus (waiver evidence: DeferredSearchPayload) — executed on local dev serve, Check 1 PASS
- [x] Record in the runbook: manual pass over highlighting, keyboard navigation, and screen-reader announcements with hybrid results (waiver evidence: PresentationPreserved) — keyboard/highlight/rollup executed and PASS; screen-reader pass waived by maintainer scope decision (keyboard + visual accepted as sufficient for v1, recorded in `design.md` § Verification Waivers and `evidence/rollout-qa.md` Check 2)
- [x] Record in the runbook: devtools-blocked artifact requests still yield clean keyword-only search in a real browser (supplementary evidence: KeywordFallback) — executed, Check 3 PASS (no user-facing error, no uncaught exception)
- [ ] Run a subset of the eval queries against the deployed site and record observed top results in the runbook
