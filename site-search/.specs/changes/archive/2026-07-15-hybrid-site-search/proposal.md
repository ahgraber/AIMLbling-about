# Proposal: hybrid-site-search

## Intent

Site search today is keyword-only (FlexSearch via Hextra) and misses queries that paraphrase content instead of quoting it — a real cost on a blog whose posts and AI-treadmill roundups are exactly the kind of material one half-remembers.
A recently fixed corpus bug (75% of sections indexed empty) also showed the search pipeline has no guardrails: silent upstream changes can gut search without anyone noticing.
This change adds client-side semantic search fused with the existing keyword engine, built by a local, testable Python pipeline, with a completeness gate so the corpus can never silently regress again.

## User Stories

Each story ladders to a numbered north-star outcome in `../../NORTH-STAR.md`; this founding change maps 1:1.

### Story: find-by-meaning

As a blog reader, I want search to surface relevant posts when my query paraphrases the content rather than quoting it, so that I can find what I half-remember without knowing the exact words used.

Ladders to: outcome 1 — Find by meaning.

### Story: no-regressions

As a blog reader, I want the existing search experience — exact-term ranking, result presentation, keyboard and screen-reader affordances — to work at least as well as it does today, so that the semantic upgrade is purely additive.

Ladders to: outcome 2 — Exact terms still win.

### Story: search-never-breaks

As a blog reader, I want search to keep working in keyword-only mode when semantic data is missing, slow, or fails to load, so that an enhancement can never take down basic search.

Ladders to: outcome 3 — Search never breaks.

### Story: zero-cost-until-used

As a blog reader, I want pages to load without paying any search-payload cost until I actually use search, so that the site stays fast.

Ladders to: outcome 4 — Zero cost until used.

### Story: trustworthy-corpus

As the site author, I want the index build to verify corpus completeness and fail loudly on regressions or theme-format changes, so that a broken search index can never silently ship again.

Ladders to: outcome 5 — Trustworthy corpus.

### Story: learnable-maintainable-pipeline

As the site author, I want the entire pipeline implemented locally in my own repo with tests and an automated build, so that I can read it, learn from it, maintain it across theme upgrades, and publish without manual steps.

Ladders to: outcome 6 — Learnable, maintainable pipeline.

## Scope

Capabilities, in build-dependency order:

1. `search-data-integrity` — corpus completeness contract and gate (format canary)
2. `search-index-artifacts` — Python index builder and the artifact set contract
3. `client-hybrid-search` — browser-side embedding, scoring, fusion, fallback
4. `search-quality-eval` — retrieval-quality floors and tokenizer parity evidence
5. `search-build-pipeline` — local recipe, Docker/CI integration, payload measurement

**In scope:**

- New uv workspace member `site-search/` (dist `site-search`, import package `site_search`, src layout) housing the index builder, corpus gate, and eval harness
- Corpus completeness gate over Hugo's generated `en.search-data.json`, runnable standalone, doubling as a theme-upgrade format canary
- Static-embedding index build (model2vec potion-base-8M), int8 quantization, artifact export to `hugo/static/search/`
- Client JS: WordPiece tokenizer port, query embedding, brute-force scoring, Reciprocal Rank Fusion into the existing `flexsearch.js` override; lazy artifact loading; keyword-only fallback
- Eval harness: labeled query set, keyword/semantic/hybrid comparison, recall@1 primary, JS/Python tokenizer parity gate
- Build integration: `just hugo search-index` recipe, two-pass Hugo Docker build, `.gitignore` for generated artifacts, and per-artifact payload measurement

**Out of scope:**

- ANN/approximate indexes (HNSW, IVF) — brute force wins at this corpus scale
- Any server-side search component — the site stays fully static
- transformers.js / ONNX / WASM neural inference — the premise is a lookup-table model, no runtime
- Embedding cache / incremental re-embedding between builds — whole-corpus embedding is milliseconds
- User-facing engine toggle — hybrid is always-on in v1; a dev-only `?searchdebug=1` comparison view is a stretch item, not committed scope

## Approach

Follow the architecture of bart.degoe.de's "Semantic search in your browser" with a local implementation (no dependency on the article's PyPI package): a Python builder reads Hugo's own `en.search-data.json` (the only place blog and module-mounted treadmill text co-exist as plain text), preserves each heading boundary, greedily packs its newline-delimited body blocks under a 256-token target, and splits only an oversized block at sentence boundaries, with no overlap.
It embeds the heading context with each chunk using potion-base-8M (a 29,528×256 lookup table — no inference runtime), quantizes to int8 (per-row scales for the token table, one global scale for chunk vectors), and exports binary artifacts plus a tokenizer config with all normalization ambiguity resolved (notably `strip_accents: null` → effective `true`).
The client ports WordPiece (~80 lines; no special tokens, OOV dropped, float accumulator to dodge int8 overflow), embeds the query, brute-force scores all chunks, rolls chunks up to pages, and fuses the top 10 results from each engine with weighted RRF (semantic weight 2, keyword weight 1, k=60) before the existing `displayResults()`.
Artifacts lazy-load on first search focus behind `Promise.allSettled`; every failure mode degrades to today's keyword-only behavior.
The Docker build gains a two-pass Hugo step (render → build index into `hugo/static/search/` → render with `--minify`) so artifacts ship with every publish.
The build reports every artifact's size, and the eval chooses the smallest dimensionality and chunk representation that meet the retrieval-quality floor rather than enforcing an arbitrary byte ceiling.

Mechanism details, thresholds, and alternatives are formalized in `design.md`; the full working brief is `._scratch/semantic-search-plan.md`.

## Open Questions

- Further payload reductions beyond the evaluated 128- and 256-dimensional candidates, provided they preserve the frozen retrieval-quality floor.
- Does the corpus contain CJK text anywhere (treadmill quotes)?
  Determines whether the tokenizer parity test needs a CJK category or marks it N/A.
- Artifact fetch trigger: focus-only (reference design, default) vs. also prefetch-on-idle — revisit only if rollout QA shows noticeable first-query delay.
