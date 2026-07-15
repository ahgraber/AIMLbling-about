# North Star — Site Search

> Product intent and enduring principles for the site-search capability — independent of blog-content and experiment specs.
> Baseline `specs/` define contracts; per-change `design.md` records decisions.
> On any conflict, this north star and the specs win.
> Every change's user stories (in `proposal.md`) MUST ladder to a north-star outcome below;
> every delta-spec requirement that advances one carries a `Serves: <story>` backlink.

## The need

Readers arrive at search with a half-memory: a paraphrase of an idea, a library someone mentioned, a fragment of an arxiv title.
Keyword-only search fails them in two compounding ways:

- **Vocabulary mismatch.**
  Keyword search returns nothing when the reader's words differ from the post's words — paraphrases and topic queries dead-end at zero results.
- **Silent corpus rot.**
  The index is built from an undocumented theme contract with no guardrails; a theme bug silently emptied 75% of the corpus, and nothing noticed until a manual audit.
  Search that degrades invisibly is worse than search that fails loudly.

## What it is

**Hybrid client-side search**: the existing keyword engine fused with static-embedding semantic retrieval, running entirely in the reader's browser against a static site — no server component, no inference runtime.
A local Python pipeline builds the semantic index from Hugo's own rendered corpus, behind a completeness gate that fails the build rather than ship a broken index.

The single measure of success: **the half-remembered query lands the right post at rank 1.**

## Who it serves

Readers first: search must work for someone who doesn't know this site's tooling and will never see an error message.
The author second, in two roles — the site's heaviest search user, and the maintainer who must be able to read, test, and evolve the pipeline alone across theme upgrades.

## North-star outcomes (what changes ladder to)

1. **Find by meaning.**
   As a reader, I find content by paraphrase or half-memory — the right post or treadmill entry surfaces without me knowing its exact words.
2. **Exact terms still win.**
   As a reader searching a library name, paper ID, or proper noun, exact matches rank at least as well as they do today — the upgrade is purely additive.
3. **Search never breaks.**
   As a reader, when semantic data is missing, slow, or failing, search silently degrades to keyword-only — an enhancement can never take down basic search.
4. **Zero cost until used.**
   As a reader, I pay no search-payload cost until I engage search, and the smallest practical cost afterward — the site stays fast.
5. **Trustworthy corpus.**
   As the author, the index build verifies corpus completeness and fails loudly on regressions or theme-format drift — a broken index can never silently ship again.
6. **Learnable, maintainable pipeline.**
   As the author, the whole pipeline is local, readable, and tested — I can learn from it, publish without manual steps, and keep it working across theme upgrades.

## Guiding principles

- **Keyword is the floor; semantic is a layer.**
  The keyword engine is the reliability baseline.
  Semantic search enhances it and must never become a dependency of it — every failure mode has a keyword-only landing.
- **Fail loudly at build time, degrade silently at run time.**
  Gate failures, unexplained payload regressions, and parity mismatches block the publish; runtime failures degrade to keyword-only with no user-facing noise.
  The build is where problems are cheap; the reader is where they are expensive.
- **Measured, not assumed.**
  Quality claims are re-verified on this corpus — recall floors, dimensionality, chunk size — not inherited from the reference design.
  A knob's default is a hypothesis until the eval confirms it.
- **Parity by construction.**
  Build-time and client behavior (tokenization, normalization, embedding) must be provably identical: ambiguity is resolved once at export, and blocking parity gates enforce it.
  Client/server drift is a silent-wrong-results bug, the worst kind.
- **Size is a feature.**
  Measure every artifact and choose the smallest representation that meets the retrieval-quality floor; quality gains must earn their bytes.
- **Own the code, borrow the ideas.**
  Libraries for primitives (model loading, arrays); local, readable implementation for the pipeline itself.
  If the author can't explain it, it doesn't ship.

## Scope boundaries (non-goals)

- No server-side search component — the site stays fully static.
- No neural inference runtime in the browser (ONNX, WASM, transformers.js) — the model is a lookup table or it doesn't ship.
- No ANN index (HNSW, IVF) — brute force wins at blog scale; revisit only if the corpus grows by orders of magnitude.
- No embedding cache or incremental re-embedding — whole-corpus builds are milliseconds; cache invalidation complexity buys nothing.
- Not a general search product — one site, one corpus (blog + treadmill), one language until the corpus says otherwise.

## Current horizon

- **v1 — hybrid always-on.**
  RRF-fused keyword + semantic with lazy-loaded artifacts, the corpus completeness gate, an eval harness enforcing recall floors and tokenizer parity, and two-pass build integration — no user-facing toggle.
- **Beyond v1, as their own changes** — a dev-only side-by-side comparison view (`?searchdebug=1`), prefetch-on-idle if first-query latency warrants it, and re-tuning dimensionality/chunking as the corpus grows.
