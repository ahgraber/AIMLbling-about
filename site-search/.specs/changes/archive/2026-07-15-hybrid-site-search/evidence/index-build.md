# Fresh-render semantic index build evidence

Recorded: 2026-07-12

## Provenance

The supported local recipe rendered the current Hugo site, gated that newly generated `hugo/public/en.search-data.json`, and built the semantic artifacts from it.
Hugo 0.155.3 rendered 408 pages.
The model resolved from the local Hugging Face cache.

## Command

<!-- rumdl-disable MD013 -->

```text
nix develop -c just hugo search-index
```

Exit status: 0.
The recipe ran Hugo first, then `site_search.gate`, then `site_search.build_index` against the freshly rendered corpus.

## Gate report

```text
blog: total=322 legitimate_empty=0 suspicious_empty=0
treadmill: total=2976 legitimate_empty=20 suspicious_empty=3
total: total=3298 legitimate_empty=20 suspicious_empty=3 suspicious_rate=0.091% limit=1.000%
suspicious: /treadmill/2025/07/11/ 'the-elements-of-differentiable-programming#The Elements of Differentiable Programming'
suspicious: /treadmill/2026/01/13/ 'applied-ai#Applied AI'
suspicious: /treadmill/2026/04/17/ 'tipsv2-advancing-vision-language-pretraining-with-enhanced-patch-text-alignment#TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment'
gate: PASS
```

<!-- rumdl-enable MD013 -->

## Build result

- Model: `minishlab/potion-base-8M`
- Dimensions: 256
- Manifest/document rows: 4,458
- Excluded chunks: 0
- Document global scale: `0.004180654883384705`
- Metadata build timestamp: `2026-07-13T01:06:39.229615+00:00`

Exact artifact sizes:

| Artifact                |          Bytes |
| ----------------------- | -------------: |
| `token-table.bin`       |      7,559,168 |
| `token-scales.bin`      |        118,112 |
| `doc-vectors.bin`       |      1,141,248 |
| `manifest.json`         |      1,455,612 |
| `tokenizer-config.json` |        446,398 |
| `meta.json`             |            327 |
| **Total**               | **10,720,865** |

`meta.json` reported the same 10,720,865-byte total and corpus statistics of 3,298 sections, 20 legitimate empties, and 3 suspicious empties.
The current corpus supersedes the proposal-time 3,072-section estimate; the 1% suspicious-rate threshold remains appropriate because the measured rate is 0.091% and all three suspicious records are printed for review.

## Local serving check

After the recipe completed, `nix develop -c hugo server --bind 127.0.0.1 --port 13137 --disableFastRender -s hugo` rendered 408 pages and served the generated static directory.
Direct HTTP requests returned status 200 with byte counts matching the build report for all six artifacts:

| URL                             | HTTP status |     Bytes |
| ------------------------------- | ----------: | --------: |
| `/search/token-table.bin`       |         200 | 7,559,168 |
| `/search/token-scales.bin`      |         200 |   118,112 |
| `/search/doc-vectors.bin`       |         200 | 1,141,248 |
| `/search/manifest.json`         |         200 | 1,455,612 |
| `/search/tokenizer-config.json` |         200 |   446,398 |
| `/search/meta.json`             |         200 |       327 |
