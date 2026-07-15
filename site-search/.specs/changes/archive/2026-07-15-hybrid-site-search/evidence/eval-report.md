# Hybrid site search evaluation report

Recorded: 2026-07-12

## Outcome

**PASSED — 256 dimensions meets both required retrieval floors with the shipped fusion policy.**

The selected production policy rolls semantic chunks up to pages, independently truncates keyword and semantic page rankings to 10, and applies weighted RRF with `k=60`, keyword weight 1, and semantic weight 2.
The 30 labels remained frozen; their SHA-256 is `6cd1e3275feef05ad15cc15a77627beb630b13e751cb5061feda798f05dd3e10`.

## Provenance and method

- Corpus: existing `hugo/public/en.search-data.json`, SHA-256 `c80a4f3aaaf2a154bedcf31965bd8dce25ecad192ade9e5dd8e7767581c6cd1d`.
- Corpus contents: 3,298 sections; 20 legitimate empties; 3 suspicious empties; gate passed at 0.091% versus the 1.000% limit.
- Model: cached `minishlab/potion-base-8M`, network disabled.
- Both candidates used the same 4,458 heading-aware chunks with zero exclusions.
- Semantic embedding, document scoring, best-chunk page rollup, truncation, and weighted RRF executed through the shipped `hugo/assets/js/semantic-search.js` from the fixed Node bridge.
- Keyword-only is the evaluator's deterministic lexical reference: case-folded Unicode word tokens, summed query-term frequency, best section per page, input order for ties.
  It is not claimed to be browser FlexSearch-identical.
- Recall is page-level against canonical `page_id`; recall@1 is primary and recall@3 supplies context.

## Retrieval results

### 128 dimensions

| Condition     | Subset       | Queries | Recall@1 | Recall@3 |
| ------------- | ------------ | ------: | -------: | -------: |
| Keyword-only  | Exact-term   |      10 |    40.0% |    50.0% |
| Keyword-only  | Paraphrase   |      10 |    20.0% |    40.0% |
| Keyword-only  | Navigational |      10 |     0.0% |    10.0% |
| Semantic-only | Exact-term   |      10 |    30.0% |    50.0% |
| Semantic-only | Paraphrase   |      10 |    40.0% |    60.0% |
| Semantic-only | Navigational |      10 |    40.0% |    70.0% |
| Hybrid        | Exact-term   |      10 |    30.0% |    60.0% |
| Hybrid        | Paraphrase   |      10 |    20.0% |    70.0% |
| Hybrid        | Navigational |      10 |    50.0% |    50.0% |

The smaller candidate fails both primary floors: exact-term hybrid 30.0% is below keyword-only 40.0%, and paraphrase hybrid 20.0% is below semantic-only 40.0%.

### 256 dimensions

| Condition     | Subset       | Queries | Recall@1 | Recall@3 |
| ------------- | ------------ | ------: | -------: | -------: |
| Keyword-only  | Exact-term   |      10 |    40.0% |    50.0% |
| Keyword-only  | Paraphrase   |      10 |    20.0% |    40.0% |
| Keyword-only  | Navigational |      10 |     0.0% |    10.0% |
| Semantic-only | Exact-term   |      10 |    40.0% |    60.0% |
| Semantic-only | Paraphrase   |      10 |    30.0% |    80.0% |
| Semantic-only | Navigational |      10 |    60.0% |    70.0% |
| Hybrid        | Exact-term   |      10 |    40.0% |    70.0% |
| Hybrid        | Paraphrase   |      10 |    30.0% |    80.0% |
| Hybrid        | Navigational |      10 |    40.0% |    70.0% |

The selected candidate meets both primary floors exactly and reports navigational metrics for every condition.

## Exact payload sizes

| Artifact                | 128 dimensions | 256 dimensions |
| ----------------------- | -------------: | -------------: |
| `token-table.bin`       |      3,779,584 |      7,559,168 |
| `token-scales.bin`      |        118,112 |        118,112 |
| `doc-vectors.bin`       |        570,624 |      1,141,248 |
| `manifest.json`         |      1,455,612 |      1,455,612 |
| `tokenizer-config.json` |        446,398 |        446,398 |
| `meta.json`             |            326 |            327 |
| **Total**               |  **6,370,656** | **10,720,865** |

The 256-dimensional candidate adds 4,350,209 bytes (68.3%) over 128 dimensions.
It is selected because it is the smallest evaluated candidate that passes the frozen retrieval floors.

Machine-readable results are stored beside this report in `eval-results.json`.
