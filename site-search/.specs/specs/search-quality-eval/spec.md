# Search Quality Eval Specification

## Purpose

Defines the offline quality checks for search: the retrieval-quality floors hybrid search must meet on a labeled query set, and the tokenizer parity gate that fails the suite on any drift between the build-time and client tokenizers.

## Requirements

### Requirement: RetrievalQualityFloor

On the labeled evaluation query set, hybrid search SHALL trail keyword-only by at most one recall@1 hit on the exact-term subset, and SHALL trail semantic-only by at most one recall@1 hit on the paraphrase subset.

The floor is a gut check that catches a broken fusion, not a release-quality gate.
Each subset has 10 queries, so one query moves recall@1 by 10 points.
Publishing new content can move one query between conditions without any code change; the one-hit tolerance absorbs that noise.
A trail of two or more hits indicates a real ranking problem.

The `semantic-only` and `hybrid` conditions run the **shipped** client logic (`semantic-search.js` executed under Node), so those measurements reflect production.
The `keyword-only` condition is a **deterministic lexical reference** (case-folded term-frequency scoring), not the shipped FlexSearch engine, chosen for reproducibility without a browser.
The floor is therefore evidence against that reference baseline, not a certification of production FlexSearch ranking; treat exact-term parity as exploratory rather than a production keyword-ranking guarantee.

#### Scenario: Exact-term subset does not regress

- **GIVEN** the labeled exact-term queries (distinctive library names, paper IDs, proper nouns from the real corpus)
- **WHEN** keyword-only and hybrid conditions are evaluated
- **THEN** hybrid has at most one fewer recall@1 hit than keyword-only

#### Scenario: Paraphrase subset does not regress

- **GIVEN** the labeled paraphrase queries (same intent, different words than the source text)
- **WHEN** semantic-only and hybrid conditions are evaluated
- **THEN** hybrid has at most one fewer recall@1 hit than semantic-only

#### Scenario: Hybrid trails by two or more hits

- **GIVEN** an evaluation in which hybrid has two or more fewer recall@1 hits than the comparison condition on the exact-term or paraphrase subset
- **WHEN** the floor is checked
- **THEN** the floor fails, naming the subset and both recall@1 values

#### Scenario: Navigational subset reported

- **GIVEN** the labeled navigational queries (topic-seeking, no exact term overlap)
- **WHEN** all three conditions are evaluated
- **THEN** recall@1 and recall@3 for the subset are reported for all three conditions

### Requirement: TokenizerParityGate

The evaluation suite SHALL fail on any token-sequence mismatch between the build-time tokenizer and the client tokenizer across the representative string set, which SHALL include accented text, mixed-case and code-identifier text, and out-of-vocabulary terms.

#### Scenario: Full parity passes

- **GIVEN** the representative string set tokenized by both implementations
- **WHEN** all token-id sequences are identical
- **THEN** the parity check passes

#### Scenario: Any mismatch fails the suite

- **GIVEN** a string on which the two implementations produce different token-id sequences
- **WHEN** the parity check runs
- **THEN** the evaluation suite fails, identifying the mismatching string and both sequences

## Technical Notes

- **Implementation**: `site-search/src/site_search/eval.py`, `eval_runner.js`; parity in `site-search/tests/test_parity.py`
- **Dependencies**: client-hybrid-search, search-index-artifacts
