# Search Quality Eval Specification

## Purpose

Defines the offline quality gates for search: the retrieval-quality floors hybrid search must meet on a labeled query set, and the tokenizer parity gate that fails the suite on any drift between the build-time and client tokenizers.

## Requirements

### Requirement: RetrievalQualityFloor

On the labeled evaluation query set, hybrid search recall@1 SHALL be at least keyword-only recall@1 on the exact-term subset and at least semantic-only recall@1 on the paraphrase subset.

The `semantic-only` and `hybrid` conditions run the **shipped** client logic (`semantic-search.js` executed under Node), so those measurements reflect production.
The `keyword-only` condition is a **deterministic lexical reference** (case-folded term-frequency scoring), not the shipped FlexSearch engine, chosen for reproducibility without a browser.
The floor is therefore evidence against that reference baseline, not a certification of production FlexSearch ranking; treat exact-term parity as exploratory rather than a production keyword-ranking guarantee.

#### Scenario: Exact-term subset does not regress

- **GIVEN** the labeled exact-term queries (distinctive library names, paper IDs, proper nouns from the real corpus)
- **WHEN** keyword-only and hybrid conditions are evaluated
- **THEN** hybrid recall@1 is greater than or equal to keyword-only recall@1

#### Scenario: Paraphrase subset does not regress

- **GIVEN** the labeled paraphrase queries (same intent, different words than the source text)
- **WHEN** semantic-only and hybrid conditions are evaluated
- **THEN** hybrid recall@1 is greater than or equal to semantic-only recall@1

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
