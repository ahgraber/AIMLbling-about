# Client Hybrid Search Specification

## Purpose

Defines the browser-side search contract: query embedding and scoring, fusion of the keyword and semantic signals, page deduplication, deferred payload loading, and — above all — silent degradation to keyword-only search whenever the semantic layer is unavailable.

## Requirements

### Requirement: KeywordFallback

Whenever semantic search is unavailable for a query — artifacts absent or failed to load, artifacts not yet loaded, or the query yielding no known tokens — search SHALL return keyword-only results identical to those produced with no semantic search present, with no user-facing error.

#### Scenario: Artifact fetch failure degrades to keyword

- **GIVEN** the search artifacts fail to load (missing file, network error, or parse error)
- **WHEN** the user searches
- **THEN** results equal today's keyword-only results and no error is surfaced to the user

#### Scenario: Query during artifact loading degrades to keyword

- **GIVEN** the artifact fetch has started but not completed
- **WHEN** the user submits a query
- **THEN** keyword-only results are returned without waiting on the fetch

#### Scenario: Fully out-of-vocabulary query degrades to keyword

- **GIVEN** artifacts are loaded and a query tokenizes to zero known tokens
- **WHEN** the user searches
- **THEN** keyword-only results are returned and no degenerate (zero or NaN) semantic scores are produced

### Requirement: HybridRanking

When semantic search is available, results SHALL be ordered by weighted Reciprocal Rank Fusion of the keyword and semantic rank lists, such that a result ranked by both signals outranks a result ranked by only one, other contributions being equal.
The semantic signal carries the higher fusion weight (to recover paraphrase and navigational queries the lexical engine misses), so a strongly-ranked semantic-only result MAY outrank a keyword-only result — the two single-signal cases are symmetric and rank-based fusion cannot tell a distinctive exact match apart from an incidental lexical hit.
Exact-term non-regression is therefore an **aggregate** guarantee, enforced by `search-quality-eval`'s `RetrievalQualityFloor` (hybrid exact-term recall is at least keyword-only's), not a per-query rule that every exact match stays at rank 1.

#### Scenario: Paraphrase query finds unmatched-keyword content

- **GIVEN** a query that paraphrases a post's content using words absent from its text
- **WHEN** the user searches with semantic search available
- **THEN** the relevant page appears in the results despite matching no keywords

#### Scenario: Exact-term ranking does not regress in aggregate

- **GIVEN** the labeled exact-term query set
- **WHEN** keyword-only and hybrid results are compared
- **THEN** hybrid exact-term recall@1 is at least keyword-only's (per `RetrievalQualityFloor`); individual exact matches not also surfaced by the semantic signal MAY be reordered, so per-query rank-1 placement is not guaranteed

#### Scenario: Agreement outranks single-signal results

- **GIVEN** one result ranked well by both signals, one ranked well only by keyword, and one ranked well only by semantic
- **WHEN** the results are fused
- **THEN** the both-signal result ranks above both single-signal results

### Requirement: ClientEmbeddingParity

For any query text, the client-side embedding SHALL be equivalent to the build-time reference pipeline's embedding of the same text — an identical token sequence and a matching vector within the design's fidelity tolerance.

#### Scenario: Plain text embeds identically

- **GIVEN** a query of common in-vocabulary words
- **WHEN** embedded by the client and by the reference pipeline
- **THEN** token sequences are identical and vectors match within tolerance

#### Scenario: Accented text embeds identically

- **GIVEN** a query containing accented characters
- **WHEN** embedded by the client and by the reference pipeline
- **THEN** both sides apply the same effective accent normalization and produce identical token sequences

#### Scenario: Partially out-of-vocabulary text embeds identically

- **GIVEN** a query mixing known terms with out-of-vocabulary terms
- **WHEN** embedded by the client and by the reference pipeline
- **THEN** both sides drop the same pieces and the resulting vectors match within tolerance

### Requirement: ScoringCorrectness

Semantic similarity scores computed by the client SHALL equal the mathematically exact similarity of the corresponding vectors, within floating-point tolerance, for every indexed chunk.

#### Scenario: Ordinary vectors score exactly

- **GIVEN** a query vector and chunk vectors of typical magnitude
- **WHEN** the client scores them
- **THEN** each score equals the reference-computed similarity within floating-point tolerance

#### Scenario: High-magnitude accumulation stays exact

- **GIVEN** vectors whose similarity accumulation exceeds the range of small fixed-width integers
- **WHEN** the client scores them
- **THEN** the score equals the reference value rather than a wrapped-around (overflowed) value

### Requirement: PageDeduplication

For any query matching multiple chunks of the same page, results SHALL present that page at most once, represented by its best-matching section.

#### Scenario: Single matching chunk

- **GIVEN** a query matching exactly one chunk of a page
- **WHEN** results are presented
- **THEN** the page appears once, showing that section

#### Scenario: Multiple chunks of one page

- **GIVEN** a query matching several chunks of the same page
- **WHEN** results are presented
- **THEN** the page appears exactly once, represented by its best-matching section

### Requirement: DeferredSearchPayload

The site SHALL NOT fetch any semantic-search artifact before the user first engages the search input, and after first engagement SHALL fetch each artifact at most once per page visit.

#### Scenario: Page load fetches nothing

- **GIVEN** a fresh page load
- **WHEN** the user reads without engaging search
- **THEN** no semantic-search artifact requests occur

#### Scenario: First engagement fetches once

- **GIVEN** a user engaging the search input for the first time and then performing several searches
- **WHEN** network activity is observed
- **THEN** each artifact is fetched exactly once

### Requirement: PresentationPreserved

Search result presentation — match highlighting, keyboard navigation, and assistive-technology announcements — SHALL behave as it does for keyword-only search.

#### Scenario: Hybrid results render through existing presentation

- **GIVEN** hybrid results for a query
- **WHEN** the user navigates results by keyboard and inspects highlighting and screen-reader announcements
- **THEN** all presentation behaviors match today's keyword-only search

## Technical Notes

- **Implementation**: `hugo/assets/js/semantic-search.js`, `hugo/assets/js/flexsearch.js`
- **Dependencies**: search-index-artifacts
