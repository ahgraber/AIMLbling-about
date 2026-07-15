# Search Index Artifacts Specification

## Purpose

Defines the semantic index build: how the corpus is partitioned into embeddable chunks, how the compressed artifact set is produced, and what self-sufficiency and fidelity guarantees the artifacts must uphold so a client can search using nothing but the artifacts themselves.

## Requirements

### Requirement: StructuralChunkIntegrity

The index builder SHALL partition each section into ordered semantic chunks without duplicating body text across chunks.
A chunk SHALL NOT cross a section-heading boundary.
Whole paragraph-like body blocks SHALL remain intact unless one block exceeds the token target defined in the design, in which case it SHALL be divided at sentence boundaries.
Each embedded chunk SHALL carry its page-title and section-heading context.

#### Scenario: Short body blocks remain intact

- **GIVEN** consecutive body blocks under one heading that fit within the configured token target
- **WHEN** the section is chunked
- **THEN** the blocks remain whole and may share one ordered chunk with the heading context

#### Scenario: Oversized body block splits at sentences

- **GIVEN** one body block that exceeds the configured token target and contains multiple sentences
- **WHEN** the section is chunked
- **THEN** it is divided only at sentence boundaries, and every resulting chunk carries the heading context

#### Scenario: Chunk partition has no overlap

- **GIVEN** any section body
- **WHEN** all emitted chunks are compared in order
- **THEN** every body segment appears exactly once, no segment is omitted, and no segment is duplicated across chunks

#### Scenario: Heading boundary is preserved

- **GIVEN** two consecutive sections with different headings
- **WHEN** they are chunked
- **THEN** no chunk contains body text from both sections

### Requirement: SelfSufficientArtifactSet

For any gate-passing corpus, the index build SHALL produce an artifact set sufficient for a client to embed queries and score them against every indexed chunk using no data source other than the artifact set itself.

#### Scenario: Complete artifact set produced

- **GIVEN** a gate-passing corpus
- **WHEN** the index build runs
- **THEN** the output contains token vectors, chunk vectors, a chunk-to-page manifest, a tokenizer configuration, and build metadata

#### Scenario: Manifest and vectors correspond

- **GIVEN** a produced artifact set
- **WHEN** the manifest and chunk vectors are compared
- **THEN** every chunk vector has exactly one manifest entry and vice versa, in matching order

#### Scenario: Correspondence survives chunk exclusion

- **GIVEN** a corpus containing chunks that the build excludes from the semantic index
- **WHEN** the index build runs
- **THEN** the remaining manifest entries and chunk vectors still correspond one-to-one in matching order

### Requirement: SimilarityFidelity

The compressed artifact representation SHALL preserve pairwise similarity between texts, relative to the uncompressed reference embeddings, within the fidelity tolerance defined in the design.

#### Scenario: Round-trip similarity within tolerance

- **GIVEN** a sample of chunk pairs embedded with the reference pipeline
- **WHEN** the same pairs are scored from the compressed artifacts
- **THEN** each pairwise similarity is within the design's fidelity tolerance of the reference value

#### Scenario: Nearest-neighbor ranking preserved

- **GIVEN** a sample of query texts with reference-embedding nearest neighbors computed
- **WHEN** nearest neighbors are computed from the compressed artifacts
- **THEN** the top-ranked neighbors match the reference ranking

### Requirement: ResolvedTokenizerConfig

The exported tokenizer configuration SHALL encode the effective normalization and tokenization behavior such that tokenizing any text with the exported configuration reproduces the reference tokenizer's token sequence.

#### Scenario: Accented text matches reference

- **GIVEN** text containing accented characters (e.g. "café", "naïve")
- **WHEN** tokenized using only the exported configuration
- **THEN** the token sequence equals the reference tokenizer's output, with accent handling resolved to its effective behavior rather than an ambiguous inherited setting

#### Scenario: Out-of-vocabulary pieces dropped as reference

- **GIVEN** text containing terms outside the model vocabulary
- **WHEN** tokenized using only the exported configuration
- **THEN** the same pieces are dropped as in the reference tokenizer, with no substitute token inserted

#### Scenario: No special tokens inserted

- **GIVEN** any input text
- **WHEN** tokenized using only the exported configuration
- **THEN** the token sequence contains no sequence-delimiter special tokens

### Requirement: EmptyChunkExclusion

For any chunk whose text yields no known tokens, the index build SHALL exclude that chunk from the semantic index and report the exclusion; the artifact set SHALL NOT contain zero-magnitude or undefined vectors.

#### Scenario: Fully out-of-vocabulary chunk excluded

- **GIVEN** a corpus containing a chunk that tokenizes to zero known tokens
- **WHEN** the index build runs
- **THEN** the chunk is absent from the semantic index and the build output reports it as excluded

#### Scenario: No degenerate vectors exported

- **GIVEN** any produced artifact set
- **WHEN** every chunk vector is inspected
- **THEN** none is zero-magnitude, NaN, or otherwise undefined

## Technical Notes

- **Implementation**: `site-search/src/site_search/chunking.py`, `embedding.py`, `quantize.py`, `export.py`, `build_index.py`
- **Dependencies**: search-data-integrity
