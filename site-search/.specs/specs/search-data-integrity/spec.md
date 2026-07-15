# Search Data Integrity Specification

## Purpose

Guarantees the generated search corpus faithfully carries every section's body text and fails the build loudly on structural regressions or unexplained empties.
It is the trust floor and theme-format canary that every downstream search capability depends on.

## Requirements

### Requirement: SectionContentCompleteness

The generated search corpus SHALL contain, for every content section whose source has body text, that body text attributed to that section's own heading.

#### Scenario: Plain heading section indexed

- **GIVEN** a page section under a plain-text heading with body text beneath it
- **WHEN** the search corpus is generated
- **THEN** the corpus entry for that heading contains the section's body text

#### Scenario: Markdown-link heading section indexed

- **GIVEN** a page section whose heading is entirely a markdown link (e.g. an arxiv-style `[[id] title](url)` treadmill entry) with body text beneath it
- **WHEN** the search corpus is generated
- **THEN** the corpus entry for that heading contains the section's body text, and no other heading's entry absorbs it

#### Scenario: Title-only entry legitimately empty

- **GIVEN** a heading immediately followed by the next heading, with no body text between them
- **WHEN** the search corpus is generated
- **THEN** the corpus entry for that heading exists with empty content and is classified as legitimately empty

#### Scenario: Pre-first-heading intro indexed

- **GIVEN** a page with introductory text before its first heading
- **WHEN** the search corpus is generated
- **THEN** the corpus's intro slot for that page contains the introductory text

### Requirement: CorpusGateBlocksSuspiciousCorpus

For any corpus in which the proportion of suspicious-empty sections (empty without a legitimate structural reason) exceeds the acceptance threshold, the index build SHALL fail without producing artifacts.

#### Scenario: Healthy corpus passes the gate

- **GIVEN** a corpus whose suspicious-empty rate is below the acceptance threshold
- **WHEN** the gate runs
- **THEN** the gate passes and index building may proceed

#### Scenario: Regressed corpus is rejected

- **GIVEN** a corpus whose suspicious-empty rate exceeds the acceptance threshold
- **WHEN** the gate runs
- **THEN** the build fails with a diagnostic identifying the suspicious sections, and no artifacts are written

### Requirement: FormatCanary

Given a search corpus that does not conform to the expected structure, the gate SHALL fail with a structural diagnostic rather than classify or embed it.

#### Scenario: Malformed corpus rejected

- **GIVEN** a corpus file whose structure deviates from the expected shape (e.g. after a theme upgrade changes the search-data format)
- **WHEN** the gate runs
- **THEN** the gate fails, reporting the structural mismatch, and no downstream index work occurs

### Requirement: EmptyClassificationReport

Every gate run SHALL report total section count and empty-section counts classified as legitimate or suspicious, broken down by content source.

#### Scenario: Gate run produces classified report

- **GIVEN** a well-formed corpus containing blog and treadmill sections, some legitimately empty
- **WHEN** the gate runs
- **THEN** the report states totals and legitimate/suspicious empty counts for each content source

## Technical Notes

- **Implementation**: `site-search/src/site_search/loader.py`, `gate.py`, `build_index.py`
- **Dependencies**: none (foundation capability)
