# Search Data Integrity Specification

## Purpose

Guarantees the generated search corpus faithfully carries every section's body text and fails the build loudly on structural regressions or a mass loss of section text.
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

#### Scenario: Title-only entry is empty

- **GIVEN** a heading immediately followed by the next heading, with no body text between them
- **WHEN** the search corpus is generated
- **THEN** the corpus entry for that heading exists with empty content, and no child heading's title or text is copied into it

#### Scenario: Pre-first-heading intro indexed

- **GIVEN** a page with introductory text before its first heading
- **WHEN** the search corpus is generated
- **THEN** the corpus's intro slot for that page contains the introductory text

### Requirement: CorpusGateBlocksMassEmptying

For any corpus in which the proportion of empty sections (no body text, whatever the reason) exceeds the acceptance threshold, or which has no sections, the index build SHALL fail without producing artifacts.

#### Scenario: Healthy corpus passes the gate

- **GIVEN** a corpus whose empty-section rate is at or below the acceptance threshold
- **WHEN** the gate runs
- **THEN** the gate passes and index building may proceed

#### Scenario: Mass-emptied corpus is rejected

- **GIVEN** a corpus whose empty-section rate exceeds the acceptance threshold
- **WHEN** the gate runs
- **THEN** the build fails with the empty-section report, and no artifacts are written

### Requirement: FormatCanary

Given a search corpus that does not conform to the expected structure, the gate SHALL fail with a structural diagnostic rather than evaluate or embed it.

#### Scenario: Malformed corpus rejected

- **GIVEN** a corpus file whose structure deviates from the expected shape (e.g. after a theme upgrade changes the search-data format)
- **WHEN** the gate runs
- **THEN** the gate fails, reporting the structural mismatch, and no downstream index work occurs

### Requirement: EmptySectionReport

Every gate run SHALL report the total and empty section counts for each content source, the overall empty-section rate, the acceptance threshold, and the verdict.

#### Scenario: Gate run produces report

- **GIVEN** a well-formed corpus containing blog and treadmill sections, some empty
- **WHEN** the gate runs
- **THEN** the report states totals and empty counts for each content source, the overall rate against the threshold, and pass or fail

## Technical Notes

- **Implementation**: `site-search/src/site_search/loader.py`, `gate.py`, `build_index.py`
- **Dependencies**: none (foundation capability)
- **Threshold**: 35% empty sections (`DEFAULT_EMPTY_RATE_LIMIT`).
  A healthy full build measured 19.5% on 2026-10-22, mostly treadmill category headings with no text of their own; the theme bug this gate was built for left 75% empty.
  The gate does not tell intended empties from lost text per section, so it misses small regressions by design.
