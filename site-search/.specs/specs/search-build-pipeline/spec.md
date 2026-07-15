# Search Build Pipeline Specification

## Purpose

Defines how the search artifacts are produced and shipped: a single local recipe, a publish build that always carries artifacts generated from that same build's content, per-build payload reporting, and the exclusion of generated artifacts from version control.

## Requirements

### Requirement: PublishedArtifactCurrency

Every published site build SHALL include a search artifact set generated from that same build's content.

#### Scenario: Fresh publish ships artifacts

- **GIVEN** a container/publish build from a clean checkout
- **WHEN** the build completes
- **THEN** the published site contains the complete artifact set generated during that build

#### Scenario: Content change regenerates artifacts

- **GIVEN** a content change followed by a publish build
- **WHEN** the build completes
- **THEN** the shipped artifacts reflect the changed content, not a stale corpus

### Requirement: LocalIndexRecipe

The repository SHALL provide a single task-runner recipe that produces the complete artifact set for local development.

#### Scenario: Recipe produces working local artifacts

- **GIVEN** a clean local checkout with the dev environment available
- **WHEN** the recipe is run
- **THEN** the complete artifact set exists locally where the dev server serves it

### Requirement: PayloadSizeVisibility

Every index build SHALL report the measured byte size of each artifact and the complete artifact set.
Quality comparisons between index configurations SHALL report payload size alongside retrieval results so the selected configuration is the smallest candidate that meets the retrieval-quality floor.

#### Scenario: Build reports exact artifact sizes

- **GIVEN** a completed artifact set
- **WHEN** the index build completes
- **THEN** it reports each artifact's measured byte size and their exact total

#### Scenario: Quality comparison includes payload cost

- **GIVEN** two or more index configurations evaluated on the same labeled query set
- **WHEN** their retrieval results are compared
- **THEN** the comparison reports each configuration's retrieval metrics and total payload size, and identifies the smallest configuration that meets the retrieval-quality floor

### Requirement: GeneratedArtifactsUntracked

Generated search artifacts SHALL NOT be tracked in version control.

#### Scenario: Artifact paths ignored

- **GIVEN** a locally generated artifact set
- **WHEN** version-control status is inspected
- **THEN** every artifact path is ignored, not untracked-and-committable

## Technical Notes

- **Implementation**: `hugo/justfile` (`search-index` recipe), `hugo/docker/Dockerfile`, `.github/workflows/publish.yaml`, `.gitignore`
- **Dependencies**: search-data-integrity, search-index-artifacts
