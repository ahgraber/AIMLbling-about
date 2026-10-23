# Delta: Search Data Integrity

## ADDED Requirements

### Requirement: TextDerivedSectionAnchors

Each section anchor on a rendered page SHALL be derived from the heading's visible text and SHALL NOT depend on the heading's position on the page.
The search corpus SHALL key each section by the same anchor that the rendered page carries.

Serves: link-lands-on-the-paper

#### Scenario: Link heading whose text begins with a bracket

- **GIVEN** a heading that is a link whose text begins with `[`, such as `[[2603.24477] Composer 2 Technical Report](https://arxiv.org/abs/2603.24477)`
- **WHEN** the site is built
- **THEN** the heading's anchor is derived from its visible text, not `heading` or `heading-N`

#### Scenario: Plain and ordinary link headings keep their anchors

- **GIVEN** a plain-text heading and a link heading whose text does not begin with `[`
- **WHEN** the site is built
- **THEN** each keeps the anchor Hugo derives from its text

#### Scenario: Adding an earlier entry does not move a later anchor

- **GIVEN** a page whose bracket-led link headings have anchors from a previous build
- **WHEN** a new bracket-led link heading is added above them and the site is rebuilt
- **THEN** the existing headings keep their anchors

#### Scenario: Rendered page and corpus agree

- **GIVEN** any heading on a rendered page
- **WHEN** the search corpus is generated
- **THEN** the corpus section key for that heading uses the anchor the rendered page carries, and the table of contents links to the same anchor
