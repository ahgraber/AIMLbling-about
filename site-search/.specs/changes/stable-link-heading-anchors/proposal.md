# Proposal: Stable Link Heading Anchors

> Status: stub.
> Proposal and delta spec only; `design.md` and `tasks.md` are written when the change is picked up.

## Intent

Hugo gives a heading an empty auto ID when the heading is a link whose text begins with `[`, and then falls back to `heading`, `heading-1`, and so on.
Every arXiv-style treadmill entry (`### [[2603.24477] Title](url)`) has this form.
Its anchor therefore depends on the entry's position on the page, not on the paper.
A search result or a shared link that points at one paper lands on a different paper when an entry above it is added or removed.
The bug still reproduces on Hugo v0.168.0 (2026-10-22); a report is drafted in `._scratch/hugo-heading-autoid-bug-report.md`.

## User Stories

### Story: link-lands-on-the-paper

As a reader who follows a search result or a shared link to a treadmill paper, I want the link to open at that paper, so that the link still works after the page changes.

Ladders to north-star outcome 1 (Find by meaning): the right treadmill entry surfaces, and the link that surfaces it must open at that entry.

## Scope

**In scope:**

- Rendered heading anchors and search-corpus section anchors for headings whose Hugo auto ID is `heading` or `heading-N`.
- Site-level overrides of Hextra v0.13 `layouts/_markup/render-heading.html` and `layouts/_partials/utils/headings.html`.

**Out of scope:**

- Content changes in the `ai-treadmill` module.
- A fix in Hugo itself (tracked by the bug report).
- Headings whose Hugo auto ID is already text-derived.
- Redirects from old `#heading-N` links.

## Approach

In a heading render hook, `.Anchor` and `.PlainText` are both truncated to `[` by the bug, but `.Text` (the heading HTML) is intact.
`.Fragments` keeps the full title with the fallback ID.
Every Hextra v0.13 consumer of heading IDs (table of contents, mobile sidebar, search index) reads them through `_partials/utils/headings.html`.

1. In the render hook, when `.Anchor` matches `^heading(-\d+)?$`, use `.Text | plainify | anchorize` as the anchor.
2. In `utils/headings.html`, rebuild the heading tree with the same rule applied to each fallback ID, using `.Title | plainify | anchorize`.

Both sides then produce, for example, `260324477-composer-2-technical-report`, the same form Hugo gives bracketed headings that are not links.
Delete both overrides when Hugo ships a fix.

## Open Questions

- Two headings with the same text on one page get the same anchor; Hugo would disambiguate them.
  Accept, or add a page-scoped counter?
- Existing `#heading-N` links break once.
  Accept, given that they are already position-dependent?
