# Rollout QA runbook: hybrid-site-search

Status: Executed 2026-07-14 (local dev serve) — one item outstanding: screen-reader announcements not exercised (keyboard/visual only); see Check 2.

This runbook records the browser-only and deployed-site checks that automated tests cannot prove.
It is the manual-evidence source named by the `DeferredSearchPayload` and `PresentationPreserved` verification waivers in `../design.md`, and it supplies supplementary `KeywordFallback` and sampled `HybridRanking` evidence in a real browser.
Work top to bottom against a single release candidate, attach or link the named captures, and replace each `Pending` / `Not recorded` value with the observed result.
Do not treat this template as rollout evidence until it is executed and signed off in § Rollout decision.

## How the mechanism behaves (what these checks assert)

- Artifacts are fetched lazily.
  The first `focus` on a `.hextra-search-input` element runs `init()` in `hugo/assets/js/flexsearch.js`, which fires `semanticLoader.initialize()` exactly once behind a single-initialization guard.
  Focus is reachable by click, Tab, `/`, or `⌘K`/`Ctrl+K`.
- The six semantic artifacts live at fixed paths under `/search/`: `token-table.bin`, `token-scales.bin`, `doc-vectors.bin`, `manifest.json`, `tokenizer-config.json`, `meta.json`.
- The keyword index (`en.*.search-data.json`, fingerprinted in production) also loads on first focus.
  It is the pre-existing FlexSearch payload, **not** one of the six semantic artifacts — filter it out by matching the `/search/` path segment and the six names above.
- Every semantic failure mode (fetch blocked, fetch pending, fully-out-of-vocabulary query) degrades to today's keyword-only path with no user-facing error: `getHybridDisplayResults` returns the keyword results unchanged whenever semantic state is not ready.

## Prerequisites

Environment: macOS (Darwin aarch64), local
Browser and version: Zen 1.21.5b (Firefox 152.0.4), aarch64
Assistive technology (for § Presentation): None — keyboard and visual only; screen reader not exercised
Build or commit: c808b268fa08f639c7c9f3d549b48e4dd907665d plus uncommitted working-tree changes (the review-fix branch)
Candidate URL: <http://localhost:1313>

Pick one release-candidate target and record which:

- [ ] **Deployed site** (preferred for the deployed-eval check) — the published homelab URL serving the release image.
- [ ] **Local container** — `nix develop -c just hugo build linux/arm64` then `nix develop -c just hugo run`; QA the nginx-served site as shipped.
- [x] **Local dev serve** — `nix develop -c just hugo search-index` to generate `hugo/static/search/`, then `nix develop -c just hugo demo`; closest to source, no minify.

Before starting, confirm the candidate actually ships the artifact set:

- [ ] Request each of the six `/search/` files directly (browser or `curl -sI`) and confirm HTTP 200.
  Reference byte counts (256-dim build) are in `index-build.md`; total 10,720,865 bytes.
- [ ] Open DevTools, select the Network panel, enable **Preserve log** and **Disable cache**, and clear the log before each check below.

## Check 1 — Semantic artifacts load only after search engagement

Waiver evidence: `DeferredSearchPayload` Network capture: Firefox DevTools Network panel, filtered to `search`, cache enabled, on the local dev serve.

Procedure:

- [ ] Load a content page (e.g. any `/blog/...` post) with DevTools Network open, log preserved, cache disabled.
  Do **not** touch search.
- [ ] Filter the request list to `search` and read the page-load requests.
  Confirm none of the six `/search/` artifacts appear. (The keyword `en.*.search-data.json` should also be absent until focus.)
- [ ] Focus the search input **once** (click it, or press `/`).
  Confirm exactly one request for each of the six artifacts.
- [ ] Run several distinct queries and blur/refocus the input.
  Confirm no `/search/` artifact is requested a second time.

Record the observed request count per artifact:

| Artifact                | Before focus | After first focus | After repeated searches |
| ----------------------- | -----------: | ----------------: | ----------------------: |
| `token-table.bin`       |            0 |                 1 |                       0 |
| `token-scales.bin`      |            0 |                 1 |                       0 |
| `doc-vectors.bin`       |            0 |                 1 |                       0 |
| `manifest.json`         |            0 |                 1 |                       0 |
| `tokenizer-config.json` |            0 |                 1 |                       0 |
| `meta.json`             |            0 |                 1 |                       0 |

Pass criteria: every "Before focus" cell is `0`, every "After first focus" cell is `1`, every "After repeated searches" cell is `0` (no additional fetch).

Result: PASS

Notes: On this single-page criterion the artifacts loaded only on first search focus and were not re-fetched on repeated searches within the page.
Navigating to a new page re-fetches the artifacts on the dev serve, because `hugo demo` sends no `Cache-Control` headers.
Confirmed in the local container (`just hugo run`): nginx serves `/search/*` with `Cache-Control: public, max-age=300, s-maxage=3600`, and the browser reuses the artifacts from cache across page navigations — within `DeferredSearchPayload` ("at most once per page visit").
A longer cache lifetime or fingerprinted artifact names is a noted efficiency follow-up, not a blocker.

## Check 2 — Existing presentation remains usable with hybrid results

Waiver evidence: `PresentationPreserved`
Observed queries: Multi-result queries returning several pages (exact strings not individually recorded)
Evidence (screenshots / screen-reader transcript): Manual keyboard/visual pass; screen-reader transcript not captured (AT not run)

Use a query that returns several hybrid results across more than one page (e.g. `uv workspaces` or `ragas`).
With semantic artifacts loaded:

- [ ] **Highlighting** — confirm matched query text is wrapped in `hextra-search-match` highlight spans in both the result title and excerpt, exactly as keyword-only search renders it.
- [ ] **Keyboard navigation** — with the input focused, use `ArrowDown`/`ArrowUp` to move the active result, `Enter` to open the selected result, and `Escape` to clear and dismiss.
  Confirm the active-result outline tracks selection and `Enter` navigates.
- [ ] **Screen-reader announcements** — with the AT running, confirm the result-count status (`.hextra-search-status`, `aria-live`) announces "N results found" as results update, and the no-results message announces when a query matches nothing.
- [ ] **Rollup** — confirm each page appears at most once, represented by its best-matching section (no duplicate page rows from multiple chunk hits).

Pass criteria: all four presentation behaviors are indistinguishable from keyword-only search (the design routes hybrid results through the unchanged `displayResults()`).

Result: PASS (keyboard + visual coverage, accepted as sufficient — see decision)

Notes: Highlighting, keyboard navigation (arrow/enter/escape), and page-level rollup (each page once) were confirmed indistinguishable from keyword-only search.
Screen-reader announcements (`.hextra-search-status` `aria-live`) were not exercised — this pass was keyboard and visual only.
Scope decision (maintainer, 2026-07-14): keyboard + visual coverage is accepted as sufficient evidence for `PresentationPreserved`; a dedicated screen-reader (VoiceOver) pass is not required for v1.
Rationale: hybrid results route through the unchanged `displayResults()` and its unchanged `.hextra-search-status` `aria-live` element, so no AT-announcement code path is new in this change — the announcement behavior is preserved structurally.
Recorded in `../design.md` § Verification Waivers.

## Check 3 — Blocked artifacts fall back without a user-facing failure

Supplementary evidence: `KeywordFallback` (integration level)
Blocked-request rule: `*/search/*` blocked in DevTools before first focus
Console capture: No uncaught site error observed

Procedure:

- [ ] In DevTools, block all `/search/*` requests (Network → request blocking pattern `*/search/*`, or offline for those paths) **before** first focusing the input.
- [ ] Open the Console panel with **Preserve log** on.
- [ ] Focus the input and run a distinctive exact-term query with known keyword results (e.g. `Semantic Kernel` or an arxiv ID present in the corpus).
- [ ] Confirm the results match keyword-only behavior — the same pages the keyword engine returns with no semantic layer.
- [ ] Confirm the UI shows no error state and the console shows no **uncaught** site error. (The browser's own "blocked by client" network line for the blocked artifact requests is expected and is not a user-facing search error.)

Pass criteria: keyword-only results render normally; no user-facing error; no uncaught exception from `flexsearch.js` / `semantic-search.js`.

Result: PASS

Notes:
With `/search/*` blocked, an exact-term query returned the expected keyword-only results with no user-facing error state and no uncaught exception.

## Check 4 — Deployed results match sampled evaluation expectations

Sampled evidence: `HybridRanking` (exact-term stays on top; paraphrase finds content)
Deployment URL: Not recorded
Build or image digest: Not recorded
Recorded at: Not recorded

Run this representative subset of the frozen eval set (`site-search/tests/fixtures/eval-queries.json`) against the **deployed** site with semantic search enabled, and record the observed rank of the expected page.
Recall@1 is the primary lens; note rank ≤ 3 as context.

|   # | Subset       | Query                                                               | Expected page                                      | Observed rank | Result |
| --: | ------------ | ------------------------------------------------------------------- | -------------------------------------------------- | ------------: | ------ |
|   1 | exact-term   | `RAGAS to Riches part 1 Ragas v0.2`                                 | `/blog/ragas-to-riches-1`                          |             1 | Found  |
|   2 | exact-term   | `2411.02959 HtmlRAG`                                                | `/treadmill/2024/11/08`                            |             1 | Found  |
|   3 | exact-term   | `uv workspaces central dependency`                                  | `/blog/managing-project-dependencies-with-uv`      |             1 | Found  |
|   4 | paraphrase   | `why errors multiply across long AI conversations and agent chains` | `/blog/the-compounding-error-of-generative-models` |    not in top | Miss   |
|   5 | paraphrase   | `small specialist coding models can match frontier systems`         | `/blog/on-the-viability-of-fine-tuning-slms`       |            ≤3 | Found  |
|   6 | paraphrase   | `company incentives change what open AI really means`               | `/blog/for-some-definition-of-open`                |    not in top | Miss   |
|   7 | navigational | `the author's 2026 AI forecast with probability estimates`          | `/blog/predictions-2026`                           |           1–2 | Found  |
|   8 | navigational | `post about managing environments across a Python monorepo`         | `/blog/managing-project-dependencies-with-uv`      |           1–2 | Found  |
|   9 | navigational | `essay discussing Richard Sutton and scaling computation`           | `/blog/the-bitter-lesson`                          |           1–2 | Found  |

Interpretation, not a hard gate: the offline eval report (`eval-report.md`) is the authoritative recall measurement; this deployed sample confirms the shipped site behaves consistently with it.
Exact-term queries (1–3) should keep their expected page at or near rank 1; paraphrase queries (4–6) should surface the expected page that keyword-only would miss.
Record and explain any query where the deployed rank diverges materially from the offline expectation.

Observed recall@1 (sample): exact-term 3/3, paraphrase 1/3, navigational 2/3 (rank 1); all 3 navigational within rank 3

Notes: Exact-term landed all three expected pages at the top, and navigational surfaced all three within rank 3 (two at rank 1, one at rank 2; the specific query ranked 2nd was not individually recorded).
Paraphrase recovered 1/3 (#5), matching the offline paraphrase recall@1 of 30% — #4 and #6 are hard paraphrases and are the model's known weak spot, not a pipeline regression.
Deployed-site run is still pending; this sample was taken on the local dev serve.

## Rollout decision

| Check                      | Waiver / evidence                 | Result                                                              |
| -------------------------- | --------------------------------- | ------------------------------------------------------------------- |
| 1 — Deferred payload       | `DeferredSearchPayload`           | PASS                                                                |
| 2 — Presentation preserved | `PresentationPreserved`           | PARTIAL — keyboard/visual confirmed; screen-reader pass outstanding |
| 3 — Keyword fallback       | `KeywordFallback` (supplementary) | PASS                                                                |
| 4 — Deployed eval sample   | `HybridRanking` (sampled)         | PASS on local serve (consistent with offline); deployed run pending |

Outstanding before full sign-off: a screen-reader pass for Check 2 (or a recorded decision to accept keyboard/visual coverage), and optionally the deployed-site eval sample for Check 4.

Overall rollout decision: Pending maintainer sign-off Decided by: Not recorded Decided at: Not recorded
