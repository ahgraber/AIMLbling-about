const assert = require("node:assert/strict")
const test = require("node:test")

const { getHybridDisplayResults } = require("./semantic-search.js")

// Keyword page groups in the shape Hextra's search dialog renders, plus the
// canonical `pageId` the fusion step joins on.
function keywordGroups() {
  return [
    {
      pageId: "/both",
      route: "/both",
      title: "Both",
      breadcrumbs: ["Keyword"],
      matches: [
        { route: "/both#keyword", title: "Keyword section", content: "keyword excerpt" },
        { route: "/both#second", title: "Second section", content: "second excerpt" },
      ],
    },
  ]
}

const pageBreadcrumbs = new Map([
  ["/both", ["Keyword"]],
  ["/semantic-only", ["Semantic"]],
])

function readyLoader(bestRoute = "/both#semantic") {
  return {
    state: "ready",
    artifacts: {
      tokenTable: new Int8Array([0, 0, 2, 0]),
      tokenScales: new Float32Array([0, 0.5]),
      documentVectors: new Int8Array([100, 0, 90, 0]),
      manifest: [
        {
          chunk_id: "both",
          page_id: "/both",
          url: bestRoute,
          title: "Both semantic",
          heading: "Semantic section",
        },
        {
          chunk_id: "semantic-only",
          page_id: "/semantic-only",
          url: "/semantic-only#best",
          title: "Semantic only",
          heading: "Best section",
        },
      ],
      tokenizerConfig: {
        version: 1,
        add_special_tokens: false,
        drop_unknown: true,
        unknown_token_id: 0,
        tokenizer: {
          normalizer: {
            type: "BertNormalizer",
            clean_text: true,
            handle_chinese_chars: true,
            lowercase: true,
            strip_accents: true,
          },
          pre_tokenizer: { type: "BertPreTokenizer" },
          model: { type: "WordPiece", vocab: { "[UNK]": 0, known: 1 } },
        },
      },
      meta: { dimensions: 2, document_global_scale: 0.01 },
    },
  }
}

test("every non-ready state returns the exact keyword groups array", () => {
  for (const state of ["unavailable", "pending", "failed"]) {
    const groups = keywordGroups()

    const results = getHybridDisplayResults(groups, "known", { state }, 20, pageBreadcrumbs)

    assert.strictEqual(results, groups)
  }
})

test("fully OOV queries return the exact keyword groups array", () => {
  const groups = keywordGroups()

  const results = getHybridDisplayResults(groups, "unknown", readyLoader(), 20, pageBreadcrumbs)

  assert.strictEqual(results, groups)
})

test("ready hybrid results show a both-engine page once, by its top keyword section and excerpt", () => {
  const results = getHybridDisplayResults(keywordGroups(), "known", readyLoader(), 20, pageBreadcrumbs)

  assert.deepEqual(results[0], {
    route: "/both",
    title: "Both",
    breadcrumbs: ["Keyword"],
    matches: [{ route: "/both#keyword", title: "Keyword section", content: "keyword excerpt" }],
  })
})

test("ready hybrid results show a semantic-only page by its best section, with breadcrumbs from the page map", () => {
  const results = getHybridDisplayResults(keywordGroups(), "known", readyLoader(), 20, pageBreadcrumbs)

  assert.deepEqual(results[1], {
    route: "/semantic-only",
    title: "Semantic only",
    breadcrumbs: ["Semantic"],
    matches: [{ route: "/semantic-only#best", title: "Best section" }],
  })
})

test("ready hybrid results honor the page cap", () => {
  const results = getHybridDisplayResults(keywordGroups(), "known", readyLoader(), 1, pageBreadcrumbs)

  assert.equal(results.length, 1)
  assert.equal(results[0].route, "/both")
})
