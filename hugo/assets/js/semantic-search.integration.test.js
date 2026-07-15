const assert = require("node:assert/strict")
const test = require("node:test")

const { getHybridDisplayResults } = require("./semantic-search.js")

function keywordData(route = "/both#keyword") {
  const displayResults = [
    {
      id: "0_0",
      route,
      prefix: "Keyword > Both",
      children: { title: "Keyword section", content: "keyword excerpt" },
    },
  ]
  const candidates = [
    {
      pageId: "/both",
      route,
      title: "Both",
      heading: "Keyword section",
      crumb: "Keyword > Both",
      prefix: "Keyword > Both",
      keywordRoute: route,
      content: "keyword excerpt",
      source: "keyword",
    },
  ]
  return { candidates, displayResults }
}

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
          crumb: "Semantic > Both",
        },
        {
          chunk_id: "semantic-only",
          page_id: "/semantic-only",
          url: "/semantic-only#best",
          title: "Semantic only",
          heading: "Best section",
          crumb: "Semantic > Only",
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

test("every non-ready state returns the exact keyword display array", () => {
  for (const state of ["unavailable", "pending", "failed"]) {
    const keyword = keywordData()

    const results = getHybridDisplayResults(keyword.displayResults, keyword.candidates, "known", { state }, 20)

    assert.strictEqual(results, keyword.displayResults)
  }
})

test("fully OOV queries return the exact keyword display array", () => {
  const keyword = keywordData()

  const results = getHybridDisplayResults(keyword.displayResults, keyword.candidates, "unknown", readyLoader(), 20)

  assert.strictEqual(results, keyword.displayResults)
})

test("ready hybrid results keep the keyword-matched section and its highlighted excerpt for both-engine pages", () => {
  const keyword = keywordData()

  const results = getHybridDisplayResults(keyword.displayResults, keyword.candidates, "known", readyLoader(), 20)

  assert.deepEqual(results[0], {
    id: "hybrid_0",
    route: "/both#keyword",
    prefix: "Keyword > Both",
    children: { title: "Keyword section", content: "keyword excerpt" },
  })
  assert.deepEqual(results[1], {
    id: "hybrid_1",
    route: "/semantic-only#best",
    prefix: "Semantic > Only",
    children: { title: "Best section" },
  })
})

test("ready hybrid results preserve a matching excerpt and honor the page cap", () => {
  const keyword = keywordData()

  const results = getHybridDisplayResults(
    keyword.displayResults,
    keyword.candidates,
    "known",
    readyLoader("/both#keyword"),
    1,
  )

  assert.deepEqual(results, [
    {
      id: "hybrid_0",
      route: "/both#keyword",
      prefix: "Keyword > Both",
      children: { title: "Keyword section", content: "keyword excerpt" },
    },
  ])
})
