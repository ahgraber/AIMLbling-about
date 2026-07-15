const assert = require("node:assert/strict")
const test = require("node:test")

const { rankHybridResults } = require("./semantic-search.js")

function keyword(pageId, route) {
  return {
    pageId,
    route,
    title: `keyword ${pageId}`,
    heading: `keyword ${pageId}`,
    crumb: `keyword crumb ${pageId}`,
    prefix: `keyword prefix ${pageId}`,
    untouched: `keyword untouched ${pageId}`,
    source: "keyword",
  }
}

function semantic(pageId, route, score) {
  return {
    pageId,
    route,
    title: `semantic ${pageId}`,
    heading: `semantic ${pageId}`,
    crumb: `semantic crumb ${pageId}`,
    score,
    source: "semantic",
  }
}

test("agreement across both engines outranks either single-signal page and keeps the keyword section", () => {
  const keywords = [keyword("keyword-only", "/keyword"), keyword("both", "/both-keyword")]
  const semantics = [semantic("semantic-only", "/semantic", 10), semantic("both", "/both-best-section", 9)]

  const ranked = rankHybridResults(keywords, semantics, "ready")

  assert.equal(ranked[0].pageId, "both")
  // A both-engine page retains the keyword-matched section it hit on, not the
  // semantic best chunk, so the exact-term highlight and anchor are preserved.
  assert.equal(ranked[0].route, "/both-keyword")
  assert.equal(ranked[0].title, "keyword both")
  assert.equal(ranked[0].heading, "keyword both")
  assert.equal(ranked[0].crumb, "keyword crumb both")
  assert.equal(ranked[0].prefix, "keyword prefix both")
  assert.equal(ranked[0].untouched, "keyword untouched both")
  assert.equal(ranked[0].source, "keyword")
  assert.equal(ranked[0].keywordRank, 2)
  assert.equal(ranked[0].semanticRank, 2)
  assert.ok(ranked[0].rrfScore > ranked[1].rrfScore)
  assert.deepEqual(new Set(ranked.map((result) => result.pageId)), new Set(["both", "keyword-only", "semantic-only"]))
})

test("semantic-only pages join the union and semantic weight resolves single-signal ties", () => {
  const ranked = rankHybridResults(
    [keyword("keyword-only", "/keyword")],
    [semantic("semantic-only", "/semantic", 10)],
    "ready",
  )

  assert.deepEqual(
    ranked.map((result) => result.pageId),
    ["semantic-only", "keyword-only"],
  )
  assert.equal(ranked[0].crumb, "semantic crumb semantic-only")
  assert.equal(ranked[0].prefix, undefined)
})

test("fusion independently truncates keyword pages and rolled-up semantic pages to ten", () => {
  const keywords = Array.from({ length: 11 }, (_, index) =>
    keyword(index === 10 ? "keyword-tail" : `keyword-${index}`, `/keyword-${index}`),
  )
  const semantics = Array.from({ length: 11 }, (_, index) =>
    semantic(index === 10 ? "semantic-tail" : `semantic-${index}`, `/semantic-${index}`, 100 - index),
  )

  const ranked = rankHybridResults(keywords, semantics, "ready")

  assert.equal(ranked.length, 20)
  assert.ok(!ranked.some((result) => result.pageId === "keyword-tail"))
  assert.ok(!ranked.some((result) => result.pageId === "semantic-tail"))
})

test("fusion weights semantic rank twice as strongly as keyword rank with k sixty", () => {
  const ranked = rankHybridResults(
    [keyword("keyword-first", "/keyword-first")],
    [semantic("semantic-first", "/semantic-first", 1)],
    "ready",
  )

  assert.deepEqual(
    ranked.map((result) => result.pageId),
    ["semantic-first", "keyword-first"],
  )
  assert.equal(ranked[0].rrfScore, 2 / 61)
  assert.equal(ranked[1].rrfScore, 1 / 61)
})

test("ready-state fusion does not mutate either input list or its candidates", () => {
  const keywords = [keyword("both", "/both-keyword")]
  const semantics = [semantic("both", "/both-semantic", 1)]
  const keywordSnapshot = structuredClone(keywords)
  const semanticSnapshot = structuredClone(semantics)
  Object.freeze(keywords[0])
  Object.freeze(semantics[0])
  Object.freeze(keywords)
  Object.freeze(semantics)

  rankHybridResults(keywords, semantics, "ready")

  assert.deepEqual(keywords, keywordSnapshot)
  assert.deepEqual(semantics, semanticSnapshot)
})

test("unavailable, pending, failed, and OOV states return keyword objects unchanged", () => {
  for (const state of ["unavailable", "pending", "failed", "oov"]) {
    const keywords = [keyword("first", "/first"), keyword("second", "/second")]

    const ranked = rankHybridResults(keywords, [semantic("semantic", "/semantic", 10)], state)

    assert.strictEqual(ranked, keywords)
    assert.strictEqual(ranked[0], keywords[0])
    assert.strictEqual(ranked[1], keywords[1])
  }
})
