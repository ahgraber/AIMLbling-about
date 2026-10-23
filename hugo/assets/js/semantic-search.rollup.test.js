const assert = require("node:assert/strict")
const test = require("node:test")

const { rollupSemanticChunks, semanticCandidates } = require("./semantic-search.js")

test("manifest rows and chunk scores map to scaled semantic candidates in manifest order", () => {
  const manifest = [
    { chunk_id: "a", page_id: "/a", url: "/a#one", title: "A", heading: "One" },
    { chunk_id: "b", page_id: "/b", url: "/b", title: "B", heading: "B" },
  ]

  const candidates = semanticCandidates(manifest, new Float32Array([2, 4]), 0.5)

  assert.deepEqual(candidates, [
    { pageId: "/a", route: "/a#one", title: "A", heading: "One", score: 1, source: "semantic" },
    { pageId: "/b", route: "/b", title: "B", heading: "B", score: 2, source: "semantic" },
  ])
})

test("single semantic chunk produces one page with that section", () => {
  const chunk = {
    pageId: "page",
    route: "/page#only",
    title: "Page",
    heading: "Only",
    score: 0.5,
  }

  const pages = rollupSemanticChunks([chunk])

  assert.deepEqual(pages, [chunk])
})

test("multiple chunks of one page retain only its highest-scoring section", () => {
  const chunks = [
    { pageId: "page", route: "/page#lower", title: "Page", heading: "Lower", score: 0.5 },
    {
      pageId: "other",
      route: "/other#section",
      title: "Other",
      heading: "Section",
      score: 0.7,
    },
    { pageId: "page", route: "/page#best", title: "Page", heading: "Best", score: 0.9 },
  ]

  const pages = rollupSemanticChunks(chunks)

  assert.deepEqual(
    pages.map((page) => [page.pageId, page.route, page.heading, page.score]),
    [
      ["page", "/page#best", "Best", 0.9],
      ["other", "/other#section", "Section", 0.7],
    ],
  )
})

test("equal semantic scores preserve input order", () => {
  const chunks = [
    { pageId: "first", route: "/first", title: "First", heading: "First", score: 1 },
    { pageId: "second", route: "/second", title: "Second", heading: "Second", score: 1 },
  ]

  assert.deepEqual(
    rollupSemanticChunks(chunks).map((page) => page.pageId),
    ["first", "second"],
  )
})

test("semantic chunks require explicit manifest page identity and a finite score", () => {
  assert.throws(
    () => rollupSemanticChunks([{ route: "/page", title: "Page", heading: "Page", score: 1 }]),
    /pageId and a finite score/,
  )
  assert.throws(
    () =>
      rollupSemanticChunks([{ pageId: "page", route: "/page", title: "Page", heading: "Page", score: Number.NaN }]),
    /pageId and a finite score/,
  )
})
