const assert = require("node:assert/strict")
const test = require("node:test")

const { rollupSemanticChunks } = require("./semantic-search.js")

test("single semantic chunk produces one page with that section", () => {
  const chunk = {
    pageId: "page",
    route: "/page#only",
    title: "Page",
    heading: "Only",
    crumb: "Root > Page",
    score: 0.5,
  }

  const pages = rollupSemanticChunks([chunk])

  assert.deepEqual(pages, [chunk])
})

test("multiple chunks of one page retain only its highest-scoring section", () => {
  const chunks = [
    { pageId: "page", route: "/page#lower", title: "Page", heading: "Lower", crumb: "Root > Page", score: 0.5 },
    {
      pageId: "other",
      route: "/other#section",
      title: "Other",
      heading: "Section",
      crumb: "Root > Other",
      score: 0.7,
    },
    { pageId: "page", route: "/page#best", title: "Page", heading: "Best", crumb: "Root > Page", score: 0.9 },
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
    { pageId: "first", route: "/first", title: "First", heading: "First", crumb: "First", score: 1 },
    { pageId: "second", route: "/second", title: "Second", heading: "Second", crumb: "Second", score: 1 },
  ]

  assert.deepEqual(
    rollupSemanticChunks(chunks).map((page) => page.pageId),
    ["first", "second"],
  )
})

test("semantic chunks require explicit manifest page identity and breadcrumb", () => {
  assert.throws(
    () => rollupSemanticChunks([{ route: "/page", title: "Page", heading: "Page", crumb: "Page", score: 1 }]),
    /pageId and crumb/,
  )
  assert.throws(
    () => rollupSemanticChunks([{ pageId: "page", route: "/page", title: "Page", heading: "Page", score: 1 }]),
    /pageId and crumb/,
  )
})
