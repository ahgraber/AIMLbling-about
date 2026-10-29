// Integration-load coverage for the flexsearch.js data plane.
//
// flexsearch.js is a Hugo-template IIFE whose only public surface is
// `window.hextraSearch` ({ preload, search }), which Hextra's search-dialog.js
// consumes. This harness loads the *real* flexsearch.js source in a `vm`
// sandbox with the `{{ ... }}` template tokens neutralized, a stub FlexSearch
// (the real library loads from a CDN at runtime), and a recording fetch, then
// asserts the contract in the states that matter:
//   1. the semantic module failed to load entirely (globalThis.SiteSemanticSearch
//      undefined) — exercises the inline failed-loader stub + passthrough shims;
//   2. the semantic module is present — exercises the real createSemanticArtifactLoader path.
//
// Scope: this covers the data plane only. Dialog rendering, keyboard navigation,
// and highlighting belong to Hextra's search-dialog.js and are checked manually.

const assert = require("node:assert/strict")
const test = require("node:test")
const fs = require("node:fs")
const path = require("node:path")
const vm = require("node:vm")

const FLEXSEARCH_SRC = path.join(__dirname, "flexsearch.js")
const SEMANTIC_SRC = path.join(__dirname, "semantic-search.js")
const SEARCH_DATA_URL = "search-data"

// Replace every Hugo template action with a non-empty placeholder so string
// literals stay truthy (the real loader rejects empty URLs). Artifact URL
// templates become the artifact file name, so a test can tell the requests
// apart; `| default X` actions become X, so result limits and the tokenizer
// get the values a site without overrides gets.
function neutralizeTemplates(source) {
  return source
    .replace(/\{\{ with resources\.Get "search\/([^"]+)" \}\}[\s\S]*?\{\{ end \}\}/g, "$1")
    .replace(/\{\{[\s\S]*?\}\}/g, (action) => {
      if (action.includes("$searchData.RelPermalink")) return SEARCH_DATA_URL
      const fallback = action.match(/\|\s*default\s+"?([^"\s]+?)"?\s*-?\}\}$/)
      return fallback ? fallback[1] : "x"
    })
}

// A small stand-in for FlexSearch.Document (the real library loads from a CDN).
// It matches by case-insensitive substring on the indexed field, honors the
// result limit and tag filter, keeps only stored fields, and returns FlexSearch's
// enriched result shape: `[{ field, result: [{ id, doc }] }]`.
class FakeDocument {
  constructor({ document }) {
    this.config = document
    this.docs = []
  }

  add(doc) {
    this.docs.push(doc)
  }

  search(query, limitOrOptions, maybeOptions) {
    const limit = typeof limitOrOptions === "number" ? limitOrOptions : Number.POSITIVE_INFINITY
    const options = (typeof limitOrOptions === "object" ? limitOrOptions : maybeOptions) ?? {}
    const tagField = this.config.tag?.[0]?.field
    const needle = query.toLowerCase()
    const result = []
    for (const doc of this.docs) {
      if (result.length >= limit) break
      if (options.tag && doc[tagField] !== options.tag[tagField]) continue
      if (!String(doc[this.config.index]).toLowerCase().includes(needle)) continue
      const stored = Object.fromEntries(this.config.store.filter((key) => key in doc).map((key) => [key, doc[key]]))
      result.push({ id: doc.id, doc: stored })
    }
    return result.length ? [{ field: this.config.index, result }] : []
  }
}

function okResponse(body = {}) {
  return {
    ok: true,
    json: () => Promise.resolve(body),
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)),
  }
}

// `artifactResponse` decides how semantic artifact requests settle;
// `searchDataResponse` decides the search data request, which by default
// succeeds with `searchData`.
function loadFlexsearch({
  withSemanticModule,
  searchData = {},
  artifactResponse = () => Promise.resolve(okResponse()),
  searchDataResponse = () => Promise.resolve(okResponse(searchData)),
}) {
  const requests = []
  const sandbox = {
    console: { debug: () => {} },
    fetch: (url) => {
      requests.push(url)
      return url === SEARCH_DATA_URL ? searchDataResponse() : artifactResponse(url)
    },
    FlexSearch: { Document: FakeDocument },
    Promise,
  }
  sandbox.window = sandbox
  vm.createContext(sandbox)

  if (withSemanticModule) {
    vm.runInContext(neutralizeTemplates(fs.readFileSync(SEMANTIC_SRC, "utf8")), sandbox, {
      filename: "semantic-search.js",
    })
    assert.ok(sandbox.SiteSemanticSearch, "semantic module should register globalThis.SiteSemanticSearch")
  }

  vm.runInContext(neutralizeTemplates(fs.readFileSync(FLEXSEARCH_SRC, "utf8")), sandbox, { filename: "flexsearch.js" })
  return { sandbox, requests }
}

for (const withSemanticModule of [false, true]) {
  const label = withSemanticModule ? "semantic module present" : "semantic module absent"

  test(`exposes window.hextraSearch with preload and search (${label})`, () => {
    const { sandbox } = loadFlexsearch({ withSemanticModule })

    assert.equal(typeof sandbox.hextraSearch?.preload, "function")
    assert.equal(typeof sandbox.hextraSearch?.search, "function")
  })

  test(`loading the script fetches nothing before search is engaged (${label})`, () => {
    const { requests } = loadFlexsearch({ withSemanticModule })

    assert.deepEqual(requests, [])
  })
}

const WIDGET_CORPUS = {
  "/blog/": { title: "Blog", data: { "": "Index of posts" } },
  "/blog/alpha/": {
    title: "Alpha",
    data: { "": "alpha intro about widgets", "setup#Setup": "configure widgets here\nsecond paragraph" },
  },
  "/blog/beta/": { title: "Beta", data: { "usage#Usage": "widgets in practice" } },
}

// Page groups for "widgets" over WIDGET_CORPUS: canonical routes (no trailing
// slash), parent-section breadcrumbs, one match per matching section with its
// paragraph as the excerpt, and option ids unique across groups and matches.
const WIDGET_GROUPS = [
  {
    id: "hextra-search-opt-0",
    route: "/blog/alpha",
    title: "Alpha",
    breadcrumbs: ["Blog"],
    matches: [
      { id: "hextra-search-opt-1", route: "/blog/alpha", title: "Alpha", content: "alpha intro about widgets" },
      { id: "hextra-search-opt-2", route: "/blog/alpha#setup", title: "Setup", content: "configure widgets here" },
    ],
  },
  {
    id: "hextra-search-opt-3",
    route: "/blog/beta",
    title: "Beta",
    breadcrumbs: ["Blog"],
    matches: [
      { id: "hextra-search-opt-4", route: "/blog/beta#usage", title: "Usage", content: "widgets in practice" },
    ],
  },
]

// Results come from the sandbox realm; round-trip them so deepEqual compares values only.
function plain(value) {
  return JSON.parse(JSON.stringify(value))
}

const KEYWORD_ONLY_STATES = {
  "semantic module absent": { withSemanticModule: false },
  "semantic artifacts pending": { withSemanticModule: true, artifactResponse: () => new Promise(() => {}) },
  "semantic artifacts failed": {
    withSemanticModule: true,
    artifactResponse: () => Promise.reject(new Error("network down")),
  },
}

for (const [state, options] of Object.entries(KEYWORD_ONLY_STATES)) {
  test(`keyword-only search returns complete page groups (${state})`, async () => {
    const { sandbox } = loadFlexsearch({ ...options, searchData: WIDGET_CORPUS })

    const results = await sandbox.hextraSearch.search("widgets")

    assert.deepEqual(plain(results), WIDGET_GROUPS)
  })
}

test("search caps page groups at the default maxPageResults of 20", async () => {
  const searchData = Object.fromEntries(
    Array.from({ length: 25 }, (_, i) => [`/p${i}/`, { title: `Page ${i}`, data: { "": "widgets" } }]),
  )
  const { sandbox } = loadFlexsearch({ withSemanticModule: false, searchData })

  const results = await sandbox.hextraSearch.search("widgets")

  assert.equal(results.length, 20)
})

test("search caps matches per page at the default maxSectionResults of 10", async () => {
  const data = Object.fromEntries(Array.from({ length: 12 }, (_, i) => [`s${i}#Section ${i}`, `widgets ${i}`]))
  const { sandbox } = loadFlexsearch({ withSemanticModule: false, searchData: { "/p/": { title: "P", data } } })

  const results = await sandbox.hextraSearch.search("widgets")

  assert.equal(results[0].matches.length, 10)
})

test("repeated preloads and searches fetch the search data and each semantic artifact once", async () => {
  const { sandbox, requests } = loadFlexsearch({ withSemanticModule: true })

  await sandbox.hextraSearch.preload()
  await sandbox.hextraSearch.preload()
  await sandbox.hextraSearch.search("one")
  await sandbox.hextraSearch.search("two")

  const counts = Object.fromEntries(requests.map((url) => [url, requests.filter((other) => other === url).length]))
  assert.deepEqual(counts, {
    [SEARCH_DATA_URL]: 1,
    "token-table.bin": 1,
    "token-scales.bin": 1,
    "doc-vectors.bin": 1,
    "manifest.json": 1,
    "tokenizer-config.json": 1,
    "meta.json": 1,
  })
})

test("search returns keyword results without waiting for pending semantic artifacts", async () => {
  const { sandbox } = loadFlexsearch({ withSemanticModule: true, artifactResponse: () => new Promise(() => {}) })

  const results = await sandbox.hextraSearch.search("anything")

  // `results` comes from the sandbox realm, so compare by length, not deepEqual.
  assert.equal(results.length, 0)
})

test("an HTTP error for the search data rejects preload, and the next search retries", async () => {
  // A JSON error body would otherwise parse as an empty corpus and be cached.
  const responses = [{ ok: false, status: 503, json: () => Promise.resolve({}) }, okResponse(WIDGET_CORPUS)]
  const { sandbox, requests } = loadFlexsearch({
    withSemanticModule: false,
    searchDataResponse: () => Promise.resolve(responses.shift()),
  })

  await assert.rejects(sandbox.hextraSearch.preload(), /503/)
  const results = await sandbox.hextraSearch.search("widgets")

  assert.deepEqual(plain(results), WIDGET_GROUPS)
  assert.equal(requests.filter((url) => url === SEARCH_DATA_URL).length, 2)
})
