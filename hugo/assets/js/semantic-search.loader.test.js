const assert = require("node:assert/strict")
const test = require("node:test")

const semanticSearch = require("./semantic-search.js")

const ARTIFACT_NAMES = [
  "token-table.bin",
  "token-scales.bin",
  "doc-vectors.bin",
  "manifest.json",
  "tokenizer-config.json",
  "meta.json",
]

function validPayloads() {
  return {
    "token-table.bin": new Int8Array([1, 2, 3, 4]).buffer,
    "token-scales.bin": new Float32Array([0.5, 0.25]).buffer,
    "doc-vectors.bin": new Int8Array([5, 6, 7, 8]).buffer,
    "manifest.json": [
      { chunk_id: "one", page_id: "/one", url: "/one", title: "One", heading: "One", crumb: "One" },
      { chunk_id: "two", page_id: "/two", url: "/two#part", title: "Two", heading: "Part", crumb: "Two" },
    ],
    "tokenizer-config.json": {
      version: 1,
      add_special_tokens: false,
      drop_unknown: true,
      unknown_token_id: 0,
      tokenizer: {
        normalizer: { type: "BertNormalizer", strip_accents: true },
        pre_tokenizer: { type: "BertPreTokenizer" },
        model: { type: "WordPiece", vocab: { "[UNK]": 0, known: 1 } },
      },
    },
    "meta.json": {
      version: 1,
      model_id: "test/model",
      dimensions: 2,
      document_global_scale: 0.1,
      chunk_count: 2,
      excluded_chunk_count: 0,
      total_payload_size: 100,
      built_at: "2026-07-12T00:00:00Z",
      corpus_stats: {},
    },
  }
}

function harness(payloads = validPayloads(), responseOverrides = {}) {
  const calls = []
  const urls = Object.fromEntries(ARTIFACT_NAMES.map((name) => [name, `/search/${name}`]))
  const fetchArtifact = async (url) => {
    calls.push(url)
    const name = url.slice(url.lastIndexOf("/") + 1)
    const override = responseOverrides[name]
    if (override instanceof Error) throw override
    const payload = payloads[name]
    return {
      ok: override?.ok ?? true,
      status: override?.status ?? 200,
      json: async () => payload,
      arrayBuffer: async () => payload,
    }
  }
  return { calls, fetchArtifact, urls }
}

test("CommonJS API is also exposed through the explicit browser namespace", () => {
  assert.strictEqual(globalThis.SiteSemanticSearch, semanticSearch)
})

test("loader fetches and validates all six artifacts before becoming ready", async () => {
  const { calls, fetchArtifact, urls } = harness()
  const loader = semanticSearch.createSemanticArtifactLoader(fetchArtifact, urls)

  assert.equal(loader.state, "unavailable")
  const initialized = loader.initialize()
  assert.equal(loader.state, "pending")
  const artifacts = await initialized

  assert.equal(loader.state, "ready")
  assert.deepEqual(new Set(calls), new Set(Object.values(urls)))
  assert.equal(calls.length, 6)
  assert.ok(artifacts.tokenTable instanceof Int8Array)
  assert.ok(artifacts.tokenScales instanceof Float32Array)
  assert.ok(artifacts.documentVectors instanceof Int8Array)
  assert.equal(artifacts.manifest.length, 2)
  assert.equal(artifacts.meta.dimensions, 2)
  assert.strictEqual(loader.artifacts, artifacts)
})

test("loader initialization and every artifact fetch happen at most once", async () => {
  const { calls, fetchArtifact, urls } = harness()
  const loader = semanticSearch.createSemanticArtifactLoader(fetchArtifact, urls)

  const first = loader.initialize()
  const second = loader.initialize()

  assert.strictEqual(second, first)
  await Promise.all([first, second, loader.initialize()])
  assert.equal(calls.length, 6)
})

test("HTTP and fetch failures settle silently into failed state after all requests start", async () => {
  for (const overrides of [
    { "manifest.json": { ok: false, status: 404 } },
    { "doc-vectors.bin": new Error("network unavailable") },
  ]) {
    const { calls, fetchArtifact, urls } = harness(validPayloads(), overrides)
    const loader = semanticSearch.createSemanticArtifactLoader(fetchArtifact, urls)

    const artifacts = await loader.initialize()

    assert.equal(artifacts, undefined)
    assert.equal(loader.state, "failed")
    assert.equal(loader.artifacts, undefined)
    assert.equal(calls.length, 6)
  }
})

test("malformed metadata and cross-artifact shape mismatches fail closed", async (t) => {
  const cases = {
    "invalid metadata dimensions": (payloads) => {
      payloads["meta.json"].dimensions = 0
    },
    "manifest row count mismatch": (payloads) => {
      payloads["meta.json"].chunk_count = 3
    },
    "invalid manifest row": (payloads) => {
      delete payloads["manifest.json"][0].page_id
    },
    "token table row mismatch": (payloads) => {
      payloads["token-table.bin"] = new Int8Array([1, 2, 3]).buffer
    },
    "document vector row mismatch": (payloads) => {
      payloads["doc-vectors.bin"] = new Int8Array([1, 2]).buffer
    },
    "invalid tokenizer configuration": (payloads) => {
      payloads["tokenizer-config.json"].drop_unknown = false
    },
  }

  for (const [name, mutate] of Object.entries(cases)) {
    await t.test(name, async () => {
      const payloads = validPayloads()
      mutate(payloads)
      const { fetchArtifact, urls } = harness(payloads)
      const loader = semanticSearch.createSemanticArtifactLoader(fetchArtifact, urls)

      assert.equal(await loader.initialize(), undefined)
      assert.equal(loader.state, "failed")
    })
  }
})
