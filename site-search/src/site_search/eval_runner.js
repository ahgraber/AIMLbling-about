const fs = require("node:fs")
const path = require("node:path")

const semanticSearchPath = path.resolve(__dirname, "../../../hugo/assets/js/semantic-search.js")
const { embedQuery, rankHybridResults, rollupSemanticChunks, scoreDocuments } = require(semanticSearchPath)

const MAX_INPUT_BYTES = 16 * 1024 * 1024
const MAX_QUERIES = 100

function decodeBase64(value, field) {
  if (
    typeof value !== "string" ||
    value.length === 0 ||
    value.length % 4 !== 0 ||
    !/^[A-Za-z0-9+/]+={0,2}$/.test(value)
  ) {
    throw new TypeError(`${field} must be canonical base64`)
  }
  const bytes = Buffer.from(value, "base64")
  if (bytes.toString("base64") !== value) throw new TypeError(`${field} must be canonical base64`)
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
}

function object(value, location) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${location} must be an object`)
  }
  return value
}

function main() {
  const input = fs.readFileSync(0, "utf8")
  if (Buffer.byteLength(input, "utf8") > MAX_INPUT_BYTES) throw new RangeError("evaluation request is too large")
  const request = object(JSON.parse(input), "request")
  const expectedKeys = [
    "dimensions",
    "document_global_scale",
    "document_vectors_base64",
    "manifest",
    "queries",
    "token_scales_base64",
    "token_table_base64",
    "tokenizer_config",
  ]
  if (
    JSON.stringify(Object.keys(request).sort()) !== JSON.stringify(expectedKeys) ||
    !Number.isInteger(request.dimensions) ||
    request.dimensions <= 0 ||
    request.dimensions > 4096 ||
    !Number.isFinite(request.document_global_scale) ||
    request.document_global_scale <= 0 ||
    !Array.isArray(request.manifest) ||
    !Array.isArray(request.queries) ||
    request.queries.length === 0 ||
    request.queries.length > MAX_QUERIES
  ) {
    throw new TypeError("invalid evaluation request")
  }

  const tokenTable = new Int8Array(decodeBase64(request.token_table_base64, "token_table_base64"))
  const tokenScales = new Float32Array(decodeBase64(request.token_scales_base64, "token_scales_base64"))
  const documentVectors = new Int8Array(decodeBase64(request.document_vectors_base64, "document_vectors_base64"))
  if (documentVectors.length !== request.manifest.length * request.dimensions) {
    throw new RangeError("document vectors do not correspond to manifest rows")
  }

  const results = request.queries.map((rawQuery) => {
    const query = object(rawQuery, "query")
    if (
      Object.keys(query).length !== 2 ||
      typeof query.query !== "string" ||
      !query.query ||
      !Array.isArray(query.keyword_page_ids) ||
      !query.keyword_page_ids.every((pageId) => typeof pageId === "string" && pageId)
    ) {
      throw new TypeError("invalid query request")
    }
    const vector = embedQuery(query.query, request.tokenizer_config, tokenTable, tokenScales, request.dimensions)
    if (vector === null) {
      return { query: query.query, "semantic-only": [], hybrid: query.keyword_page_ids }
    }
    const scores = scoreDocuments(vector, documentVectors, request.dimensions)
    const semanticChunks = request.manifest.map((row, index) => ({
      pageId: row.page_id,
      route: row.url,
      title: row.title,
      heading: row.heading,
      crumb: row.crumb,
      score: scores[index] * request.document_global_scale,
    }))
    const semanticPages = rollupSemanticChunks(semanticChunks)
    const keywordPages = query.keyword_page_ids.map((pageId) => ({ pageId }))
    return {
      query: query.query,
      "semantic-only": semanticPages.map((candidate) => candidate.pageId),
      hybrid: rankHybridResults(keywordPages, semanticChunks, "ready").map((candidate) => candidate.pageId),
    }
  })
  process.stdout.write(JSON.stringify({ results }))
}

try {
  main()
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : "evaluation bridge failed"}\n`)
  process.exitCode = 1
}
