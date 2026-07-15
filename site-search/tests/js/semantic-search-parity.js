const fs = require("node:fs")
const path = require("node:path")

const semanticSearchPath = path.resolve(__dirname, "../../../hugo/assets/js/semantic-search.js")
const { embedQuery, tokenizeWordPiece } = require(semanticSearchPath)

const MAX_INPUT_BYTES = 16 * 1024 * 1024
const MAX_TEXTS = 100
const MAX_TEXT_LENGTH = 10_000

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

function validateRequest(request) {
  const expectedKeys = ["dimensions", "texts", "token_scales_base64", "token_table_base64", "tokenizer_config"]
  if (
    request === null ||
    typeof request !== "object" ||
    Array.isArray(request) ||
    JSON.stringify(Object.keys(request).sort()) !== JSON.stringify(expectedKeys) ||
    !Number.isInteger(request.dimensions) ||
    request.dimensions <= 0 ||
    request.dimensions > 4096 ||
    !Array.isArray(request.texts) ||
    request.texts.length === 0 ||
    request.texts.length > MAX_TEXTS ||
    request.texts.some((text) => typeof text !== "string" || text.length > MAX_TEXT_LENGTH) ||
    request.tokenizer_config === null ||
    typeof request.tokenizer_config !== "object" ||
    Array.isArray(request.tokenizer_config)
  ) {
    throw new TypeError("invalid parity request")
  }
}

function main() {
  const input = fs.readFileSync(0, "utf8")
  if (Buffer.byteLength(input, "utf8") > MAX_INPUT_BYTES) throw new RangeError("parity request is too large")
  const request = JSON.parse(input)
  validateRequest(request)

  const tokenTable = new Int8Array(decodeBase64(request.token_table_base64, "token_table_base64"))
  const tokenScales = new Float32Array(decodeBase64(request.token_scales_base64, "token_scales_base64"))
  const results = request.texts.map((text) => {
    const tokenIds = tokenizeWordPiece(text, request.tokenizer_config)
    const vector = embedQuery(text, request.tokenizer_config, tokenTable, tokenScales, request.dimensions)
    return { text, token_ids: tokenIds, vector: vector === null ? null : Array.from(vector) }
  })
  process.stdout.write(JSON.stringify({ results }))
}

try {
  main()
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : "parity bridge failed"}\n`)
  process.exitCode = 1
}
