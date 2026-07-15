const assert = require("node:assert/strict")
const test = require("node:test")

const { embedQuery, parseFloat32LE, parseInt8, scoreDocuments, tokenizeWordPiece } = require("./semantic-search.js")

function tokenizerConfig() {
  return {
    add_special_tokens: false,
    drop_unknown: true,
    unknown_token_id: 0,
    tokenizer: {
      normalizer: {
        type: "BertNormalizer",
        clean_text: true,
        handle_chinese_chars: true,
        strip_accents: true,
        lowercase: true,
      },
      pre_tokenizer: { type: "BertPreTokenizer" },
      model: {
        type: "WordPiece",
        vocab: {
          "[UNK]": 0,
          semantic: 1,
          search: 2,
          cafe: 3,
          naive: 4,
          code: 5,
          "##base": 6,
        },
        unk_token: "[UNK]",
        continuing_subword_prefix: "##",
        max_input_chars_per_word: 100,
      },
    },
  }
}

test("tokenizer handles plain, accented, subword, and partial-OOV text without special tokens", () => {
  const config = tokenizerConfig()

  assert.deepEqual(tokenizeWordPiece("Semantic search", config), [1, 2])
  assert.deepEqual(tokenizeWordPiece("café naïve", config), [3, 4])
  assert.deepEqual(tokenizeWordPiece("CodeBase xyzzy search", config), [5, 6, 2])
  assert.equal(tokenizeWordPiece("Semantic search", config).includes(0), false)
})

test("binary parsing uses signed int8 and little-endian float32", () => {
  const signed = parseInt8(Uint8Array.from([255, 0, 127, 128]).buffer)
  const scaleBytes = new ArrayBuffer(8)
  const view = new DataView(scaleBytes)
  view.setFloat32(0, 0.5, true)
  view.setFloat32(4, -2.25, true)

  assert.deepEqual(Array.from(signed), [-1, 0, 127, -128])
  assert.deepEqual(Array.from(parseFloat32LE(scaleBytes)), [0.5, -2.25])
})

test("query embedding dequantizes per row, mean-pools, and normalizes", () => {
  const config = tokenizerConfig()
  const dimensions = 2
  const rows = 7
  const table = new Int8Array(rows * dimensions)
  table.set([127, 0], 3 * dimensions)
  table.set([0, 127], 4 * dimensions)
  const scales = new Float32Array(rows)
  scales[3] = 1 / 127
  scales[4] = 1 / 127

  const embedded = embedQuery("café naïve", config, table, scales, dimensions)

  assert.ok(embedded instanceof Float32Array)
  assert.ok(Math.abs(embedded[0] - Math.SQRT1_2) < 1e-6)
  assert.ok(Math.abs(embedded[1] - Math.SQRT1_2) < 1e-6)
  assert.equal(embedQuery("xyzzy", config, table, scales, dimensions), null)
})

test("ordinary document vectors score exactly", () => {
  const query = new Float32Array([0.6, 0.8])
  const documents = new Int8Array([3, 4, -3, 4])

  const scores = scoreDocuments(query, documents, 2)

  assert.ok(Math.abs(scores[0] - 5) < 1e-6)
  assert.ok(Math.abs(scores[1] - 1.4) < 1e-6)
})

test("high-magnitude scoring uses a Number accumulator without integer wraparound", () => {
  const dimensions = 512
  const query = new Float32Array(dimensions).fill(127)
  const documents = new Int8Array(dimensions).fill(127)

  const [score] = scoreDocuments(query, documents, dimensions)

  assert.equal(score, 512 * 127 * 127)
  assert.ok(score > 32767)
})
