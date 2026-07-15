// Pure semantic-search primitives and a dependency-injected artifact loader.

const SEMANTIC_ARTIFACT_NAMES = [
  "token-table.bin",
  "token-scales.bin",
  "doc-vectors.bin",
  "manifest.json",
  "tokenizer-config.json",
  "meta.json",
]
const FUSION_INPUT_LIMIT = 10
const RRF_K = 60
const KEYWORD_RRF_WEIGHT = 1
const SEMANTIC_RRF_WEIGHT = 2
// Semantic pages contribute to fusion only above this cosine floor. A
// non-positive similarity is orthogonal or anti-correlated — never a genuine
// match — so excluding it keeps the 2x-weighted semantic signal from injecting
// noise pages ahead of precise keyword hits. A tuned positive threshold is a
// follow-up to validate against the eval; zero is the safe, no-retune floor.
const SEMANTIC_SCORE_FLOOR = 0

function parseInt8(buffer) {
  if (!(buffer instanceof ArrayBuffer)) throw new TypeError("int8 artifact must be an ArrayBuffer")
  return new Int8Array(buffer)
}

function parseFloat32LE(buffer) {
  if (!(buffer instanceof ArrayBuffer)) throw new TypeError("float32 artifact must be an ArrayBuffer")
  if (buffer.byteLength % 4 !== 0) throw new RangeError("float32 artifact length must be divisible by four")
  const view = new DataView(buffer)
  const values = new Float32Array(buffer.byteLength / 4)
  for (let index = 0; index < values.length; index++) {
    values[index] = view.getFloat32(index * 4, true)
  }
  return values
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

function validateMetadata(meta) {
  if (
    !isObject(meta) ||
    meta.version !== 1 ||
    typeof meta.model_id !== "string" ||
    !meta.model_id ||
    !Number.isInteger(meta.dimensions) ||
    meta.dimensions <= 0 ||
    !Number.isFinite(meta.document_global_scale) ||
    meta.document_global_scale <= 0 ||
    !Number.isInteger(meta.chunk_count) ||
    meta.chunk_count <= 0 ||
    !Number.isInteger(meta.excluded_chunk_count) ||
    meta.excluded_chunk_count < 0 ||
    !Number.isInteger(meta.total_payload_size) ||
    meta.total_payload_size <= 0 ||
    typeof meta.built_at !== "string" ||
    !meta.built_at ||
    !isObject(meta.corpus_stats)
  ) {
    throw new TypeError("semantic metadata is invalid")
  }
}

function validateManifest(manifest, chunkCount) {
  if (!Array.isArray(manifest) || manifest.length !== chunkCount) {
    throw new RangeError("manifest rows must correspond to metadata")
  }
  const stringFields = ["chunk_id", "page_id", "url", "title", "heading", "crumb"]
  for (const row of manifest) {
    if (
      !isObject(row) ||
      stringFields.some((field) => typeof row[field] !== "string") ||
      !row.chunk_id ||
      !row.page_id ||
      !row.url
    ) {
      throw new TypeError("manifest rows are invalid")
    }
  }
}

function validateTokenizerConfig(config) {
  const tokenizer = config?.tokenizer
  const unknownTokenId = config?.unknown_token_id
  if (
    !isObject(config) ||
    config.version !== 1 ||
    config.add_special_tokens !== false ||
    config.drop_unknown !== true ||
    (unknownTokenId !== null && !Number.isInteger(unknownTokenId)) ||
    tokenizer?.normalizer?.type !== "BertNormalizer" ||
    typeof tokenizer.normalizer.strip_accents !== "boolean" ||
    tokenizer?.pre_tokenizer?.type !== "BertPreTokenizer" ||
    tokenizer?.model?.type !== "WordPiece" ||
    !isObject(tokenizer.model.vocab)
  ) {
    throw new TypeError("tokenizer configuration is invalid")
  }
}

function validateArtifactPayloads(payloads) {
  const meta = payloads["meta.json"]
  const manifest = payloads["manifest.json"]
  const tokenizerConfig = payloads["tokenizer-config.json"]
  validateMetadata(meta)
  validateManifest(manifest, meta.chunk_count)
  validateTokenizerConfig(tokenizerConfig)

  const tokenTable = parseInt8(payloads["token-table.bin"])
  const tokenScales = parseFloat32LE(payloads["token-scales.bin"])
  const documentVectors = parseInt8(payloads["doc-vectors.bin"])
  if (tokenScales.length === 0 || tokenTable.length !== tokenScales.length * meta.dimensions) {
    throw new RangeError("token artifacts do not match metadata dimensions")
  }
  if (documentVectors.length !== manifest.length * meta.dimensions) {
    throw new RangeError("document vectors do not correspond to manifest rows")
  }
  if (Array.from(tokenScales).some((scale) => !Number.isFinite(scale) || scale < 0)) {
    throw new RangeError("token scales must be finite and non-negative")
  }
  return { tokenTable, tokenScales, documentVectors, manifest, tokenizerConfig, meta }
}

function createSemanticArtifactLoader(fetchArtifact, artifactURLs) {
  if (typeof fetchArtifact !== "function" || !isObject(artifactURLs)) {
    throw new TypeError("artifact loader requires fetch and URL dependencies")
  }
  for (const name of SEMANTIC_ARTIFACT_NAMES) {
    if (typeof artifactURLs[name] !== "string" || !artifactURLs[name]) {
      throw new TypeError(`missing artifact URL for ${name}`)
    }
  }

  let state = "unavailable"
  let artifacts
  let initialization

  async function fetchOne(name) {
    const response = await fetchArtifact(artifactURLs[name])
    if (response?.ok !== true) {
      throw new Error(`artifact request failed with status ${response?.status ?? "unknown"}`)
    }
    const method = name.endsWith(".json") ? "json" : "arrayBuffer"
    if (typeof response[method] !== "function") {
      throw new TypeError(`artifact response does not support ${method}`)
    }
    return response[method]()
  }

  function initialize() {
    if (initialization) return initialization
    state = "pending"
    initialization = Promise.allSettled(SEMANTIC_ARTIFACT_NAMES.map((name) => fetchOne(name)))
      .then((results) => {
        if (results.some((result) => result.status === "rejected")) {
          throw new Error("one or more semantic artifacts failed to load")
        }
        const payloads = Object.fromEntries(SEMANTIC_ARTIFACT_NAMES.map((name, index) => [name, results[index].value]))
        artifacts = validateArtifactPayloads(payloads)
        state = "ready"
        return artifacts
      })
      .catch(() => {
        artifacts = undefined
        state = "failed"
        return undefined
      })
    return initialization
  }

  return {
    get state() {
      return state
    },
    get artifacts() {
      return artifacts
    },
    initialize,
  }
}

function canonicalizeRoute(route) {
  // Single definition of the keyword/semantic fusion join key. The Python
  // builder derives page_id and section url with route.rstrip("/"); the client
  // must match exactly or fusion never merges a page found by both engines.
  if (typeof route !== "string" || !route) throw new TypeError("route must be a non-empty string")
  return route === "/" ? route : route.replace(/\/+$/, "")
}

function isChinese(codePoint) {
  return (
    (codePoint >= 0x4e00 && codePoint <= 0x9fff) ||
    (codePoint >= 0x3400 && codePoint <= 0x4dbf) ||
    (codePoint >= 0x20000 && codePoint <= 0x2a6df) ||
    (codePoint >= 0x2a700 && codePoint <= 0x2b73f) ||
    (codePoint >= 0x2b740 && codePoint <= 0x2b81f) ||
    (codePoint >= 0x2b820 && codePoint <= 0x2ceaf) ||
    (codePoint >= 0xf900 && codePoint <= 0xfaff) ||
    (codePoint >= 0x2f800 && codePoint <= 0x2fa1f)
  )
}

function cleanAndSpace(text, normalizer) {
  let output = ""
  for (const character of text) {
    const codePoint = character.codePointAt(0)
    if (normalizer.clean_text) {
      if (codePoint === 0 || codePoint === 0xfffd) continue
      // Match BertNormalizer._clean_text ordering exactly: control/format chars
      // (Cc/Cf) are removed first — except tab/newline/CR — so vertical tab and
      // form feed are dropped, not turned into spaces. Only then is BERT's
      // whitespace set (space, tab, newline, CR, and Unicode Zs separators)
      // collapsed to a space. Using JS `\s` here would misclassify \v, \f, and
      // other Unicode whitespace that BERT does not treat as spacing.
      if (character !== "\t" && character !== "\n" && character !== "\r" && /[\p{Cc}\p{Cf}]/u.test(character)) continue
      if (
        character === " " ||
        character === "\t" ||
        character === "\n" ||
        character === "\r" ||
        /\p{Zs}/u.test(character)
      ) {
        output += " "
        continue
      }
    }
    if (normalizer.handle_chinese_chars && isChinese(codePoint)) {
      output += ` ${character} `
    } else {
      output += character
    }
  }
  if (normalizer.lowercase) output = output.toLowerCase()
  // HF BertNormalizer strips only nonspacing marks (category Mn), matching the
  // Python reference tokenizer; \p{M} would also drop spacing/enclosing marks
  // and silently diverge on non-Latin scripts.
  if (normalizer.strip_accents) output = output.normalize("NFD").replace(/\p{Mn}/gu, "")
  return output
}

function isPunctuation(character) {
  const codePoint = character.codePointAt(0)
  return (
    (codePoint >= 33 && codePoint <= 47) ||
    (codePoint >= 58 && codePoint <= 64) ||
    (codePoint >= 91 && codePoint <= 96) ||
    (codePoint >= 123 && codePoint <= 126) ||
    /\p{P}/u.test(character)
  )
}

function basicTokens(text) {
  const tokens = []
  let current = ""
  const flush = () => {
    if (current) tokens.push(current)
    current = ""
  }
  for (const character of text) {
    if (/\s/u.test(character)) {
      flush()
    } else if (isPunctuation(character)) {
      flush()
      tokens.push(character)
    } else {
      current += character
    }
  }
  flush()
  return tokens
}

function tokenizeWordPiece(text, config) {
  if (config.add_special_tokens !== false || config.drop_unknown !== true) {
    throw new Error("tokenizer config must disable special tokens and drop unknown pieces")
  }
  const tokenizer = config.tokenizer
  const normalizer = tokenizer?.normalizer
  const model = tokenizer?.model
  if (normalizer?.type !== "BertNormalizer" || typeof normalizer.strip_accents !== "boolean") {
    throw new Error("tokenizer config must contain resolved Bert normalization")
  }
  if (tokenizer?.pre_tokenizer?.type !== "BertPreTokenizer" || model?.type !== "WordPiece") {
    throw new Error("tokenizer config must contain Bert pre-tokenization and WordPiece")
  }

  const vocabulary = model.vocab
  const prefix = model.continuing_subword_prefix || "##"
  const maxCharacters = model.max_input_chars_per_word || 100
  const ids = []
  for (const token of basicTokens(cleanAndSpace(String(text), normalizer))) {
    const characters = Array.from(token)
    if (characters.length > maxCharacters) continue
    const pieces = []
    let start = 0
    let unknown = false
    while (start < characters.length) {
      let end = characters.length
      let found
      while (end > start) {
        const piece = `${start === 0 ? "" : prefix}${characters.slice(start, end).join("")}`
        if (Object.hasOwn(vocabulary, piece)) {
          found = piece
          break
        }
        end--
      }
      if (found === undefined) {
        unknown = true
        break
      }
      pieces.push(vocabulary[found])
      start = end
    }
    if (!unknown) ids.push(...pieces)
  }
  return ids
}

function embedQuery(text, config, tokenTable, tokenScales, dimensions) {
  if (!(tokenTable instanceof Int8Array)) throw new TypeError("token table must be signed int8")
  if (!(tokenScales instanceof Float32Array)) throw new TypeError("token scales must be float32")
  if (!Number.isInteger(dimensions) || dimensions <= 0) throw new RangeError("dimensions must be positive")
  if (tokenTable.length !== tokenScales.length * dimensions) {
    throw new RangeError("token rows and scales must correspond")
  }
  const tokenIds = tokenizeWordPiece(text, config)
  if (tokenIds.length === 0) return null

  const vector = new Float32Array(dimensions)
  for (const tokenId of tokenIds) {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= tokenScales.length) {
      throw new RangeError("token ID is outside the exported table")
    }
    const rowOffset = tokenId * dimensions
    const scale = tokenScales[tokenId]
    for (let dimension = 0; dimension < dimensions; dimension++) {
      vector[dimension] += tokenTable[rowOffset + dimension] * scale
    }
  }

  let squaredNorm = 0
  for (let dimension = 0; dimension < dimensions; dimension++) {
    vector[dimension] /= tokenIds.length
    squaredNorm += vector[dimension] * vector[dimension]
  }
  const norm = Math.sqrt(squaredNorm)
  if (norm === 0 || !Number.isFinite(norm)) return null
  for (let dimension = 0; dimension < dimensions; dimension++) vector[dimension] /= norm
  return vector
}

function scoreDocuments(queryVector, documentVectors, dimensions) {
  if (!(queryVector instanceof Float32Array)) throw new TypeError("query vector must be float32")
  if (!(documentVectors instanceof Int8Array)) throw new TypeError("document vectors must be signed int8")
  if (queryVector.length !== dimensions || documentVectors.length % dimensions !== 0) {
    throw new RangeError("vector dimensions do not match")
  }
  const scores = new Float64Array(documentVectors.length / dimensions)
  for (let row = 0; row < scores.length; row++) {
    let score = 0
    const rowOffset = row * dimensions
    for (let dimension = 0; dimension < dimensions; dimension++) {
      score += queryVector[dimension] * documentVectors[rowOffset + dimension]
    }
    scores[row] = score
  }
  return scores
}

function rollupSemanticChunks(chunkResults) {
  const rankedChunks = chunkResults
    .map((candidate, index) => ({ candidate, index }))
    .sort((left, right) => right.candidate.score - left.candidate.score || left.index - right.index)
  const seenPages = new Set()
  const pages = []
  for (const { candidate } of rankedChunks) {
    if (
      typeof candidate.pageId !== "string" ||
      !candidate.pageId ||
      typeof candidate.crumb !== "string" ||
      !Number.isFinite(candidate.score)
    ) {
      throw new TypeError("semantic candidates require a pageId and crumb plus a finite score")
    }
    if (seenPages.has(candidate.pageId)) continue
    seenPages.add(candidate.pageId)
    pages.push(candidate)
  }
  return pages
}

function rankHybridResults(keywordPages, semanticChunks, semanticState) {
  if (semanticState !== "ready") return keywordPages

  const boundedKeywordPages = keywordPages.slice(0, FUSION_INPUT_LIMIT)
  const semanticPages = rollupSemanticChunks(semanticChunks)
    .filter((page) => page.score > SEMANTIC_SCORE_FLOOR)
    .slice(0, FUSION_INPUT_LIMIT)
  const byPage = new Map()
  let firstSeen = 0
  for (let index = 0; index < boundedKeywordPages.length; index++) {
    const candidate = boundedKeywordPages[index]
    if (typeof candidate.pageId !== "string" || !candidate.pageId) {
      throw new TypeError("keyword candidates require a pageId")
    }
    byPage.set(candidate.pageId, {
      candidate,
      firstSeen: firstSeen++,
      keywordRank: index + 1,
      semanticRank: undefined,
    })
  }
  for (let index = 0; index < semanticPages.length; index++) {
    const candidate = semanticPages[index]
    const existing = byPage.get(candidate.pageId)
    if (existing) {
      // A page matched by both engines keeps its keyword section, route, and
      // highlighted excerpt — the reader typed a term that matched there, so
      // exact-term relevance and the highlight are preserved. Semantic only
      // contributes its rank to the fused score.
      existing.semanticRank = index + 1
    } else {
      byPage.set(candidate.pageId, {
        candidate,
        firstSeen: firstSeen++,
        keywordRank: undefined,
        semanticRank: index + 1,
      })
    }
  }

  return Array.from(byPage.values())
    .map((entry) => ({
      ...entry.candidate,
      keywordRank: entry.keywordRank,
      semanticRank: entry.semanticRank,
      rrfScore:
        (entry.keywordRank === undefined ? 0 : KEYWORD_RRF_WEIGHT / (RRF_K + entry.keywordRank)) +
        (entry.semanticRank === undefined ? 0 : SEMANTIC_RRF_WEIGHT / (RRF_K + entry.semanticRank)),
      firstSeen: entry.firstSeen,
    }))
    .sort((left, right) => right.rrfScore - left.rrfScore || left.firstSeen - right.firstSeen)
    .map(({ firstSeen: _, ...candidate }) => candidate)
}

function getHybridDisplayResults(keywordDisplayResults, keywordCandidates, query, semanticLoader, maxPageResults) {
  if (semanticLoader?.state !== "ready") return keywordDisplayResults
  try {
    const { tokenTable, tokenScales, documentVectors, manifest, tokenizerConfig, meta } = semanticLoader.artifacts
    const queryVector = embedQuery(query, tokenizerConfig, tokenTable, tokenScales, meta.dimensions)
    if (queryVector === null) return keywordDisplayResults

    const scores = scoreDocuments(queryVector, documentVectors, meta.dimensions)
    const semanticCandidates = manifest.map((row, index) => ({
      pageId: row.page_id,
      route: row.url,
      title: row.title,
      heading: row.heading,
      crumb: row.crumb,
      score: scores[index] * meta.document_global_scale,
      source: "semantic",
    }))
    return rankHybridResults(keywordCandidates, semanticCandidates, "ready")
      .slice(0, maxPageResults)
      .map((candidate, index) => ({
        id: `hybrid_${index}`,
        route: candidate.route,
        prefix: candidate.prefix ?? candidate.crumb,
        children: {
          title: candidate.heading,
          ...(candidate.keywordRoute === candidate.route && candidate.content && { content: candidate.content }),
        },
      }))
  } catch {
    return keywordDisplayResults
  }
}

const semanticSearchAPI = {
  canonicalizeRoute,
  createSemanticArtifactLoader,
  embedQuery,
  getHybridDisplayResults,
  parseFloat32LE,
  parseInt8,
  rankHybridResults,
  rollupSemanticChunks,
  scoreDocuments,
  tokenizeWordPiece,
}

if (typeof globalThis !== "undefined") globalThis.SiteSemanticSearch = semanticSearchAPI
if (typeof module !== "undefined") module.exports = semanticSearchAPI
