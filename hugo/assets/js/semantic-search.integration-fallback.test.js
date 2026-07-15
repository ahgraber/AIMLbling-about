// Integration-load coverage for the flexsearch.js fallback wiring.
//
// flexsearch.js is a Hugo-template IIFE with closure-private functions, so its
// semantic-integration glue (the `? : failed-stub` loader selection and the
// `?? passthrough` shims at the top of the IIFE) cannot be require()'d like the
// pure semantic-search.js API. This harness loads the *real* flexsearch.js
// source in a `vm` sandbox with the `{{ ... }}` template tokens neutralized and
// a minimal DOM stub, then asserts the module wires up keyword-only fallback
// without throwing in the two states that matter:
//   1. the semantic module failed to load entirely (globalThis.SiteSemanticSearch
//      undefined) — exercises the inline failed-loader stub + passthrough shims;
//   2. the semantic module is present — exercises the real createSemanticArtifactLoader path.
//
// Scope: this covers construction-time integration (KeywordFallback at the
// module-absent boundary). It does not drive a full focus -> search -> DOM
// render; that needs a real DOM (jsdom) and is covered manually by rollout-qa.md
// Check 3.

const assert = require("node:assert/strict")
const test = require("node:test")
const fs = require("node:fs")
const path = require("node:path")
const vm = require("node:vm")

const FLEXSEARCH_SRC = path.join(__dirname, "flexsearch.js")
const SEMANTIC_SRC = path.join(__dirname, "semantic-search.js")

// Replace every Hugo template action with a non-empty placeholder so string
// literals like "{{ ... }}" stay truthy (the real loader rejects empty URLs).
function neutralizeTemplates(source) {
  return source.replace(/\{\{[\s\S]*?\}\}/g, "x")
}

function makeDomStub() {
  const listeners = []
  const document = {
    addEventListener: (type, handler) => listeners.push({ type, handler }),
    // No search inputs/kbd elements at load time; the input-wiring loops become
    // no-ops, which is all we need to exercise the top-of-IIFE integration glue.
    querySelectorAll: () => [],
  }
  return { document, navigator: { userAgent: "" } }
}

function loadFlexsearch({ withSemanticModule }) {
  const sandbox = {
    ...makeDomStub(),
    console: { error: () => {}, warn: () => {}, log: () => {} },
    fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve({}) }),
    FlexSearch: { Document: class {} },
    setTimeout,
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
  return sandbox
}

test("flexsearch.js loads and wires keyword-only fallback when the semantic module is absent", () => {
  const sandbox = loadFlexsearch({ withSemanticModule: false })
  assert.strictEqual(sandbox.SiteSemanticSearch, undefined)
})

test("flexsearch.js loads without throwing when the semantic module is present", () => {
  const sandbox = loadFlexsearch({ withSemanticModule: true })
  assert.strictEqual(typeof sandbox.SiteSemanticSearch.getHybridDisplayResults, "function")
})
