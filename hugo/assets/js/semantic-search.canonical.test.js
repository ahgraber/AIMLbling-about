const assert = require("node:assert/strict")
const test = require("node:test")

const { canonicalizeRoute } = require("./semantic-search.js")

// The fusion join key: the keyword engine derives a page's identity from its
// route, and it MUST equal the semantic manifest's page_id, which the Python
// builder computes as `route if route == "/" else route.rstrip("/")`
// (build_index.py). A prior `route.trimEnd("/")` was a no-op (trimEnd ignores
// its argument and strips only whitespace), so keyword ids kept a trailing
// slash that manifest ids did not — and no page ever fused. These expected
// values are hard-coded to the Python reference so any such regression fails.
test("canonicalizeRoute strips trailing slashes to match the Python builder's page_id", () => {
  const cases = [
    ["/", "/"],
    ["/glossary", "/glossary"],
    ["/blog/the-bitter-lesson/", "/blog/the-bitter-lesson"],
    ["/treadmill/2024/11/08/", "/treadmill/2024/11/08"],
    ["/blog/post//", "/blog/post"],
  ]
  for (const [route, expected] of cases) {
    assert.equal(canonicalizeRoute(route), expected, `route ${route}`)
    // Guards specifically against the trimEnd("/") no-op regression.
    assert.ok(route === "/" || !canonicalizeRoute(route).endsWith("/"), `no trailing slash for ${route}`)
  }
})

test("canonicalizeRoute equals a rstrip reference across trailing-slash routes", () => {
  const rstrip = (route) => (route === "/" ? route : route.replace(/\/+$/, ""))
  for (const route of ["/", "/a/", "/a/b/", "/a/b/c/", "/a/b/c"]) {
    assert.equal(canonicalizeRoute(route), rstrip(route))
  }
})

test("canonicalizeRoute rejects empty or non-string routes", () => {
  assert.throws(() => canonicalizeRoute(""), /non-empty string/)
  assert.throws(() => canonicalizeRoute(undefined), /non-empty string/)
})
