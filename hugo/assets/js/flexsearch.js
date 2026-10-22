// FlexSearch data plane with hybrid semantic ranking.
//
// Site override of the Hextra "assets/js/flexsearch.js" asset. Hextra's
// search-dialog.js owns the UI and consumes `window.hextraSearch`
// ({ preload, search }); this file keeps that contract and returns the same
// page groups, `{ id, route, title, breadcrumbs, matches: [{ id, route, title, content }] }`.
// It differs from the theme's version:
//   - Routes are canonicalized (trailing slash removed) to match the page_ids
//     and URLs that the Python index builder writes to the semantic manifest.
//   - preload() also starts loading the semantic artifacts. search() never
//     waits for them; until they are ready, results are keyword-only.
//   - When the artifacts are ready, page groups are reordered and extended by
//     weighted reciprocal rank fusion (see semantic-search.js).

// The search data is not fingerprinted: the Docker build and `just hugo search-index`
// read public/en.search-data.json by that fixed name.
// {{ $searchDataFile := printf "%s.search-data.json" .Language.Lang }}
// {{ $searchData := resources.Get "json/search-data.json" | resources.ExecuteAsTemplate $searchDataFile . }}
;(() => {
  const searchDataURL = "{{ $searchData.RelPermalink }}"
  const semanticSearch = globalThis.SiteSemanticSearch
  // Fingerprinted, content-addressed artifact URLs resolved at build time. Hugo
  // hashes each file (`resources.Get | fingerprint`) and bakes its immutable URL
  // into this script, so every page references exactly one build's complete
  // set — no runtime URL lookup, no cross-build cache mixing.
  // When artifacts are absent each `with` yields an empty string, the loader
  // constructor rejects it, and the catch leaves search in keyword-only mode.
  let semanticLoader = { state: "failed", initialize: () => Promise.resolve(undefined) }
  try {
    if (semanticSearch?.createSemanticArtifactLoader) {
      semanticLoader = semanticSearch.createSemanticArtifactLoader(fetch, {
        "token-table.bin":
          '{{ with resources.Get "search/token-table.bin" }}{{ (. | fingerprint).RelPermalink }}{{ end }}',
        "token-scales.bin":
          '{{ with resources.Get "search/token-scales.bin" }}{{ (. | fingerprint).RelPermalink }}{{ end }}',
        "doc-vectors.bin":
          '{{ with resources.Get "search/doc-vectors.bin" }}{{ (. | fingerprint).RelPermalink }}{{ end }}',
        "manifest.json":
          '{{ with resources.Get "search/manifest.json" }}{{ (. | fingerprint).RelPermalink }}{{ end }}',
        "tokenizer-config.json":
          '{{ with resources.Get "search/tokenizer-config.json" }}{{ (. | fingerprint).RelPermalink }}{{ end }}',
        "meta.json": '{{ with resources.Get "search/meta.json" }}{{ (. | fingerprint).RelPermalink }}{{ end }}',
      })
    }
  } catch {
    // Artifacts absent or misconfigured: keep the keyword-only fallback loader.
  }
  const getHybridDisplayResults = semanticSearch?.getHybridDisplayResults ?? ((keywordGroups) => keywordGroups)
  // Shared with the semantic module so keyword page_ids/URLs match the Python
  // builder's route.rstrip("/") exactly; the identical inline fallback keeps
  // keyword-only search working when the semantic module fails to load.
  const canonicalizeRoute =
    semanticSearch?.canonicalizeRoute ?? ((route) => (route === "/" ? route : route.replace(/\/+$/, "")))

  let pageIndex
  let sectionIndex
  let indexPromise = null
  // Canonical pageId -> breadcrumb trail, for semantic-only results.
  const pageBreadcrumbs = new Map()

  /**
   * Starts loading the semantic artifacts and builds the keyword index on first use.
   * @returns {Promise<void>} Resolves when the keyword index is ready; rejects if the
   *   search data fails to load, and a later call retries.
   */
  function preloadIndex() {
    // Semantic failures degrade silently by design; the loader is idempotent.
    semanticLoader.initialize()
    if (indexPromise) return indexPromise
    const attempt = (async () => {
      const tokenize = '{{- site.Params.search.flexsearch.tokenize | default  "forward" -}}'

      // https://github.com/TryGhost/Ghost/pull/21148
      const regex =
        /[\u{4E00}-\u{9FFF}\u{3040}-\u{309F}\u{30A0}-\u{30FF}\u{AC00}-\u{D7A3}\u{3400}-\u{4DBF}\u{20000}-\u{2A6DF}\u{2A700}-\u{2B73F}\u{2B740}-\u{2B81F}\u{2B820}-\u{2CEAF}\u{2CEB0}-\u{2EBEF}\u{30000}-\u{3134F}\u{31350}-\u{323AF}\u{2EBF0}-\u{2EE5F}\u{F900}-\u{FAFF}\u{2F800}-\u{2FA1F}]|[0-9A-Za-zа-яÀ-ſЀ-ӿ؀-ۿঀ-৿Ḁ-ỿ֐-׿]+/gmu
      const encode = (str) => {
        return `${str}`.toLowerCase().match(regex) ?? []
      }

      pageIndex = new FlexSearch.Document({
        tokenize,
        encode,
        cache: 100,
        document: {
          id: "id",
          store: ["title", "breadcrumbs", "pageId"],
          index: "content",
        },
      })

      sectionIndex = new FlexSearch.Document({
        tokenize,
        encode,
        cache: 100,
        document: {
          id: "id",
          store: ["title", "content", "url", "display", "crumb"],
          index: "content",
          tag: [
            {
              field: "pageId",
            },
          ],
        },
      })

      const resp = await fetch(searchDataURL)
      const data = await resp.json()
      let pageId = 0
      for (const route in data) {
        let pageContent = ""
        ++pageId
        const canonicalPageId = canonicalizeRoute(route)
        const urlParts = route.split("/").filter((x) => x !== "" && !x.startsWith("#"))

        let crumb = ""
        const crumbParts = []
        let searchUrl = "/"
        for (let i = 0; i < urlParts.length; i++) {
          const urlPart = urlParts[i]
          searchUrl += `${urlPart}/`

          const crumbData = data[searchUrl]
          if (!crumbData) {
            console.debug("Excluded page", searchUrl, "- will not be included for search result breadcrumb for", route)
            continue
          }

          let title = data[searchUrl].title
          if (title === "_index") {
            title = urlPart
              .split("-")
              .map((x) => x)
              .join(" ")
          }
          crumbParts.push(title)
          crumb += title

          if (i < urlParts.length - 1) {
            crumb += " > "
          }
        }
        const breadcrumbs = crumbParts.slice(0, -1)
        pageBreadcrumbs.set(canonicalPageId, breadcrumbs)

        for (const heading in data[route].data) {
          const separator = heading.indexOf("#")
          const hash = separator === -1 ? heading : heading.slice(0, separator)
          const text = separator === -1 ? "" : heading.slice(separator + 1)
          const url = canonicalPageId + (hash ? `#${hash}` : "")
          const title = text || data[route].title

          const content = data[route].data[heading] || ""
          const paragraphs = content.split("\n").filter(Boolean)

          sectionIndex.add({
            id: url,
            url,
            title,
            crumb,
            pageId: `page_${pageId}`,
            content: title,
            ...(paragraphs[0] && { display: paragraphs[0] }),
          })

          for (let i = 0; i < paragraphs.length; i++) {
            sectionIndex.add({
              id: `${url}_${i}`,
              url,
              title,
              crumb,
              pageId: `page_${pageId}`,
              content: paragraphs[i],
            })
          }

          pageContent += ` ${title} ${content}`
        }

        pageIndex.add({
          id: pageId,
          pageId: canonicalPageId,
          title: data[route].title,
          breadcrumbs,
          content: pageContent,
        })
      }
    })()
    // Clear the cached promise on failure so the next call retries instead of
    // returning the rejection forever.
    const wrapped = attempt.catch((err) => {
      if (indexPromise === wrapped) indexPromise = null
      pageIndex = undefined
      sectionIndex = undefined
      throw err
    })
    indexPromise = wrapped
    return indexPromise
  }

  /**
   * Returns keyword page groups ordered by title-match count, then FlexSearch page
   * rank, with duplicate sections removed. Call only after preloadIndex() resolves.
   * Each group carries the canonical `pageId` that hybrid fusion joins on.
   */
  function performKeywordSearch(query, maxPageResults) {
    const maxSectionResults = parseInt("{{- site.Params.search.flexsearch.maxSectionResults | default 10 -}}", 10)
    const pageResults = pageIndex.search(query, maxPageResults, { enrich: true, suggest: true })[0]?.result || []

    const groups = []
    const pageTitleMatches = {}

    for (let i = 0; i < pageResults.length; i++) {
      const result = pageResults[i]
      pageTitleMatches[i] = 0

      const sectionResults =
        sectionIndex.search(query, { enrich: true, suggest: true, tag: { pageId: `page_${result.id}` } })[0]?.result ||
        []
      const occurred = {}
      const group = {
        _page_rk: i,
        pageId: result.doc.pageId,
        route: "",
        title: result.doc.title,
        breadcrumbs: result.doc.breadcrumbs || [],
        matches: [],
      }

      const nResults = Math.min(sectionResults.length, maxSectionResults)
      for (let j = 0; j < nResults; j++) {
        const { doc } = sectionResults[j]
        const isMatchingTitle = doc.display !== undefined
        if (isMatchingTitle) {
          pageTitleMatches[i]++
        }
        const { url, title } = doc
        const content = doc.display || doc.content

        if (occurred[`${url}@${content}`]) continue
        occurred[`${url}@${content}`] = true
        if (!group.route) group.route = url.split("#")[0]
        group.matches.push({ route: url, title, content })
      }

      if (group.matches.length) groups.push(group)
    }

    return groups
      .sort((a, b) => {
        if (pageTitleMatches[a._page_rk] !== pageTitleMatches[b._page_rk]) {
          return pageTitleMatches[b._page_rk] - pageTitleMatches[a._page_rk]
        }
        return a._page_rk - b._page_rk
      })
      .map(({ _page_rk: _, ...group }) => group)
  }

  // Assigns the unique option ids the dialog uses for keyboard navigation.
  function withOptionIds(groups) {
    let optionId = 0
    return groups.map((group) => ({
      id: `hextra-search-opt-${optionId++}`,
      route: group.route,
      title: group.title,
      breadcrumbs: group.breadcrumbs,
      matches: group.matches.map((match) => ({
        id: `hextra-search-opt-${optionId++}`,
        route: match.route,
        title: match.title,
        content: match.content,
      })),
    }))
  }

  window.hextraSearch = {
    preload: preloadIndex,
    async search(query) {
      await preloadIndex()
      const maxPageResults = parseInt("{{- site.Params.search.flexsearch.maxPageResults | default 20 -}}", 10)
      const keywordGroups = performKeywordSearch(query, maxPageResults)
      return withOptionIds(
        getHybridDisplayResults(keywordGroups, query, semanticLoader, maxPageResults, pageBreadcrumbs),
      )
    },
  }
})()
