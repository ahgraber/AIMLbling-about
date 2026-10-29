from __future__ import annotations

from pathlib import Path

import pytest


def test_real_labeled_fixture_is_balanced_unique_and_resolves_to_corpus() -> None:
    from site_search.eval import load_query_labels
    from site_search.loader import load_search_data, section_location

    repo = Path(__file__).resolve().parents[2]
    corpus_path = repo / "hugo" / "public" / "en.search-data.json"
    if not corpus_path.exists():
        pytest.skip("rendered corpus is required; run `just hugo search-index` first")
    labels = load_query_labels(Path(__file__).parent / "fixtures" / "eval-queries.json")
    section_urls: dict[str, set[str]] = {}
    for record in load_search_data(corpus_path):
        page_id, url, _ = section_location(record)
        section_urls.setdefault(page_id, set()).add(url)

    assert len(labels) == 30
    assert len({label.query for label in labels}) == 30
    assert {
        subset: sum(label.subset == subset for label in labels) for subset in {label.subset for label in labels}
    } == {
        "exact-term": 10,
        "paraphrase": 10,
        "navigational": 10,
    }
    for label in labels:
        assert label.page_id in section_urls, f"{label.query!r} names absent page {label.page_id!r}"
        assert label.rationale
        assert label.source_text
        if label.section_url is None:
            continue
        route = label.section_url.split("#", 1)[0]
        assert route == label.page_id
        assert label.section_url in section_urls[label.page_id], (
            f"{label.query!r} names absent section {label.section_url!r}"
        )
