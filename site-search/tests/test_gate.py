from __future__ import annotations

from pathlib import Path

import pytest

from site_search.gate import EmptyKind, evaluate_corpus, render_report
from site_search.loader import SearchDataStructureError, SearchRecord, load_search_data

FIXTURE = Path(__file__).parent / "fixtures" / "search-data.sample.json"
MALFORMED_FIXTURE = Path(__file__).parent / "fixtures" / "search-data.malformed.json"
REAL_CORPUS = Path(__file__).resolve().parents[2] / "hugo" / "public" / "en.search-data.json"


def test_section_content_completeness() -> None:
    records = load_search_data(FIXTURE)
    by_section = {(record.route, record.heading_key): record for record in records}

    assert by_section[("/blog/example/", "plain-heading#Plain heading")].fragment_text == "Plain body text."
    assert by_section[("/blog/example/", "heading-2#[1234.56789] Linked heading")].fragment_text == "Linked body text."
    assert by_section[("/blog/example/", "heading-3#Title only")].fragment_text == ""
    assert by_section[("/blog/example/", "")].fragment_text == "Introductory text."

    result = evaluate_corpus(records)
    assert result.kind_for("/blog/example/", "heading-3#Title only") is EmptyKind.LEGITIMATE


def test_markdown_link_headings_carry_body_in_real_corpus() -> None:
    """Guard the fragments.html attribution against the corpus-emptying regression.

    The hand-authored fixture in test_section_content_completeness assumes correct
    upstream attribution; it cannot catch a theme-template regression. This checks
    the *actual* rendered corpus: arxiv-style markdown-link headings (`[id] title`,
    the treadmill's heaviest structure) must carry their body text. The pre-fix
    bug emptied ~75% of these sections, so a low empty-rate ceiling trips on a
    regression without pinning to volatile treadmill page URLs. Skips on a fresh
    checkout / CI without a Hugo build.
    """
    if not REAL_CORPUS.exists():
        pytest.skip("rendered corpus required; run `just hugo search-index` (or `hugo -s hugo`) first")

    records = load_search_data(REAL_CORPUS)
    link_headings = [
        record
        for record in records
        if "#" in record.heading_key and record.heading_key.split("#", 1)[1].startswith("[")
    ]
    assert link_headings, "expected arxiv-style markdown-link headings in the treadmill corpus"

    empty = [record for record in link_headings if not record.fragment_text.strip()]
    empty_rate = len(empty) / len(link_headings)
    assert empty_rate < 0.15, (
        f"{empty_rate:.1%} of markdown-link headings are empty ({len(empty)}/{len(link_headings)}) — "
        "possible fragments.html heading-link attribution regression"
    )


def test_gate_threshold() -> None:
    # The tiny fixture is a classification corpus, not a rate corpus (2/6 empties
    # are legitimate), so the legitimate ceiling is relaxed to isolate the
    # suspicious-rate behavior under test.
    healthy = evaluate_corpus(load_search_data(FIXTURE), suspicious_rate_limit=0.5, legitimate_rate_limit=0.5)
    regressed = evaluate_corpus(load_search_data(FIXTURE), suspicious_rate_limit=0.1, legitimate_rate_limit=0.5)

    assert healthy.passed
    assert not regressed.passed
    assert "named-anchor#Unverified empty" in render_report(regressed)


def test_legitimate_empty_spike_fails_gate() -> None:
    # A theme/format regression that re-empties link-heading sections presents as
    # legitimately-classified `heading-N` title-only entries. None are suspicious,
    # but the legitimate-empty rate ceiling still trips the gate — closing the
    # blind spot where a mass re-emptying could hide inside the legitimate class.
    records = [SearchRecord("/treadmill/x/", f"heading-{i}#[{i}] Entry", "Roundup", "") for i in range(9)]
    records.append(SearchRecord("/treadmill/x/", "heading-9#[9] Kept", "Roundup", "Body text."))

    result = evaluate_corpus(records)

    assert result.suspicious_count == 0
    assert result.legitimate_rate > result.legitimate_rate_limit
    assert not result.passed
    assert "gate: FAIL" in render_report(result)


def test_whitespace_only_named_section_is_suspicious() -> None:
    result = evaluate_corpus([SearchRecord("/blog/example/", "named#Named", "Example", " \n\t")])

    assert result.kind_for("/blog/example/", "named#Named") is EmptyKind.SUSPICIOUS
    assert not result.passed


def test_format_canary() -> None:
    with pytest.raises(SearchDataStructureError, match="data.*object"):
        load_search_data(MALFORMED_FIXTURE)


# FormatCanary is a universal claim: *any* corpus deviating from the expected
# shape must fail with a structural diagnostic before classification/embedding.
# The fixture above samples one branch (non-object `data`); these parametrized
# cases exercise every other structural guard in the loader so a theme-format
# change that mutates any level of the contract trips the canary, not a silent
# misparse downstream.
@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("not json {", r"cannot read valid JSON"),
        ("[]", r"search corpus must be a JSON object"),
        ('{"bad-route":{"title":"T","data":{}}}', r"must begin with '/'"),
        ('{"/p/":"not-a-page"}', r"page '/p/' must be a JSON object"),
        ('{"/p/":{"title":"T","data":{},"extra":1}}', r"exactly 'title' and 'data'"),
        ('{"/p/":{"title":"T"}}', r"exactly 'title' and 'data'"),
        ('{"/p/":{"title":5,"data":{}}}', r"title must be a string"),
        ('{"/p/":{"title":"T","data":{"h#H":5}}}', r"content must be a string"),
    ],
)
def test_format_canary_covers_every_structural_branch(tmp_path: Path, payload: str, match: str) -> None:
    corpus = tmp_path / "malformed.json"
    corpus.write_text(payload, encoding="utf-8")

    with pytest.raises(SearchDataStructureError, match=match):
        load_search_data(corpus)


def test_loader_accepts_route_without_trailing_slash(tmp_path: Path) -> None:
    corpus = tmp_path / "route-without-trailing-slash.json"
    corpus.write_text('{"/glossary":{"title":"Glossary","data":{"":"Terms"}}}', encoding="utf-8")

    records = load_search_data(corpus)

    assert records[0].route == "/glossary"


def test_classification_report() -> None:
    report = render_report(evaluate_corpus(load_search_data(FIXTURE), suspicious_rate_limit=0.5))

    assert "blog: total=4 legitimate_empty=1 suspicious_empty=0" in report
    assert "treadmill: total=2 legitimate_empty=1 suspicious_empty=1" in report
    assert "total: total=6 legitimate_empty=2 suspicious_empty=1" in report
