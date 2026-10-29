from __future__ import annotations

from pathlib import Path

import pytest

from site_search.gate import DEFAULT_EMPTY_RATE_LIMIT, evaluate_corpus, main as gate_main, render_report
from site_search.loader import SearchDataStructureError, SearchRecord, load_search_data

FIXTURE = Path(__file__).parent / "fixtures" / "search-data.sample.json"
MALFORMED_FIXTURE = Path(__file__).parent / "fixtures" / "search-data.malformed.json"
REAL_CORPUS = Path(__file__).resolve().parents[2] / "hugo" / "public" / "en.search-data.json"


def _corpus(empty: int, total: int, route: str = "/blog/p/") -> list[SearchRecord]:
    return [SearchRecord(route, f"s{i}#S{i}", "P", "" if i < empty else "body") for i in range(total)]


def test_section_content_completeness() -> None:
    records = load_search_data(FIXTURE)
    by_section = {(record.route, record.heading_key): record for record in records}

    assert by_section[("/blog/example/", "plain-heading#Plain heading")].fragment_text == "Plain body text."
    assert by_section[("/blog/example/", "heading-2#[1234.56789] Linked heading")].fragment_text == "Linked body text."
    assert by_section[("/blog/example/", "heading-3#Title only")].fragment_text == ""
    assert by_section[("/blog/example/", "")].fragment_text == "Introductory text."


def test_markdown_link_headings_carry_body_in_real_corpus() -> None:
    """Guard the theme's section attribution against the corpus-emptying regression.

    The hand-authored fixture in test_section_content_completeness assumes correct
    upstream attribution; it cannot catch a theme-template regression. This checks
    the *actual* rendered corpus: arxiv-style markdown-link headings (`[id] title`,
    the treadmill's heaviest structure) must carry their body text. A low
    empty-rate ceiling trips on a regression without pinning to volatile
    treadmill page URLs. Skips on a fresh checkout / CI without a Hugo build.
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
        "possible theme heading-link attribution regression"
    )


def test_gate_default_limit_is_35_percent() -> None:
    assert DEFAULT_EMPTY_RATE_LIMIT == 0.35


def test_gate_passes_at_the_limit() -> None:
    assert evaluate_corpus(_corpus(empty=35, total=100)).passed


def test_gate_fails_above_the_limit() -> None:
    result = evaluate_corpus(_corpus(empty=36, total=100))

    assert not result.passed
    assert result.empty_count == 36
    assert result.empty_rate == pytest.approx(0.36)


def test_whitespace_only_section_counts_as_empty() -> None:
    result = evaluate_corpus([SearchRecord("/blog/p/", "s#S", "P", " \n\t")])

    assert result.empty_count == 1
    assert not result.passed


def test_empty_corpus_fails() -> None:
    assert not evaluate_corpus([]).passed


def test_limit_must_be_a_fraction() -> None:
    with pytest.raises(ValueError, match="empty_rate_limit"):
        evaluate_corpus(_corpus(empty=0, total=1), empty_rate_limit=1.5)


def test_report_states_counts_per_source_and_the_verdict() -> None:
    report = render_report(evaluate_corpus(load_search_data(FIXTURE)))

    assert "blog: total=4 empty=1" in report
    assert "treadmill: total=2 empty=2" in report
    assert "total: total=6 empty=3 empty_rate=50.000% limit=35.000%" in report
    assert report.endswith("gate: FAIL")


def test_gate_cli_exit_code_follows_the_verdict(tmp_path: Path) -> None:
    corpus = tmp_path / "en.search-data.json"
    corpus.write_text('{"/blog/p/":{"title":"P","data":{"a#A":"","b#B":"Body"}}}', encoding="utf-8")

    assert gate_main([str(corpus)]) == 1
    assert gate_main([str(corpus), "--empty-rate-limit", "0.5"]) == 0


def test_format_canary() -> None:
    with pytest.raises(SearchDataStructureError, match=r"data.*object"):
        load_search_data(MALFORMED_FIXTURE)


# FormatCanary is a universal claim: *any* corpus deviating from the expected
# shape must fail with a structural diagnostic before the gate or embedding runs.
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
