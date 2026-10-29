from __future__ import annotations

import pytest

from site_search.loader import SearchRecord, section_location


@pytest.mark.parametrize(
    ("route", "heading_key", "expected"),
    [
        ("/", "", ("/", "/", "Page")),
        ("/blog/post/", "", ("/blog/post", "/blog/post", "Page")),
        ("/blog/post/", "setup#Setup", ("/blog/post", "/blog/post#setup", "Setup")),
        (
            "/blog/post/",
            "heading-2#[1234.56789] Linked",
            ("/blog/post", "/blog/post#heading-2", "[1234.56789] Linked"),
        ),
        ("/blog/post/", "anchor#", ("/blog/post", "/blog/post#anchor", "Page")),
        ("/blog/post/", "#Title only", ("/blog/post", "/blog/post", "Title only")),
        ("/glossary", "term#Term", ("/glossary", "/glossary#term", "Term")),
    ],
)
def test_section_location(route: str, heading_key: str, expected: tuple[str, str, str]) -> None:
    assert section_location(SearchRecord(route, heading_key, "Page", "body")) == expected
