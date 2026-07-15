from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


class SearchDataStructureError(ValueError):
    """Raised when Hugo search data violates the expected JSON contract."""


@dataclass(frozen=True, slots=True)
class SearchRecord:
    route: str
    heading_key: str
    title: str
    fragment_text: str


@dataclass(frozen=True, slots=True)
class SearchCorpus:
    records: tuple[SearchRecord, ...]
    titles_by_route: Mapping[str, str]


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SearchDataStructureError(f"{location} must be a JSON object with string keys")
    return value


def load_search_corpus(path: Path | str) -> SearchCorpus:
    """Load searchable records and the complete validated route/title hierarchy."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SearchDataStructureError(f"cannot read valid JSON from {source}: {exc}") from exc

    corpus = _object(raw, "search corpus")
    records: list[SearchRecord] = []
    titles_by_route: dict[str, str] = {}
    for route, raw_page in corpus.items():
        if not route.startswith("/"):
            raise SearchDataStructureError(f"route {route!r} must begin with '/'")
        page = _object(raw_page, f"page {route!r}")
        if set(page) != {"title", "data"}:
            raise SearchDataStructureError(f"page {route!r} must contain exactly 'title' and 'data'")
        title = page["title"]
        if not isinstance(title, str):
            raise SearchDataStructureError(f"page {route!r} title must be a string")
        titles_by_route[route] = title
        data = _object(page["data"], f"page {route!r} data")
        for heading_key, fragment_text in data.items():
            if not isinstance(fragment_text, str):
                raise SearchDataStructureError(f"page {route!r} section {heading_key!r} content must be a string")
            records.append(SearchRecord(route, heading_key, title, fragment_text))
    return SearchCorpus(tuple(records), titles_by_route)


def load_search_data(path: Path | str) -> list[SearchRecord]:
    """Load and strictly validate Hextra's generated searchable records."""
    return list(load_search_corpus(path).records)
