from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class SearchDataStructureError(ValueError):
    """Raised when Hugo search data violates the expected JSON contract."""


@dataclass(frozen=True, slots=True)
class SearchRecord:
    route: str
    heading_key: str
    title: str
    fragment_text: str


def section_location(record: SearchRecord) -> tuple[str, str, str]:
    """Return a record's ``(page_id, url, heading)``.

    ``page_id`` is the route without its trailing slash (``/`` stays ``/``), the
    key the client's ``canonicalizeRoute`` also produces; ``url`` adds the
    section anchor when there is one; ``heading`` falls back to the page title.
    The build manifest and the eval's keyword documents both use this, so their
    page ids join by equality.
    """
    anchor, _, heading = record.heading_key.partition("#")
    page_id = record.route if record.route == "/" else record.route.rstrip("/")
    url = f"{page_id}#{anchor}" if anchor else page_id
    return page_id, url, heading or record.title


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SearchDataStructureError(f"{location} must be a JSON object with string keys")
    return value


def load_search_data(path: Path | str) -> list[SearchRecord]:
    """Load and strictly validate Hextra's generated searchable records."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SearchDataStructureError(f"cannot read valid JSON from {source}: {exc}") from exc

    corpus = _object(raw, "search corpus")
    records: list[SearchRecord] = []
    for route, raw_page in corpus.items():
        if not route.startswith("/"):
            raise SearchDataStructureError(f"route {route!r} must begin with '/'")
        page = _object(raw_page, f"page {route!r}")
        if set(page) != {"title", "data"}:
            raise SearchDataStructureError(f"page {route!r} must contain exactly 'title' and 'data'")
        title = page["title"]
        if not isinstance(title, str):
            raise SearchDataStructureError(f"page {route!r} title must be a string")
        data = _object(page["data"], f"page {route!r} data")
        for heading_key, fragment_text in data.items():
            if not isinstance(fragment_text, str):
                raise SearchDataStructureError(f"page {route!r} section {heading_key!r} content must be a string")
            records.append(SearchRecord(route, heading_key, title, fragment_text))
    return records
