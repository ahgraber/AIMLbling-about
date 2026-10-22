from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import TYPE_CHECKING

from site_search.loader import SearchDataStructureError, SearchRecord, load_search_data

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

DEFAULT_SUSPICIOUS_RATE_LIMIT = 0.01
# An empty section under a synthetic `heading-N` anchor is classified
# legitimate, but a link-heading section that a theme/format regression has
# stripped of its body looks the same in the flattened corpus. The
# suspicious-rate gate therefore cannot see a mass re-emptying of those
# sections (the 75%-empty failure this gate exists to catch). This ceiling is
# the tripwire for that class: a healthy corpus sits at a few percent
# legitimate-empty, mostly parent headings followed directly by a sub-heading,
# so a spike far above it signals a regression regardless of per-section
# classification.
DEFAULT_LEGITIMATE_RATE_LIMIT = 0.10
_SYNTHETIC_HEADING = re.compile(r"heading-\d+#")
# A heading as Hextra's render hook emits it (`data-hextra-search-id` holds the
# anchor), followed by only whitespace and then the next heading. The body never
# crosses the heading's own closing tag, and the lookahead leaves the next
# heading unconsumed so runs of consecutive headings all match.
_TITLE_ONLY_HEADING = re.compile(
    r'<h([1-6]) data-hextra-search-id="([^"]+)"(?:(?!</h\1>).)*</h\1>(?=\s*<h[1-6][\s>])',
    re.DOTALL,
)


class EmptyKind(str, Enum):
    NON_EMPTY = "non-empty"
    LEGITIMATE = "legitimate-empty"
    SUSPICIOUS = "suspicious-empty"


@dataclass(frozen=True, slots=True)
class ClassifiedRecord:
    record: SearchRecord
    kind: EmptyKind


@dataclass(frozen=True, slots=True)
class GateResult:
    records: tuple[ClassifiedRecord, ...]
    suspicious_rate_limit: float
    legitimate_rate_limit: float = DEFAULT_LEGITIMATE_RATE_LIMIT

    @property
    def suspicious_count(self) -> int:
        """Return the number of empty sections lacking a structural exemption."""
        return sum(item.kind is EmptyKind.SUSPICIOUS for item in self.records)

    @property
    def legitimate_count(self) -> int:
        """Return the number of empty sections with a structural exemption."""
        return sum(item.kind is EmptyKind.LEGITIMATE for item in self.records)

    @property
    def suspicious_rate(self) -> float:
        """Return suspicious empty sections as a fraction of all sections."""
        return self.suspicious_count / len(self.records) if self.records else 0.0

    @property
    def legitimate_rate(self) -> float:
        """Return legitimately-empty sections as a fraction of all sections."""
        return self.legitimate_count / len(self.records) if self.records else 0.0

    @property
    def passed(self) -> bool:
        """Report whether the corpus stays within both empty-rate ceilings.

        The legitimate-empty ceiling catches a mass re-emptying that would
        otherwise hide inside the legitimate class (see DEFAULT_LEGITIMATE_RATE_LIMIT).
        """
        return (
            bool(self.records)
            and self.suspicious_rate <= self.suspicious_rate_limit
            and self.legitimate_rate <= self.legitimate_rate_limit
        )

    def kind_for(self, route: str, heading_key: str) -> EmptyKind:
        """Return the classification for one uniquely identified section."""
        for item in self.records:
            if item.record.route == route and item.record.heading_key == heading_key:
                return item.kind
        raise KeyError((route, heading_key))


def _source(route: str) -> str:
    if route.startswith("/treadmill/"):
        return "treadmill"
    return "blog"


def _classify(record: SearchRecord, title_only_sections: Collection[tuple[str, str]]) -> EmptyKind:
    if record.fragment_text.strip():
        return EmptyKind.NON_EMPTY
    if record.heading_key == "" or record.heading_key.startswith("#"):
        return EmptyKind.LEGITIMATE
    if _SYNTHETIC_HEADING.fullmatch(record.heading_key.split("#", 1)[0] + "#"):
        return EmptyKind.LEGITIMATE
    if (record.route, record.heading_key) in title_only_sections:
        return EmptyKind.LEGITIMATE
    return EmptyKind.SUSPICIOUS


def find_title_only_sections(public_dir: Path, records: Sequence[SearchRecord]) -> frozenset[tuple[str, str]]:
    """Return the empty sections whose rendered heading is followed directly by another heading.

    Reads ``public_dir/<route>/index.html`` for each route that has otherwise
    suspicious empty sections. A route without that file contributes no
    exemptions, so its empty sections stay suspicious.

    Returns:
        ``(route, heading_key)`` pairs.
    """
    candidates: dict[str, list[SearchRecord]] = {}
    for record in records:
        if _classify(record, frozenset()) is EmptyKind.SUSPICIOUS:
            candidates.setdefault(record.route, []).append(record)

    found: set[tuple[str, str]] = set()
    for route, page_records in candidates.items():
        page = public_dir / route.strip("/") / "index.html"
        if not page.is_file():
            continue
        anchors = {match.group(2) for match in _TITLE_ONLY_HEADING.finditer(page.read_text(encoding="utf-8"))}
        found.update(
            (route, record.heading_key) for record in page_records if record.heading_key.split("#", 1)[0] in anchors
        )
    return frozenset(found)


def evaluate_corpus(
    records: Sequence[SearchRecord],
    *,
    title_only_sections: Collection[tuple[str, str]] = frozenset(),
    suspicious_rate_limit: float = DEFAULT_SUSPICIOUS_RATE_LIMIT,
    legitimate_rate_limit: float = DEFAULT_LEGITIMATE_RATE_LIMIT,
) -> GateResult:
    """Classify validated records and apply the suspicious- and legitimate-empty limits.

    ``title_only_sections`` lists ``(route, heading_key)`` pairs from
    `find_title_only_sections`; empty sections in it are legitimate.
    """
    if not 0.0 <= suspicious_rate_limit <= 1.0:
        raise ValueError("suspicious_rate_limit must be between 0 and 1")
    if not 0.0 <= legitimate_rate_limit <= 1.0:
        raise ValueError("legitimate_rate_limit must be between 0 and 1")
    return GateResult(
        tuple(ClassifiedRecord(record, _classify(record, title_only_sections)) for record in records),
        suspicious_rate_limit,
        legitimate_rate_limit,
    )


def render_report(result: GateResult) -> str:
    """Render stable human-readable totals and suspicious-section diagnostics."""
    lines: list[str] = []
    for source in ("blog", "treadmill"):
        items = [item for item in result.records if _source(item.record.route) == source]
        counts = Counter(item.kind for item in items)
        lines.append(
            f"{source}: total={len(items)} legitimate_empty={counts[EmptyKind.LEGITIMATE]} "
            f"suspicious_empty={counts[EmptyKind.SUSPICIOUS]}"
        )
    counts = Counter(item.kind for item in result.records)
    lines.append(
        f"total: total={len(result.records)} legitimate_empty={counts[EmptyKind.LEGITIMATE]} "
        f"suspicious_empty={counts[EmptyKind.SUSPICIOUS]} "
        f"suspicious_rate={result.suspicious_rate:.3%} limit={result.suspicious_rate_limit:.3%} "
        f"legitimate_rate={result.legitimate_rate:.3%} legit_limit={result.legitimate_rate_limit:.3%}"
    )
    lines.extend(
        f"suspicious: {item.record.route} {item.record.heading_key!r}"
        for item in result.records
        if item.kind is EmptyKind.SUSPICIOUS
    )
    lines.append("gate: PASS" if result.passed else "gate: FAIL")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the corpus gate command-line interface."""
    parser = argparse.ArgumentParser(description="Validate Hugo search corpus completeness")
    parser.add_argument("path", type=Path)
    parser.add_argument("--suspicious-rate-limit", type=float, default=DEFAULT_SUSPICIOUS_RATE_LIMIT)
    parser.add_argument("--legitimate-rate-limit", type=float, default=DEFAULT_LEGITIMATE_RATE_LIMIT)
    args = parser.parse_args(argv)
    try:
        records = load_search_data(args.path)
        result = evaluate_corpus(
            records,
            # Hugo writes the corpus at the root of the rendered site.
            title_only_sections=find_title_only_sections(args.path.parent, records),
            suspicious_rate_limit=args.suspicious_rate_limit,
            legitimate_rate_limit=args.legitimate_rate_limit,
        )
    except (SearchDataStructureError, ValueError) as exc:
        parser.exit(2, f"structural error: {exc}\n")
    print(render_report(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
