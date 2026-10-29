from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from site_search.loader import SearchDataStructureError, SearchRecord, load_search_data

if TYPE_CHECKING:
    from collections.abc import Sequence

# The gate catches a mass loss of section text, like the theme bug that left 75%
# of sections empty. Some sections are empty by design (a heading followed
# directly by a sub-heading); a healthy full build sits near 20% empty, mostly
# treadmill category headings. 35% sits between the two with room on each side.
# Raise it if the content format adds many more text-less headings.
DEFAULT_EMPTY_RATE_LIMIT = 0.35


@dataclass(frozen=True, slots=True)
class GateResult:
    records: tuple[SearchRecord, ...]
    empty_rate_limit: float = DEFAULT_EMPTY_RATE_LIMIT

    @property
    def empty_count(self) -> int:
        """Return the number of sections with no body text."""
        return sum(_is_empty(record) for record in self.records)

    @property
    def empty_rate(self) -> float:
        """Return empty sections as a fraction of all sections."""
        return self.empty_count / len(self.records) if self.records else 0.0

    @property
    def passed(self) -> bool:
        """Report whether the corpus is non-empty and within the empty-rate limit."""
        return bool(self.records) and self.empty_rate <= self.empty_rate_limit


def _is_empty(record: SearchRecord) -> bool:
    return not record.fragment_text.strip()


def _source(route: str) -> str:
    if route.startswith("/treadmill/"):
        return "treadmill"
    return "blog"


def evaluate_corpus(
    records: Sequence[SearchRecord], *, empty_rate_limit: float = DEFAULT_EMPTY_RATE_LIMIT
) -> GateResult:
    """Apply the empty-rate limit to validated records."""
    if not 0.0 <= empty_rate_limit <= 1.0:
        raise ValueError("empty_rate_limit must be between 0 and 1")
    return GateResult(tuple(records), empty_rate_limit)


def render_report(result: GateResult) -> str:
    """Render section and empty counts per content source, the overall rate, and the verdict."""
    totals = Counter(_source(record.route) for record in result.records)
    empties = Counter(_source(record.route) for record in result.records if _is_empty(record))
    lines = [f"{source}: total={totals[source]} empty={empties[source]}" for source in ("blog", "treadmill")]
    lines.append(
        f"total: total={len(result.records)} empty={result.empty_count} "
        f"empty_rate={result.empty_rate:.3%} limit={result.empty_rate_limit:.3%}"
    )
    lines.append("gate: PASS" if result.passed else "gate: FAIL")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the corpus gate command-line interface."""
    parser = argparse.ArgumentParser(description="Validate Hugo search corpus completeness")
    parser.add_argument("path", type=Path)
    parser.add_argument("--empty-rate-limit", type=float, default=DEFAULT_EMPTY_RATE_LIMIT)
    args = parser.parse_args(argv)
    try:
        result = evaluate_corpus(load_search_data(args.path), empty_rate_limit=args.empty_rate_limit)
    except (SearchDataStructureError, ValueError) as exc:
        parser.exit(2, f"structural error: {exc}\n")
    print(render_report(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
