from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

from site_search.chunking import partition_body
from site_search.embedding import StaticEmbedding, load_embedding_model
from site_search.export import Chunk, export_artifacts
from site_search.gate import (
    DEFAULT_SUSPICIOUS_RATE_LIMIT,
    EmptyKind,
    GateResult,
    evaluate_corpus,
    render_report,
)
from site_search.loader import SearchDataStructureError, SearchRecord, load_search_corpus


class CorpusRejectedError(RuntimeError):
    """Raised when the completeness gate rejects a validated corpus."""


@dataclass(frozen=True, slots=True)
class BuildSummary:
    gate_result: GateResult
    chunk_count: int
    excluded_chunk_ids: tuple[str, ...]
    artifact_sizes: tuple[tuple[str, int], ...]
    total_payload_size: int


def _section_location(record: SearchRecord) -> tuple[str, str]:
    if "#" in record.heading_key:
        anchor, heading = record.heading_key.split("#", 1)
    else:
        anchor, heading = record.heading_key, ""
    base_url = record.route if record.route == "/" else record.route.rstrip("/")
    url = f"{base_url}#{anchor}" if anchor else base_url
    return url, heading or record.title


def _page_metadata(route: str, titles_by_route: Mapping[str, str]) -> tuple[str, str]:
    page_id = route if route == "/" else route.rstrip("/")
    route_parts = [part for part in route.split("/") if part and not part.startswith("#")]
    crumb = ""
    search_route = "/"
    for index, route_part in enumerate(route_parts):
        search_route += f"{route_part}/"
        title = titles_by_route.get(search_route)
        if title is None:
            continue
        if title == "_index":
            title = " ".join(route_part.split("-"))
        crumb += title
        if index < len(route_parts) - 1:
            crumb += " > "
    return page_id, crumb


def _chunks(
    records: Sequence[SearchRecord], model: StaticEmbedding, titles_by_route: Mapping[str, str]
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in records:
        url, heading = _section_location(record)
        page_id, crumb = _page_metadata(record.route, titles_by_route)
        context = record.title if heading == record.title else f"{record.title}\n{heading}"
        bodies = partition_body(
            record.fragment_text,
            context=context,
            count_tokens=lambda text: len(model.tokenize([text])[0]),
        )
        for index, body in enumerate(bodies):
            chunks.append(
                Chunk(
                    chunk_id=f"{record.route}|{record.heading_key}|{index}",
                    page_id=page_id,
                    url=url,
                    title=record.title,
                    heading=heading,
                    crumb=crumb,
                    text=f"{context}\n\n{body}",
                )
            )
    return chunks


def build_index(
    search_data: Path,
    output: Path,
    *,
    model_factory: Callable[[], StaticEmbedding] = load_embedding_model,
    suspicious_rate_limit: float = DEFAULT_SUSPICIOUS_RATE_LIMIT,
) -> BuildSummary:
    """Validate a Hugo corpus and publish a semantic artifact set."""
    corpus = load_search_corpus(search_data)
    records = corpus.records
    gate_result = evaluate_corpus(records, suspicious_rate_limit=suspicious_rate_limit)
    if not gate_result.passed:
        raise CorpusRejectedError(render_report(gate_result))

    model = model_factory()
    chunks = _chunks(records, model, corpus.titles_by_route)
    corpus_stats = {
        "section_count": len(records),
        "legitimate_empty_count": sum(item.kind is EmptyKind.LEGITIMATE for item in gate_result.records),
        "suspicious_empty_count": gate_result.suspicious_count,
    }
    export_result = export_artifacts(chunks, model, output, corpus_stats=corpus_stats)
    return BuildSummary(
        gate_result=gate_result,
        chunk_count=export_result.manifest_count,
        excluded_chunk_ids=export_result.excluded_chunk_ids,
        artifact_sizes=export_result.artifact_sizes,
        total_payload_size=export_result.total_payload_size,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    model_factory: Callable[[], StaticEmbedding] = load_embedding_model,
) -> int:
    """Run the semantic index builder command-line interface."""
    parser = argparse.ArgumentParser(description="Build static semantic-search artifacts")
    _ = parser.add_argument("--search-data", required=True, type=Path)
    _ = parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = build_index(args.search_data, args.out, model_factory=model_factory)
    except SearchDataStructureError as exc:
        print(f"structural error: {exc}", file=sys.stderr)
        return 2
    except CorpusRejectedError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(render_report(summary.gate_result))
    if summary.excluded_chunk_ids:
        print(f"excluded chunks ({len(summary.excluded_chunk_ids)}):")
        for chunk_id in summary.excluded_chunk_ids:
            print(f"- {chunk_id}")
    for name, size in summary.artifact_sizes:
        print(f"artifact-size name={name} bytes={size}")
    print(f"artifact-total bytes={summary.total_payload_size}")
    print(f"artifacts: chunks={summary.chunk_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
