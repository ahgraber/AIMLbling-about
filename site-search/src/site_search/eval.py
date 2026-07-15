from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Literal

from site_search.loader import SearchDataStructureError, load_search_corpus

Subset = Literal["exact-term", "paraphrase", "navigational"]
Condition = Literal["keyword-only", "semantic-only", "hybrid"]
SUBSETS: tuple[Subset, ...] = ("exact-term", "paraphrase", "navigational")
CONDITIONS: tuple[Condition, ...] = ("keyword-only", "semantic-only", "hybrid")
KEYWORD_BASELINE = (
    "deterministic lexical reference: case-folded Unicode word tokens, summed query-term frequency, "
    "best section per page, input order for ties"
)
RUNNER = Path(__file__).with_name("eval_runner.js")


class EvaluationDataError(ValueError):
    """Raised when evaluation labels, documents, artifacts, or bridge output are invalid."""


@dataclass(frozen=True, slots=True)
class QueryLabel:
    query: str
    subset: Subset
    page_id: str
    section_url: str | None
    rationale: str
    source_text: str = ""


@dataclass(frozen=True, slots=True)
class KeywordDocument:
    page_id: str
    url: str
    text: str


@dataclass(frozen=True, slots=True)
class RecallMetric:
    queries: int
    recall_at_1: float
    recall_at_3: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    metrics: Mapping[Condition, Mapping[Subset, RecallMetric]]
    rankings: Mapping[Condition, Mapping[str, Sequence[str]]]
    keyword_baseline: str = KEYWORD_BASELINE


def _object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvaluationDataError(f"{location} must be an object with string keys")
    return value


def load_query_labels(path: Path) -> tuple[QueryLabel, ...]:
    """Load and strictly validate the labeled real-corpus query fixture."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"cannot read query labels from {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise EvaluationDataError("query labels must be an array")
    labels: list[QueryLabel] = []
    expected_keys = {
        "query",
        "subset",
        "expected_page_id",
        "expected_section_url",
        "rationale",
        "source_text",
    }
    for index, value in enumerate(raw):
        row = _object(value, f"query label {index}")
        if set(row) != expected_keys:
            raise EvaluationDataError(f"query label {index} must contain exactly {sorted(expected_keys)}")
        query = row["query"]
        subset = row["subset"]
        page_id = row["expected_page_id"]
        section_url = row["expected_section_url"]
        rationale = row["rationale"]
        source_text = row["source_text"]
        if (
            not isinstance(query, str)
            or not query
            or subset not in SUBSETS
            or not isinstance(page_id, str)
            or not page_id.startswith("/")
            or (section_url is not None and (not isinstance(section_url, str) or not section_url.startswith("/")))
            or not isinstance(rationale, str)
            or not rationale
            or not isinstance(source_text, str)
            or not source_text
        ):
            raise EvaluationDataError(f"query label {index} has invalid fields")
        labels.append(QueryLabel(query, subset, page_id, section_url, rationale, source_text))
    if len({label.query for label in labels}) != len(labels):
        raise EvaluationDataError("query labels must have unique query strings")
    return tuple(labels)


def _terms(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def rank_keyword_pages(query: str, documents: Sequence[KeywordDocument]) -> list[str]:
    """Rank the documented lexical reference baseline with stable input-order ties."""
    query_terms = Counter(_terms(query))
    if not query_terms:
        return []
    page_scores: dict[str, tuple[int, int]] = {}
    for index, document in enumerate(documents):
        document_terms = Counter(_terms(document.text))
        score = sum(count * document_terms[term] for term, count in query_terms.items())
        if score <= 0:
            continue
        existing = page_scores.get(document.page_id)
        if existing is None:
            page_scores[document.page_id] = (score, index)
        elif score > existing[0]:
            page_scores[document.page_id] = (score, existing[1])
    return [page_id for page_id, _ in sorted(page_scores.items(), key=lambda item: (-item[1][0], item[1][1]))]


def validate_label_pages(labels: Sequence[QueryLabel], available_pages: set[str]) -> None:
    """Fail with query-specific diagnostics when a label cannot be evaluated."""
    for label in labels:
        if label.page_id not in available_pages:
            raise EvaluationDataError(f"query {label.query!r} expects missing page {label.page_id!r}")


def _load_artifact_request(artifact_dir: Path) -> dict[str, object]:
    try:
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        tokenizer = json.loads((artifact_dir / "tokenizer-config.json").read_text(encoding="utf-8"))
        meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
        token_table = base64.b64encode((artifact_dir / "token-table.bin").read_bytes()).decode("ascii")
        token_scales = base64.b64encode((artifact_dir / "token-scales.bin").read_bytes()).decode("ascii")
        document_vectors = base64.b64encode((artifact_dir / "doc-vectors.bin").read_bytes()).decode("ascii")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"cannot read semantic artifacts from {artifact_dir}: {exc}") from exc
    meta = _object(meta, "semantic metadata")
    dimensions = meta.get("dimensions")
    global_scale = meta.get("document_global_scale")
    chunk_count = meta.get("chunk_count")
    if (
        not isinstance(manifest, list)
        or not isinstance(tokenizer, dict)
        or not isinstance(dimensions, int)
        or dimensions <= 0
        or not isinstance(global_scale, (int, float))
        or global_scale <= 0
        or chunk_count != len(manifest)
    ):
        raise EvaluationDataError("semantic artifact metadata and manifest are inconsistent")
    return {
        "dimensions": dimensions,
        "document_global_scale": global_scale,
        "manifest": manifest,
        "tokenizer_config": tokenizer,
        "token_table_base64": token_table,
        "token_scales_base64": token_scales,
        "document_vectors_base64": document_vectors,
    }


def run_client_rankings(
    queries: Sequence[str], keyword_rankings: Mapping[str, Sequence[str]], artifact_dir: Path
) -> dict[Condition, dict[str, list[str]]]:
    """Execute shipped client semantic scoring, rollup, and fusion through Node."""
    node = shutil.which("node")
    if node is None:
        raise EvaluationDataError("Node is required to execute shipped client search logic")
    request = _load_artifact_request(artifact_dir)
    request["queries"] = [
        {"query": query, "keyword_page_ids": list(keyword_rankings.get(query, ()))} for query in queries
    ]
    completed = subprocess.run(  # noqa: S603 -- absolute Node executable and fixed runner; JSON stdin, no shell.
        [node, str(RUNNER)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise EvaluationDataError(
            f"client evaluator failed with exit {completed.returncode}: {completed.stderr.strip() or 'no diagnostic'}"
        )
    try:
        output = _object(json.loads(completed.stdout), "client evaluator output")
    except (json.JSONDecodeError, EvaluationDataError) as exc:
        raise EvaluationDataError(f"client evaluator returned invalid JSON: {exc}") from exc
    results = output.get("results")
    if not isinstance(results, list) or len(results) != len(queries):
        raise EvaluationDataError("client evaluator returned the wrong result count")
    rankings: dict[Condition, dict[str, list[str]]] = {"semantic-only": {}, "hybrid": {}}
    for query, value in zip(queries, results, strict=True):
        row = _object(value, f"client result for {query!r}")
        if row.get("query") != query:
            raise EvaluationDataError(f"client result query mismatch for {query!r}")
        for condition in ("semantic-only", "hybrid"):
            page_ids = row.get(condition)
            if not isinstance(page_ids, list) or not all(isinstance(page_id, str) for page_id in page_ids):
                raise EvaluationDataError(f"client result {condition} ranking is invalid for {query!r}")
            rankings[condition][query] = page_ids
    return rankings


def summarize_rankings(
    labels: Sequence[QueryLabel], rankings: Mapping[str, Mapping[str, Sequence[str]]]
) -> EvaluationReport:
    """Compute page recall@1 and recall@3 for every condition/subset pair."""
    metrics: dict[Condition, dict[Subset, RecallMetric]] = {}
    for condition in CONDITIONS:
        condition_rankings = rankings.get(condition)
        if condition_rankings is None:
            raise EvaluationDataError(f"missing rankings for condition {condition!r}")
        metrics[condition] = {}
        for subset in SUBSETS:
            subset_labels = [label for label in labels if label.subset == subset]
            if not subset_labels:
                raise EvaluationDataError(f"no labels for subset {subset!r}")
            hits_at_1 = 0
            hits_at_3 = 0
            for label in subset_labels:
                if label.query not in condition_rankings:
                    raise EvaluationDataError(f"missing {condition} ranking for query {label.query!r}")
                ranked_pages = condition_rankings[label.query]
                hits_at_1 += label.page_id in ranked_pages[:1]
                hits_at_3 += label.page_id in ranked_pages[:3]
            count = len(subset_labels)
            metrics[condition][subset] = RecallMetric(count, hits_at_1 / count, hits_at_3 / count)
    typed_rankings: dict[Condition, Mapping[str, Sequence[str]]] = {
        condition: rankings[condition] for condition in CONDITIONS
    }
    return EvaluationReport(metrics, typed_rankings)


def format_report(report: EvaluationReport) -> str:
    """Render a deterministic compact metrics table."""
    lines = ["condition     subset       queries  recall@1  recall@3"]
    for condition in CONDITIONS:
        for subset in SUBSETS:
            metric = report.metrics[condition][subset]
            lines.append(
                f"{condition:<13} {subset:<13} {metric.queries:>7} {metric.recall_at_1:>9.1%} {metric.recall_at_3:>9.1%}"
            )
    return "\n".join(lines)


def keyword_documents_from_corpus(path: Path) -> tuple[KeywordDocument, ...]:
    """Build the lexical reference documents from the canonical validated corpus.

    Uses the single shared loader (the same parser the gate's FormatCanary relies
    on) so the eval and the build agree on the search-data contract rather than
    forking a second, divergent parser.
    """
    try:
        corpus = load_search_corpus(path)
    except SearchDataStructureError as exc:
        raise EvaluationDataError(str(exc)) from exc
    documents: list[KeywordDocument] = []
    for record in corpus.records:
        page_id = record.route if record.route == "/" else record.route.rstrip("/")
        anchor, _, heading = record.heading_key.partition("#")
        url = page_id + (f"#{anchor}" if anchor else "")
        documents.append(
            KeywordDocument(page_id, url, f"{record.title}\n{heading or record.title}\n{record.fragment_text}")
        )
    return tuple(documents)


def evaluate(
    labels: Sequence[QueryLabel], documents: Sequence[KeywordDocument], artifact_dir: Path
) -> EvaluationReport:
    """Evaluate lexical-reference, shipped semantic, and shipped hybrid conditions."""
    validate_label_pages(labels, {document.page_id for document in documents})
    keyword_rankings = {label.query: rank_keyword_pages(label.query, documents) for label in labels}
    client_rankings = run_client_rankings(tuple(label.query for label in labels), keyword_rankings, artifact_dir)
    return summarize_rankings(labels, {"keyword-only": keyword_rankings, **client_rankings})


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluator without applying release quality floors."""
    parser = argparse.ArgumentParser(description="Evaluate static site search ranking conditions")
    _ = parser.add_argument("--queries", required=True, type=Path)
    _ = parser.add_argument("--search-data", required=True, type=Path)
    _ = parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            load_query_labels(args.queries), keyword_documents_from_corpus(args.search_data), args.artifacts
        )
    except EvaluationDataError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    print(f"keyword baseline: {report.keyword_baseline}")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
