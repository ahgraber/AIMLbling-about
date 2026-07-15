from __future__ import annotations

import json
from pathlib import Path

import pytest

import numpy as np


def test_lexical_keyword_reference_is_stable_and_page_deduplicated() -> None:
    from site_search.eval import KeywordDocument, rank_keyword_pages

    documents = (
        KeywordDocument("/first", "/first#one", "shared exact term"),
        KeywordDocument("/first", "/first#two", "shared exact term"),
        KeywordDocument("/second", "/second", "shared exact term"),
    )

    assert rank_keyword_pages("shared", documents) == ["/first", "/second"]
    assert rank_keyword_pages("absent", documents) == []


def test_recall_metrics_and_report_are_computed_per_subset() -> None:
    from site_search.eval import QueryLabel, format_report, summarize_rankings

    labels = (
        QueryLabel("exact", "exact-term", "/a", "/a#section", "exact rationale"),
        QueryLabel("meaning", "paraphrase", "/b", None, "paraphrase rationale"),
        QueryLabel("topic", "navigational", "/c", "/c", "navigation rationale"),
    )
    rankings = {
        "keyword-only": {"exact": ["/a"], "meaning": ["/x", "/b"], "topic": []},
        "semantic-only": {"exact": ["/x", "/a"], "meaning": ["/b"], "topic": ["/c"]},
        "hybrid": {"exact": ["/a"], "meaning": ["/x", "/b"], "topic": ["/x", "/y", "/c"]},
    }

    report = summarize_rankings(labels, rankings)

    assert report.metrics["keyword-only"]["exact-term"].recall_at_1 == 1.0
    assert report.metrics["semantic-only"]["exact-term"].recall_at_1 == 0.0
    assert report.metrics["semantic-only"]["exact-term"].recall_at_3 == 1.0
    assert report.metrics["hybrid"]["navigational"].recall_at_1 == 0.0
    assert report.metrics["hybrid"]["navigational"].recall_at_3 == 1.0
    assert format_report(report) == (
        "condition     subset       queries  recall@1  recall@3\n"
        "keyword-only  exact-term          1    100.0%    100.0%\n"
        "keyword-only  paraphrase          1      0.0%    100.0%\n"
        "keyword-only  navigational        1      0.0%      0.0%\n"
        "semantic-only exact-term          1      0.0%    100.0%\n"
        "semantic-only paraphrase          1    100.0%    100.0%\n"
        "semantic-only navigational        1    100.0%    100.0%\n"
        "hybrid        exact-term          1    100.0%    100.0%\n"
        "hybrid        paraphrase          1      0.0%    100.0%\n"
        "hybrid        navigational        1      0.0%    100.0%"
    )


def test_missing_label_page_has_query_and_page_diagnostic() -> None:
    from site_search.eval import EvaluationDataError, QueryLabel, validate_label_pages

    labels = (QueryLabel("find the missing essay", "navigational", "/missing", None, "fixture"),)

    with pytest.raises(EvaluationDataError, match="find the missing essay.*?/missing"):
        validate_label_pages(labels, {"/present"})


def test_node_bridge_executes_shipped_semantic_rollup_and_hybrid_fusion(tmp_path: Path) -> None:
    from site_search.eval import run_client_rankings

    tokenizer_config = {
        "version": 1,
        "add_special_tokens": False,
        "drop_unknown": True,
        "unknown_token_id": 0,
        "tokenizer": {
            "normalizer": {
                "type": "BertNormalizer",
                "clean_text": True,
                "handle_chinese_chars": True,
                "lowercase": True,
                "strip_accents": True,
            },
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "model": {
                "type": "WordPiece",
                "vocab": {"[UNK]": 0, "alpha": 1, "beta": 2, "both": 3},
            },
        },
    }
    manifest = [
        {"chunk_id": "a", "page_id": "/a", "url": "/a", "title": "A", "heading": "A", "crumb": "A"},
        {"chunk_id": "b", "page_id": "/b", "url": "/b", "title": "B", "heading": "B", "crumb": "B"},
    ]
    (tmp_path / "token-table.bin").write_bytes(
        np.asarray([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.int8).tobytes()
    )
    (tmp_path / "token-scales.bin").write_bytes(np.asarray([0, 0.1, 0.1, 0.1], dtype="<f4").tobytes())
    (tmp_path / "doc-vectors.bin").write_bytes(np.asarray([[100, 0], [0, 100]], dtype=np.int8).tobytes())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "tokenizer-config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (tmp_path / "meta.json").write_text(
        json.dumps({"version": 1, "dimensions": 2, "document_global_scale": 0.01, "chunk_count": 2}),
        encoding="utf-8",
    )

    rankings = run_client_rankings(
        ("alpha", "both"),
        {"alpha": ["/a", "/b"], "both": ["/b", "/a"]},
        tmp_path,
    )

    assert rankings["semantic-only"]["alpha"] == ["/a", "/b"]
    assert rankings["semantic-only"]["both"] == ["/a", "/b"]
    assert rankings["hybrid"]["alpha"][0] == "/a"
    assert rankings["hybrid"]["both"] == ["/a", "/b"]
