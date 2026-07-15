from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from site_search.eval import evaluate, keyword_documents_from_corpus, load_query_labels

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "hugo" / "public" / "en.search-data.json"
ARTIFACTS = REPO / "hugo" / "assets" / "search"
QUERIES = Path(__file__).parent / "fixtures" / "eval-queries.json"
_REQUIRED_ARTIFACTS = (
    "meta.json",
    "manifest.json",
    "doc-vectors.bin",
    "token-table.bin",
    "token-scales.bin",
    "tokenizer-config.json",
)


def test_live_eval_meets_retrieval_floors() -> None:
    """Re-run the eval against the shipped artifacts and enforce the floors live.

    Complements the frozen-evidence guard in test_eval_floors: this exercises the
    real embedding, scoring, and fusion (the shipped semantic-search.js via Node)
    end-to-end, so a regression in the pipeline fails here rather than only when
    someone regenerates the recorded numbers. Skips when Node or the built
    artifacts/corpus are absent (e.g. a fresh checkout or CI without a Hugo build).
    """
    if shutil.which("node") is None:
        pytest.skip("Node is required to run the shipped client search logic")
    if not CORPUS.exists() or not all((ARTIFACTS / name).exists() for name in _REQUIRED_ARTIFACTS):
        pytest.skip("built search artifacts and rendered corpus are required; run `just hugo search-index` first")

    report = evaluate(load_query_labels(QUERIES), keyword_documents_from_corpus(CORPUS), ARTIFACTS)
    metrics = report.metrics

    hybrid_exact = metrics["hybrid"]["exact-term"].recall_at_1
    keyword_exact = metrics["keyword-only"]["exact-term"].recall_at_1
    hybrid_paraphrase = metrics["hybrid"]["paraphrase"].recall_at_1
    semantic_paraphrase = metrics["semantic-only"]["paraphrase"].recall_at_1

    assert hybrid_exact >= keyword_exact, f"exact-term hybrid {hybrid_exact:.1%} < keyword {keyword_exact:.1%}"
    assert (
        hybrid_paraphrase >= semantic_paraphrase
    ), f"paraphrase hybrid {hybrid_paraphrase:.1%} < semantic {semantic_paraphrase:.1%}"
    for condition in ("keyword-only", "semantic-only", "hybrid"):
        assert metrics[condition]["navigational"].queries == 10
