from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# The frozen evaluation oracle lives with the test suite (not in the change's
# evidence/ directory) so the retrieval-floor guard survives the change being
# archived; the archived evidence copy remains the historical record.
RESULTS = Path(__file__).parent / "fixtures" / "eval-results.json"


def _measured_results() -> dict[str, Any]:
    assert RESULTS.exists(), f"missing durable evaluation results: {RESULTS}"
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_selected_candidate_meets_retrieval_quality_floors() -> None:
    measured_results = _measured_results()
    dimensions = measured_results["selected_dimensions"]
    assert dimensions == 256
    candidate = measured_results["candidates"][str(dimensions)]
    metrics = candidate["metrics"]
    hybrid_exact = metrics["hybrid"]["exact-term"]["recall_at_1"]
    keyword_exact = metrics["keyword-only"]["exact-term"]["recall_at_1"]
    hybrid_paraphrase = metrics["hybrid"]["paraphrase"]["recall_at_1"]
    semantic_paraphrase = metrics["semantic-only"]["paraphrase"]["recall_at_1"]

    failures = []
    if hybrid_exact < keyword_exact:
        failures.append(f"exact-term hybrid {hybrid_exact:.1%} < keyword {keyword_exact:.1%}")
    if hybrid_paraphrase < semantic_paraphrase:
        failures.append(f"paraphrase hybrid {hybrid_paraphrase:.1%} < semantic {semantic_paraphrase:.1%}")
    assert candidate["meets_floors"] is True
    assert not failures, f"{dimensions}-dimension retrieval floor failed: {'; '.join(failures)}"


def test_smaller_candidate_is_rejected_by_frozen_floors() -> None:
    candidate = _measured_results()["candidates"]["128"]

    assert candidate["meets_floors"] is False


@pytest.mark.parametrize("dimensions", ["128", "256"])
def test_navigational_metrics_exist_for_every_condition(
    dimensions: str,
) -> None:
    measured_results = _measured_results()
    metrics = measured_results["candidates"][dimensions]["metrics"]

    for condition in ("keyword-only", "semantic-only", "hybrid"):
        navigational = metrics[condition]["navigational"]
        assert navigational["queries"] == 10
        assert isinstance(navigational["recall_at_1"], float)
        assert isinstance(navigational["recall_at_3"], float)
