from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

import numpy as np

from site_search.embedding import StaticEmbeddingModel, load_embedding_model
from site_search.export import Chunk, export_artifacts, tokenize_from_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "site-search" / "tests" / "js" / "semantic-search-parity.js"
FIDELITY_TOLERANCE = 0.999
PARITY_CASES = (
    "semantic search",
    "café naïve",
    "StaticModel HTTPResponse tokenizer_config",
    "semantic xyzzyplugh☃",
    "☃☃☃",
    # Control/whitespace edge cases: vertical tab and form feed are control
    # chars BERT removes (not spacing); NBSP is a Zs separator it collapses to a
    # space; tab/newline are BERT whitespace. Guards the clean_text ordering.
    "vertical\x0btab and\x0cform feed",
    "non\u00a0breaking\tspace\nlines",
    # CJK: BERT splits each CJK character into its own token (handle_chinese_chars);
    # the model's vocabulary has none, so both sides must drop them identically.
    # The real corpus quotes one ("Real" (\u5b9e), treadmill 2026-07-21).
    "comes down to a single word: Real (\u5b9e)",
    "\u5b9e\u65f6 semantic search",
)


@pytest.fixture(scope="module")
def parity_result(
    tmp_path_factory: pytest.TempPathFactory, potion_model_path: Path
) -> tuple[StaticEmbeddingModel, dict[str, Any], subprocess.CompletedProcess[str]]:
    model = load_embedding_model(source=potion_model_path, dimensions=128)
    artifact_dir = tmp_path_factory.mktemp("parity-artifacts")
    export_artifacts(
        [Chunk("probe", "/probe", "/probe", "Probe", "Probe", "semantic search")],
        model,
        artifact_dir,
    )
    tokenizer_config = json.loads((artifact_dir / "tokenizer-config.json").read_text(encoding="utf-8"))
    request = {
        "texts": list(PARITY_CASES),
        "dimensions": 128,
        "tokenizer_config": tokenizer_config,
        "token_table_base64": base64.b64encode((artifact_dir / "token-table.bin").read_bytes()).decode("ascii"),
        "token_scales_base64": base64.b64encode((artifact_dir / "token-scales.bin").read_bytes()).decode("ascii"),
    }
    node = shutil.which("node")
    assert node is not None, "Node is required for client parity tests"
    completed = subprocess.run(  # noqa: S603 -- fixed Node executable and runner with JSON stdin; no shell.
        [node, str(RUNNER)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    return model, tokenizer_config, completed


def _bridge_output(
    parity_result: tuple[StaticEmbeddingModel, dict[str, Any], subprocess.CompletedProcess[str]],
) -> tuple[StaticEmbeddingModel, dict[str, Any]]:
    model, _, completed = parity_result
    assert completed.returncode == 0, (
        f"Node parity bridge failed with exit {completed.returncode}: stderr={completed.stderr!r} stdout={completed.stdout!r}"
    )
    return model, json.loads(completed.stdout)


def test_token_sequences_match_exported_config_and_python_reference(
    parity_result: tuple[StaticEmbeddingModel, dict[str, Any], subprocess.CompletedProcess[str]],
) -> None:
    model, javascript = _bridge_output(parity_result)
    config = parity_result[1]
    results = javascript["results"]

    for text, result in zip(PARITY_CASES, results, strict=True):
        python_ids = model.tokenize([text])[0]
        exported_ids = tokenize_from_config(text, config)
        javascript_ids = result["token_ids"]
        assert exported_ids == python_ids, (
            f"exported tokenizer mismatch for {text!r}: python={python_ids!r}, exported={exported_ids!r}"
        )
        assert javascript_ids == python_ids, (
            f"token mismatch for {text!r}: python={python_ids!r}, javascript={javascript_ids!r}"
        )


def test_embedding_vectors(
    parity_result: tuple[StaticEmbeddingModel, dict[str, Any], subprocess.CompletedProcess[str]],
) -> None:
    model, javascript = _bridge_output(parity_result)
    python_vectors = model.embed(PARITY_CASES)

    for text, python_vector, result in zip(PARITY_CASES, python_vectors, javascript["results"], strict=True):
        javascript_vector = result["vector"]
        if not result["token_ids"]:
            mismatch = (
                f"OOV vector mismatch for {text!r}: python={python_vector.tolist()!r}, "
                f"javascript={javascript_vector!r}"
            )
            assert javascript_vector is None, mismatch
            assert np.count_nonzero(python_vector) == 0, mismatch
            continue
        client_vector = np.asarray(javascript_vector, dtype=np.float32)
        cosine = float(np.dot(python_vector, client_vector))
        assert cosine >= FIDELITY_TOLERANCE, (
            f"embedding mismatch for {text!r}: cosine={cosine}, python={python_vector.tolist()!r}, "
            f"javascript={javascript_vector!r}"
        )
