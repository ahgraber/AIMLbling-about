from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import pytest

import numpy as np

from site_search.embedding import StaticEmbeddingModel, load_embedding_model
from site_search.export import Chunk, export_artifacts, tokenize_from_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "site-search" / "tests" / "js" / "semantic-search-parity.js"
CORPUS = REPO_ROOT / "hugo" / "public" / "en.search-data.json"
FIDELITY_TOLERANCE = 0.999
PARITY_CASES = (
    "semantic search",
    "café naïve",
    "StaticModel HTTPResponse tokenizer_config",
    "semantic xyzzyplugh☃",
    "☃☃☃",
)
CJK = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7a3]")


@pytest.fixture(scope="module")
def parity_result(
    tmp_path_factory: pytest.TempPathFactory, potion_model_path: Path
) -> tuple[StaticEmbeddingModel, dict[str, Any], subprocess.CompletedProcess[str]]:
    model = load_embedding_model(source=potion_model_path, dimensions=128)
    artifact_dir = tmp_path_factory.mktemp("parity-artifacts")
    export_artifacts(
        [Chunk("probe", "/probe", "/probe", "Probe", "Probe", "Probe", "semantic search")],
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
    assert (
        completed.returncode == 0
    ), f"Node parity bridge failed with exit {completed.returncode}: stderr={completed.stderr!r} stdout={completed.stdout!r}"
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
        assert (
            exported_ids == python_ids
        ), f"exported tokenizer mismatch for {text!r}: python={python_ids!r}, exported={exported_ids!r}"
        assert (
            javascript_ids == python_ids
        ), f"token mismatch for {text!r}: python={python_ids!r}, javascript={javascript_ids!r}"


def test_embedding_vectors(
    parity_result: tuple[StaticEmbeddingModel, dict[str, Any], subprocess.CompletedProcess[str]],
) -> None:
    model, javascript = _bridge_output(parity_result)
    python_vectors = model.embed(PARITY_CASES)

    for text, python_vector, result in zip(PARITY_CASES, python_vectors, javascript["results"], strict=True):
        javascript_vector = result["vector"]
        if not result["token_ids"]:
            assert javascript_vector is None and np.count_nonzero(python_vector) == 0, (
                f"OOV vector mismatch for {text!r}: python={python_vector.tolist()!r}, "
                f"javascript={javascript_vector!r}"
            )
            continue
        client_vector = np.asarray(javascript_vector, dtype=np.float32)
        cosine = float(np.dot(python_vector, client_vector))
        assert cosine >= FIDELITY_TOLERANCE, (
            f"embedding mismatch for {text!r}: cosine={cosine}, python={python_vector.tolist()!r}, "
            f"javascript={javascript_vector!r}"
        )


def test_real_corpus_has_no_cjk_case_and_keeps_cjk_parity_explicitly_na() -> None:
    if not CORPUS.exists():
        pytest.skip("rendered corpus is required; run `just hugo search-index` first")
    corpus_text = CORPUS.read_text(encoding="utf-8")

    assert (
        CJK.search(corpus_text) is None
    ), "CJK is now present in the real corpus; replace this N/A canary with a representative CJK parity case"
