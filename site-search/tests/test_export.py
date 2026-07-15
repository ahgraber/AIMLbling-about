from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model2vec import StaticModel

import numpy as np

from site_search.embedding import StaticEmbeddingModel
from site_search.export import Chunk, export_artifacts


def _real_model(path: Path) -> StaticEmbeddingModel:
    return StaticEmbeddingModel(StaticModel.from_pretrained(path, dimensionality=128), model_id="potion")


def _chunks() -> list[Chunk]:
    return [
        Chunk("first", "/first", "/first/", "First", "Intro", "Root > First", "semantic search"),
        Chunk("second", "/second", "/second/", "Second", "Details", "Root > Second", "exact library term"),
    ]


def test_artifact_set_complete(tmp_path: Path, potion_model_path: Path) -> None:
    out = tmp_path / "search"

    result = export_artifacts(_chunks(), _real_model(potion_model_path), out, built_at="2026-07-12T00:00:00Z")

    assert {path.name for path in out.iterdir()} == {
        "token-table.bin",
        "token-scales.bin",
        "doc-vectors.bin",
        "manifest.json",
        "tokenizer-config.json",
        "meta.json",
    }
    manifest = json.loads((out / "manifest.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    assert manifest == [
        {
            "chunk_id": "first",
            "page_id": "/first",
            "url": "/first/",
            "title": "First",
            "heading": "Intro",
            "crumb": "Root > First",
        },
        {
            "chunk_id": "second",
            "page_id": "/second",
            "url": "/second/",
            "title": "Second",
            "heading": "Details",
            "crumb": "Root > Second",
        },
    ]
    assert (out / "doc-vectors.bin").stat().st_size == len(manifest) * meta["dimensions"]
    assert result.manifest_count == len(manifest)
    assert meta["total_payload_size"] == sum(path.stat().st_size for path in out.iterdir())


def test_correspondence_after_exclusion(tmp_path: Path, potion_model_path: Path) -> None:
    chunks = [
        *_chunks()[:1],
        Chunk("empty", "/empty", "/empty/", "Empty", "", "Empty", "☃☃☃"),
        *_chunks()[1:],
    ]
    out = tmp_path / "search"

    result = export_artifacts(chunks, _real_model(potion_model_path), out)

    manifest = json.loads((out / "manifest.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    assert [entry["chunk_id"] for entry in manifest] == ["first", "second"]
    assert result.excluded_chunk_ids == ("empty",)
    assert (out / "doc-vectors.bin").stat().st_size == len(manifest) * meta["dimensions"]


def test_empty_chunk_exclusion(tmp_path: Path, potion_model_path: Path) -> None:
    out = tmp_path / "search"
    chunks = [Chunk("empty", "/empty", "/empty/", "Empty", "", "Empty", "☃☃☃"), *_chunks()]

    result = export_artifacts(chunks, _real_model(potion_model_path), out)

    meta = json.loads((out / "meta.json").read_text())
    values = np.fromfile(out / "doc-vectors.bin", dtype=np.int8).reshape(-1, meta["dimensions"])
    assert result.excluded_chunk_ids == ("empty",)
    assert np.all(np.linalg.norm(values.astype(np.float32), axis=1) > 0)
    assert np.isfinite(values).all()


def test_failed_export_preserves_existing_set(tmp_path: Path, potion_model_path: Path) -> None:
    out = tmp_path / "search"
    out.mkdir()
    (out / "meta.json").write_text('{"complete":true}')
    invalid = Chunk("bad", "/bad", "/bad/", "Bad", "", "Bad", "semantic search")
    invalid_metadata: dict[str, Any] = {"not_json": object()}

    try:
        export_artifacts([invalid], _real_model(potion_model_path), out, corpus_stats=invalid_metadata)
    except TypeError:
        pass
    else:
        raise AssertionError("invalid metadata unexpectedly exported")

    assert [path.name for path in out.iterdir()] == ["meta.json"]
    assert (out / "meta.json").read_text() == '{"complete":true}'
