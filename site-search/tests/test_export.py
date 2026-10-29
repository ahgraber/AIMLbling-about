from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from model2vec import StaticModel
import pytest

import numpy as np

from site_search.embedding import StaticEmbeddingModel
from site_search.export import Chunk, export_artifacts

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt


def _real_model(path: Path) -> StaticEmbeddingModel:
    return StaticEmbeddingModel(StaticModel.from_pretrained(path, dimensionality=128), model_id="potion")


def _chunks() -> list[Chunk]:
    return [
        Chunk("first", "/first", "/first/", "First", "Intro", "semantic search"),
        Chunk("second", "/second", "/second/", "Second", "Details", "exact library term"),
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
        },
        {
            "chunk_id": "second",
            "page_id": "/second",
            "url": "/second/",
            "title": "Second",
            "heading": "Details",
        },
    ]
    assert (out / "doc-vectors.bin").stat().st_size == len(manifest) * meta["dimensions"]
    assert result.manifest_count == len(manifest)
    assert meta["total_payload_size"] == sum(path.stat().st_size for path in out.iterdir())


def test_correspondence_after_exclusion(tmp_path: Path, potion_model_path: Path) -> None:
    chunks = [
        *_chunks()[:1],
        Chunk("empty", "/empty", "/empty/", "Empty", "", "☃☃☃"),
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
    chunks = [Chunk("empty", "/empty", "/empty/", "Empty", "", "☃☃☃"), *_chunks()]

    result = export_artifacts(chunks, _real_model(potion_model_path), out)

    meta = json.loads((out / "meta.json").read_text())
    values = np.fromfile(out / "doc-vectors.bin", dtype=np.int8).reshape(-1, meta["dimensions"])
    assert result.excluded_chunk_ids == ("empty",)
    assert np.all(np.linalg.norm(values.astype(np.float32), axis=1) > 0)
    assert np.isfinite(values).all()


class _ShortEmbeddingModel(StaticEmbeddingModel):
    """Returns one vector fewer than the texts it embeds."""

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        return super().embed(texts)[:-1]


def test_export__provider_returns_fewer_vectors_than_chunks__rejects_and_preserves_existing_set(
    tmp_path: Path, potion_model_path: Path
) -> None:
    out = tmp_path / "search"
    out.mkdir()
    (out / "meta.json").write_text('{"complete":true}')
    short_model = _ShortEmbeddingModel(
        StaticModel.from_pretrained(potion_model_path, dimensionality=128), model_id="potion"
    )

    with pytest.raises(ValueError, match="one document vector per chunk"):
        export_artifacts(_chunks(), short_model, out)

    assert [path.name for path in out.iterdir()] == ["meta.json"]


def test_failed_export_preserves_existing_set(tmp_path: Path, potion_model_path: Path) -> None:
    out = tmp_path / "search"
    out.mkdir()
    (out / "meta.json").write_text('{"complete":true}')
    invalid = Chunk("bad", "/bad", "/bad/", "Bad", "", "semantic search")
    invalid_metadata: dict[str, Any] = {"not_json": object()}

    try:
        export_artifacts([invalid], _real_model(potion_model_path), out, corpus_stats=invalid_metadata)
    except TypeError:
        pass
    else:
        raise AssertionError("invalid metadata unexpectedly exported")

    assert [path.name for path in out.iterdir()] == ["meta.json"]
    assert (out / "meta.json").read_text() == '{"complete":true}'
