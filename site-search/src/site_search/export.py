from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

from tokenizers import Tokenizer

import numpy as np

from site_search.embedding import StaticEmbedding
from site_search.quantize import quantize_global, quantize_rows

ARTIFACT_NAMES = (
    "token-table.bin",
    "token-scales.bin",
    "doc-vectors.bin",
    "manifest.json",
    "tokenizer-config.json",
    "meta.json",
)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    page_id: str
    url: str
    title: str
    heading: str
    crumb: str
    text: str


@dataclass(frozen=True, slots=True)
class ExportResult:
    manifest_count: int
    excluded_chunk_ids: tuple[str, ...]
    artifact_sizes: tuple[tuple[str, int], ...]
    total_payload_size: int


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def resolve_tokenizer_config(tokenizer_json: str, unknown_token_id: int | None) -> dict[str, Any]:
    """Resolve ambiguous tokenizer behavior into an explicit client contract."""
    raw = json.loads(tokenizer_json)
    if not isinstance(raw, dict):
        raise TypeError("tokenizer JSON must be an object")
    normalizer = raw.get("normalizer")
    if not isinstance(normalizer, dict) or normalizer.get("type") != "BertNormalizer":
        raise ValueError("expected a BertNormalizer tokenizer")
    strip_accents = normalizer.get("strip_accents")
    if strip_accents is None:
        normalizer["strip_accents"] = bool(normalizer.get("lowercase"))
    elif not isinstance(strip_accents, bool):
        raise ValueError("strip_accents must be boolean or null")
    return {
        "version": 1,
        "tokenizer": raw,
        "unknown_token_id": unknown_token_id,
        "add_special_tokens": False,
        "drop_unknown": True,
    }


def tokenize_from_config(text: str, config: Mapping[str, Any]) -> list[int]:
    """Tokenize using only the resolved exported configuration."""
    tokenizer_data = config.get("tokenizer")
    if not isinstance(tokenizer_data, dict):
        raise TypeError("tokenizer config must contain a tokenizer object")
    tokenizer = Tokenizer.from_str(json.dumps(tokenizer_data))
    add_special_tokens = config.get("add_special_tokens")
    if not isinstance(add_special_tokens, bool):
        raise TypeError("add_special_tokens must be boolean")
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
    if config.get("drop_unknown") is True:
        unknown = config.get("unknown_token_id")
        if unknown is not None and not isinstance(unknown, int):
            raise ValueError("unknown_token_id must be an integer or null")
        ids = [token_id for token_id in ids if token_id != unknown]
    return ids


def _fixed_meta_bytes(base: dict[str, Any], other_payload_size: int) -> tuple[bytes, int]:
    total = other_payload_size
    while True:
        encoded = _json_bytes({**base, "total_payload_size": total})
        next_total = other_payload_size + len(encoded)
        if next_total == total:
            return encoded, total
        total = next_total


def _publish_staging(staging: Path, output: Path) -> None:
    backup = output.parent / f".{output.name}.backup-{uuid4().hex}"
    if output.exists():
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        if backup.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def export_artifacts(
    chunks: Sequence[Chunk],
    model: StaticEmbedding,
    output: Path,
    *,
    built_at: str | None = None,
    corpus_stats: Mapping[str, Any] | None = None,
) -> ExportResult:
    """Build and atomically publish a complete browser-search artifact set."""
    token_ids = model.tokenize([chunk.text for chunk in chunks])
    included = [(chunk, ids) for chunk, ids in zip(chunks, token_ids, strict=True) if ids]
    excluded = tuple(chunk.chunk_id for chunk, ids in zip(chunks, token_ids, strict=True) if not ids)
    if not included:
        raise ValueError("no chunks contain known model tokens")

    included_chunks = [chunk for chunk, _ in included]
    vectors = model.embed([chunk.text for chunk in included_chunks])
    if vectors.ndim != 2 or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("document vectors must be finite and non-zero")

    token_values, token_scales = quantize_rows(model.token_vectors)
    document_values, document_scale = quantize_global(vectors)
    manifest = [
        {
            "chunk_id": chunk.chunk_id,
            "page_id": chunk.page_id,
            "url": chunk.url,
            "title": chunk.title,
            "heading": chunk.heading,
            "crumb": chunk.crumb,
        }
        for chunk in included_chunks
    ]
    tokenizer_config = resolve_tokenizer_config(model.tokenizer_json, model.unknown_token_id)
    payloads = {
        "token-table.bin": token_values.tobytes(order="C"),
        "token-scales.bin": token_scales.astype("<f4", copy=False).tobytes(order="C"),
        "doc-vectors.bin": document_values.tobytes(order="C"),
        "manifest.json": _json_bytes(manifest),
        "tokenizer-config.json": _json_bytes(tokenizer_config),
    }
    base_meta = {
        "version": 1,
        "model_id": model.model_id,
        "dimensions": int(vectors.shape[1]),
        "document_global_scale": document_scale,
        "built_at": built_at or datetime.now(timezone.utc).isoformat(),
        "corpus_stats": dict(corpus_stats or {}),
        "chunk_count": len(manifest),
        "excluded_chunk_count": len(excluded),
    }
    meta_bytes, total_size = _fixed_meta_bytes(base_meta, sum(len(data) for data in payloads.values()))
    payloads["meta.json"] = meta_bytes

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for name in ARTIFACT_NAMES:
            (staging / name).write_bytes(payloads[name])
        _publish_staging(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    artifact_sizes = tuple((name, len(payloads[name])) for name in ARTIFACT_NAMES)
    return ExportResult(len(manifest), excluded, artifact_sizes, total_size)
