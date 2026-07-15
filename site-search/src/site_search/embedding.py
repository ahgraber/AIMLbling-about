from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from model2vec import StaticModel

from tokenizers import Tokenizer

import numpy as np
import numpy.typing as npt

MODEL_ID = "minishlab/potion-base-8M"
# Pin the Hub revision so the shipped token table is reproducible. model2vec's
# from_pretrained has no revision argument and would otherwise track the model
# repo's default branch, silently changing every artifact between builds of the
# same commit. Bump this deliberately when adopting a new model release.
MODEL_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"
DEFAULT_DIMENSIONS = 256
FloatArray = npt.NDArray[np.float32]


class StaticEmbeddingProvider(Protocol):
    """Provider behavior needed by the static-search build pipeline."""

    @property
    def embedding(self) -> npt.NDArray[np.generic]:
        """Return the provider token table."""
        ...

    @property
    def token_mapping(self) -> npt.NDArray[np.generic] | None:
        """Return any indirection between tokenizer IDs and table rows."""
        ...

    @property
    def weights(self) -> npt.NDArray[np.generic] | None:
        """Return optional token weights."""
        ...

    @property
    def normalize(self) -> bool:
        """Return whether provider output is L2 normalized."""
        ...

    @property
    def tokenizer(self) -> Tokenizer:
        """Return the provider tokenizer."""
        ...

    @property
    def unk_token_id(self) -> int | None:
        """Return the provider unknown-token ID."""
        ...

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        """Return known token IDs without special or unknown tokens."""
        ...

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.generic]:
        """Return provider reference embeddings."""
        ...


class StaticEmbedding(Protocol):
    """Structural boundary consumed by chunking and artifact export."""

    model_id: str

    @property
    def token_vectors(self) -> FloatArray:
        """Return raw token rows in tokenizer-ID order."""
        ...

    @property
    def tokenizer_json(self) -> str:
        """Serialize the tokenizer used to produce token IDs."""
        ...

    @property
    def unknown_token_id(self) -> int | None:
        """Return the token ID excluded from known-token sequences."""
        ...

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        """Tokenize input text into known token IDs."""
        ...

    def embed(self, texts: Sequence[str]) -> FloatArray:
        """Embed input text as normalized vectors."""
        ...


def embed_token_ids(token_ids: Sequence[Sequence[int]], token_vectors: FloatArray) -> FloatArray:
    """Mean-pool known token rows and L2-normalize each non-empty result."""
    embedded = np.zeros((len(token_ids), token_vectors.shape[1]), dtype=np.float32)
    for row, ids in enumerate(token_ids):
        if not ids:
            continue
        embedded[row] = token_vectors[np.asarray(ids, dtype=np.intp)].mean(axis=0)
    norms = np.linalg.norm(embedded, axis=1, keepdims=True) + np.float32(1e-32)
    return np.asarray(embedded / norms, dtype=np.float32)


class StaticEmbeddingModel:
    """Model-neutral boundary around a static token embedding provider."""

    def __init__(self, provider: object, *, model_id: str) -> None:
        provider = cast(StaticEmbeddingProvider, provider)
        if provider.token_mapping is not None or provider.weights is not None:
            raise ValueError("the browser artifact requires direct, unweighted token-row lookup")
        if not provider.normalize:
            raise ValueError("the browser artifact requires normalized provider embeddings")
        self._provider: StaticEmbeddingProvider = provider
        self.model_id: str = model_id

    @property
    def token_vectors(self) -> FloatArray:
        """Return raw token rows in tokenizer-ID order."""
        return np.asarray(self._provider.embedding, dtype=np.float32)

    @property
    def tokenizer_json(self) -> str:
        """Serialize tokenizer behavior for resolved artifact export."""
        return self._provider.tokenizer.to_str()

    @property
    def unknown_token_id(self) -> int | None:
        """Return the token ID removed by provider tokenization."""
        return self._provider.unk_token_id

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        """Tokenize text with provider unknown-token removal semantics."""
        return self._provider.tokenize(texts)

    def embed(self, texts: Sequence[str]) -> FloatArray:
        """Embed text through the provider reference implementation."""
        return np.asarray(self._provider.encode(texts), dtype=np.float32)


def load_embedding_model(
    *, source: str | Path = MODEL_ID, revision: str | None = MODEL_REVISION, dimensions: int = DEFAULT_DIMENSIONS
) -> StaticEmbeddingModel:
    """Load the pinned model2vec model behind the embedding boundary.

    A Hub id is resolved to its pinned `revision` snapshot (cache first, fetching
    only when absent) so the token table cannot drift between builds; a local
    `source` path is loaded as-is.
    """
    resolved: str | Path = source
    if revision is not None and not Path(source).exists():
        try:
            resolved = snapshot_download(str(source), revision=revision, local_files_only=True)
        except LocalEntryNotFoundError:
            resolved = snapshot_download(str(source), revision=revision)
    provider = StaticModel.from_pretrained(resolved, dimensionality=dimensions, force_download=False)
    return StaticEmbeddingModel(provider, model_id=MODEL_ID)
