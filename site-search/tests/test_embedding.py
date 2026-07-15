from collections.abc import Sequence
from pathlib import Path

from model2vec import StaticModel

from tokenizers import Tokenizer

import numpy as np
import numpy.typing as npt

from site_search.embedding import StaticEmbeddingModel, embed_token_ids, load_embedding_model


class FakeProvider:
    def __init__(self) -> None:
        self.embedding: npt.NDArray[np.float32] = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self.token_mapping: None = None
        self.weights: None = None
        self.normalize: bool = True

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        return [[0, 1] if text == "alpha beta" else [2] for text in texts]

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        return embed_token_ids(self.tokenize(texts), self.embedding)


def test_adapter_mean_pools_and_normalizes_fake_provider() -> None:
    model = StaticEmbeddingModel(FakeProvider(), model_id="fake")

    embedded = model.embed(["alpha beta", "both"])

    expected = np.asarray([[2**-0.5, 2**-0.5], [2**-0.5, 2**-0.5]], dtype=np.float32)
    np.testing.assert_allclose(embedded, expected, rtol=0, atol=1e-7)


def test_model2vec_exposes_raw_assets_and_first_128_dimensions(potion_model_path: Path) -> None:
    full = StaticModel.from_pretrained(potion_model_path)
    reduced = StaticModel.from_pretrained(potion_model_path, dimensionality=128)

    assert isinstance(full.embedding, np.ndarray)
    assert isinstance(full.tokenizer, Tokenizer)
    assert len(full.tokens) == 29_528
    assert full.embedding.shape == (29_528, 256)
    assert reduced.embedding.shape == (29_528, 128)
    assert full.normalize is True
    assert full.token_mapping is None
    assert full.weights is None
    np.testing.assert_array_equal(reduced.embedding, full.embedding[:, :128])


def test_manual_embedding_matches_model2vec_encode(potion_model_path: Path) -> None:
    model = load_embedding_model(source=potion_model_path, dimensions=128)
    samples = ["semantic search", "café naïve", "semantic xyzzyplugh☃"]

    provider_vectors = model.embed(samples)
    manual_vectors = embed_token_ids(model.tokenize(samples), model.token_vectors)

    np.testing.assert_array_equal(manual_vectors, provider_vectors)


def test_release_model_uses_full_256_dimensions_by_default(potion_model_path: Path) -> None:
    model = load_embedding_model(source=potion_model_path)

    assert model.token_vectors.shape == (29_528, 256)
