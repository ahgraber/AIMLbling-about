import numpy as np
import numpy.typing as npt

from site_search.quantize import dequantize_global, dequantize_rows, quantize_global, quantize_rows


def _row_cosines(left: npt.NDArray[np.float32], right: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))


def test_roundtrip_fidelity() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(32, 128)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    token_values, token_scales = quantize_rows(vectors)
    document_values, document_scale = quantize_global(vectors)

    assert np.all(_row_cosines(vectors, dequantize_rows(token_values, token_scales)) >= 0.999)
    assert np.all(_row_cosines(vectors, dequantize_global(document_values, document_scale)) >= 0.999)


def test_ranking_preserved() -> None:
    documents = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    documents /= np.linalg.norm(documents, axis=1, keepdims=True)
    queries = np.asarray([[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]], dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    quantized, scale = quantize_global(documents)

    reference = np.argsort(-(queries @ documents.T), axis=1)
    compressed = np.argsort(-(queries @ dequantize_global(quantized, scale).T), axis=1)

    np.testing.assert_array_equal(compressed, reference)
