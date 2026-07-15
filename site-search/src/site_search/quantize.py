from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
Int8Array = npt.NDArray[np.int8]


def _float_matrix(vectors: npt.ArrayLike) -> FloatArray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    return matrix


def quantize_rows(vectors: npt.ArrayLike) -> tuple[Int8Array, FloatArray]:
    """Quantize each token vector with its own float32 scale."""
    matrix = _float_matrix(vectors)
    maxima = np.max(np.abs(matrix), axis=1)
    scales = np.where(maxima == 0, np.float32(1.0), maxima / np.float32(127.0)).astype(np.float32)
    values = np.clip(np.rint(matrix / scales[:, None]), -127, 127).astype(np.int8)
    return values, scales


def dequantize_rows(values: Int8Array, scales: FloatArray) -> FloatArray:
    """Restore row-scaled values to float32 for verification."""
    return (values.astype(np.float32) * scales[:, None]).astype(np.float32)


def quantize_global(vectors: npt.ArrayLike) -> tuple[Int8Array, float]:
    """Quantize document vectors with one rank-preserving scale."""
    matrix = _float_matrix(vectors)
    maximum = float(np.max(np.abs(matrix)))
    scale = np.float32(maximum / 127.0 if maximum else 1.0)
    values = np.clip(np.rint(matrix / scale), -127, 127).astype(np.int8)
    return values, float(scale)


def dequantize_global(values: Int8Array, scale: float) -> FloatArray:
    """Restore globally scaled values to float32 for verification."""
    return (values.astype(np.float32) * np.float32(scale)).astype(np.float32)
