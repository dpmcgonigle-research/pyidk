"""Similarity utilities for finite Isolation Kernel embeddings."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse


def _validate_embeddings(embeddings: csr_matrix, n_partitions: int) -> csr_matrix:
    if isinstance(n_partitions, (bool, np.bool_)) or not isinstance(
        n_partitions, (int, np.integer)
    ):
        raise TypeError("n_partitions must be an integer.")
    if n_partitions < 1:
        raise ValueError("n_partitions must be at least 1.")
    if not issparse(embeddings):
        raise TypeError("embeddings must be a SciPy sparse matrix.")
    if embeddings.ndim != 2 or not embeddings.shape[1]:
        raise ValueError("embeddings must have two dimensions and nonzero feature width.")
    if embeddings.shape[1] % n_partitions:
        raise ValueError("Feature width must be divisible by n_partitions.")
    if np.iscomplexobj(embeddings):
        raise ValueError("embeddings must contain real values.")
    result = csr_matrix(embeddings, dtype=float)
    if not np.isfinite(result.data).all():
        raise ValueError("embeddings must contain finite values.")
    return result


def pairwise_similarity(
    embeddings: csr_matrix,
    other: csr_matrix | None = None,
    *,
    n_partitions: int,
    dense: bool = True,
):
    """Compute pairwise Isolation Kernel similarities from feature embeddings.

    ``embeddings`` and ``other`` are sparse matrices of shape ``(n, k)`` and
    ``(m, k)`` from the same fitted basis; rows represent points or distributions.
    If ``other`` is omitted, compare ``embeddings`` with itself. Return shape
    ``(n, m)`` as a NumPy array if ``dense=True``, otherwise as a CSR matrix.
    ``n_partitions`` must be positive and match the basis used for the embeddings.

    The finite Isolation Kernel feature map uses the convention

        K(x, y) = <Phi(x), Phi(y)> / t

    and the same scaling applies to mean distribution embeddings.
    """
    embeddings = _validate_embeddings(embeddings, n_partitions)
    if other is None:
        other = embeddings
    else:
        other = _validate_embeddings(other, n_partitions)

    if embeddings.shape[1] != other.shape[1]:
        raise ValueError("Embedding matrices must have the same number of columns.")

    result = (embeddings @ other.T) / float(n_partitions)
    if dense:
        return np.asarray(result.toarray())
    return result.tocsr()


def pairwise_distance(
    embeddings: csr_matrix,
    other: csr_matrix | None = None,
    *,
    n_partitions: int,
) -> np.ndarray:
    """Return dense IDK-induced distances, with shape ``(n, m)``.

    Compute ``sqrt(max(0, K(x,x) + K(y,y) - 2*K(x,y)))``, equivalent to
    Euclidean distance between feature rows divided by ``sqrt(n_partitions)``.
    Inputs must use the same fitted basis; raw matrices cannot establish its
    identity. Use ``WindowEmbeddings.distance`` for metadata validation or
    ``reduce_references`` to avoid materializing a full pairwise matrix.

    Two all-zero embeddings have distance zero, including fully outside windows.
    Inspect outside mass as well when measuring nominal support.
    """
    left = _validate_embeddings(embeddings, n_partitions)
    right = left if other is None else _validate_embeddings(other, n_partitions)
    if left.shape[1] != right.shape[1]:
        raise ValueError("Embedding matrices must have the same number of columns.")
    return _distance_block(left, right, n_partitions)


def _distance_block(
    left: csr_matrix,
    right: csr_matrix,
    n_partitions: int,
    left_norm: np.ndarray | None = None,
    right_norm: np.ndarray | None = None,
) -> np.ndarray:
    if left_norm is None:
        left_norm = np.asarray(left.multiply(left).sum(axis=1)).ravel() / n_partitions
    if right_norm is None:
        right_norm = np.asarray(right.multiply(right).sum(axis=1)).ravel() / n_partitions
    squared = (left @ right.T).toarray() * (-2.0 / n_partitions)
    squared += left_norm[:, None]
    squared += right_norm[None, :]
    # The dot-product identity loses precision for almost identical rows. Only
    # those pairs need the equivalent sparse-difference calculation. Zero/zero
    # pairs already have exact distance zero and do not need this fallback.
    scale = left_norm[:, None] + right_norm[None, :]
    near = (scale > 0) & (squared <= 16 * np.finfo(float).eps * scale)
    for i, j in zip(*np.nonzero(near), strict=True):
        difference = left[i] - right[j]
        squared[i, j] = difference.multiply(difference).sum() / n_partitions
    return np.sqrt(np.maximum(squared, 0.0))
