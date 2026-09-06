"""Similarity utilities for finite Isolation Kernel embeddings."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse


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
    if n_partitions < 1:
        raise ValueError("n_partitions must be at least 1.")
    if not issparse(embeddings):
        raise TypeError("embeddings must be a SciPy sparse matrix.")
    if other is None:
        other = embeddings
    elif not issparse(other):
        raise TypeError("other must be a SciPy sparse matrix.")

    if embeddings.shape[1] != other.shape[1]:
        raise ValueError("Embedding matrices must have the same number of columns.")

    result = (embeddings @ other.T) / float(n_partitions)
    if dense:
        return np.asarray(result.toarray())
    return result.tocsr()
