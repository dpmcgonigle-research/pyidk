"""Support diagnostics for mean isolation occupancy embeddings."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .similarity import _validate_embeddings


@dataclass(frozen=True, slots=True)
class OutsideMass:
    """Missing occupancy: ``per_partition`` has shape (n, t), ``mean`` (n,)."""

    per_partition: NDArray[np.float64]
    mean: NDArray[np.float64]


def outside_mass(embeddings: csr_matrix, *, n_partitions: int, atol: float = 1e-12) -> OutsideMass:
    """Measure basis support, without references or neighbor selection.

    For mean occupancies, return ``1 - sum(cells)`` per partition and its
    mean across partitions. Occupied cells are never renormalized. Negative
    cells or partition totals above one by more than ``atol`` are rejected;
    roundoff within this absolute tolerance is clipped to the valid range.
    Raw count embeddings are not supported (counts that happen to satisfy the
    bounds cannot be distinguished from means, so the caller must supply means).
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative.")
    values = _validate_embeddings(embeddings, n_partitions).copy()
    values.sum_duplicates()
    if np.any(values.data < -atol):
        raise ValueError("Mean occupancy entries must be non-negative.")
    values.data = np.maximum(values.data, 0.0)
    width = values.shape[1]
    psi = width // n_partitions
    grouping = csr_matrix(
        (np.ones(width), (np.arange(width), np.arange(width) // psi)),
        shape=(width, n_partitions),
    )
    totals = (values @ grouping).toarray()
    if np.any(totals > 1.0 + atol):
        raise ValueError("Mean occupancy per partition must not exceed one.")
    missing = np.clip(1.0 - totals, 0.0, 1.0)
    return OutsideMass(missing, missing.mean(axis=1))
