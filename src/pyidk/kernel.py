"""Isolation Kernel point feature map."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from .partition import (
    HyperspherePartitioner,
    IsolationBasis,
    PartitionSampler,
    UniformPartitionSampler,
)


def _as_2d_float(X: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError("X must be a 2-D numeric array.")
    if array.shape[0] == 0:
        raise ValueError("X must contain at least one row.")
    return array


@dataclass(slots=True)
class IsolationKernel:
    """Finite feature map for an iNNE-style Isolation Kernel."""

    n_partitions: int = 100
    samples_per_partition: int = 16
    random_state: int | None = None
    chunk_size: int = 4096
    sampler: PartitionSampler = field(default_factory=UniformPartitionSampler)
    partitioner: HyperspherePartitioner = field(default_factory=HyperspherePartitioner)

    basis_: IsolationBasis | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_partitions < 1:
            raise ValueError("n_partitions must be at least 1.")
        if self.samples_per_partition < 2:
            raise ValueError("samples_per_partition must be at least 2.")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")

    @property
    def n_embedding_features(self) -> int:
        """Dimensionality of the finite point feature map."""
        return self.n_partitions * self.samples_per_partition

    def fit(self, X: ArrayLike) -> IsolationKernel:
        """Sample training observations and fit the isolation basis; return self.

        ``X`` has shape ``(n_samples, n_features)``: rows are observations and
        columns are features. For one feature, use ``(n_samples, 1)``. Require
        at least ``samples_per_partition`` rows. Store the result in ``basis_``.
        """
        array = _as_2d_float(X)
        if self.samples_per_partition > array.shape[0]:
            raise ValueError("samples_per_partition cannot exceed the number of training rows.")

        rng = np.random.default_rng(self.random_state)
        sample_indices = self.sampler.sample(
            array,
            n_partitions=self.n_partitions,
            samples_per_partition=self.samples_per_partition,
            rng=rng,
        )
        self.basis_ = self.partitioner.fit_basis(array, sample_indices)
        return self

    def transform(self, X: ArrayLike) -> csr_matrix:
        """Encode observations using the fitted hyperspheres.

        ``X`` has shape ``(n_samples, n_features)`` with the same feature columns
        used in ``fit``. Return a sparse CSR matrix of shape
        ``(n_samples, n_partitions * samples_per_partition)``. Columns represent
        hyperspheres, grouped by partition; each row contains at most one 1 per
        partition, or none when the observation is outside its selected sphere.
        Raise ``RuntimeError`` if the kernel has not been fitted.
        """
        if self.basis_ is None:
            raise RuntimeError("IsolationKernel must be fit before transform.")

        array = _as_2d_float(X)
        assignments = self.partitioner.assign(
            array,
            self.basis_,
            chunk_size=self.chunk_size,
        )

        # Each point can activate at most one cell per partition. Build the sparse
        # matrix from only active assignments.
        point_rows, partitions = np.nonzero(assignments >= 0)
        local_cells = assignments[point_rows, partitions]
        columns = partitions * self.samples_per_partition + local_cells
        data = np.ones(point_rows.shape[0], dtype=float)

        return csr_matrix(
            (data, (point_rows, columns)),
            shape=(array.shape[0], self.n_embedding_features),
            dtype=float,
        )

    def fit_transform(self, X: ArrayLike) -> csr_matrix:
        """Fit on ``X`` and return its sparse point embeddings.

        Input shape is ``(n_samples, n_features)``; output shape is
        ``(n_samples, n_partitions * samples_per_partition)``. Equivalent to
        ``fit(X).transform(X)``.
        """
        return self.fit(X).transform(X)
