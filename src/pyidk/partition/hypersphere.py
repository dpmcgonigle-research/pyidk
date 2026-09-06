"""iNNE-style hypersphere isolation partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist


@dataclass(frozen=True, slots=True)
class IsolationBasis:
    """Fitted hypersphere basis for an Isolation Kernel.

    Attributes
    ----------
    centers:
        Shape ``(t, psi, d)``.
    radii:
        Shape ``(t, psi)``. Each radius is the distance from a sampled center to
        its nearest sampled neighbor in the same partition.
    sample_indices:
        Shape ``(t, psi)``: training-row indices used to create each partition.
    metric:
        SciPy `cdist` metric used for fitting and assignment.
    """

    centers: NDArray[np.float64]
    radii: NDArray[np.float64]
    sample_indices: NDArray[np.int64]
    metric: str

    @property
    def n_partitions(self) -> int:
        """Number of independently sampled partitions, ``t``."""
        return self.centers.shape[0]

    @property
    def samples_per_partition(self) -> int:
        """Number of sampled centers in each partition, ``psi``."""
        return self.centers.shape[1]

    @property
    def n_features_in(self) -> int:
        """Number of input features per observation, ``d``."""
        return self.centers.shape[2]


@dataclass(frozen=True, slots=True)
class HyperspherePartitioner:
    """Construct iNNE-style hyperspheres from sampled training observations."""

    metric: str = "euclidean"

    def fit_basis(
        self,
        X: ArrayLike,
        sample_indices: NDArray[np.int64],
    ) -> IsolationBasis:
        """Build hyperspheres around the selected training observations.

        Parameters
        ----------
        X:
            Shape ``(n_samples, n_features)``: one observation per row and one
            feature per column. A single feature still needs shape ``(n, 1)``;
        sample_indices:
            Integer array of shape ``(n_partitions, samples_per_partition)``.
            Each row selects the training rows used as centers for one partition.
            For example, ``[[0, 2], [1, 3]]`` creates two partitions, each with two
            centers. A single partition still needs shape ``(1, psi)``.
            Indices must refer to rows of ``X``, with at least two per partition.

        Returns
        -------
        IsolationBasis
            Centers of shape ``(n_partitions, samples_per_partition, n_features)``
            and radii of shape ``(n_partitions, samples_per_partition)``. Each
            radius is the distance to the nearest other center in that partition.
        """
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("X must be a 2-D array.")
        if sample_indices.ndim != 2:
            raise ValueError("sample_indices must be a 2-D integer array.")
        if sample_indices.shape[1] < 2:
            raise ValueError("At least two samples per partition are required.")

        centers = array[sample_indices].copy()
        t, psi, _ = centers.shape
        radii = np.empty((t, psi), dtype=float)

        for partition in range(t):
            # SciPy validates metric strings at runtime; its stubs require literals.
            distances = cdist(  # type: ignore[call-overload]
                centers[partition],
                centers[partition],
                metric=self.metric,
            )
            np.fill_diagonal(distances, np.inf)
            radii[partition] = distances.min(axis=1)

        return IsolationBasis(
            centers=centers,
            radii=radii,
            sample_indices=np.asarray(sample_indices, dtype=np.int64).copy(),
            metric=self.metric,
        )

    def assign(
        self,
        X: ArrayLike,
        basis: IsolationBasis,
        *,
        chunk_size: int = 4096,
    ) -> NDArray[np.int64]:
        """Assign each point to at most one hypersphere per partition.

        Parameters
        ----------
        X:
            Shape ``(n_samples, n_features)`` with the same feature columns used
            to fit ``basis``.
        basis:
            Fitted centers, radii, and distance metric; ``t`` is the number of
            partitions and ``psi`` is the number of centers per partition.
        chunk_size:
            Maximum number of query rows processed at once; must be positive.

        Returns
        -------
        assignments:
            Integer array of shape ``(n_samples, t)``. Each entry is a local cell
            index in ``[0, psi)`` or ``-1`` when the point is outside the selected
            hypersphere for that partition.

        Notes
        -----
        The query point is first associated with its nearest sampled center. That
        cell is active only when the query-center distance does not exceed that
        center's fitted nearest-neighbor radius.
        """
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("X must be a 2-D array.")
        if array.shape[1] != basis.n_features_in:
            raise ValueError("X has a different number of features than the basis.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")

        n = array.shape[0]
        assignments = np.full((n, basis.n_partitions), -1, dtype=np.int64)

        for partition in range(basis.n_partitions):
            centers = basis.centers[partition]
            radii = basis.radii[partition]

            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                # SciPy validates metric strings at runtime; its stubs require literals.
                distances = cdist(  # type: ignore[call-overload]
                    array[start:stop], centers, metric=basis.metric
                )
                nearest = distances.argmin(axis=1)
                nearest_distance = distances[
                    np.arange(stop - start),
                    nearest,
                ]
                active = nearest_distance <= radii[nearest]
                rows = np.nonzero(active)[0]
                assignments[start + rows, partition] = nearest[rows]

        return assignments
