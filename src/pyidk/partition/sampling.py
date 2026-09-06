"""Partition sampling abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class PartitionSampler(ABC):
    """Choose training rows used to construct each isolation partition."""

    @abstractmethod
    def sample(
        self,
        X: NDArray[np.float64],
        *,
        n_partitions: int,
        samples_per_partition: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Select row indices from ``X``, shaped ``(n_samples, n_features)``.

        Return an integer array of shape
        ``(n_partitions, samples_per_partition)``: one row of selected training
        indices per partition. Use ``rng`` for any random choices.
        """


@dataclass(frozen=True, slots=True)
class UniformPartitionSampler(PartitionSampler):
    """Uniformly sample distinct observations inside each partition."""

    def sample(
        self,
        X: NDArray[np.float64],
        *,
        n_partitions: int,
        samples_per_partition: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Sample training-row indices uniformly without replacement per partition.

        ``X`` has shape ``(n_samples, n_features)``. Return integer indices of
        shape ``(n_partitions, samples_per_partition)`` using ``rng``. A training
        row may appear in multiple partitions, but only once within each one.
        Require positive ``n_partitions`` and ``2 <= samples_per_partition <=
        n_samples``.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        if n_partitions < 1:
            raise ValueError("n_partitions must be at least 1.")
        if samples_per_partition < 2:
            raise ValueError("samples_per_partition must be at least 2.")
        if samples_per_partition > X.shape[0]:
            raise ValueError("samples_per_partition cannot exceed the number of training rows.")

        samples = np.empty(
            (n_partitions, samples_per_partition),
            dtype=np.int64,
        )
        for partition in range(n_partitions):
            samples[partition] = rng.choice(
                X.shape[0],
                size=samples_per_partition,
                replace=False,
            )
        return samples
