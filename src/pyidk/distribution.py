"""Isolation Distributional Kernel group embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

from .data import SequenceBatch
from .kernel import IsolationKernel
from .similarity import pairwise_similarity

Aggregation = Literal["mean", "count"]


@dataclass(slots=True)
class IsolationDistributionalKernel:
    """Kernel-mean embedding of groups using an `IsolationKernel` feature map."""

    point_kernel: IsolationKernel = field(default_factory=IsolationKernel)

    def fit(self, data: SequenceBatch) -> IsolationDistributionalKernel:
        """Fit the point-level isolation basis on all rows of ``data.values``.

        ``data`` is a sequence batch; sequence boundaries do not affect fitting.
        Return self with its point kernel fitted.
        """
        self.point_kernel.fit(data.values)
        return self

    def transform(
        self,
        data: SequenceBatch,
        *,
        aggregation: Aggregation = "mean",
    ) -> csr_matrix:
        """Combine point features into one sparse row per sequence in ``data``.

        Return a CSR matrix of shape ``(data.n_sequences, t * psi)``, where
        ``t`` and ``psi`` are the point kernel's partitions and centers per
        partition. With ``aggregation='count'``, each entry counts points assigned
        to that hypersphere. With ``'mean'``, divide each count by the sequence's
        total number of points, giving a fraction. Point order is not retained.
        Mean aggregation rejects empty sequences; count produces zero rows for
        them. The point kernel must already be fitted.
        """
        if aggregation not in {"mean", "count"}:
            raise ValueError("aggregation must be 'mean' or 'count'.")

        point_embeddings = self.point_kernel.transform(data.values)
        lengths = data.lengths

        if np.any(lengths == 0) and aggregation == "mean":
            raise ValueError("Mean distribution embeddings are undefined for empty sequences.")

        group_ids = np.repeat(np.arange(data.n_sequences, dtype=np.int64), lengths)
        point_ids = np.arange(data.n_samples, dtype=np.int64)

        if aggregation == "count":
            weights = np.ones(data.n_samples, dtype=float)
        else:
            weights = np.repeat(1.0 / lengths, lengths)

        grouping = csr_matrix(
            (weights, (group_ids, point_ids)),
            shape=(data.n_sequences, data.n_samples),
            dtype=float,
        )
        return (grouping @ point_embeddings).tocsr()

    def fit_transform(
        self,
        data: SequenceBatch,
        *,
        aggregation: Aggregation = "mean",
    ) -> csr_matrix:
        """Fit on all observations in ``data``, then embed each sequence.

        Return a CSR matrix with one row per sequence and one column per fitted
        hypersphere. ``aggregation`` selects counts or fractions as described in
        ``transform``. Equivalent to ``fit(data).transform(data, aggregation=...)``.
        """
        return self.fit(data).transform(data, aggregation=aggregation)

    def similarity(
        self,
        embeddings: csr_matrix,
        other: csr_matrix | None = None,
        *,
        dense: bool = True,
    ):
        """Return row-by-row dot products divided by the number of partitions.

        ``embeddings`` and ``other`` are sparse matrices from the same fitted
        basis, with shapes ``(n, t * psi)`` and ``(m, t * psi)``. If ``other`` is
        omitted, compare ``embeddings`` with itself. Return shape ``(n, m)`` as
        a NumPy array when ``dense=True``, otherwise a CSR matrix. This is raw
        IDK similarity, not cosine similarity; self-similarity need not be 1.
        """
        return pairwise_similarity(
            embeddings,
            other,
            n_partitions=self.point_kernel.n_partitions,
            dense=dense,
        )
