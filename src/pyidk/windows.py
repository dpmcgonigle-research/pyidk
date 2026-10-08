"""Reusable point features and sequence-local moving analysis intervals."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, vstack

from .data import SequenceBatch
from .kernel import IsolationKernel
from .metrics import OutsideMass, outside_mass
from .partition import IsolationBasis
from .similarity import pairwise_distance, pairwise_similarity

Interval = tuple[int, int, int]


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")


def _basis_digest(basis: IsolationBasis) -> bytes:
    digest = hashlib.sha256(basis.metric.encode())
    for array in (basis.centers, basis.radii, basis.sample_indices):
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.digest()


@dataclass(frozen=True, slots=True)
class _BasisBinding:
    kernel: IsolationKernel
    basis: IsolationBasis
    digest: bytes
    n_partitions: int
    samples_per_partition: int
    n_features_in: int

    @classmethod
    def capture(cls, kernel: IsolationKernel) -> _BasisBinding:
        basis = kernel.basis_
        if basis is None:
            raise RuntimeError("The isolation kernel must already be fitted.")
        result = cls(
            kernel,
            basis,
            _basis_digest(basis),
            basis.n_partitions,
            basis.samples_per_partition,
            basis.n_features_in,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.kernel.basis_ is not self.basis
            or self.kernel.n_partitions != self.n_partitions
            or self.kernel.samples_per_partition != self.samples_per_partition
            or _basis_digest(self.basis) != self.digest
        ):
            raise ValueError("Cached features are incompatible with the changed fitted basis.")

    def assert_compatible(self, other: _BasisBinding) -> None:
        self.validate()
        other.validate()
        if self.basis is not other.basis or self.digest != other.digest:
            raise ValueError("Features must come from the same fitted basis instance.")


@dataclass(frozen=True, slots=True)
class WindowEmbeddings:
    """Mean occupancy rows with aligned ``(sequence_index, start, stop)`` metadata.

    Construct via ``PointFeatureCache``. Intervals are sequence-local, half-open
    representation-row coordinates. ``stop - 1`` is the last consumed row.
    Treat the embeddings and metadata as read-only. No time alignment, padding,
    sequence weighting, or physical transition semantics are inferred.
    """

    embeddings: csr_matrix
    intervals: NDArray[np.int64]
    _binding: _BasisBinding

    @property
    def n_partitions(self) -> int:
        return self._binding.n_partitions

    @property
    def sequence_index(self) -> NDArray[np.int64]:
        return self.intervals[:, 0]

    @property
    def start(self) -> NDArray[np.int64]:
        return self.intervals[:, 1]

    @property
    def stop(self) -> NDArray[np.int64]:
        return self.intervals[:, 2]

    @property
    def row_count(self) -> NDArray[np.int64]:
        return self.stop - self.start

    def assert_compatible(self, other: WindowEmbeddings) -> None:
        """Reject different, refitted, or mutated bases, even with equal widths."""
        self._validate()
        other._validate()
        self._binding.assert_compatible(other._binding)

    def _validate(self) -> None:
        self._binding.validate()
        if (
            self.intervals.ndim != 2
            or self.intervals.shape[1] != 3
            or not np.issubdtype(self.intervals.dtype, np.integer)
            or np.any(self.intervals[:, :2] < 0)
            or np.any(self.intervals[:, 2] <= self.intervals[:, 1])
        ):
            raise ValueError("Invalid window interval metadata.")
        expected_width = self.n_partitions * self._binding.samples_per_partition
        if self.embeddings.shape != (len(self.intervals), expected_width):
            raise ValueError(
                "Embedding dimensions disagree with interval or fitted basis metadata."
            )

    def similarity(self, other: WindowEmbeddings, *, dense: bool = True):
        """Native IDK similarities, after checking fitted-basis compatibility."""
        self.assert_compatible(other)
        return pairwise_similarity(
            self.embeddings, other.embeddings, n_partitions=self.n_partitions, dense=dense
        )

    def distance(self, other: WindowEmbeddings) -> NDArray[np.float64]:
        """Dense IDK-induced distances, after checking basis compatibility."""
        self.assert_compatible(other)
        return pairwise_distance(self.embeddings, other.embeddings, n_partitions=self.n_partitions)

    def outside_mass(self, *, atol: float = 1e-12) -> OutsideMass:
        """Return support diagnostics in exactly this object's interval order."""
        self._validate()
        return outside_mass(self.embeddings, n_partitions=self.n_partitions, atol=atol)


@dataclass(frozen=True, slots=True)
class PointFeatureCache:
    """In-memory sparse point features tied to one fitted basis.

    Use ``from_sequences`` on already represented/scaled rows. Their feature
    meaning and scaler identity are the caller's responsibility; changing either
    requires a new cache. Basis refits, parameter changes, and in-place basis
    mutations are detected. Never modify the cache's arrays or sparse matrix.
    """

    features: csr_matrix
    offsets: NDArray[np.int64]
    _binding: _BasisBinding

    @classmethod
    def from_sequences(cls, kernel: IsolationKernel, data: SequenceBatch) -> PointFeatureCache:
        """Transform each row once with an already fitted kernel; never fit it.

        Empty sequences retain their offsets, including an entirely empty batch.
        Subsequent interval/window operations do not call point transform again.
        """
        binding = _BasisBinding.capture(kernel)
        if data.n_features != binding.n_features_in:
            raise ValueError("Sequence feature dimension differs from the fitted basis.")
        if not np.isfinite(data.values).all():
            raise ValueError("Sequence values must be finite.")
        features = (
            kernel.transform(data.values)
            if data.n_samples
            else csr_matrix((0, binding.n_partitions * binding.samples_per_partition))
        )
        offsets = np.array(data.offsets, dtype=np.int64, copy=True)
        offsets.flags.writeable = False
        for array in (features.data, features.indices, features.indptr):
            array.flags.writeable = False
        return cls(features, offsets, binding)

    @property
    def n_partitions(self) -> int:
        return self._binding.n_partitions

    @property
    def samples_per_partition(self) -> int:
        return self._binding.samples_per_partition

    @property
    def n_features_in(self) -> int:
        return self._binding.n_features_in

    def assert_compatible(self, other: PointFeatureCache | IsolationKernel) -> None:
        """Validate against another cache or the kernel that will consume it."""
        binding = (
            other._binding if isinstance(other, PointFeatureCache) else _BasisBinding.capture(other)
        )
        self._binding.assert_compatible(binding)

    def window_intervals(self, length: int, *, stride: int = 1) -> Iterator[Interval]:
        """Enumerate complete windows, in sequence then start order.

        Starts are ``0, stride, 2*stride, ...`` while ``start + length <= n``.
        Short/empty sequences contribute none. No extra off-stride terminal
        window is appended; use explicit intervals if one is desired.
        """
        _positive_integer(length, "length")
        _positive_integer(stride, "stride")
        for sequence, size in enumerate(np.diff(self.offsets)):
            for start in range(0, int(size) - length + 1, stride):
                yield sequence, start, start + length

    def iter_windows(
        self, length: int, *, stride: int = 1, batch_size: int = 256
    ) -> Iterator[WindowEmbeddings]:
        """Yield bounded-size embedding batches; reuse rolling point-feature sums."""
        return self.iter_intervals(
            self.window_intervals(length, stride=stride), batch_size=batch_size
        )

    def windows(self, length: int, *, stride: int = 1) -> WindowEmbeddings:
        """Materialize all complete windows. Use ``iter_windows`` for bounded output."""
        return self.aggregate(self.window_intervals(length, stride=stride))

    def aggregate(self, intervals: Iterable[Interval]) -> WindowEmbeddings:
        """Materialize explicit intervals in caller order, allowing overlaps/repeats."""
        chunks = list(self.iter_intervals(intervals))
        if not chunks:
            return self._result([], [])
        return WindowEmbeddings(
            vstack([chunk.embeddings for chunk in chunks], format="csr"),
            np.concatenate([chunk.intervals for chunk in chunks]),
            self._binding,
        )

    def _result(self, rows: list[csr_matrix], intervals: list[Interval]) -> WindowEmbeddings:
        return WindowEmbeddings(
            vstack(rows, format="csr") if rows else csr_matrix((0, self.features.shape[1])),
            np.asarray(intervals, dtype=np.int64).reshape(-1, 3),
            self._binding,
        )

    def iter_intervals(
        self, intervals: Iterable[Interval], *, batch_size: int = 256
    ) -> Iterator[WindowEmbeddings]:
        """Aggregate intervals with O(t*psi) working sums plus one output batch.

        Incrementally add/remove sparse rows for forward overlapping intervals;
        restart the sum for arbitrary jumps. No dense rows-by-features prefix
        matrix is built. Validation is lazy; invalid intervals raise when read.
        """
        _positive_integer(batch_size, "batch_size")
        self._binding.validate()
        width = self.features.shape[1]
        counts = np.zeros(width)
        previous: Interval | None = None
        rows: list[csr_matrix] = []
        metadata: list[Interval] = []

        def update(start: int, stop: int, sign: int) -> None:
            a, b = self.features.indptr[[start, stop]]
            operation = np.add if sign == 1 else np.subtract
            operation.at(counts, self.features.indices[a:b], self.features.data[a:b])

        for interval in intervals:
            if len(interval) != 3 or any(
                isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                for x in interval
            ):
                raise ValueError("Intervals must be integer (sequence_index, start, stop) triples.")
            sequence, start, stop = map(int, interval)
            if sequence < 0 or sequence >= len(self.offsets) - 1:
                raise ValueError("Interval sequence index out of range.")
            base, end = self.offsets[sequence : sequence + 2]
            if not 0 <= start < stop <= end - base:
                raise ValueError("Interval must satisfy 0 <= start < stop <= sequence length.")
            if (
                previous is not None
                and previous[0] == sequence
                and previous[1] <= start < previous[2] <= stop
            ):
                update(base + previous[1], base + start, -1)
                update(base + previous[2], base + stop, 1)
            else:
                counts.fill(0)
                update(base + start, base + stop, 1)
            indices = np.flatnonzero(counts)
            rows.append(
                csr_matrix(
                    (counts[indices] / (stop - start), indices, np.array([0, len(indices)])),
                    shape=(1, width),
                )
            )
            previous = (sequence, start, stop)
            metadata.append(previous)
            if len(rows) == batch_size:
                yield self._result(rows, metadata)
                self._binding.validate()
                rows, metadata = [], []
        if rows:
            yield self._result(rows, metadata)
