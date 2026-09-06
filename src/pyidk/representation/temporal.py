"""Temporal representations for sequential data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyidk.data import SequenceBatch

from .base import Representation


@dataclass(frozen=True, slots=True)
class TransitionRepresentation(Representation):
    """Concatenate observations at arbitrary non-negative lags.

    Examples
    --------
    ``lags=(0, 1)`` yields ``[x_t, x_{t+1}]``.

    ``lags=(0, 2)`` yields ``[x_t, x_{t+2}]``.

    ``lags=(0, 1, 2)`` yields a three-step representation.
    """

    lags: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        if not self.lags:
            raise ValueError("At least one lag is required.")
        if any(not isinstance(lag, int) for lag in self.lags):
            raise TypeError("lags must contain integers.")
        if any(lag < 0 for lag in self.lags):
            raise ValueError("lags must be non-negative.")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError("lags must be unique.")
        if tuple(sorted(self.lags)) != self.lags:
            raise ValueError("lags must be provided in increasing order.")

    def transform(self, batch: SequenceBatch) -> SequenceBatch:
        """Join lagged observations into feature rows within each sequence.

        An input sequence of shape ``(n, d)`` becomes
        ``(max(0, n - max(lags)), d * len(lags))``. Each output row concatenates
        observations in lag order. Return a new batch with the same number of
        sequences, including empty outputs for sequences too short for the lags.
        """
        max_lag = self.lags[-1]
        output = []

        for i in range(batch.n_sequences):
            sequence = batch.sequence(i)
            n_rows = max(0, sequence.shape[0] - max_lag)

            if n_rows == 0:
                output.append(np.empty((0, batch.n_features * len(self.lags)), dtype=float))
                continue

            parts = [sequence[lag : lag + n_rows] for lag in self.lags]
            output.append(np.concatenate(parts, axis=1))

        return SequenceBatch.from_sequences(output)


@dataclass(frozen=True, slots=True)
class WindowRepresentation(Representation):
    """Flatten fixed-length windows into one feature vector per window."""

    length: int
    stride: int = 1

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError("length must be at least 1.")
        if self.stride < 1:
            raise ValueError("stride must be at least 1.")

    def transform(self, batch: SequenceBatch) -> SequenceBatch:
        """Flatten consecutive windows without crossing sequence boundaries.

        For each input sequence of shape ``(n, d)``, take full windows starting
        at rows ``0, stride, 2 * stride, ...``. Each window becomes a row with
        ``length * d`` columns, ordered by time and then feature. Return a new
        batch with one output sequence per input; sequences shorter than
        ``length`` produce empty outputs.
        """
        output = []
        width = self.length * batch.n_features

        for i in range(batch.n_sequences):
            sequence = batch.sequence(i)
            if sequence.shape[0] < self.length:
                output.append(np.empty((0, width), dtype=float))
                continue

            starts = range(0, sequence.shape[0] - self.length + 1, self.stride)
            windows = [sequence[start : start + self.length].reshape(-1) for start in starts]
            output.append(np.asarray(windows, dtype=float))

        return SequenceBatch.from_sequences(output)


@dataclass(frozen=True, slots=True)
class PhaseAugmentation(Representation):
    """Append normalized sequence phase to each observation.

    A sequence with more than one observation receives phases linearly spaced from
    0.0 through 1.0. A one-observation sequence receives phase 0.0.
    """

    def transform(self, batch: SequenceBatch) -> SequenceBatch:
        """Append relative position within each sequence as its last feature.

        Each sequence of shape ``(n, d)`` becomes ``(n, d + 1)``. The added column
        runs from 0 to 1 at equal intervals, or is 0 for a single observation.
        Return a new batch preserving sequence lengths, including empty ones.
        """
        output = []

        for i in range(batch.n_sequences):
            sequence = batch.sequence(i)
            n = sequence.shape[0]

            if n == 0:
                output.append(np.empty((0, batch.n_features + 1), dtype=float))
            elif n == 1:
                output.append(np.column_stack((sequence, np.array([0.0]))))
            else:
                phase = np.linspace(0.0, 1.0, n)
                output.append(np.column_stack((sequence, phase)))

        return SequenceBatch.from_sequences(output)
