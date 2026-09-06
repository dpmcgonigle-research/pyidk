"""Data containers for variable-length groups and sequences."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """A compact batch of variable-length sequences.

    Parameters
    ----------
    values:
        A 2-D array with shape ``(n_total_samples, n_features)`` containing all
        sequence samples concatenated row-wise.
    offsets:
        A 1-D integer array of length ``n_sequences + 1``. Sequence ``i`` occupies
        ``values[offsets[i]:offsets[i + 1]]``.

    Notes
    -----
    This representation avoids padding variable-length sequences and makes sequence
    boundaries explicit so temporal transforms cannot accidentally cross boundaries.
    """

    values: NDArray[np.float64]
    offsets: NDArray[np.integer]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        offsets = np.asarray(self.offsets)

        if values.ndim != 2:
            raise ValueError("values must be a 2-D array.")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("values must contain numeric data.")
        if offsets.ndim != 1:
            raise ValueError("offsets must be a 1-D array.")
        if not np.issubdtype(offsets.dtype, np.integer):
            raise TypeError("offsets must contain integers.")
        if offsets.size < 2:
            raise ValueError("offsets must contain at least a start and end.")
        if offsets[0] != 0:
            raise ValueError("offsets must start at 0.")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("offsets must be monotonically non-decreasing.")
        if offsets[-1] != values.shape[0]:
            raise ValueError("The final offset must equal the number of rows in values.")

        # Store predictable dtypes and make sure callers cannot mutate arrays through
        # references passed to the constructor.
        object.__setattr__(self, "values", np.asarray(values, dtype=float).copy())
        object.__setattr__(self, "offsets", np.asarray(offsets, dtype=np.int64).copy())

    @classmethod
    def from_sequences(cls, sequences: Iterable[ArrayLike]) -> SequenceBatch:
        """Construct a batch from an iterable of arrays shaped ``(length, n_features)``.

        Rows are observations and columns are features. Sequence lengths may
        differ, including zero, but all arrays must have the same feature count.
        Require at least one sequence; concatenate rows and record their boundaries.
        """
        arrays = [np.asarray(sequence, dtype=float) for sequence in sequences]
        if not arrays:
            raise ValueError("At least one sequence is required.")

        for array in arrays:
            if array.ndim != 2:
                raise ValueError("Each sequence must be a 2-D array.")

        n_features = arrays[0].shape[1]
        if any(array.shape[1] != n_features for array in arrays):
            raise ValueError("All sequences must have the same number of features.")

        lengths = np.fromiter((array.shape[0] for array in arrays), dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        values = np.concatenate(arrays, axis=0)
        return cls(values=values, offsets=offsets)

    @property
    def n_sequences(self) -> int:
        """Number of sequences in the batch."""
        return self.offsets.size - 1

    @property
    def n_samples(self) -> int:
        """Total number of samples across all sequences."""
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features per sample."""
        return self.values.shape[1]

    @property
    def lengths(self) -> NDArray[np.int64]:
        """Number of observations per sequence, as an array of shape ``(n_sequences,)``."""
        return np.diff(self.offsets)

    def sequence(self, index: int) -> NDArray[np.float64]:
        """Return sequence ``index`` as a view of shape ``(length, n_features)``.

        Negative indices count from the end. Raise ``IndexError`` if out of range.
        """
        if index < 0:
            index += self.n_sequences
        if index < 0 or index >= self.n_sequences:
            raise IndexError("sequence index out of range")
        start, stop = self.offsets[index : index + 2]
        return self.values[start:stop]

    def with_values(self, values: ArrayLike) -> SequenceBatch:
        """Return a new batch with the same boundaries and replacement values.

        ``values`` must have shape ``(n_samples, new_n_features)``. The feature
        count may change, but the total row count and sequence lengths may not.
        """
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError("replacement values must be a 2-D array.")
        if array.shape[0] != self.n_samples:
            raise ValueError("replacement values must contain the same number of rows.")
        return SequenceBatch(array, self.offsets)
