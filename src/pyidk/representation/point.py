"""Point-wise sequence representation."""

from __future__ import annotations

from pyidk.data import SequenceBatch

from .base import Representation


class PointRepresentation(Representation):
    """Use each observation directly as one kernel sample."""

    def transform(self, batch: SequenceBatch) -> SequenceBatch:
        """Return ``batch`` itself, with its values and sequence boundaries unchanged."""
        # SequenceBatch is immutable from the public API, so returning it is safe.
        return batch
