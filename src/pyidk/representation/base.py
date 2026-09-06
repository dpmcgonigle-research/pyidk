"""Representation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pyidk.data import SequenceBatch


class Representation(ABC):
    """Convert one `SequenceBatch` into another feature representation."""

    @abstractmethod
    def transform(self, batch: SequenceBatch) -> SequenceBatch:
        """Transform a sequence batch while preserving sequence boundaries."""
