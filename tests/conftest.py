"""Shared pytest fixtures."""

import numpy as np
import pytest

from pyidk import SequenceBatch


@pytest.fixture
def simple_sequences() -> SequenceBatch:
    """Three short 2-D sequences with intentionally different lengths."""
    return SequenceBatch.from_sequences(
        [
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                ]
            ),
            np.array(
                [
                    [10.0, 100.0],
                    [11.0, 110.0],
                    [12.0, 120.0],
                ]
            ),
            np.array(
                [
                    [20.0, 200.0],
                    [21.0, 210.0],
                    [22.0, 220.0],
                    [23.0, 230.0],
                    [24.0, 240.0],
                ]
            ),
        ]
    )


@pytest.fixture
def clustered_sequences() -> SequenceBatch:
    """Small groups suitable for deterministic kernel tests."""
    return SequenceBatch.from_sequences(
        [
            np.array([[0.0], [0.1], [0.2], [0.3]]),
            np.array([[0.05], [0.15], [0.25]]),
            np.array([[5.0], [5.1], [5.2], [5.3]]),
        ]
    )
