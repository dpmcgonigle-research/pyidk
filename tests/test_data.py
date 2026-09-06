import numpy as np
import pytest

from pyidk import SequenceBatch


def test_from_sequences_builds_offsets(simple_sequences):
    assert simple_sequences.n_sequences == 3
    assert simple_sequences.n_samples == 12
    assert simple_sequences.n_features == 2
    np.testing.assert_array_equal(simple_sequences.lengths, [4, 3, 5])
    np.testing.assert_array_equal(simple_sequences.offsets, [0, 4, 7, 12])


def test_sequence_returns_expected_rows(simple_sequences):
    np.testing.assert_array_equal(
        simple_sequences.sequence(1),
        [[10.0, 100.0], [11.0, 110.0], [12.0, 120.0]],
    )


def test_negative_sequence_index(simple_sequences):
    np.testing.assert_array_equal(
        simple_sequences.sequence(-1),
        simple_sequences.sequence(2),
    )


def test_rejects_invalid_final_offset():
    with pytest.raises(ValueError, match="final offset"):
        SequenceBatch(np.zeros((3, 2)), np.array([0, 2]))


def test_with_values_preserves_boundaries(simple_sequences):
    replacement = np.ones((12, 4))
    updated = simple_sequences.with_values(replacement)
    np.testing.assert_array_equal(updated.offsets, simple_sequences.offsets)
    assert updated.values.shape == (12, 4)
