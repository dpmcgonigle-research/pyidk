import numpy as np

from pyidk import (
    PhaseAugmentation,
    PointRepresentation,
    TransitionRepresentation,
    WindowRepresentation,
)


def test_point_representation_returns_same_batch(simple_sequences):
    assert PointRepresentation().transform(simple_sequences) is simple_sequences


def test_transition_representation_does_not_cross_boundaries(simple_sequences):
    transitions = TransitionRepresentation(lags=(0, 1)).transform(simple_sequences)

    np.testing.assert_array_equal(transitions.lengths, [3, 2, 4])
    assert transitions.n_features == 4

    # Last transition of sequence 0 must end at [3, 30], not sequence 1's first row.
    np.testing.assert_array_equal(
        transitions.sequence(0)[-1],
        [2.0, 20.0, 3.0, 30.0],
    )


def test_transition_arbitrary_lag(simple_sequences):
    transitions = TransitionRepresentation(lags=(0, 2)).transform(simple_sequences)
    np.testing.assert_array_equal(transitions.lengths, [2, 1, 3])
    np.testing.assert_array_equal(
        transitions.sequence(0)[0],
        [0.0, 0.0, 2.0, 20.0],
    )


def test_window_representation(simple_sequences):
    windows = WindowRepresentation(length=3, stride=1).transform(simple_sequences)
    np.testing.assert_array_equal(windows.lengths, [2, 1, 3])
    assert windows.n_features == 6
    np.testing.assert_array_equal(
        windows.sequence(0)[0],
        [0.0, 0.0, 1.0, 10.0, 2.0, 20.0],
    )


def test_window_stride(simple_sequences):
    windows = WindowRepresentation(length=2, stride=2).transform(simple_sequences)
    np.testing.assert_array_equal(windows.lengths, [2, 1, 2])


def test_phase_augmentation(simple_sequences):
    augmented = PhaseAugmentation().transform(simple_sequences)
    assert augmented.n_features == 3
    np.testing.assert_allclose(augmented.sequence(0)[:, -1], [0.0, 1 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(augmented.sequence(1)[[0, -1], -1], [0.0, 1.0])
