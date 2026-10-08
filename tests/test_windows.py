from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix, vstack

from pyidk import (
    IsolationDistributionalKernel,
    IsolationKernel,
    PointFeatureCache,
    SequenceBatch,
    TransitionRepresentation,
    reduce_references,
)


def make_data(d=4):
    rng = np.random.default_rng(56)
    return SequenceBatch.from_sequences([rng.normal(size=(n, d)) for n in (19, 0, 11, 2)])


def make_model(data):
    return IsolationDistributionalKernel(
        IsolationKernel(n_partitions=7, samples_per_partition=4, random_state=12)
    ).fit(data)


@pytest.mark.parametrize("d", [4, 8, 9])
def test_windows_match_independent_slices_and_reuse_point_features(d):
    data = make_data(d)
    model = make_model(data)
    transform = IsolationKernel.transform
    with patch.object(IsolationKernel, "transform", autospec=True, side_effect=transform) as spy:
        cache = PointFeatureCache.from_sequences(model.point_kernel, data)
        results = [
            cache.windows(length, stride=stride) for length, stride in [(1, 1), (4, 2), (8, 3)]
        ]
        assert spy.call_count == 1
        assert spy.call_args.args[1].shape == data.values.shape
    assert cache.n_features_in == d
    assert cache.n_partitions == 7
    assert cache.samples_per_partition == 4
    np.testing.assert_array_equal(cache.offsets, data.offsets)
    for result in results:
        slices = [data.sequence(i)[a:b] for i, a, b in result.intervals]
        expected = model.transform(SequenceBatch.from_sequences(slices))
        np.testing.assert_allclose(result.embeddings.toarray(), expected.toarray(), atol=1e-14)
        np.testing.assert_array_equal(result.row_count, result.stop - result.start)
        assert not np.any(result.sequence_index == 1)


def test_explicit_intervals_whole_sequences_and_arbitrary_order():
    data = make_data()
    model = make_model(data)
    cache = PointFeatureCache.from_sequences(model.point_kernel, data)
    windows = cache.windows(4, stride=2)
    explicit = cache.aggregate(windows.intervals)
    np.testing.assert_array_equal(explicit.embeddings.toarray(), windows.embeddings.toarray())
    full = [(i, 0, int(n)) for i, n in enumerate(data.lengths) if n]
    # Reverse order, repeat, shrink, jump, and cross sequence boundaries.
    intervals = full + [(0, 8, 14), (0, 2, 4), (0, 2, 4), (0, 2, 3), (2, 1, 6)]
    actual = cache.aggregate(intervals)
    expected = model.transform(
        SequenceBatch.from_sequences([data.sequence(i)[a:b] for i, a, b in intervals])
    )
    np.testing.assert_allclose(actual.embeddings.toarray(), expected.toarray(), atol=1e-14)
    chunks = list(cache.iter_intervals(iter(intervals), batch_size=2))
    assert max(len(chunk.intervals) for chunk in chunks) <= 2
    np.testing.assert_array_equal(
        vstack([c.embeddings for c in chunks]).toarray(), actual.embeddings.toarray()
    )
    np.testing.assert_array_equal(np.concatenate([c.intervals for c in chunks]), actual.intervals)


def test_complete_windows_and_empty_results():
    data = make_data()
    cache = PointFeatureCache.from_sequences(make_model(data).point_kernel, data)
    assert list(cache.window_intervals(10, stride=6)) == [(0, 0, 10), (0, 6, 16), (2, 0, 10)]
    assert list(cache.iter_windows(100)) == []
    result = cache.windows(100)
    assert result.embeddings.shape == (0, 28)
    assert result.intervals.shape == (0, 3)
    assert result.outside_mass().mean.shape == (0,)
    empty = SequenceBatch.from_sequences([np.empty((0, 4)), np.empty((0, 4))])
    empty_cache = PointFeatureCache.from_sequences(make_model(data).point_kernel, empty)
    assert empty_cache.features.shape == (0, 28)
    np.testing.assert_array_equal(empty_cache.offsets, [0, 0, 0])
    assert empty_cache.windows(1).embeddings.shape == (0, 28)


@pytest.mark.parametrize(
    "interval",
    [
        (-1, 0, 1),
        (4, 0, 1),
        (0, -1, 2),
        (0, 0, 20),
        (0, 2, 2),
        (0, 3, 2),
        (1, 0, 1),
        (0.0, 0, 1),
        (False, 0, 1),
        (0, 1),
    ],
)
def test_invalid_intervals(interval):
    data = make_data()
    cache = PointFeatureCache.from_sequences(make_model(data).point_kernel, data)
    with pytest.raises(ValueError):
        cache.aggregate([interval])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 0},
        {"length": 2, "stride": 0},
        {"length": 2, "batch_size": -1},
        {"length": 1.5},
        {"length": True},
    ],
)
def test_invalid_window_parameters(kwargs):
    data = make_data()
    cache = PointFeatureCache.from_sequences(make_model(data).point_kernel, data)
    with pytest.raises((ValueError, TypeError)):
        list(cache.iter_windows(**kwargs))


@pytest.mark.parametrize("change", ["refit", "centers", "radii", "psi", "t"])
def test_cache_rejects_changed_basis(change):
    data = make_data()
    kernel = make_model(data).point_kernel
    cache = PointFeatureCache.from_sequences(kernel, data)
    windows = cache.windows(3)
    if change == "refit":
        kernel.fit(data.values)
    elif change in ("centers", "radii"):
        getattr(kernel.basis_, change).flat[0] += 0.1
    elif change == "psi":
        kernel.samples_per_partition += 1
    else:
        kernel.n_partitions += 1
    with pytest.raises(ValueError, match="basis"):
        cache.windows(3)
    with pytest.raises(ValueError, match="basis"):
        windows.outside_mass()
    with pytest.raises(ValueError, match="basis"):
        reduce_references(windows, windows)


def test_distinct_basis_instances_rejected_even_with_same_seed_and_shape():
    data = make_data()
    a = PointFeatureCache.from_sequences(make_model(data).point_kernel, data)
    b = PointFeatureCache.from_sequences(make_model(data).point_kernel, data)
    with pytest.raises(ValueError, match="same fitted basis"):
        a.assert_compatible(b)
    for operation in (lambda x, y: x.similarity(y), lambda x, y: x.distance(y), reduce_references):
        with pytest.raises(ValueError, match="same fitted basis"):
            operation(a.windows(3), b.windows(3))


def test_cache_validation_and_compatible_results():
    data = make_data()
    kernel = make_model(data).point_kernel
    a = PointFeatureCache.from_sequences(kernel, data)
    b = PointFeatureCache.from_sequences(kernel, data)
    a.assert_compatible(b)
    a.assert_compatible(kernel)
    x, y = a.windows(3), b.windows(5)
    np.testing.assert_allclose(x.similarity(y), (x.embeddings @ y.embeddings.T).toarray() / 7)
    assert x.distance(y).shape == (len(x.intervals), len(y.intervals))
    with pytest.raises(RuntimeError, match="fitted"):
        PointFeatureCache.from_sequences(IsolationKernel(), data)
    with pytest.raises(ValueError, match="dimension"):
        PointFeatureCache.from_sequences(kernel, make_data(9))
    data.values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PointFeatureCache.from_sequences(kernel, data)


def test_state_and_transition_row_intervals_for_same_physical_transitions():
    states = make_data(4)
    transitions = TransitionRepresentation().transform(states)
    state_model, transition_model = make_model(states), make_model(transitions)
    state_cache = PointFeatureCache.from_sequences(state_model.point_kernel, states)
    transition_cache = PointFeatureCache.from_sequences(transition_model.point_kernel, transitions)
    # Physical transitions 2->3, 3->4, 4->5 require state rows [2,6), transition rows [2,5).
    state_window = state_cache.aggregate([(0, 2, 6)])
    transition_window = transition_cache.aggregate([(0, 2, 5)])
    assert state_window.row_count[0] == 4
    assert transition_window.row_count[0] == 3
    assert state_window.stop[0] - 1 == transition_window.stop[0]
    np.testing.assert_allclose(transitions.sequence(0)[2:5, :4], states.sequence(0)[2:5])
    np.testing.assert_allclose(transitions.sequence(0)[2:5, 4:], states.sequence(0)[3:6])


def test_iterator_is_lazy_and_detects_refit_between_batches():
    data = make_data()
    kernel = make_model(data).point_kernel
    cache = PointFeatureCache.from_sequences(kernel, data)
    consumed = []

    def intervals():
        for item in cache.window_intervals(2):
            consumed.append(item)
            yield item

    iterator = cache.iter_intervals(intervals(), batch_size=3)
    assert len(next(iterator).intervals) == 3
    assert len(consumed) == 3
    kernel.fit(data.values)
    with pytest.raises(ValueError, match="basis"):
        next(iterator)


def test_aggregation_never_densifies_point_features(monkeypatch):
    data = make_data()
    cache = PointFeatureCache.from_sequences(make_model(data).point_kernel, data)

    def forbidden(*args, **kwargs):
        raise AssertionError("Point features must remain sparse")

    monkeypatch.setattr(csr_matrix, "toarray", forbidden)
    for result in cache.iter_windows(5, stride=2, batch_size=3):
        assert result.embeddings.shape[0] <= 3


def test_cached_outside_mass_agrees_with_point_occupancy():
    data = make_data()
    kernel = make_model(data).point_kernel
    cache = PointFeatureCache.from_sequences(kernel, data)
    windows = cache.windows(4, stride=2)
    observed = windows.outside_mass()
    for i, (sequence, start, stop) in enumerate(windows.intervals):
        base = cache.offsets[sequence]
        points = cache.features[base + start : base + stop].toarray().reshape(-1, 7, 4)
        np.testing.assert_allclose(observed.per_partition[i], 1 - points.sum(axis=2).mean(axis=0))
    outside = SequenceBatch.from_sequences([np.full((5, 4), 1e10)])
    empty_features = PointFeatureCache.from_sequences(kernel, outside).windows(3)
    assert empty_features.embeddings.nnz == 0
    np.testing.assert_array_equal(empty_features.distance(empty_features), 0)
    np.testing.assert_array_equal(empty_features.outside_mass().mean, 1)


def test_window_metadata_validation():
    data = make_data()
    windows = PointFeatureCache.from_sequences(make_model(data).point_kernel, data).windows(4)
    invalid = replace(windows, embeddings=csr_matrix((len(windows.intervals), 35)))
    with pytest.raises(ValueError, match="dimensions"):
        invalid.outside_mass()
    for intervals in [np.array([0, 1, 2]), np.array([[0.0, 1.0, 2.0]]), np.array([[0, 2, 1]])]:
        invalid = replace(windows, intervals=intervals)
        with pytest.raises(ValueError, match="interval metadata"):
            reduce_references(invalid, windows)
