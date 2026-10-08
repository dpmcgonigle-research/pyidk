import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyidk import (
    IsolationKernel,
    PointFeatureCache,
    SequenceBatch,
    pairwise_distance,
    pairwise_similarity,
    reduce_references,
)
from pyidk.scoring import Eligibility


@pytest.mark.parametrize("metric", ["similarity", "distance"])
@pytest.mark.parametrize(
    "aggregation,k", [("mean", None), ("top_k", 1), ("top_k", 3), ("top_k", 99)]
)
def test_block_reductions_dense_oracle_ties_exclusions_and_block_sizes(metric, aggregation, k):
    rng = np.random.default_rng(123)
    # Quantized rows give exact ties, including two identical references.
    query = csr_matrix(rng.integers(0, 3, size=(7, 12)) / 8)
    references = rng.integers(0, 3, size=(11, 12)) / 8
    references[4] = references[0]
    references[7] = references[0]
    reference = csr_matrix(references)
    allowed = rng.random((7, 11)) > 0.25
    allowed[1] = False
    allowed[2] = False
    allowed[2, 4] = True
    allowed[0] = True
    function = pairwise_similarity if metric == "similarity" else pairwise_distance
    dense = function(query, reference, n_partitions=3)
    expected_scores = []
    expected_indices = []
    for i, row in enumerate(dense):
        indices = np.flatnonzero(allowed[i])
        if aggregation == "top_k":
            ranks = -row[indices] if metric == "similarity" else row[indices]
            indices = indices[np.lexsort((indices, ranks))[:k]]
        expected_indices.append(indices)
        expected_scores.append(row[indices].mean() if len(indices) else np.nan)
    baseline = None
    for qblock, rblock in [(1, 1), (3, 2), (5, 7), (99, 99)]:
        policies: list[Eligibility] = [allowed, lambda q, r: allowed[np.ix_(q, r)]]
        for eligibility in policies:
            result = reduce_references(
                query,
                reference,
                n_partitions=3,
                metric=metric,
                aggregation=aggregation,
                k=k,
                eligibility=eligibility,
                query_block_size=qblock,
                reference_block_size=rblock,
                return_selected=True,
            )
            np.testing.assert_allclose(result.scores, expected_scores, rtol=1e-14, atol=1e-14)
            np.testing.assert_array_equal(result.eligible_reference_count, allowed.sum(axis=1))
            np.testing.assert_array_equal(result.selected_count, [len(i) for i in expected_indices])
            assert result.selected_indices is not None
            assert result.selected_values is not None
            for i, indices in enumerate(expected_indices):
                np.testing.assert_array_equal(result.selected_indices[i], indices)
                np.testing.assert_array_equal(result.selected_values[i], dense[i, indices])
            if baseline is None:
                baseline = result
            else:
                np.testing.assert_array_equal(result.scores, baseline.scores)


def test_similarity_and_distance_select_their_own_neighbors():
    query = csr_matrix([[0.2, 0]])
    reference = csr_matrix([[1, 0], [0.2, 0]])
    similarity = reduce_references(
        query,
        reference,
        n_partitions=1,
        metric="similarity",
        aggregation="top_k",
        k=1,
        return_selected=True,
    )
    distance = reduce_references(
        query,
        reference,
        n_partitions=1,
        metric="distance",
        aggregation="top_k",
        k=1,
        return_selected=True,
    )
    assert similarity.selected_indices is not None
    assert distance.selected_indices is not None
    assert similarity.selected_indices[0].tolist() == [0]
    assert distance.selected_indices[0].tolist() == [1]


@pytest.mark.parametrize("metric", ["similarity", "distance"])
def test_exact_ties_prefer_lowest_reference_index(metric):
    result = reduce_references(
        csr_matrix((2, 4)),
        csr_matrix((7, 4)),
        n_partitions=2,
        metric=metric,
        aggregation="top_k",
        k=3,
        reference_block_size=2,
        return_selected=True,
    )
    assert result.selected_indices is not None
    for indices in result.selected_indices:
        np.testing.assert_array_equal(indices, [0, 1, 2])


@pytest.mark.parametrize("aggregation,k", [("mean", None), ("top_k", 3)])
def test_no_references_and_empty_queries(aggregation, k):
    result = reduce_references(
        csr_matrix((3, 4)),
        csr_matrix((0, 4)),
        n_partitions=2,
        aggregation=aggregation,
        k=k,
        return_selected=True,
    )
    assert np.isnan(result.scores).all()
    assert (result.eligible_reference_count == 0).all()
    assert (result.selected_count == 0).all()
    assert result.selected_indices is not None
    assert all(len(x) == 0 for x in result.selected_indices)
    empty = reduce_references(csr_matrix((0, 4)), csr_matrix((3, 4)), n_partitions=2)
    assert empty.scores.shape == (0,)
    assert empty.selected_indices is None


def test_source_exclusion_interval_metadata_and_no_time_alignment():
    data = SequenceBatch.from_sequences([np.arange(12).reshape(6, 2), np.arange(12).reshape(6, 2)])
    kernel = IsolationKernel(n_partitions=3, samples_per_partition=3, random_state=9).fit(
        data.values
    )
    cache = PointFeatureCache.from_sequences(kernel, data)
    query = cache.aggregate([(0, 0, 2)])
    # Each source has several offsets, including copied content under a separate sequence index.
    reference = cache.aggregate([(0, 0, 2), (1, 0, 2), (1, 4, 6)])
    query_sources = np.array([10])
    reference_sources = np.array([10, 10, 20])
    result = reduce_references(
        query,
        reference,
        aggregation="top_k",
        k=5,
        return_selected=True,
        eligibility=lambda q, r: query_sources[q, None] != reference_sources[None, r],
    )
    assert result.selected_indices is not None
    assert result.selected_indices[0].tolist() == [2]  # A late window can match an early query.
    assert result.eligible_reference_count.tolist() == [1]
    assert result.selected_count.tolist() == [1]
    row = list(result.records())[0]
    assert row == dict(
        query_index=0,
        query_sequence=0,
        start=0,
        stop=2,
        row_count=2,
        metric="similarity",
        aggregation="top_k",
        score=result.scores[0],
        eligible_reference_count=1,
        selected_count=1,
    )
    outside = query.outside_mass()
    assert outside.mean.shape == result.scores.shape
    raw = reduce_references(query.embeddings, reference.embeddings, n_partitions=3)
    assert "start" not in list(raw.records())[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "bad"},
        {"aggregation": "bad"},
        {"aggregation": "top_k"},
        {"aggregation": "top_k", "k": 0},
        {"aggregation": "top_k", "k": 1.5},
        {"k": 2},
        {"query_block_size": 0},
        {"reference_block_size": 0},
        {"eligibility": np.ones((2, 3))},
        {"eligibility": np.ones((3, 3), dtype=bool)},
        {"eligibility": lambda q, r: np.ones((len(q), len(r)))},
        {"eligibility": lambda q, r: np.ones((1, 1), dtype=bool)},
    ],
)
def test_reducer_validation(kwargs):
    with pytest.raises((ValueError, TypeError)):
        reduce_references(csr_matrix((2, 4)), csr_matrix((3, 4)), n_partitions=2, **kwargs)


def test_metadata_and_raw_parameter_validation():
    data = SequenceBatch.from_sequences([np.arange(20).reshape(10, 2)])
    kernel = IsolationKernel(n_partitions=2, samples_per_partition=3).fit(data.values)
    windows = PointFeatureCache.from_sequences(kernel, data).windows(3)
    with pytest.raises(ValueError, match="metadata"):
        reduce_references(windows, windows, n_partitions=3)
    with pytest.raises(TypeError, match="two"):
        reduce_references(windows, windows.embeddings)
    with pytest.raises(ValueError, match="require"):
        reduce_references(windows.embeddings, windows.embeddings)
    with pytest.raises(ValueError, match="columns"):
        reduce_references(csr_matrix((2, 4)), csr_matrix((3, 6)), n_partitions=2)


def test_pairwise_work_is_bounded_by_requested_blocks(monkeypatch):
    shapes = []
    original = csr_matrix.toarray

    def tracked(self, *args, **kwargs):
        shapes.append(self.shape)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", tracked)
    reduce_references(
        csr_matrix(np.ones((13, 8))),
        csr_matrix(np.ones((29, 8))),
        n_partitions=2,
        query_block_size=3,
        reference_block_size=5,
    )
    assert shapes
    assert all(n <= 3 and m <= 5 for n, m in shapes)
