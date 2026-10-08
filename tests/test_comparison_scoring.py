import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyidk import (
    IsolationKernel,
    PointFeatureCache,
    SequenceBatch,
    pairwise_cosine,
    pairwise_js_divergence,
    pairwise_kl_divergence,
    reduce_references,
)
from pyidk.scoring import Eligibility, Metric


def scalar_oracle(x, y, metric, t, epsilon=0.01):
    if metric == "cosine":
        denominator = math.sqrt(math.fsum(v * v for v in x) * math.fsum(v * v for v in y))
        return (
            math.fsum(a * b for a, b in zip(x, y, strict=True)) / denominator
            if denominator
            else 0.0
        )
    terms: list[float] = []
    for p, q in zip(np.split(x, t), np.split(y, t), strict=True):
        a, b = list(p) + [1 - math.fsum(p)], list(q) + [1 - math.fsum(q)]
        if metric == "js":
            middle = [(u + v) / 2 for u, v in zip(a, b, strict=True)]
            terms.extend(0.5 * u * math.log(u / v) for u, v in zip(a, middle, strict=True) if u)
            terms.extend(0.5 * u * math.log(u / v) for u, v in zip(b, middle, strict=True) if u)
        else:
            a = [(v + epsilon) / (1 + epsilon * len(a)) for v in a]
            b = [(v + epsilon) / (1 + epsilon * len(b)) for v in b]
            terms.extend(u * math.log(u / v) for u, v in zip(a, b, strict=True))
    return math.fsum(terms) / t


@pytest.mark.parametrize("metric", ["cosine", "js", "kl"])
@pytest.mark.parametrize(
    "aggregation,k", [("mean", None), ("top_k", 1), ("top_k", 3), ("top_k", 99)]
)
def test_reductions_scalar_oracle_selections_masks_and_blocks(metric, aggregation, k):
    rng = np.random.default_rng(726)
    x = rng.dirichlet(np.ones(4), size=(5, 2))[:, :, :3].reshape(5, 6)
    y = rng.dirichlet(np.ones(4), size=(7, 2))[:, :, :3].reshape(7, 6)
    x[0] = 0
    y[2] = y[0]
    y[6] = y[0]
    allowed = rng.random((5, 7)) > 0.3
    allowed[0] = True
    allowed[1] = False
    allowed[2] = False
    allowed[2, 2] = True
    oracle = np.array([[scalar_oracle(a, b, metric, 2) for b in y] for a in x])
    expected_indices = []
    expected_scores = []
    for i, values in enumerate(oracle):
        indices = np.flatnonzero(allowed[i])
        if aggregation == "top_k":
            ranks = -values[indices] if metric == "cosine" else values[indices]
            indices = indices[np.lexsort((indices, ranks))[:k]]
        expected_indices.append(indices)
        expected_scores.append(values[indices].mean() if len(indices) else np.nan)
    baseline = None
    for qb, rb in [(1, 1), (2, 3), (5, 2), (99, 99)]:
        policies: list[Eligibility] = [allowed, lambda q, r: allowed[np.ix_(q, r)]]
        for policy in policies:
            result = reduce_references(
                csr_matrix(x),
                csr_matrix(y),
                n_partitions=2,
                metric=metric,
                metric_parameters={"epsilon": 0.01} if metric == "kl" else {},
                aggregation=aggregation,
                k=k,
                eligibility=policy,
                query_block_size=qb,
                reference_block_size=rb,
                return_selected=True,
            )
            np.testing.assert_allclose(result.scores, expected_scores, atol=2e-15, rtol=1e-13)
            np.testing.assert_array_equal(result.eligible_reference_count, allowed.sum(axis=1))
            np.testing.assert_array_equal(result.selected_count, [len(i) for i in expected_indices])
            assert result.selected_indices is not None and result.selected_values is not None
            for i, indices in enumerate(expected_indices):
                np.testing.assert_array_equal(result.selected_indices[i], indices)
                np.testing.assert_allclose(
                    result.selected_values[i], oracle[i, indices], atol=2e-15
                )
            if baseline is None:
                baseline = result
            else:
                np.testing.assert_array_equal(result.scores, baseline.scores)
                assert baseline.selected_values is not None
                for actual, expected in zip(
                    result.selected_values, baseline.selected_values, strict=True
                ):
                    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("metric", ["cosine", "js", "kl"])
@pytest.mark.parametrize("aggregation,k", [("mean", None), ("top_k", 3)])
def test_empty_inputs_no_matches_and_exact_ties(metric, aggregation, k):
    parameters = {"epsilon": 0.01} if metric == "kl" else None
    for nq, nr in [(0, 0), (0, 3), (3, 0)]:
        result = reduce_references(
            csr_matrix((nq, 4)),
            csr_matrix((nr, 4)),
            n_partitions=2,
            metric=metric,
            metric_parameters=parameters,
            aggregation=aggregation,
            k=k,
            return_selected=True,
        )
        assert result.scores.shape == (nq,)
        assert np.isnan(result.scores).all()
        assert (result.selected_count == 0).all()
        assert (result.eligible_reference_count == 0).all()
        assert result.selected_indices is not None
        assert all(len(i) == 0 for i in result.selected_indices)
    ties = reduce_references(
        csr_matrix((2, 4)),
        csr_matrix((6, 4)),
        n_partitions=2,
        metric=metric,
        metric_parameters=parameters,
        aggregation=aggregation,
        k=k,
        reference_block_size=2,
        return_selected=True,
    )
    assert ties.selected_indices is not None
    for indices in ties.selected_indices:
        np.testing.assert_array_equal(indices, np.arange(6 if k is None else k))
    np.testing.assert_array_equal(ties.scores, 0)


@pytest.mark.parametrize(
    "metric,parameters",
    [
        ("cosine", {"epsilon": 0.01}),
        ("similarity", {"atol": 1e-12}),
        ("distance", {"extra": 1}),
        ("js", {"epsilon": 0.1}),
        ("js", {"atol": np.nan}),
        ("js", {"atol": -1}),
        ("kl", None),
        ("kl", {}),
        ("kl", {"epsilon": 0}),
        ("kl", {"epsilon": -1}),
        ("kl", {"epsilon": np.inf}),
        ("kl", {"epsilon": np.nan}),
        ("kl", {"epsilon": None}),
        ("kl", {"epsilon": 0.1, "unknown": 1}),
        ("cosine", []),
        ("kl", {"epsilon": True}),
    ],
)
def test_reducer_metric_parameters_validate_even_without_rows(metric, parameters):
    with pytest.raises((ValueError, TypeError)):
        reduce_references(
            csr_matrix((0, 4)),
            csr_matrix((0, 4)),
            n_partitions=2,
            metric=metric,
            metric_parameters=parameters,
        )


@pytest.mark.parametrize("metric", ["js", "kl"])
def test_all_occupancies_validated_even_without_pairs_or_eligible_references(metric):
    parameters = {"epsilon": 0.01} if metric == "kl" else None
    invalid = csr_matrix([[2.0, 0.0]])
    for q, r in [(csr_matrix((0, 2)), invalid), (invalid, csr_matrix((0, 2))), (invalid, invalid)]:
        with pytest.raises(ValueError, match="occupancy"):
            reduce_references(
                q,
                r,
                n_partitions=1,
                metric=metric,
                metric_parameters=parameters,
                eligibility=np.zeros((q.shape[0], r.shape[0]), dtype=bool),
            )
    drift = csr_matrix([[-5e-11, 1.0]])
    with pytest.raises(ValueError, match="occupancy"):
        reduce_references(drift, drift, n_partitions=1, metric=metric, metric_parameters=parameters)
    result = reduce_references(
        drift,
        drift,
        n_partitions=1,
        metric=metric,
        metric_parameters={**(parameters or {}), "atol": 1e-10},
    )
    np.testing.assert_array_equal(result.scores, [0])


def test_metric_ranking_differs_and_kl_preserves_direction():
    x, y = csr_matrix([[0.2, 0]]), csr_matrix([[1, 0], [0.2, 0], [0.19, 0.001]])
    expectations: list[tuple[Metric, int]] = [
        ("similarity", 0),
        ("cosine", 0),
        ("distance", 1),
        ("js", 1),
        ("kl", 1),
    ]
    for metric, expected in expectations:
        result = reduce_references(
            x,
            y,
            n_partitions=1,
            metric=metric,
            metric_parameters={"epsilon": 0.01} if metric == "kl" else None,
            aggregation="top_k",
            k=1,
            return_selected=True,
        )
        assert result.selected_indices is not None
        assert result.selected_indices[0].tolist() == [expected]
    result = reduce_references(
        csr_matrix([[0.5, 0]]),
        csr_matrix([[0, 1]]),
        n_partitions=1,
        metric="kl",
        metric_parameters={"epsilon": 0.25},
    )
    assert result.scores[0] == pytest.approx(6 / 7 * math.log(3) + 1 / 7 * math.log(1 / 5))


@pytest.mark.parametrize("metric", ["cosine", "js", "kl"])
def test_window_metadata_and_basis_compatibility(metric):
    data = SequenceBatch.from_sequences([np.arange(20).reshape(10, 2)])
    kernel = IsolationKernel(n_partitions=2, samples_per_partition=3, random_state=2).fit(
        data.values
    )
    cache = PointFeatureCache.from_sequences(kernel, data)
    query, reference = cache.windows(3, stride=2), cache.windows(4, stride=2)
    parameters = {"epsilon": 0.01} if metric == "kl" else None
    result = reduce_references(query, reference, metric=metric, metric_parameters=parameters)
    np.testing.assert_array_equal(result.intervals, query.intervals)
    assert all(row["metric"] == metric for row in result.records())
    other = PointFeatureCache.from_sequences(
        IsolationKernel(n_partitions=2, samples_per_partition=3, random_state=2).fit(data.values),
        data,
    ).windows(4)
    with pytest.raises(ValueError, match="basis"):
        reduce_references(query, other, metric=metric, metric_parameters=parameters)


@pytest.mark.parametrize("metric", ["cosine", "js", "kl"])
def test_pairwise_and_partition_category_workspaces_are_bounded(monkeypatch, metric):
    import pyidk.comparison as module

    observed = []
    original_block = module._MetricEvaluator.block
    original_partition = module._partition_probabilities
    original_relative = module.rel_entr

    def block(self, qa, qb, ra, rb):
        assert qb - qa <= 3 and rb - ra <= 5
        observed.append((qb - qa, rb - ra))
        return original_block(self, qa, qb, ra, rb)

    def partition(values, part, psi):
        assert values.shape[0] <= 5
        assert psi == 7  # Only one of four partitions can become dense.
        return original_partition(values, part, psi)

    def relative(a, b):
        assert a.ndim == b.ndim == 2  # No pair-by-category tensor.
        assert np.broadcast_shapes(a.shape, b.shape)[0] <= 3
        assert np.broadcast_shapes(a.shape, b.shape)[1] <= 5
        return original_relative(a, b)

    monkeypatch.setattr(module._MetricEvaluator, "block", block)
    monkeypatch.setattr(module, "_partition_probabilities", partition)
    monkeypatch.setattr(module, "rel_entr", relative)
    reduce_references(
        csr_matrix(np.full((13, 28), 0.05)),
        csr_matrix(np.full((29, 28), 0.04)),
        n_partitions=4,
        metric=metric,
        metric_parameters={"epsilon": 0.01} if metric == "kl" else None,
        query_block_size=3,
        reference_block_size=5,
    )
    assert observed


@pytest.mark.parametrize(
    "operation,parameters",
    [
        (pairwise_cosine, {}),
        (pairwise_js_divergence, {}),
        (pairwise_kl_divergence, {"epsilon": 0.01}),
    ],
)
def test_explicit_pairwise_calls_bound_workspace_and_use_all_rows(
    monkeypatch, operation, parameters
):
    import pyidk.comparison as module

    original = module._MetricEvaluator.block
    calls = []

    def block(self, qa, qb, ra, rb):
        calls.append((qa, qb, ra, rb))
        assert qb - qa <= 64 and rb - ra <= 64
        return original(self, qa, qb, ra, rb)

    monkeypatch.setattr(module._MetricEvaluator, "block", block)
    x, y = csr_matrix(np.full((65, 4), 0.1)), csr_matrix(np.full((67, 4), 0.2))
    result = operation(x, y, n_partitions=2, **parameters)
    assert result.shape == (65, 67)
    assert len(calls) == 4
    np.testing.assert_allclose(result, result[0, 0])
