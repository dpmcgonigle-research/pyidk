import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyidk import (
    outside_mass,
    pairwise_cosine,
    pairwise_js_divergence,
    pairwise_kl_divergence,
)


def categorical_oracle(x, y, partitions, epsilon=None):
    """Independent scalar probability formulas, including explicit outside mass."""
    width = len(x) // partitions
    values = []
    for i in range(partitions):
        p = list(x[i * width : (i + 1) * width])
        q = list(y[i * width : (i + 1) * width])
        p.append(1 - math.fsum(p))
        q.append(1 - math.fsum(q))
        if epsilon is None:
            m = [(a + b) / 2 for a, b in zip(p, q, strict=True)]
            value = 0.5 * math.fsum(a * math.log(a / b) for a, b in zip(p, m, strict=True) if a)
            value += 0.5 * math.fsum(a * math.log(a / b) for a, b in zip(q, m, strict=True) if a)
        else:
            p = [(a + epsilon) / (1 + epsilon * (width + 1)) for a in p]
            q = [(a + epsilon) / (1 + epsilon * (width + 1)) for a in q]
            value = math.fsum(a * math.log(a / b) for a, b in zip(p, q, strict=True))
        values.append(value)
    return math.fsum(values) / partitions


def test_cosine_identical_orthogonal_rescaling_zero_and_rectangular():
    x = csr_matrix([[3, 4], [0, 0], [-3, -4], [4, -3]])
    y = csr_matrix([[6, 8], [0, 0], [3, 4]])
    expected = [[1, 0, 1], [0, 0, 0], [-1, 0, -1], [0, 0, 0]]
    np.testing.assert_allclose(pairwise_cosine(x, y, n_partitions=1), expected, atol=1e-15)
    np.testing.assert_allclose(np.diag(pairwise_cosine(x, n_partitions=1)), [1, 0, 1, 1])
    np.testing.assert_allclose(pairwise_cosine(x, y, n_partitions=2), expected, atol=1e-15)


def test_cosine_extreme_finite_scales_and_stored_zeros():
    x = csr_matrix([[1e300, -1e300], [1e-300, -1e-300]])
    np.testing.assert_allclose(pairwise_cosine(x, n_partitions=1), 1, atol=1e-15)
    zero = csr_matrix(([0.0], [0], [0, 1]), shape=(1, 2))
    np.testing.assert_array_equal(pairwise_cosine(zero, n_partitions=1), [[0]])


def test_js_known_values_outside_and_partition_average():
    # Same fixture as sibling cartpole-idk's test_js_preserves_outside_mass.
    x = csr_matrix([[1, 0], [0, 0], [0.5, 0], [0, 1]])
    actual = pairwise_js_divergence(x, n_partitions=1)
    np.testing.assert_array_equal(actual.diagonal(), 0)
    np.testing.assert_array_equal(actual, actual.T)
    assert np.isfinite(actual).all()
    assert actual[0, 1] == pytest.approx(math.log(2))
    assert actual[0, 3] == pytest.approx(math.log(2))
    assert actual[0, 2] == pytest.approx(0.21576155433883565)
    both = csr_matrix([[1, 0, 1, 0], [1, 0, 0, 1]])
    assert pairwise_js_divergence(both, n_partitions=2)[0, 1] == pytest.approx(math.log(2) / 2)
    assert pairwise_js_divergence(csr_matrix((1, 4)), n_partitions=2)[0, 0] == 0


def test_js_known_nontrivial_and_nearly_identical_probabilities():
    x, y = np.array([0.2, 0.3, 0.1, 0.1]), np.array([0.4, 0.1, 0.6, 0.3])
    actual = pairwise_js_divergence(csr_matrix([x]), csr_matrix([y]), n_partitions=2)
    assert actual[0, 0] == pytest.approx(categorical_oracle(x, y, 2), abs=1e-15)
    near = csr_matrix([[0.5, 0.5, 0, 0], [0.5 + 1e-14, 0.5 - 1e-14, 0, 0], [0.5, 0.5, 1e-300, 0]])
    result = pairwise_js_divergence(near, n_partitions=2)
    assert np.isfinite(result).all()
    assert (result >= 0).all()
    np.testing.assert_allclose(result, 0, atol=1e-15)
    smallest = csr_matrix([[np.nextafter(0.0, 1.0), 0], [0, 0]])
    assert np.isfinite(pairwise_js_divergence(smallest, n_partitions=1)).all()


def test_kl_hand_calculation_direction_and_outside_smoothing():
    # epsilon=.25 -> p_smooth=[3/7,1/7,3/7], q_smooth=[1/7,5/7,1/7].
    x, y = csr_matrix([[0.5, 0]]), csr_matrix([[0, 1]])
    actual = pairwise_kl_divergence(x, y, n_partitions=1, epsilon=0.25)[0, 0]
    expected = 6 / 7 * math.log(3) + 1 / 7 * math.log(1 / 5)
    assert actual == pytest.approx(expected)
    reverse = pairwise_kl_divergence(y, x, n_partitions=1, epsilon=0.25)[0, 0]
    assert reverse == pytest.approx(2 / 7 * math.log(1 / 3) + 5 / 7 * math.log(5))
    assert reverse != pytest.approx(actual)
    # The outside category receives the same smoothing as each occupied cell.
    occupied = csr_matrix([[1, 0]])
    outside = csr_matrix([[0, 0]])
    assert pairwise_kl_divergence(occupied, outside, n_partitions=1, epsilon=0.125)[
        0, 0
    ] == pytest.approx((8 / 11) * math.log(9))


def test_sibling_occupancy_fixture_matches_independent_scalar_formulas():
    # Copied fixture values keep parity tests runnable without a sibling checkout.
    x = np.array([[0.5, 0, 0.25, 0.25], [0, 1, 0.5, 0], [0, 0, 0, 0]])
    js = pairwise_js_divergence(csr_matrix(x), n_partitions=2)
    kl = pairwise_kl_divergence(csr_matrix(x), n_partitions=2, epsilon=0.01)
    for i in range(len(x)):
        for j in range(len(x)):
            assert js[i, j] == pytest.approx(categorical_oracle(x[i], x[j], 2), abs=1e-15)
            assert kl[i, j] == pytest.approx(categorical_oracle(x[i], x[j], 2, 0.01), abs=1e-15)
    np.testing.assert_allclose(kl.diagonal(), 0, atol=1e-15)
    assert np.isfinite(kl).all()


@pytest.mark.parametrize("epsilon", [1e-300, np.nextafter(0.0, 1.0), 2.0, 1e300])
def test_kl_finite_positive_smoothing_extremes(epsilon):
    x = csr_matrix([[1, 0], [0, 0]])
    result = pairwise_kl_divergence(x, n_partitions=1, epsilon=epsilon)
    assert np.isfinite(result).all()
    assert (result >= 0).all()
    np.testing.assert_array_equal(result.diagonal(), 0)


@pytest.mark.parametrize("epsilon", [0, -0.1, np.inf, -np.inf, np.nan, None, True, 1j, ".1", [0.1]])
def test_kl_requires_valid_explicit_epsilon_even_for_empty_inputs(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        pairwise_kl_divergence(csr_matrix((0, 2)), n_partitions=1, epsilon=epsilon)


def test_kl_missing_epsilon():
    with pytest.raises(TypeError, match="epsilon"):
        pairwise_kl_divergence(csr_matrix((0, 2)), n_partitions=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "operation,parameters",
    [
        (pairwise_cosine, {}),
        (pairwise_js_divergence, {}),
        (pairwise_kl_divergence, {"epsilon": 0.01}),
    ],
)
def test_new_pairwise_validation_shapes_and_no_mutation(operation, parameters):
    for value in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="finite"):
            operation(csr_matrix([[value, 0.0]]), n_partitions=1, **parameters)
    with pytest.raises(ValueError, match="real"):
        operation(csr_matrix([[1j, 0j]]), n_partitions=1, **parameters)
    for partitions in [0, -1, 1.5, True, 3]:
        with pytest.raises((ValueError, TypeError)):
            operation(csr_matrix([[0, 1]]), n_partitions=partitions, **parameters)
    with pytest.raises(ValueError, match="columns"):
        operation(csr_matrix((1, 2)), csr_matrix((1, 4)), n_partitions=1, **parameters)
    with pytest.raises(TypeError, match="sparse"):
        operation(np.array([[0, 1]]), n_partitions=1, **parameters)
    for n, m in [(0, 0), (0, 3), (3, 0), (1, 1), (2, 3)]:
        assert operation(
            csr_matrix((n, 4)), csr_matrix((m, 4)), n_partitions=2, **parameters
        ).shape == (n, m)
    x = csr_matrix([[0.2, 0, 0.4, 0.1]])
    original = x.copy()
    operation(x, n_partitions=2, **parameters)
    np.testing.assert_array_equal(x.toarray(), original.toarray())


@pytest.mark.parametrize(
    "operation,parameters",
    [(outside_mass, {}), (pairwise_js_divergence, {}), (pairwise_kl_divergence, {"epsilon": 0.01})],
)
def test_common_occupancy_validation_and_roundoff(operation, parameters):
    for data in [[-0.01, 0.5], [0.7, 0.4], [2.0, 0.0], [-5e-11, 0.5]]:
        with pytest.raises(ValueError, match="occupancy"):
            operation(csr_matrix([data]), n_partitions=1, **parameters)
    # Explicit negative drift and mass just above one follow the same policy.
    x = csr_matrix([[-1e-14, 1 + 1e-14], [0.5, 0.5]])
    operation(x, n_partitions=1, **parameters)
    with pytest.raises(ValueError, match="occupancy"):
        operation(x, n_partitions=1, atol=0, **parameters)
    for tolerance in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="atol"):
            operation(x, n_partitions=1, atol=tolerance, **parameters)


@pytest.mark.parametrize(
    "operation,parameters",
    [(pairwise_js_divergence, {}), (pairwise_kl_divergence, {"epsilon": 0.01})],
)
def test_invalid_occupancies_are_rejected_even_when_other_matrix_is_empty(operation, parameters):
    invalid = csr_matrix([[2, 0]])
    empty = csr_matrix((0, 2))
    for left, right in [(invalid, empty), (empty, invalid)]:
        with pytest.raises(ValueError, match="occupancy"):
            operation(left, right, n_partitions=1, **parameters)


def test_js_roundoff_correction_includes_outside_not_region_only_normalization():
    drift = csr_matrix([[-1e-14, 1 + 1e-14]])
    corrected = csr_matrix([[0, 1]])
    np.testing.assert_allclose(
        pairwise_js_divergence(drift, corrected, n_partitions=1), 0, atol=1e-15
    )
    half = csr_matrix([[0.5, 0]])
    full = csr_matrix([[1, 0]])
    assert pairwise_js_divergence(half, full, n_partitions=1)[0, 0] > 0.2
