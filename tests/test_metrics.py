import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from pyidk import outside_mass, pairwise_distance, pairwise_similarity


def test_distance_matches_direct_scaled_euclidean_and_similarity():
    rng = np.random.default_rng(23)
    x, y = rng.normal(size=(8, 12)), rng.normal(size=(6, 12))
    np.testing.assert_allclose(
        pairwise_distance(csr_matrix(x), csr_matrix(y), n_partitions=3),
        cdist(x, y) / np.sqrt(3),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        pairwise_similarity(csr_matrix(x), csr_matrix(y), n_partitions=3), x @ y.T / 3
    )
    np.testing.assert_allclose(
        pairwise_distance(csr_matrix(x), n_partitions=3), cdist(x, x) / np.sqrt(3), atol=1e-14
    )


def test_outside_mass_support_partial_and_zero_embeddings():
    features = csr_matrix([[1, 0, 0.4, 0.6], [0.3, 0.2, 0, 0], [0, 0, 0, 0]])
    result = outside_mass(features, n_partitions=2)
    np.testing.assert_allclose(result.per_partition, [[0.0, 0.0], [0.5, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(result.mean, [0, 0.75, 1])
    assert pairwise_distance(features[2:], features[2:], n_partitions=2)[0, 0] == 0
    assert pairwise_similarity(features[2:], n_partitions=2)[0, 0] == 0


def test_distance_of_nearly_identical_rows_avoids_cancellation():
    left = np.array([[0.25, 0.75, 0.3, 0.7]])
    right = left.copy()
    right[0, 0] += 1e-10
    actual = pairwise_distance(csr_matrix(left), csr_matrix(right), n_partitions=2)
    np.testing.assert_allclose(actual, cdist(left, right) / np.sqrt(2), rtol=1e-14, atol=0)


def test_outside_mass_roundoff_and_invalid_occupancies():
    features = csr_matrix([[-1e-14, 1 + 1e-14, 0.3, 0.7]])
    np.testing.assert_allclose(outside_mass(features, n_partitions=2).mean, [0], atol=1e-14)
    # Validation does not mutate caller data.
    assert features.data[0] < 0
    for values in [[-0.01, 1, 0, 0], [0.6, 0.5, 0, 0]]:
        with pytest.raises(ValueError, match="occupancy"):
            outside_mass(csr_matrix([values]), n_partitions=2)
    for tolerance in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="atol"):
            outside_mass(features, n_partitions=2, atol=tolerance)


@pytest.mark.parametrize("operation", [pairwise_similarity, pairwise_distance, outside_mass])
def test_metric_input_validation(operation):
    for value in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="finite"):
            operation(csr_matrix([[value, 0]]), n_partitions=1)
    with pytest.raises(ValueError, match="divisible"):
        operation(csr_matrix([[1, 2, 3]]), n_partitions=2)
    with pytest.raises(ValueError, match="nonzero"):
        operation(csr_matrix((2, 0)), n_partitions=1)
    with pytest.raises(ValueError, match="real"):
        operation(csr_matrix([[1j, 0]]), n_partitions=1)
    with pytest.raises(TypeError, match="sparse"):
        operation(np.ones((2, 2)), n_partitions=1)
    for count in [0, -1, 1.5, True]:
        with pytest.raises((ValueError, TypeError)):
            operation(csr_matrix([[1, 0]]), n_partitions=count)


@pytest.mark.parametrize("operation", [pairwise_similarity, pairwise_distance])
def test_pairwise_shapes_empty_and_mismatched(operation):
    assert operation(csr_matrix((0, 4)), csr_matrix((3, 4)), n_partitions=2).shape == (0, 3)
    with pytest.raises(ValueError, match="columns"):
        operation(csr_matrix((2, 4)), csr_matrix((3, 6)), n_partitions=2)
