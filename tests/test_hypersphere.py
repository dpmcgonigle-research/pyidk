import numpy as np

from pyidk.partition import HyperspherePartitioner


def test_radii_are_nearest_sampled_neighbor_distances():
    X = np.array([[0.0], [2.0], [5.0]])
    sample_indices = np.array([[0, 1, 2]])
    basis = HyperspherePartitioner().fit_basis(X, sample_indices)

    np.testing.assert_allclose(basis.radii[0], [2.0, 2.0, 3.0])


def test_sampled_centers_assign_to_themselves():
    X = np.array([[0.0], [2.0], [5.0]])
    sample_indices = np.array([[0, 1, 2]])
    partitioner = HyperspherePartitioner()
    basis = partitioner.fit_basis(X, sample_indices)

    assignments = partitioner.assign(X, basis)
    np.testing.assert_array_equal(assignments[:, 0], [0, 1, 2])


def test_far_point_can_map_to_no_cell():
    X = np.array([[0.0], [1.0], [10.0]])
    sample_indices = np.array([[0, 1]])
    partitioner = HyperspherePartitioner()
    basis = partitioner.fit_basis(X, sample_indices)

    assignments = partitioner.assign(np.array([[100.0]]), basis)
    assert assignments[0, 0] == -1
