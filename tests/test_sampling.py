import numpy as np

from pyidk.partition import UniformPartitionSampler


def test_uniform_sampler_shape_and_uniqueness():
    X = np.arange(20, dtype=float).reshape(10, 2)
    sampler = UniformPartitionSampler()
    samples = sampler.sample(
        X,
        n_partitions=5,
        samples_per_partition=4,
        rng=np.random.default_rng(7),
    )

    assert samples.shape == (5, 4)
    for row in samples:
        assert np.unique(row).size == 4


def test_uniform_sampler_is_seed_deterministic():
    X = np.arange(20, dtype=float).reshape(10, 2)
    sampler = UniformPartitionSampler()

    a = sampler.sample(
        X,
        n_partitions=3,
        samples_per_partition=4,
        rng=np.random.default_rng(42),
    )
    b = sampler.sample(
        X,
        n_partitions=3,
        samples_per_partition=4,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(a, b)
