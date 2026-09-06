import numpy as np

from pyidk import IsolationDistributionalKernel, IsolationKernel, pairwise_similarity


def test_distribution_mean_matches_manual_point_average(clustered_sequences):
    point_kernel = IsolationKernel(
        n_partitions=10,
        samples_per_partition=3,
        random_state=5,
    )
    idk = IsolationDistributionalKernel(point_kernel).fit(clustered_sequences)

    group_embeddings = idk.transform(clustered_sequences, aggregation="mean")
    point_embeddings = point_kernel.transform(clustered_sequences.values)

    for i in range(clustered_sequences.n_sequences):
        start, stop = clustered_sequences.offsets[i : i + 2]
        manual = np.asarray(point_embeddings[start:stop].mean(axis=0))
        np.testing.assert_allclose(group_embeddings[i].toarray(), manual)


def test_distribution_count_matches_manual_sum(clustered_sequences):
    point_kernel = IsolationKernel(
        n_partitions=6,
        samples_per_partition=3,
        random_state=5,
    )
    idk = IsolationDistributionalKernel(point_kernel).fit(clustered_sequences)

    groups = idk.transform(clustered_sequences, aggregation="count")
    points = point_kernel.transform(clustered_sequences.values)

    start, stop = clustered_sequences.offsets[0:2]
    manual = np.asarray(points[start:stop].sum(axis=0))
    np.testing.assert_allclose(groups[0].toarray(), manual)


def test_distribution_similarity_equals_average_pairwise_point_similarity(
    clustered_sequences,
):
    point_kernel = IsolationKernel(
        n_partitions=12,
        samples_per_partition=3,
        random_state=11,
    )
    idk = IsolationDistributionalKernel(point_kernel).fit(clustered_sequences)

    group_embeddings = idk.transform(clustered_sequences)
    group_similarity = idk.similarity(group_embeddings)

    point_embeddings = point_kernel.transform(clustered_sequences.values)
    point_similarity = pairwise_similarity(
        point_embeddings,
        n_partitions=point_kernel.n_partitions,
        dense=True,
    )

    for i in range(clustered_sequences.n_sequences):
        i0, i1 = clustered_sequences.offsets[i : i + 2]
        for j in range(clustered_sequences.n_sequences):
            j0, j1 = clustered_sequences.offsets[j : j + 2]
            expected = point_similarity[i0:i1, j0:j1].mean()
            assert np.isclose(group_similarity[i, j], expected)


def test_similarity_can_remain_sparse(clustered_sequences):
    point_kernel = IsolationKernel(
        n_partitions=5,
        samples_per_partition=3,
        random_state=9,
    )
    idk = IsolationDistributionalKernel(point_kernel).fit(clustered_sequences)
    embeddings = idk.transform(clustered_sequences)

    result = idk.similarity(embeddings, dense=False)
    assert result.shape == (3, 3)
