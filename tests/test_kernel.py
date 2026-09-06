import numpy as np
from scipy.sparse import csr_matrix

from pyidk import IsolationKernel


def test_kernel_embedding_shape_and_sparsity():
    X = np.arange(30, dtype=float).reshape(15, 2)
    kernel = IsolationKernel(
        n_partitions=7,
        samples_per_partition=4,
        random_state=1,
    )
    embedding = kernel.fit_transform(X)

    assert isinstance(embedding, csr_matrix)
    assert embedding.shape == (15, 28)
    assert embedding.nnz <= 15 * 7

    # No row can have more than one active cell per partition.
    assert np.all(np.diff(embedding.indptr) <= 7)


def test_kernel_is_deterministic_given_seed():
    X = np.arange(40, dtype=float).reshape(20, 2)

    a = IsolationKernel(8, 5, random_state=42).fit_transform(X)
    b = IsolationKernel(8, 5, random_state=42).fit_transform(X)

    np.testing.assert_array_equal(a.toarray(), b.toarray())


def test_identical_queries_have_identical_embeddings():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    kernel = IsolationKernel(6, 3, random_state=3).fit(X)
    query = np.array([[1.25], [1.25]])

    embedding = kernel.transform(query)
    np.testing.assert_array_equal(embedding[0].toarray(), embedding[1].toarray())
