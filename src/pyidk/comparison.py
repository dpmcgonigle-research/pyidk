"""Cosine and partition-averaged divergences, with shared block metric evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import rel_entr

from .metrics import _validate_occupancies
from .similarity import _distance_block, _similarity_block, _validate_embeddings

Metric = Literal["similarity", "distance", "cosine", "js", "kl"]


def _metric_parameters(metric: Metric, parameters: Mapping[str, float] | None) -> dict[str, float]:
    if metric not in ("similarity", "distance", "cosine", "js", "kl"):
        raise ValueError("metric must be 'similarity', 'distance', 'cosine', 'js', or 'kl'.")
    if parameters is not None and not isinstance(parameters, Mapping):
        raise TypeError("metric_parameters must be a mapping.")
    result = dict(parameters) if parameters is not None else {}
    allowed = {"atol", "epsilon"} if metric == "kl" else {"atol"} if metric == "js" else set()
    if result.keys() - allowed:
        raise ValueError(f"Unknown parameters for {metric}: {list(result.keys() - allowed)}")
    if metric == "kl" and "epsilon" not in result:
        raise ValueError("KL requires explicit epsilon smoothing.")
    if metric in ("js", "kl"):
        result.setdefault("atol", 1e-12)
    for name, value in result.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real scalar.")
        if not np.isfinite(value) or (value <= 0 if name == "epsilon" else value < 0):
            bound = "positive" if name == "epsilon" else "non-negative"
            raise ValueError(f"{name} must be finite and {bound}.")
        result[name] = float(value)
    return result


def _unit_rows(values: csr_matrix) -> csr_matrix:
    """Normalize full vectors, using row scaling to avoid overflow/underflow."""
    result = values.copy()
    result.sum_duplicates()
    lengths = np.diff(result.indptr)
    indices = np.repeat(np.arange(result.shape[0]), lengths)
    scale = np.zeros(result.shape[0])
    np.maximum.at(scale, indices, np.abs(result.data))
    scale[scale == 0] = 1
    result.data /= scale[indices]
    norms = np.sqrt(np.asarray(result.multiply(result).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    result.data /= norms[indices]
    return result


def _cosine_block(left: csr_matrix, right: csr_matrix) -> np.ndarray:
    """Inputs are unit-normalized rows; zero rows remain zero."""
    return np.clip((left @ right.T).toarray(), -1.0, 1.0)


def _partition_probabilities(values: csr_matrix, partition: int, psi: int) -> np.ndarray:
    """Augment validated occupancies for one partition in one row block."""
    regions = values[:, partition * psi : (partition + 1) * psi].toarray()
    outside = np.maximum(1.0 - regions.sum(axis=1), 0.0)
    probabilities = np.column_stack((regions, outside))
    # Correct only accepted floating-point drift. Outside remains a category.
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _smooth(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    categories = probabilities.shape[1]
    if epsilon <= 1:
        return (probabilities + epsilon) / (1.0 + epsilon * categories)
    # Algebraically identical, without overflow for large finite epsilon.
    return (probabilities / epsilon + 1.0) / (1.0 / epsilon + categories)


def _divergence_block(
    left: csr_matrix, right: csr_matrix, n_partitions: int, epsilon: float | None
) -> np.ndarray:
    """One row-pair block, one partition, and one category at a time.

    Extra dense memory is O(Q*R + (Q+R)*(psi+1)), never Q*R*embedding_width.
    Compute JS directly as a divergence to avoid sqrt-of-negative roundoff.
    """
    psi = left.shape[1] // n_partitions
    result = np.zeros((left.shape[0], right.shape[0]))
    for partition in range(n_partitions):
        p = _partition_probabilities(left, partition, psi)
        q = _partition_probabilities(right, partition, psi)
        if epsilon is not None:
            p, q = _smooth(p, epsilon), _smooth(q, epsilon)
        contribution = np.zeros_like(result)
        for category in range(psi + 1):
            a, b = p[:, category, None], q[None, :, category]
            if epsilon is None:
                # Equivalent to .5*(rel_entr(a,m)+rel_entr(b,m)), but avoids
                # underflow in m=(a+b)/2 for subnormal positive categories.
                total = a + b
                contribution += 0.25 * (rel_entr(2 * a, total) + rel_entr(2 * b, total))
            else:
                contribution += rel_entr(a, b)
        # True JS/KL cannot be negative; direct accumulation can lose a few ulps.
        result += np.maximum(contribution, 0.0) / n_partitions
    return result


class _MetricEvaluator:
    """Validate/prepare once, then share the same block code across all APIs."""

    def __init__(
        self,
        query: csr_matrix,
        reference: csr_matrix | None,
        n_partitions: int,
        metric: Metric,
        parameters: Mapping[str, float] | None = None,
    ) -> None:
        self.parameters = _metric_parameters(metric, parameters)
        left = _validate_embeddings(query, n_partitions)
        right = left if reference is None else _validate_embeddings(reference, n_partitions)
        if left.shape[1] != right.shape[1]:
            raise ValueError("Embedding matrices must have the same number of columns.")
        self.n_partitions = n_partitions
        self.metric = metric
        self.higher_is_better = metric in ("similarity", "cosine")
        self.left_norm = self.right_norm = None
        if metric in ("js", "kl"):
            left, _ = _validate_occupancies(left, n_partitions, self.parameters["atol"])
            right, _ = _validate_occupancies(right, n_partitions, self.parameters["atol"])
        elif metric == "cosine":
            left, right = _unit_rows(left), _unit_rows(right)
        elif metric == "distance":
            self.left_norm = np.asarray(left.multiply(left).sum(axis=1)).ravel() / n_partitions
            self.right_norm = np.asarray(right.multiply(right).sum(axis=1)).ravel() / n_partitions
        self.left, self.right = left, right

    def block(self, qa: int, qb: int, ra: int, rb: int) -> np.ndarray:
        left, right = self.left[qa:qb], self.right[ra:rb]
        if self.metric == "similarity":
            return _similarity_block(left, right, self.n_partitions).toarray()
        if self.metric == "distance":
            assert self.left_norm is not None and self.right_norm is not None
            return _distance_block(
                left, right, self.n_partitions, self.left_norm[qa:qb], self.right_norm[ra:rb]
            )
        if self.metric == "cosine":
            return _cosine_block(left, right)
        return _divergence_block(left, right, self.n_partitions, self.parameters.get("epsilon"))


def _dense_result(evaluator: _MetricEvaluator) -> np.ndarray:
    """Explicitly allocate the requested pairwise output; bound metric workspace."""
    nq, nr = evaluator.left.shape[0], evaluator.right.shape[0]
    result = np.empty((nq, nr))
    for qa in range(0, nq, 64):
        for ra in range(0, nr, 64):
            qb, rb = min(qa + 64, nq), min(ra + 64, nr)
            result[qa:qb, ra:rb] = evaluator.block(qa, qb, ra, rb)
    return result


def pairwise_cosine(
    embeddings: csr_matrix, other: csr_matrix | None = None, *, n_partitions: int
) -> np.ndarray:
    """Return full-vector cosine similarities of shape (n_query, n_reference).

    Compare with self when ``other`` is omitted. Zero-vector comparisons return
    zero, including their diagonal. No outside category is appended. Signed
    finite vectors are supported; probability validation does not apply.
    ``n_partitions`` validates feature width but does not scale cosine.
    Raw matrices must use the same fitted basis; its identity is caller-managed.
    Use ``reduce_references(metric='cosine')`` to avoid the full output matrix.
    """
    return _dense_result(_MetricEvaluator(embeddings, other, n_partitions, "cosine"))


def pairwise_js_divergence(
    embeddings: csr_matrix,
    other: csr_matrix | None = None,
    *,
    n_partitions: int,
    atol: float = 1e-12,
) -> np.ndarray:
    """Return partition-mean Jensen-Shannon divergence in nats, not its square root.

    Each partition's mean occupancies are augmented with outside probability
    ``1 - sum(regions)``. Zeros are supported without smoothing. Materially
    negative occupancies or partition totals above one are rejected; ``atol``
    permits only absolute floating-point drift, consistent with ``outside_mass``.
    After validation, clip negative roundoff, append outside, and normalize only
    to correct drift. Counts are not means and must not be passed to this API.

    Return shape (n_query, n_reference), including empty inputs; omitted ``other``
    compares with self. Raw matrices require caller-managed basis compatibility.
    Metric workspace uses row blocks and one partition/category at a time.
    Use ``reduce_references(metric='js')`` to avoid the full output matrix.
    """
    return _dense_result(_MetricEvaluator(embeddings, other, n_partitions, "js", {"atol": atol}))


def pairwise_kl_divergence(
    embeddings: csr_matrix,
    other: csr_matrix | None = None,
    *,
    n_partitions: int,
    epsilon: float,
    atol: float = 1e-12,
) -> np.ndarray:
    """Return mean partition KL(query || reference) in nats, with explicit smoothing.

    Use the same outside-augmented mean occupancies and ``atol`` validation as
    ``pairwise_js_divergence``. For each partition, smooth *all* psi+1 categories
    by ``(p + epsilon)/(1 + epsilon*(psi+1))``. ``epsilon`` must be finite and
    positive; no default is chosen. This divergence is directional.

    Return shape (n_query, n_reference); omitted ``other`` compares with self.
    Raw matrices require caller-managed basis compatibility. Empty inputs still
    undergo parameter and occupancy validation. Metric workspace is blocked.
    Use ``reduce_references(metric='kl', metric_parameters={'epsilon': value})``
    to avoid the full pairwise output matrix.
    """
    return _dense_result(
        _MetricEvaluator(embeddings, other, n_partitions, "kl", {"epsilon": epsilon, "atol": atol})
    )
