"""Blockwise reference reductions for numeric or metadata-bearing embeddings."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .comparison import Metric, _MetricEvaluator
from .windows import WindowEmbeddings, _positive_integer

Reduction = Literal["mean", "top_k"]
Eligibility = (
    NDArray[np.bool_] | Callable[[NDArray[np.int64], NDArray[np.int64]], NDArray[np.bool_]]
)


@dataclass(frozen=True, slots=True)
class ReferenceScores:
    """One reference-reduced score per query row, in original query order.

    Missing matches yield NaN and zero counts. Optional selections use global
    reference row indices. Mean selections are in reference-index order; top-K
    selections are best first, breaking exact metric ties by reference index.
    ``intervals`` is populated when scoring ``WindowEmbeddings``.
    """

    scores: NDArray[np.float64]
    eligible_reference_count: NDArray[np.int64]
    selected_count: NDArray[np.int64]
    metric: Metric
    aggregation: Reduction
    k: int | None
    selected_indices: tuple[NDArray[np.int64], ...] | None = None
    selected_values: tuple[NDArray[np.float64], ...] | None = None
    intervals: NDArray[np.int64] | None = None

    def records(self) -> Iterator[dict]:
        """Yield plain dictionaries for downstream tables; no pandas dependency."""
        for i, score in enumerate(self.scores):
            row = {
                "query_index": i,
                "metric": self.metric,
                "aggregation": self.aggregation,
                "score": float(score),
                "eligible_reference_count": int(self.eligible_reference_count[i]),
                "selected_count": int(self.selected_count[i]),
            }
            if self.intervals is not None:
                sequence, start, stop = map(int, self.intervals[i])
                row.update(query_sequence=sequence, start=start, stop=stop, row_count=stop - start)
            yield row


def reduce_references(
    query: csr_matrix | WindowEmbeddings,
    reference: csr_matrix | WindowEmbeddings,
    *,
    n_partitions: int | None = None,
    metric: Metric = "similarity",
    metric_parameters: Mapping[str, float] | None = None,
    aggregation: Reduction = "mean",
    k: int | None = None,
    eligibility: Eligibility | None = None,
    query_block_size: int = 128,
    reference_block_size: int = 1024,
    return_selected: bool = False,
) -> ReferenceScores:
    """Reduce similarities or distances without a full query-by-reference matrix.

    Pass two ``WindowEmbeddings`` for automatic basis and interval validation,
    or two sparse matrices plus ``n_partitions`` (basis identity is then the
    caller's responsibility). ``mean`` uses all eligible references. ``top_k``
    requires positive ``k`` and selects highest similarity/cosine or lowest
    distance/JS/KL, using fewer than k when necessary. No implicit alignment or
    weighting by source sequence is applied.

    ``metric_parameters={'epsilon': value}`` is required for KL. JS/KL also accept
    ``atol`` (default 1e-12) for mean-occupancy validation. Other metrics accept
    no parameters. Unknown/missing/invalid parameters are rejected even on empty
    inputs. JS is divergence in nats; KL is smoothed query-to-reference divergence.

    ``eligibility`` is either a full boolean (n_query, n_reference) mask or a
    callback receiving global query/reference index vectors and returning a
    boolean block of shape (len(query_indices), len(reference_indices)). Apply
    source identity/exclusion policy there, before neighbor selection.

    Working pairwise memory is O(query_block_size * reference_block_size), plus
    O(query_block_size * k) for top-K. JS/KL additionally hold one partition's
    categories for each row block, O((query_block_size+reference_block_size)*psi),
    and process categories individually, never a pair-by-feature tensor.
    Preparation may copy sparse inputs; it never densifies all feature rows.
    Output scores/counts are O(n_query).
    Optional selections cost O(n_query*k) for top-K or O(number of eligible
    pairs) for means; request them only when inspection warrants that memory.
    For fully streamed queries, call this on batches from ``iter_windows``.
    Callbacks must depend on indices, not on block size or invocation history.
    """
    if aggregation not in ("mean", "top_k"):
        raise ValueError("aggregation must be 'mean' or 'top_k'.")
    if aggregation == "top_k":
        if k is None:
            raise ValueError("top_k aggregation requires k.")
        _positive_integer(k, "k")
    elif k is not None:
        raise ValueError("k applies only to top_k aggregation.")
    _positive_integer(query_block_size, "query_block_size")
    _positive_integer(reference_block_size, "reference_block_size")
    if n_partitions is not None:
        _positive_integer(n_partitions, "n_partitions")
    intervals = None
    if isinstance(query, WindowEmbeddings) or isinstance(reference, WindowEmbeddings):
        if not isinstance(query, WindowEmbeddings) or not isinstance(reference, WindowEmbeddings):
            raise TypeError("Pass two WindowEmbeddings or two sparse matrices.")
        query.assert_compatible(reference)
        if n_partitions is not None and n_partitions != query.n_partitions:
            raise ValueError("n_partitions disagrees with the fitted basis metadata.")
        n_partitions = query.n_partitions
        intervals = query.intervals.copy()
        query, reference = query.embeddings, reference.embeddings
    if n_partitions is None:
        raise ValueError("Raw sparse matrices require n_partitions.")
    evaluator = _MetricEvaluator(query, reference, n_partitions, metric, metric_parameters)
    left, right = evaluator.left, evaluator.right
    nq, nr = left.shape[0], right.shape[0]
    if eligibility is not None and not callable(eligibility):
        eligibility = np.asarray(eligibility)
        if eligibility.dtype != np.bool_ or eligibility.shape != (nq, nr):
            raise ValueError("eligibility must be a boolean (n_query, n_reference) mask.")
    scores = np.full(nq, np.nan)
    eligible_counts = np.zeros(nq, dtype=np.int64)
    selected_counts = np.zeros(nq, dtype=np.int64)
    inspected_indices: list[NDArray[np.int64]] = []
    inspected_values: list[NDArray[np.float64]] = []
    for qa in range(0, nq, query_block_size):
        qb = min(qa + query_block_size, nq)
        q_indices = np.arange(qa, qb, dtype=np.int64)
        total = np.zeros(qb - qa)
        correction = np.zeros(qb - qa)
        best_indices = [np.empty(0, dtype=np.int64) for _ in q_indices]
        best_values = [np.empty(0) for _ in q_indices]
        mean_index_chunks: list[list[NDArray[np.int64]]] = [[] for _ in q_indices]
        mean_value_chunks: list[list[NDArray[np.float64]]] = [[] for _ in q_indices]
        for ra in range(0, nr, reference_block_size):
            rb = min(ra + reference_block_size, nr)
            r_indices = np.arange(ra, rb, dtype=np.int64)
            block = evaluator.block(qa, qb, ra, rb)
            if eligibility is None:
                allowed = np.ones(block.shape, dtype=bool)
            elif callable(eligibility):
                allowed = np.asarray(eligibility(q_indices, r_indices))
            else:
                allowed = eligibility[qa:qb, ra:rb]
            if allowed.dtype != np.bool_ or allowed.shape != block.shape:
                raise ValueError(
                    "Eligibility callback must return a boolean block of matching shape."
                )
            eligible_counts[qa:qb] += allowed.sum(axis=1)
            if aggregation == "mean":
                # Sum in global reference order, independent of block boundaries.
                # Compensation also reduces cancellation for generic numeric rows.
                for j in range(rb - ra):
                    value = np.where(allowed[:, j], block[:, j], 0.0) - correction
                    updated = total + value
                    correction = (updated - total) - value
                    total = updated
            if aggregation == "mean" and return_selected:
                for i in range(qb - qa):
                    mean_index_chunks[i].append(r_indices[allowed[i]])
                    mean_value_chunks[i].append(block[i, allowed[i]])
            if aggregation == "top_k":
                for i in range(qb - qa):
                    indices = np.concatenate((best_indices[i], r_indices[allowed[i]]))
                    values = np.concatenate((best_values[i], block[i, allowed[i]]))
                    ranks = -values if evaluator.higher_is_better else values
                    order = np.lexsort((indices, ranks))[:k]
                    indices, values = indices[order], values[order]
                    best_indices[i], best_values[i] = indices, values
        for i, qi in enumerate(q_indices):
            count = int(eligible_counts[qi])
            if aggregation == "mean":
                selected_counts[qi] = count
                if count:
                    scores[qi] = total[i] / count
            else:
                selected_counts[qi] = len(best_values[i])
                if len(best_values[i]):
                    scores[qi] = best_values[i].mean()
            if return_selected:
                if aggregation == "mean" and mean_index_chunks[i]:
                    best_indices[i] = np.concatenate(mean_index_chunks[i])
                    best_values[i] = np.concatenate(mean_value_chunks[i])
                inspected_indices.append(best_indices[i])
                inspected_values.append(best_values[i])
    return ReferenceScores(
        scores,
        eligible_counts,
        selected_counts,
        metric,
        aggregation,
        k,
        tuple(inspected_indices) if return_selected else None,
        tuple(inspected_values) if return_selected else None,
        intervals,
    )
