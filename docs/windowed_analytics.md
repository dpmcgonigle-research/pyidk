# Windowed IDK analytics

These APIs analyze fixed-length moving windows of arbitrary numeric features.
They do not build cumulative query prefixes or infer physical timestep semantics.
The existing `SequenceBatch`, representation, kernel, and whole-sequence APIs remain
available. `WindowRepresentation` **constructs temporal feature vectors**; the
analysis intervals below **average already constructed feature rows**.

Run the complete example from the repository root:

```sh
PYTHONPATH=src python examples/windowed_analytics.py
```

It demonstrates 4D states, 8D consecutive-state features, and those features plus
one scalar input (9D), with a separate fitting population and variable-length
query/reference sequences. It fits preprocessing and a basis once per feature
representation, sweeps several durations using the same caches, excludes copied
source trajectories, and emits summaries without pandas or plotting dependencies.

## Public API

```python
from pyidk import PointFeatureCache, reduce_references

# kernel is already fitted; batches contain represented, scaled feature rows.
queries = PointFeatureCache.from_sequences(kernel, query_batch)
references = PointFeatureCache.from_sequences(kernel, reference_batch)

reference_windows = references.windows(length=12, stride=6)
for query_windows in queries.iter_windows(length=12, stride=3, batch_size=128):
    scores = reduce_references(
        query_windows, reference_windows,
        metric="similarity", aggregation="top_k", k=5,
        query_block_size=64, reference_block_size=512,
        return_selected=True,
    )
    table_rows = list(scores.records())
    support = query_windows.outside_mass()
    # support.mean[i] and table_rows[i] describe exactly the same query interval.
```

- `PointFeatureCache.from_sequences(kernel, data)` transforms each row once with
  the fitted `IsolationKernel`. It retains sparse `features`, sequence `offsets`,
  and `n_partitions`, `samples_per_partition`, and `n_features_in` metadata.
- `cache.window_intervals(length, stride=1)` enumerates complete windows as
  `(sequence_index, start, stop)` triples. Starts are `0, stride, 2*stride, ...`.
  Short/empty sequences contribute no windows. There is no padding, partial
  window, or automatically appended off-stride terminal window.
- `cache.windows(length, stride=1)` materializes their `WindowEmbeddings`.
  `cache.iter_windows(..., batch_size=256)` yields bounded batches instead.
- `cache.aggregate(intervals)` aggregates explicit triples in caller order.
  Overlaps, duplicates, and arbitrary order are supported.
  `cache.iter_intervals(intervals, batch_size=256)` consumes an iterable lazily;
  invalid intervals raise when consumed. Windows never cross sequence boundaries.
- `WindowEmbeddings` contains sparse mean occupancy `embeddings` and aligned
  `intervals`, with `sequence_index`, `start`, `stop`, and `row_count` properties.
  Empty results have zero rows and preserve the fitted feature width.
- `windows.similarity(other, dense=True)` and `windows.distance(other)` check
  basis compatibility before computing explicitly requested pairwise matrices.
  `windows.outside_mass()` returns `OutsideMass(per_partition, mean)`.
- `reduce_references(query, reference, ...)` returns `ReferenceScores`: `scores`,
  `eligible_reference_count`, `selected_count`, metric/aggregation names, and
  query `intervals`. With `return_selected=True`, it also returns tuples of
  `selected_indices` and `selected_values`, one variable-length array per query.
  `records()` yields plain dictionaries suitable for downstream tidy tables.
- The raw numerical helpers `pairwise_similarity`, `pairwise_distance`, and
  `outside_mass` take sparse matrices and an explicit `n_partitions`. The reducer
  also accepts two raw sparse matrices with `n_partitions`; raw matrices cannot
  establish basis identity, so compatibility is the caller's responsibility.

## Indexing and physical duration

All intervals are sequence-local, zero-based, half-open **representation-row**
intervals `[start, stop)`, requiring `0 <= start < stop <= sequence_length`.
Empty intervals are rejected because a mean over zero rows is undefined. An empty
interval list is valid. `row_count = stop - start`; the last representation row
consumed is `stop - 1`.

For physical transition interval `[s, e)`:

| Representation | Selected rows | Row count | Last raw state consumed |
|---|---|---|---|
| State `x_i` | `[s, e+1)` | `e-s+1` | `e` |
| Transition `[x_i, x_(i+1)]` | `[s, e)` | `e-s` | `e` |
| Transition plus scalar action/input | `[s, e)` | `e-s` | `e` |

For example, transitions `2→3`, `3→4`, `4→5` require state interval `(seq, 2, 6)`
or transition interval `(seq, 2, 5)`. A k-transition moving window uses length
`k+1` for states or `k` for consecutive-transition features. Other lags/strides
require the caller's own mapping. Feature dimension alone never determines it.

Query/reference lengths and strides may differ. References can occur anywhere;
no matching of sequence index, timestep, or phase is implicit.

## Numerical definitions

For t partitions and psi cells per partition, every mean occupancy row has
`t*psi` entries. It is the arithmetic mean of point-level isolation features.

```text
K(mu, nu) = dot(mu, nu) / t
d(mu, nu) = sqrt(max(0, K(mu,mu) + K(nu,nu) - 2*K(mu,nu)))
outside_r(mu) = 1 - sum(mu[r*psi : (r+1)*psi])
outside_mean(mu) = mean_r(outside_r(mu))
```

Similarity is native IDK, not cosine similarity, and self-similarity need not be
one. Distance is Euclidean feature distance divided by `sqrt(t)`. For nearly
identical rows, distance uses the equivalent sparse-difference calculation to
avoid cancellation in the dot-product formula.

Outside mass measures support under the fitted basis and uses no reference bank
or neighbor selection. It returns dense `(n_windows, t)` per-partition values and
`(n_windows,)` means. Occupied cells are never renormalized. Inputs must be **mean
occupancies**, not counts. Nonfinite values, materially negative cells, or
partition occupancy sums above one are rejected. Absolute roundoff tolerance
defaults to `1e-12`; deviations within it are clipped to the valid range. Counts
that happen to satisfy these bounds cannot be distinguished from means.

Two fully outside windows both have zero embeddings: their similarity and
distance are both zero, while their outside mass is one. Always inspect support
alongside distance when interpreting nominality. The finite basis and averaging
can map distinct inputs to identical embeddings; a zero distance does not establish
identical trajectories. Temporal information depends on the input representation.

## Reference aggregation, ties, and exclusions

Choose `metric="similarity"` or `"distance"` and `aggregation="mean"` or
`"top_k"`. Means use all eligible references. Top-K requires positive `k`, picks
the largest similarities or smallest distances **using each metric's own
ranking**, and averages those values. Distance means average distances, not
squared distances or distances to an averaged reference embedding.

If fewer than k references are eligible, all eligible references are selected.
No eligible references yields `NaN`, eligible count zero, selected count zero,
and empty inspection arrays. Exact ties select lower reference row indices
first, independent of block sizes. Top-K inspection arrays are best-first; mean
inspection arrays use reference-index order. No sequence balancing or sampling
is imposed: a source supplying more windows has more opportunities to contribute.

Eligibility is applied before reduction. Supply either a boolean
`(n_query, n_reference)` mask or a callback returning a boolean block. The callback
receives query/reference index vectors **within the inputs of that reducer call**:

```python
# Source IDs come from downstream provenance, not sequence numbers alone.
q_source = query_source_ids[query_windows.sequence_index]
r_source = reference_source_ids[reference_windows.sequence_index]

def eligible(q_indices, r_indices):
    return q_source[q_indices, None] != r_source[None, r_indices]

scores = reduce_references(
    query_windows, reference_windows, metric="distance",
    aggregation="top_k", k=5, eligibility=eligible, return_selected=True,
)
matching_intervals = reference_windows.intervals[scores.selected_indices[0]]
```

Assign copied/overlapping segments their original source identity to exclude
every reference from that source. Index callbacks must give the same eligibility
regardless of block size or invocation history. When iterating query batches,
build the query-source mapping for each batch, as in the runnable example.

## Reuse and memory

Cache creation never fits or updates a basis. Caches/windows retain the originating
kernel and basis instance; compatibility checks reject refits, changed t/psi, and
in-place mutations of fitted centers/radii. A basis fingerprint is checked at
aggregation/scoring boundaries and between yielded window batches. Independently
fitted bases are considered incompatible even if seeded identically. Do not
mutate cache/result arrays or change the kernel concurrently with an operation.
These are in-memory objects, not a persistence format.

Changing only analysis length, stride, or explicit intervals reuses the cache.
Changing representation, scaler, psi, t, or fitted data requires appropriate new
point features and, when the feature space or basis changes, a newly fitted model.
The library cannot identify external representation/scaler semantics; callers
must keep them fixed and track their identity, even when feature dimensions match.

For D=t*psi, the cache costs O(point-feature nonzeros + sequence offsets). Moving
window aggregation keeps one dense O(D) running sum and sparse output batches;
it never creates a dense `total_rows × D` prefix matrix. Forward overlapping
intervals add/remove sparse rows; arbitrary jumps restart the sum from cached
features. Each output scans D entries to build its sparse row. Materializing
`windows()`/`aggregate()` retains all requested embeddings; use the iterators to
avoid that. No point transform occurs during window-size sweeps.

Reference scoring retains the supplied sparse reference bank, uses pairwise
blocks of O(query_block_size × reference_block_size), and keeps up to K candidates
per query in the active block. Norm vectors cost O(n_query+n_reference). Scores
and counts cost O(n_query). Optional top-K inspection costs O(n_query*K); inspecting
all matches for mean reduction can cost O(n_query*n_reference). Full pairwise
matrices are only built by the explicit pairwise APIs. Persisting window artifacts
is optional; regenerate them from the cache when useful.

## Downstream plots and remaining integration

Reference aggregation yields **one score per query window**. At a common physical
endpoint, compute mean/median and 25th–75th percentiles across those per-query
scores. Outside-mass IQR similarly uses one partition-mean support score per query
trajectory; it is not spread across reference matches or across partitions.
IQR describes spread, not a confidence interval. Report the count of contributing
trajectories at each endpoint, especially for variable lengths and missing scores.
The example computes such summaries after mapping representation rows to endpoints.

The downstream cartpole-idk harness still owns feature/action construction,
dataset loading, fit/calibration/test splits, source provenance, reference interval
selection/balancing, physical time mapping, parameter sweeps, alarm calibration,
outcome evaluation, and plotting. Pyidk supplies generic numerical results and
interval metadata without selecting those research policies.
