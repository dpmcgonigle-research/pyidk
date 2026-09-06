# Design notes

## Core boundary

The library is intentionally divided into three conceptual stages:

```text
SequenceBatch
    -> Representation
    -> IsolationKernel
    -> IsolationDistributionalKernel
```

The kernel receives only numeric feature vectors. It has no knowledge of robotics,
trajectory semantics, timestamps, force, pose, action, or other modalities.

## Why temporal handling is external

A state, transition, temporal window, or phase-augmented observation is just a different
point in feature space. Treating temporal construction as a representation stage lets the
same fitted-kernel API support:

- `x_t`
- `[x_t, x_{t+1}]`
- `[x_t, x_{t+2}]`
- `[x_t, x_{t+1}, ..., x_{t+k}]`
- `[x_t, phase_t]`

without adding separate kernel classes.

## Why sampling is an abstraction

The initial implementation uses `UniformPartitionSampler`, but the kernel depends only on
the `PartitionSampler` interface. Future samplers can therefore change which training rows
construct partitions without changing the hypersphere or embedding implementation.

Candidate future samplers:

- weighted-point sampling
- uniform-per-sequence sampling
- stratified / regional sampling

## Why CSR

The point feature map has `t * psi` columns but at most `t` non-zero entries per row.
CSR therefore stores the embedding compactly and works well for row-oriented operations
and sparse dot products.

## Why no Numba in 0.1

The computational path is intentionally readable:

1. sample rows;
2. fit nearest-neighbor radii;
3. compute query-to-center distances in chunks;
4. build sparse embeddings;
5. aggregate distributions through sparse matrix multiplication.

This gives a trustworthy reference implementation. Optimization should follow profiling,
with compiled paths tested against this implementation.

## Regional / stratified IDK

Regional behavior is intentionally deferred. The proposed extension point is a
`StratifiedPartitionSampler` rather than a spatially named abstraction. A stratum could
represent physical location, task phase, contact state, behavior cluster, or another
research-defined regime.

The first version of that extension should probably require one stratum per observation.
Overlapping strata introduce normalization questions and should be added only if a concrete
research need appears.
