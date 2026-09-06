# pyidk

`pyidk` is a small, extensible Python implementation of the **Isolation Kernel (IK)**
and **Isolation Distributional Kernel (IDK)**.

The first release is intentionally narrow. It provides the pieces needed to experiment
with IDK on arbitrary numeric feature vectors and variable-length sequential data without
committing the package to any particular application domain.

## Design goals

1. **Arbitrary numeric features** — the kernel operates on ordinary `N x d` arrays.
2. **Temporal representations are separate from the kernel** — states, transitions,
   windows, and phase augmentation are preprocessing choices, not separate IDK algorithms.
3. **Variable-length sequences without padding** — `SequenceBatch` stores concatenated
   observations plus sequence offsets.
4. **Sparse embeddings** — point and distribution embeddings use SciPy CSR matrices.
5. **Extensible partition construction** — sampling and partitioning are independent
   abstractions so regional/stratified sampling can be added later.
6. **Reasonable performance without premature optimization** — NumPy/SciPy vectorization
   and chunked distance calculations first; compiled/JIT paths can be added only after
   profiling demonstrates a need.
7. **Clean-room implementation** — this project is intended to be implemented from
   published algorithm descriptions and mathematical definitions, not copied from other
   IDK software implementations.

## Status

This is an **alpha-quality research implementation**. The API may change.

Implemented in 0.1.0:

- `SequenceBatch` for variable-length sequences
- feature standardization and min-max scaling
- point representations
- lagged/transition representations
- fixed-length temporal windows
- normalized phase augmentation
- uniform partition sampling
- iNNE-style hypersphere isolation partitions
- sparse Isolation Kernel point embeddings
- sparse Isolation Distributional Kernel group embeddings
- pairwise kernel similarity
- mathematical and behavioral unit tests

Deliberately deferred:

- anomaly detectors
- recovery logic
- regional / stratified partition sampling
- learned feature encoders
- modality-specific kernels
- alternative isolation mechanisms
- Numba/JIT acceleration
- streaming algorithms
- IDK² and other higher-level algorithms

## Installation

Python 3.14 is required.

Runtime installation:

```bash
pip install .
```

Development installation:

```bash
pip install .[dev]
```

Run the tests:

```bash
pytest
```

## Intuition notebooks

[Dartboard](notebooks/01_idk_intuition.ipynb) and
[sinusoids](notebooks/02_idk_sinusoids.ipynb) demonstrate IK, IDK, and temporal
representations. Each opens with a numbered outline and a Summary of the executed figures.

Install notebook dependencies with `pip install -e ".[nb]"`, then run `jupyter lab`
from the repository root using the same Python environment.

## Quick start

```python
import numpy as np

from pyidk import (
    IsolationDistributionalKernel,
    IsolationKernel,
    SequenceBatch,
    Standardizer,
    TransitionRepresentation,
)

demonstrations = SequenceBatch.from_sequences(
    [
        np.array([
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.4],
            [0.3, 0.5],
        ]),
        np.array([
            [1.0, 1.0],
            [1.1, 0.9],
            [1.2, 0.8],
        ]),
    ]
)

# Build a temporal representation: [x_t, x_{t+1}]
transitions = TransitionRepresentation(lags=(0, 1)).transform(demonstrations)

# Fit scaling only on training data.
scaler = Standardizer().fit(transitions.values)
transitions = transitions.with_values(scaler.transform(transitions.values))

point_kernel = IsolationKernel(
    n_partitions=100,
    samples_per_partition=3,
    random_state=42,
)

idk = IsolationDistributionalKernel(point_kernel)
idk.fit(transitions)

# One sparse row per demonstration.
group_embeddings = idk.transform(transitions)

# Distributional similarity matrix.
similarities = idk.similarity(group_embeddings)
```

## SequenceBatch

Real trajectories usually have different lengths. Instead of padding them to a common
length, `SequenceBatch` stores all observations contiguously:

```text
values =
    sequence 0 rows
    sequence 1 rows
    sequence 2 rows
    ...

offsets = [0, end_of_seq_0, end_of_seq_1, end_of_seq_2, ...]
```

For sequence lengths `[4, 3, 5]`:

```python
offsets == [0, 4, 7, 12]
```

This representation is compact and ensures temporal transforms never cross sequence
boundaries.

## Representations

Temporal structure is intentionally modeled *before* the kernel.

### Points

```python
PointRepresentation()
```

produces:

```text
x_t
```

### Transitions / arbitrary lags

```python
TransitionRepresentation(lags=(0, 1))
```

produces:

```text
[x_t, x_{t+1}]
```

while:

```python
TransitionRepresentation(lags=(0, 2))
```

produces:

```text
[x_t, x_{t+2}]
```

and:

```python
TransitionRepresentation(lags=(0, 1, 2))
```

produces:

```text
[x_t, x_{t+1}, x_{t+2}]
```

### Windows

```python
WindowRepresentation(length=8, stride=1)
```

flattens each eight-step window into one feature vector.

### Phase

```python
PhaseAugmentation()
```

appends normalized sequence phase in `[0, 1]` to every observation.

Representations can be composed explicitly. For example, augment observations with phase
and then construct transitions.

## Feature scaling

The default Euclidean geometry is sensitive to feature scale. Position measured in meters,
force measured in newtons, and actions measured in unrelated units should not be expected
to contribute equally without preprocessing.

The package includes two intentionally simple scalers:

```python
Standardizer()
MinMaxScaler()
```

`Standardizer` computes:

```text
x' = (x - mean) / standard_deviation
```

and is the recommended baseline.

Scaling is a research decision, not an IDK requirement. More sophisticated modality
weighting can be added later without changing the kernel.

## Isolation Kernel implementation

For each of `t` independent partitions:

1. Uniformly sample `psi` distinct training observations.
2. Use the sampled observations as hypersphere centers.
3. Give each center a radius equal to its distance to its nearest sampled neighbor.
4. For a query observation, find its nearest center.
5. Activate that center's feature if the query lies within that center's radius.
6. Otherwise the partition contributes no active feature.

The point embedding has `t * psi` columns and at most one active feature per partition.

## Isolation Distributional Kernel

For a group or trajectory `D`, the distributional embedding is the mean of its point
embeddings:

```text
Phi(D) = (1 / |D|) * sum_x Phi(x)
```

`IsolationDistributionalKernel.transform()` performs that aggregation using sequence
boundaries.

## Sparse representation

Embeddings are returned as SciPy `csr_matrix` objects. CSR is a natural representation
because each row contains very few non-zero values relative to the full `t * psi`
dimensional feature space and most downstream operations are row-oriented.

## Development roadmap

Likely post-0.1 additions:

### 0.2+
- weighted point sampling
- uniform-per-sequence sampling
- `StratifiedPartitionSampler`
- persistence / serialization
- richer diagnostics / pullback

### Later
- modality-specific distance composition
- optimized direct group aggregation
- optional JIT acceleration based on profiling
- anomaly/failure-detection utilities once research experiments establish which interfaces
  are actually useful

## References

The mathematical basis for this implementation is the published Isolation Kernel /
Isolation Distributional Kernel literature, including:

- Ting, K. M., Xu, B.-C., Washio, T., & Zhou, Z.-H. (2020).
  *Isolation Distributional Kernel: A New Tool for Point & Group Anomaly Detection.*
- Isolation Kernel work introducing finite data-dependent feature maps and iNNE-style
  hypersphere isolation mechanisms.

This README intentionally describes behavior rather than attempting to reproduce another
software package's implementation.
