# pyidk

`pyidk` is a small, extensible Python implementation of the **Isolation Kernel (IK)**
and **Isolation Distributional Kernel (IDK)**. It provides the pieces needed to experiment
with IDK on arbitrary numeric feature vectors and variable-length sequential data without
committing the package to any particular application domain.

## Design goals

1. **Arbitrary numeric features** — the kernel operates on ordinary `N x d` arrays.
2. **Temporal representations are separate from the kernel** — states, transitions,
   windows, and phase augmentation are part of preprocessing.
3. **Variable-length sequences without padding** — `SequenceBatch` stores concatenated
   observations plus sequence offsets.
4. **Sparse embeddings** — point and distribution embeddings use SciPy CSR matrices.
5. **Extensible partition construction** — sampling and partitioning are independent
   abstractions so regional/stratified sampling can be added later.

## Status

This is an **alpha-quality research implementation**. The API will likely change.

Potential future additions:

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

### Via PIP
```bash
pip install .
```

Development installation:

```bash
pip install .[dev]
```

Notebook installation:

```bash
pip install .[nb]
```

### Via Makefile
```bash
make env
```

Run the tests:

```bash
pytest
```

## Intuition/Exploration notebooks

- [dartboard](notebooks/01_idk_intuition.ipynb)
- [sinusoids](notebooks/02_idk_sinusoids.ipynb) 
- [benchmarking](notebooks/03_idk_benchmarking.ipynb) 

Install notebook dependencies with `pip install -e ".[nb]"`, then run `jupyter lab`
from the repository root using the same Python environment.

## Quick start

See `examples/basic_usage.py`

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
- anomaly/failure-detection utilities once research experiments establish which interfaces are actually useful

## References

The mathematical basis for this implementation is the published Isolation Kernel /
Isolation Distributional Kernel literature, including:

- Ting, K. M., Xu, B.-C., Washio, T., & Zhou, Z.-H. (2020).
  *Isolation Distributional Kernel: A New Tool for Point & Group Anomaly Detection.*
- Isolation Kernel work introducing finite data-dependent feature maps and iNNE-style
  hypersphere isolation mechanisms.

This README intentionally describes behavior rather than attempting to reproduce another
software package's implementation.
