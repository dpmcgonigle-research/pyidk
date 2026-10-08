# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project intends to follow semantic versioning once the public API stabilizes.

## [0.2.0] - Unreleased

### Added

- Reusable sparse point-feature caches bound to a fitted isolation basis, with
  compatibility validation and reuse across window lengths and strides.
- Mean occupancy embeddings for complete moving windows and explicit
  sequence-local, half-open intervals, with aligned interval metadata and
  bounded-memory processing.
- Public IDK-induced distance and per-partition/mean outside-mass metrics,
  alongside existing native IDK similarity, with occupancy validation.
- Blockwise all-reference and top-K similarity/distance reductions, supporting
  caller-defined exclusions, deterministic ties, reference counts, and optional
  selected-match inspection without implicit time alignment.
- Tests, a public API guide, and a runnable example covering cache reuse,
  indexing conventions, numerical correctness, and downstream plotting integration without adding
  plotting or pandas dependencies.

## [0.1.2] - 2026-09-09

### Added

- Time benchmarking in `03_idk_benchmarking.ipynb`

## [0.1.1] - 2026-09-08

### Changed

- Support Python 3.11 and newer.

## [0.1.0] - 2026-09-04

### Added

- Initial research implementation with scipy and numpy.

### Deferred

- Stratified / regional partition sampling.
- Weighted and sequence-balanced sampling.
- Anomaly detection APIs.
- Alternative isolation mechanisms.
- JIT/Numba acceleration.
