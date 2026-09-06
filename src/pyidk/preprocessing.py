"""Simple feature scaling utilities.

These classes intentionally cover only the minimal behavior needed by the initial
research codebase. They avoid adding scikit-learn as a runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_2d_float(X: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError("X must be a 2-D numeric array.")
    if array.shape[0] == 0:
        raise ValueError("X must contain at least one row.")
    return array


@dataclass(slots=True)
class Standardizer:
    """Per-feature zero-mean, unit-scale standardization."""

    mean_: NDArray[np.float64] | None = None
    scale_: NDArray[np.float64] | None = None

    def fit(self, X: ArrayLike) -> Standardizer:
        """Learn each feature's mean and population standard deviation; return self.

        ``X`` has shape ``(n_samples, n_features)`` with at least one row. Store
        one value per feature in ``mean_`` and ``scale_``. Constant features use
        a scale of 1 to avoid division by zero.
        """
        array = _as_2d_float(X)
        self.mean_ = array.mean(axis=0)
        scale = array.std(axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return ``(X - mean_) / scale_`` using the fitted feature statistics.

        ``X`` has shape ``(n_samples, n_features)`` with the same feature columns
        used in ``fit``. Return a float array of the same shape without changing
        ``X``. Raise ``RuntimeError`` if not fitted.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fit before transform.")
        array = _as_2d_float(X)
        if array.shape[1] != self.mean_.shape[0]:
            raise ValueError("X has a different number of features than fit data.")
        return (array - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Fit and standardize ``X``, shaped ``(n_samples, n_features)``.

        Return a float array of the same shape; equivalent to
        ``fit(X).transform(X)``.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Undo standardization with ``X * scale_ + mean_``.

        ``X`` contains standardized values of shape ``(n_samples, n_features)``
        in the fitted feature order. Return a float array of the same shape in
        the original units. Raise ``RuntimeError`` if not fitted.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fit before inverse_transform.")
        array = _as_2d_float(X)
        return array * self.scale_ + self.mean_


@dataclass(slots=True)
class MinMaxScaler:
    """Per-feature scaling to the interval [0, 1] for the fit data."""

    min_: NDArray[np.float64] | None = None
    range_: NDArray[np.float64] | None = None

    def fit(self, X: ArrayLike) -> MinMaxScaler:
        """Learn each feature's minimum and range; return self.

        ``X`` has shape ``(n_samples, n_features)`` with at least one row. Store
        one value per feature in ``min_`` and ``range_``. Constant features use
        a range of 1 to avoid division by zero.
        """
        array = _as_2d_float(X)
        self.min_ = array.min(axis=0)
        maximum = array.max(axis=0)
        feature_range = maximum - self.min_
        self.range_ = np.where(feature_range == 0.0, 1.0, feature_range)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return ``(X - min_) / range_`` using the fitted feature statistics.

        ``X`` has shape ``(n_samples, n_features)`` with the same feature columns
        used in ``fit``. Return a float array of the same shape without changing
        ``X``. Values outside the training range are not clipped to [0, 1].
        Raise ``RuntimeError`` if not fitted.
        """
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("MinMaxScaler must be fit before transform.")
        array = _as_2d_float(X)
        if array.shape[1] != self.min_.shape[0]:
            raise ValueError("X has a different number of features than fit data.")
        return (array - self.min_) / self.range_

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Fit and scale ``X``, shaped ``(n_samples, n_features)``, to [0, 1].

        Return a float array of the same shape, with constant feature columns
        mapped to zero. Equivalent to ``fit(X).transform(X)``.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Undo scaling with ``X * range_ + min_``.

        ``X`` contains scaled values of shape ``(n_samples, n_features)`` in the
        fitted feature order. Return a float array of the same shape in the
        original units. Raise ``RuntimeError`` if not fitted.
        """
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("MinMaxScaler must be fit before inverse_transform.")
        array = _as_2d_float(X)
        return array * self.range_ + self.min_
