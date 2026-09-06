import numpy as np

from pyidk import MinMaxScaler, Standardizer


def test_standardizer_zero_mean_unit_scale():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = Standardizer()
    Z = scaler.fit_transform(X)

    np.testing.assert_allclose(Z.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(scaler.inverse_transform(Z), X)


def test_standardizer_handles_constant_feature():
    X = np.array([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]])
    Z = Standardizer().fit_transform(X)
    np.testing.assert_allclose(Z[:, 1], 0.0)


def test_minmax_scaler_maps_training_range():
    X = np.array([[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]])
    scaler = MinMaxScaler()
    Z = scaler.fit_transform(X)

    np.testing.assert_allclose(Z.min(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(Z.max(axis=0), [1.0, 1.0])
    np.testing.assert_allclose(scaler.inverse_transform(Z), X)
