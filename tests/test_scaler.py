import pytest
import numpy as np
from fastms.preprocessing import SequenceScaler

def test_scaler_sequence_scaled_correct_order():
    X = np.ndarray((3, 3, 3), buffer=np.arange(3.**3.))
    scaler = SequenceScaler()
    X_hat = scaler.fit_transform(X)
    X_hat_expected = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    np.testing.assert_array_almost_equal(X_hat, X_hat_expected)

def test_scaler_inverse_transform():
    X = np.ndarray((3, 3, 3), buffer=np.arange(3.**3.))
    scaler = SequenceScaler().fit(X)
    X_hat = scaler.inverse_transform(scaler.transform(X))
    np.testing.assert_array_equal(X_hat, X)

def test_scaler_no_variance():
    X = np.full((3, 3, 3), 1)
    scaler = SequenceScaler().fit(X)
    X_hat = scaler.inverse_transform(scaler.transform(X))
    np.testing.assert_array_equal(X_hat, X)

def test_scaler_variable_sequences():
    X = np.ndarray((3, 3, 3), buffer=np.arange(3.**3.))
    scaler = SequenceScaler().fit(X)
    X_trunc = X[:, 0:2, :]
    X_hat = scaler.transform(X)
    X_hat_trunc = scaler.transform(X_trunc)
    np.testing.assert_array_equal(X_hat[:, 0:2, :], X_hat_trunc)

