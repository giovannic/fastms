import math
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

def format_runs(runs):
    # Time invariant parameters
    X = np.stack([entry['parameters'] for entry in runs])

    # Outputs
    y = np.stack([entry['outputs'] for entry in runs])

    # Time varying parameters
    period = y.shape[1]
    X = np.concatenate(
        [
            np.repeat(X[:, None, :], period, axis=1),
            [entry['timed_parameters'] for entry in runs]
        ],
        axis = 2
    )

    return (X, y)

def create_scaler(data):
    scaler = NDStandardScaler()
    scaler.fit(data)
    return scaler

class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self
    
    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X
    
    def inverse_transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.inverse_transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X
    
    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X
