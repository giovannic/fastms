import math
import numpy as np
from sklearn.base import TransformerMixin

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

class GlobalScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._mean = None
        self._std = None

    def fit(self, X, **kwargs):
        self._mean = np.mean(X)
        self._std = np.std(X)
        if self._std == 0.:
            self._std = 1.
        return self
    
    def transform(self, X, **kwargs):
        return (X - self._mean) / self._std
    
    def inverse_transform(self, X, **kwargs):
        return X * self._std + self._mean

class SequenceScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._mean = None
        self._std = None

    def fit(self, X, **kwargs):
        self._mean = np.mean(X, axis=(0, 1))
        self._std = np.std(X, axis=(0, 1))
        self._std[self._std == 0.] = 1.
        return self
    
    def transform(self, X, **kwargs):
        return (X - self._mean) / self._std
    
    def inverse_transform(self, X, **kwargs):
        return X * self._std + self._mean
