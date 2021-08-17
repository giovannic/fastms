import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

def format_runs(runs):
    eir = np.stack([entry['eir'] for entry in runs])
    actual_eir = np.mean(eir[:,-365:], axis=1)

    s_inputs = [
        'seasonal_a0',
        'seasonal_a1', 'seasonal_b1',
        'seasonal_a2', 'seasonal_b2',
        'seasonal_a3', 'seasonal_b3'
    ]

    inputs = ['average_age', 'Q0'] + s_inputs

    X = np.concatenate([
        np.stack([np.array([entry[i] for i in inputs]) for entry in runs ]),
        actual_eir[:, None]
    ], axis = 1)
        
    y = np.stack([entry['prev'] for entry in runs])[:,-365:]

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
