import math
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

WARMUP = 5
PERIOD = 39
INPUTS = ['average_age', 'Q0']
SEASONAL_INPUTS = [
    'seasonal_a0',
    'seasonal_a1', 'seasonal_b1',
    'seasonal_a2', 'seasonal_b2',
    'seasonal_a3', 'seasonal_b3'
]
TIMED_INPUTS = ['llin', 'irs', 'smc', 'rtss', 'tx', 'prop_act']
OUTPUTS = ['inc', 'prev', 'eir']
N_FEATURES = 365 + len(TIMED_INPUTS) + len(INPUTS) + 1
N_OUTPUTS = 365 * len(OUTPUTS)

def to_rainfall(s):
    g = np.array(s[1:6:2])
    h = np.array(s[2:7:2])
    return np.array([
        s[0] + sum([
            g[i] * np.cos(2 * math.pi * t * (i + 1) / 365) +
            h[i] * np.sin(2 * math.pi * t * (i + 1) / 365)
            for i in range(len(g))
        ])
        for t in range(365)
    ])

def format_runs(runs):
    rainfall = np.stack([
        to_rainfall(np.array([entry[i] for i in SEASONAL_INPUTS]))
        for entry in runs
    ])

    # Time invariant parameters
    actual_eir = np.mean(
        np.stack([entry['eir'][4*365:5*365] for entry in runs]),
        axis=1
    )
    X = np.concatenate(
        [
            np.stack([
                np.array([entry[i] for i in INPUTS])
                for entry in runs
            ]),
            rainfall,
            actual_eir[:, None]
        ],
        axis=1
    )

    # Time varying parameters
    X = np.concatenate(
        [
            np.repeat(X[:, None, :], PERIOD, axis=1),
            np.array(
                [
                    [
                        entry[i] if i != 'rtss' else [0] * 19 + entry[i]
                        for i in TIMED_INPUTS
                    ]
                    for entry in runs
                ]
            ).swapaxes(1, 2)
        ],
        axis = 2
    )

    # Outputs
    y = np.stack(
        [
            [
                [v if v != 'NA' else 0 for v in entry[o][WARMUP*365:]]
                for o in OUTPUTS
            ]
            for entry in runs
        ]
    ).swapaxes(1, 2)

    # combine outputs into (samples, timesteps, (len(outputs) * year))
    y_yearly = np.stack(
        np.split(y, PERIOD, axis=1)
    ).transpose((1, 0, 3, 2)).reshape((X.shape[0], PERIOD, -1))

    return (X, y_yearly)

def split_y(y):
    return dict(zip(
        OUTPUTS,
        map(lambda a: a.tolist(), np.split(y, len(OUTPUTS)))
    ))

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
