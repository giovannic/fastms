import json
import pickle
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.utils import custom_object_scope
import numpy as np
from .prob_model import RepeatLayer, beta_negative_log_likelihood

def load_model(path):
    custom_objects = {
        'beta_negative_log_likelihood': beta_negative_log_likelihood,
        'RepeatLayer': RepeatLayer
    }
    with custom_object_scope(custom_objects):
        return load_keras_model(path, compile=False)

def load_spec(path):
    with open(path, 'rb') as f:
        return json.load(f)

def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def check_parameters(parameters, spec):
    period = len(parameters[0][spec['timed_parameters'][0]])
    for p in parameters:
        assert(isinstance(p, dict))
        assert(all(name in p.keys() for name in spec['parameters']))
        assert(all(name in p.keys() for name in spec['timed_parameters']))
        ts_lengths = [
            len(p[name])
            for name in spec['timed_parameters']
        ]
        assert(all(period == ts_length for ts_length in ts_lengths))

def vectorise(parameters, spec):
    if isinstance(parameters, dict):
        parameters = [parameters]

    check_parameters(parameters, spec)

    X = np.array([
        [p[key] for key in spec['parameters']]
        for p in parameters
    ])

    period = len(parameters[0][spec['timed_parameters'][0]])
    X = np.repeat(X[:, None, :], period, axis=1)
    timed_parameters = np.array([
        [
            p[key]
            for key in spec['timed_parameters']
        ]
        for p in parameters
    ]).swapaxes(1, 2)
    X = np.concatenate(
        [
            X,
            timed_parameters
        ],
        axis = 2
    )

    return X

def predict(model, X_scaler, y_scaler, parameters, spec):
    return y_scaler.inverse_transform(
        model.predict(
            X_scaler.transform(
                vectorise(parameters, spec)
            )
        )
    )
