import json
import pickle
from tensorflow.keras.models import load_model

def load_model(path):
    return load_model(path)

def load_spec(path):
    with open(path, 'rb') as f:
        return json.load(f)

def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def check_parameters(parameters, spec):
    period = len(parameters[0]['timed_parameters'])
    for p in parameters:
        assert(isinstance(p, dict))
        assert('p' in p.keys())
        assert('timed_parameters' in p.keys())
        assert(len(p['parameters']) == len(spec['parameters']))
        assert(len(p['timed_parameters'][0]) == len(spec['timed_parameters']))
        ts_lengths = [
            len(timeseries)
            for timeseries in p['timed_parameters'].values()
        ]
        assert(all(period== ts_length for ts_length in ts_lengths))

def vectorise(parameters, spec):
    if isinstance(parameters, dict):
        parameters = [parameters]

    check_parameters(parameters, spec)

    X = np.stack([p['parameters'][key] for key in spec['parameters']])

    period = len(parameters[0]['timed_parameters'])
    X = np.repeat(X[:, None, :], period, axis=1)
    timed_parameters = np.array([
        [
            p['timed_parameters'][key]
            for key in spec['timed_parameters']
        ]
        for p in parameters
    ]).T
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
