import numpy as np
import pandas as pd
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from .log import logging

def test_model(model, X_test, X_seq_test, y_test, y_scaler):
    predictions = model.predict((X_test, X_seq_test))
    error = mean_squared_error(
        predictions.reshape(predictions.shape[0], -1),
        y_test.reshape(y_test.shape[0], -1)
    )
    logging.info(f'pre-scaling error: {error}')
    actual_error = mean_squared_error(
        y_scaler.inverse_transform(predictions).reshape(
            predictions.shape[0],
            -1
        ),
        y_scaler.inverse_transform(y_test).reshape(y_test.shape[0], -1)
    )
    logging.info(f'actual error: {actual_error}')

def test_prob_model(model, X_test, X_seq_test, y_test, y_scaler, n):
    predictions = model.predict((X_test, X_seq_test) * n)
    error = mean_squared_error(
        predictions.reshape(predictions.shape[0], -1),
        y_test.reshape(y_test.shape[0], -1)
    )
    logging.info(f'pre-scaling error: {error}')
    actual_error = mean_squared_error(
        y_scaler.inverse_transform(predictions).reshape(
            predictions.shape[0],
            -1
        ),
        y_scaler.inverse_transform(y_test).reshape(y_test.shape[0], -1)
    )
    logging.info(f'actual error: {actual_error}')

    # calibration error
    # space of CDF quantiles to observe
    p = np.linspace(0.001, 1, endpoint=False)

    # observed probabilities
    dist = model((X_test, X_seq_test) * n)
    F = dist.cdf(y_test).numpy()
    p_hat = np.array([np.sum(F < pj) for pj in p]) / F.size
    cal = np.sum(np.square(p - p_hat))

    logging.info(f'calibration error: {cal}')

    try:
        shar = np.mean(dist.stddev())
    except NotImplementedError:
        try:
            shar = np.mean(dist.stddev_approx())
        except AttributeError:
            logging.warn(
                'Tensorflow probability needs to be upgraded to get sharpness info'
            )
            return

    logging.info(f'sharpness: {shar}')
