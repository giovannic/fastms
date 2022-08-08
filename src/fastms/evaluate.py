import numpy as np
import pandas as pd
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from .log import logging

def test_model(model, X_test, X_seq_test, y_test, y_scaler):
    predictions = model.predict({'input_1': X_test, 'input_2': X_seq_test})
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
