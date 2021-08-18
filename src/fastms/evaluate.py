import numpy as np
import pandas as pd
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error

def model_error(model, gen):
    return model.evaluate(gen)[1]

def test_model(model, gen):
    print(model_error(model, gen))
