import numpy as np
import pandas as pd
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error

def convergence(train_generator):
    n = int(n)
    model = KerasRegressor(
        build_fn=create_model,
        **params
    )
    model.fit(X_train[:n], y_train[:n], verbose=False)
    error = mean_squared_error(
        y_scaler.inverse_transform(model.predict(X_test)),
        y_test
    )
    return {'samples': n, 'mse': error}
    
def convergence_stats(params, X_train, y_train):
    return pd.DataFrame(
        data=[
            convergence_stats(grid_result.best_params_, n)
            for n in np.linspace(10, X_train.shape[0], 20)
        ]
    )

# TODO does this take into account scaling?
def test_model(model, gen):
    print(model.evaluate(gen))
