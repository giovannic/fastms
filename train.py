import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import json
import argparse
from pickle import dump

seed = 42
np.random.seed(seed)

# take I/O from cmdline
parser = argparse.ArgumentParser(description='Do some magic')
parser.add_argument('indir', type=str, default='./tp_ibm_realisations')
parser.add_argument('outdir', type=str, default='./')
args = parser.parse_args()
indir = args.indir
outdir = args.outdir

print('preprocessing')

eir = np.stack([entry['eir'] for node in dataset for entry in node])
actual_eir = np.mean(eir[:,-365:], axis=1)

s_inputs = [
    'seasonal_a0',
    'seasonal_a1', 'seasonal_b1',
    'seasonal_a2', 'seasonal_b2',
    'seasonal_a3', 'seasonal_b3'
]

inputs = ['average_age', 'Q0'] + s_inputs

X = np.concatenate([
    np.stack([
        np.array([entry[i] for i in inputs])
        for node in dataset
        for entry in node
    ]),
    actual_eir[:, None]
], axis = 1)
    

y = np.stack([
    entry['prev']
    for node in dataset
    for entry in node
])[:,-365:]

idx_train, idx_test = train_test_split(
    np.arange(y.shape[0]),
    test_size=0.2,
    random_state=seed
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X[idx_train])
X_test = scaler.transform(X[idx_test])
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y[idx_train])
y_test = y[idx_test]

print('hyperparameters')

n_features = X_train.shape[1]
n_output = y_train.shape[1]

def create_model(optimiser='adam', n_layer=[n_features, n_output], dropout=.0, loss='mse'):
    model = keras.Sequential()
    model.add(layers.Dense(n_layer[0], activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_layer[1], activation='tanh'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_output))
    model.compile(loss=loss, optimizer=optimiser)
    return model

losses = ['mse', 'log_cosh']
batches = [50, 100]
dropout = [0., .1]
epochs = [100, 1000]

param_grid = dict(
    optimiser=optimisers,
    epochs=epochs,
    loss=losses,
    batch_size=batches,
    dropout=dropout
)

model = KerasRegressor(build_fn=create_model, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print('convergence stats')

def convergence_stats(params, n):
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
    
pd.DataFrame(
    data=[
        convergence_stats(grid_result.best_params_, n)
        for n in np.linspace(10, X_train.shape[0], 20)
    ]
).to_csv(os.path.join(outdir, 'convergence.csv'), index=False)

print('training best model')
model = KerasRegressor(
    build_fn=create_model,
    **grid_result.best_params_
)
model.fit(X, y, verbose=False)

print('saving outputs')
model.model.save(os.path.join(outdir, 'base'))
with open(os.path.join(outdir, 'y_scaler.pkl'), 'wb') as f:
    dump(y_scaler, f)
with open(os.path.join(outdir, 'scaler.pkl'), 'wb') as f:
    dump(scaler, f)

print('done')
