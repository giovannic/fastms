from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import re
import os
import math
import argparse
from pickle import dump

seed = 42
np.random.seed(seed)

# take I/O from cmdline
parser = argparse.ArgumentParser(description='Do some magic')
parser.add_argument('gts', type=str, default='./gts')
parser.add_argument('n', type=int)
parser.add_argument('outdir', type=str, default='./')
args = parser.parse_args()
gts = args.gts
n = args.n
outdir = args.outdir

# Create some simulations data
print('generating simulations')

det = importr('ICDMM')
base = importr('base')
data = base.readRDS(gts)

def get_run(Q0, eir, seasonality):
    return det.run_model(
        ssa0 = seasonality[0],
        ssa1 = seasonality[1],
        ssa2 = seasonality[2],
        ssa3 = seasonality[3],
        ssb1 = seasonality[4],
        ssb2 = seasonality[5],
        ssb3 = seasonality[6],
        eta  = 1 / (21 * 365),
        Q0   = Q0,
        time = sim_length,
        init_EIR = eir
    )

seasonality = data[10]
n_locations = len(seasonality)
seasons = [
    seasonality[i].rx[[
        'seasonal_a0'
        'seasonal_a1',
        'seasonal_a2',
        'seasonal_a3',
        'seasonal_b1',
        'seasonal_b2',
        'seasonal_b3'
    ]]
    for i in np.random.randint(0, n_locations, n)
]

def run_row(row):
    return get_run(*row)

idx_train, idx_test = train_test_split(
    np.arange(len(dataset)),
    test_size=0.2,
    random_state=seed
)

print('done')

print('doing rainfall')

def to_rainfall(s):
    g = np.array(s[1:6:2])
    h = np.array(s[2:7:2])
    r = np.array([
        s[0] + sum([
            g[i] * np.cos(2 * math.pi * t * (i + 1) / 365) +
            h[i] * np.sin(2 * math.pi * t * (i + 1) / 365)
            for i in range(len(g))
        ])
        for t in range(365)
    ])
    return r / np.mean(r)

s_inputs = [
    'seasonal_a0',
    'seasonal_a1', 'seasonal_b1',
    'seasonal_a2', 'seasonal_b2',
    'seasonal_a3', 'seasonal_b3'
]

rainfall_scaler = StandardScaler()
rainfall = np.stack(dataset[s_inputs].apply(to_rainfall, axis = 1))
rainfall = rainfall_scaler.fit_transform(rainfall)

print('rainfall done')

print('doing y')

n_timesteps = 365
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(np.stack(dataset.iloc[idx_train].prev))
y_train = y_train[:,:n_timesteps]
y_train_batch = y_train.T.flatten()

print('y done')

print('batching sequences')

n_in = 7
default = -200

history = np.roll(y_train[:,:,None], 1, 1)
history[:,0,:] = default
x_train_sequence = np.concatenate([rainfall[idx_train,:,None], history], 2)

x_seq_batch = np.concatenate(
    [
        np.concatenate(
            [
                x_train_sequence[:, u, None] if u >= 0 else np.full(
                    (x_train_sequence.shape[0], 1, x_train_sequence.shape[2]),
                    default
                )
                for u in t - np.arange(n_in)
            ],
            1
        )
        for t in range(x_train_sequence.shape[1])
    ],
    0
)

print('batching sequences done')

print('putting it altogether')

params = ['average_age', 'total_M', 'Q0']
# preprocess data
param_scaler = StandardScaler()
x_params = param_scaler.fit_transform(dataset.iloc[idx_train][params])

x_params_batch = np.repeat(
    np.repeat(x_params[:,None,:], 365, 0),
    n_in,
    1
)

x_train_batch = np.concatenate([x_seq_batch, x_params_batch], 2)

print('done')

print('training LSTM')

def create_sequence_model(optimiser='adam', n_layer=[5, 1], dropout=.2, loss='mse', stateful=False):
    model = keras.Sequential()
    model.add(layers.Masking(mask_value=default))
    model.add(layers.LSTM(n_layer[0], dropout=dropout))
    model.add(layers.Dense(n_layer[1]))
    model.compile(loss=loss, optimizer=optimiser)
    return model

seq_model = create_sequence_model(dropout=0., stateful=True)
for epoch in range(10):
    print('epoch', epoch)
    for X, y in zip(np.split(x_train_batch, len(idx_train)), np.split(y_train_batch, len(idx_train))):
        seq_model.fit(X, y, epochs=1, batch_size=100, shuffle=False, verbose=False)
        seq_model.reset_states()


print('saving outputs')
seq_model.save(os.path.join(outdir, 'seq'))
with open(os.path.join(outdir, 'rainfall_scaler.pkl'), 'wb') as f:
    dump(rainfall_scaler, f)
with open(os.path.join(outdir, 'y_scaler.pkl'), 'wb') as f:
    dump(y_scaler, f)
with open(os.path.join(outdir, 'param_scaler.pkl'), 'wb') as f:
    dump(param_scaler, f)
dataset.iloc[idx_test][['node', 'p_row']].to_csv(os.path.join(outdir, 'test.csv'), index=False)

print('done')
