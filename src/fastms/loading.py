import glob
import json
import os
import math
import multiprocessing
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from .preprocessing import format_runs, SequenceScaler, DummyScaler
import logging

ENTRIES_PER_PATH = 10

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def truncate_run(run, t):
    if (t != -1):
        run['timed_parameters'] = run['timed_parameters'][:t]
        run['outputs'] = run['outputs'][:t]
    return run

def load_samples(indir, start, end, truncate):
    paths = sorted(
        glob.glob(os.path.join(indir, 'realisation_*.json'))
    )
    start_path = start // ENTRIES_PER_PATH
    if end == -1:
        end_path = len(paths)
    else:
        end_path = end // ENTRIES_PER_PATH + 1
    paths = paths[start_path:end_path]

    logging.info("reading in data")
    runs = [
        truncate_run(run, truncate)
        for runs in map(load_json, paths)
        for run in runs
    ]

    logging.info("formatting")
    ncpus = multiprocessing.cpu_count()
    n_chunks = len(runs) // ncpus
    with multiprocessing.Pool(ncpus) as p:
        dataset = p.map(format_runs, chunks(runs, n_chunks))

    X, X_seq, y = zip(*dataset)
    X = np.concatenate(X)
    X_seq = np.concatenate(X_seq)
    y = np.concatenate(y)
    start_index = start % ENTRIES_PER_PATH

    if end == -1:
        end_index = X.shape[0]
    else:
        end_index = end_path * ENTRIES_PER_PATH + end % ENTRIES_PER_PATH

    return X[start_index:end_index], X_seq[start_index:end_index], y[start_index:end_index]

def create_training_generator(*args):
    return TrainingGenerator(*args)

class TrainingGenerator(object):

    split = .8
    X_scaler = None
    X_seq_scaler = None
    y_scaler = None
    n_features = None
    n_outputs = None
    n_timesteps = None

    def __init__(self, indir, n, split, seed, truncate, scale_y):
        """
        -- indir the directory to scan for samples
        -- n the total number of samples to generate
        -- split the test train split
        -- seed for the sample allocation
        -- truncate whether to truncate the timeseries
        """
        X, X_seq, y = load_samples(indir, 0, n, truncate)
        self.n_static_features = X.shape[1]
        self.n_seq_features = X_seq.shape[2]
        self.n_outputs = y.shape[2]
        self.n_timesteps = y.shape[1]
        self.seed = seed

        (
            X_train,
            X_test,
            X_seq_train,
            X_seq_test,
            y_train,
            y_test
        ) = train_test_split(
            X,
            X_seq,
            y,
            train_size=split,
            random_state=self.seed
        )
        self.X_scaler = StandardScaler().fit(X_train)
        self.X_seq_scaler = SequenceScaler().fit(X_seq_train)
        self.X_train = self.X_scaler.transform(X_train)
        self.X_test = self.X_scaler.transform(X_test)
        self.X_seq_train = self.X_seq_scaler.transform(X_seq_train)
        self.X_seq_test = self.X_seq_scaler.transform(X_seq_test)
        if scale_y:
            self.y_scaler = SequenceScaler().fit(y_train)
        else:
            self.y_scaler = DummyScaler().fit(y_train)
        self.y_train = self.y_scaler.transform(y_train)
        self.y_test = self.y_scaler.transform(y_test)
        logging.info("data processed")

    def train_generator(self, batch_size, subsample=None, seed=None):
        if seed is None:
            seed = self.seed

        d = Dataset.from_tensor_slices(
            (
                (self.X_train, self.X_seq_train),
                self.y_train
            )
        ).shuffle(
            self.X_train.shape[0],
            seed=self.seed,
            reshuffle_each_iteration=True
        )
        if subsample is None:
            return d.batch(batch_size)
        return d.take(subsample).batch(batch_size)

    def truth(self):
        return self.y_scaler.inverse_transform(self.y_test)
