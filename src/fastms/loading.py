import glob
import json
import os
import math
import multiprocessing
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from .preprocessing import format_runs, create_scaler
import logging

ENTRIES_PER_PATH = 10

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_samples(indir, start, end):
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
    runs = [run for runs in map(load_json, paths) for run in runs]

    logging.info("formatting")
    ncpus = multiprocessing.cpu_count()
    n_chunks = len(runs) // ncpus
    with multiprocessing.Pool(ncpus) as p:
        dataset = p.map(format_runs, chunks(runs, n_chunks))

    X, y = zip(*dataset)
    X = np.concatenate(X)
    y = np.concatenate(y)
    start_index = start % ENTRIES_PER_PATH

    if end == -1:
        end_index = X.shape[0]
    else:
        end_index = end_path * ENTRIES_PER_PATH + end % ENTRIES_PER_PATH

    return X[start_index:end_index], y[start_index:end_index]

def create_training_generator(*args):
    return TrainingGenerator(*args)

def create_evaluating_generator(*args):
    return EvaluatingGenerator(*args)

class TrainingGenerator(object):

    split = .8
    X_scaler = None
    y_scaler = None
    n_features = None
    n_outputs = None

    def __init__(self, indir, n, split, seed):
        """
        -- indir the directory to scan for samples
        -- n the total number of samples to generate
        -- split the test train split
        -- seed for the sample allocation
        """
        X, y = load_samples(indir, 0, n)
        self.n_features = X.shape[2]
        self.n_outputs = y.shape[2]
        self.seed = seed

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=split,
            random_state=self.seed
        )
        self.X_scaler = create_scaler(X_train)
        self.y_scaler = create_scaler(y_train)
        self.X_train = self.X_scaler.transform(X_train)
        self.X_test = self.X_scaler.transform(X_test)
        self.y_train = self.y_scaler.transform(y_train)
        self.y_test = self.y_scaler.transform(y_test)
        logging.info("data processed")

    def train_generator(self, batch_size, subsample=None):
        d = Dataset.from_tensor_slices((self.X_train, self.y_train)).shuffle(
            self.X_train.shape[0],
            seed=self.seed,
            reshuffle_each_iteration=True
        )
        if subsample is None:
            return d.batch(batch_size)
        return d.take(subsample).batch(batch_size)

    def test_generator(self, batch_size):
        return Dataset.from_tensor_slices((self.X_test, self.y_test)).shuffle(
            self.X_test.shape[0],
            seed=self.seed,
            reshuffle_each_iteration=True
        ).batch(batch_size)

class EvaluatingGenerator(object):

    def __init__(self, indir, n, split, seed, X_scaler):
        X, self.y = load_samples(indir, math.ceil(n * split), n)
        self.n_features = X.shape[2]
        self.n_outputs = self.y.shape[2]
        self.X = X_scaler.transform(X)

    def evaluating_generator(self, batch_size=100):
        return Dataset.from_tensor_slices(self.X).batch(batch_size)

    def truth(self):
        return self.y.reshape(self.y.shape[0], -1)
