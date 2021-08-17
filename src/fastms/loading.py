import glob
import json
import os
import math
from tensorflow.data import Dataset
from tensorflow import TensorSpec, float32
from sklearn.model_selection import train_test_split
from .preprocessing import format_runs, create_scaler

ENTRIES_PER_PATH = 10

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_sample_generator(*args):
    return SampleGenerator(*args)

class SampleGenerator(object):

    paths = list()
    split = .8
    X_scaler = None
    y_scaler = None

    def __init__(self, indir, n, split, seed):
        """
        -- indir the directory to scan for samples
        -- n the total number of samples to generate
        -- split the test train split
        -- seed for the sample allocation
        """
        if n == -1:
            self.paths = sorted(
                glob.glob(os.path.join(indir, 'realisation_*.json'))
            )
        else:
            n_paths = n // ENTRIES_PER_PATH + 1
            self.paths = sorted(
                glob.glob(os.path.join(indir, 'realisation_*.json'))
            )[:n_paths]

        self.seed = seed

        # TODO: multiprocessing
        X, y = format_runs(
            [run for runs in map(load_json, self.paths) for run in runs]
        )

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

    def train_generator(self, batch_size):
        return Dataset.from_tensor_slices((self.X_train, self.y_train)).shuffle(
            self.X_train.shape[0],
            seed=self.seed,
            reshuffle_each_iteration=True
        ).batch(batch_size)

    def test_generator(self, batch_size):
        return Dataset.from_tensor_slices((self.X_test, self.y_test)).shuffle(
            self.X_test.shape[0],
            seed=self.seed,
            reshuffle_each_iteration=True
        ).batch(batch_size)

class LazySampleGenerator(object):

    paths = list()
    split = .8
    X_scaler = None
    y_scaler = None

    def __init__(self, indir, n, split, block_size):
        """
        -- indir the directory to scan for samples
        -- n the total number of samples to generate
        -- split the test train split
        -- block_size the number of samples to read at once
        """
        n_paths = n // ENTRIES_PER_PATH + 1
        self.paths = sorted(
            glob.glob(os.path.join(indir, 'realisation_*.json'))
        )[:n_paths]
        self.n = n
        self.split = split
        self.block_size = block_size
        self.block_stride = block_size // ENTRIES_PER_PATH + 1

    def _load_block_from(self, i):
        return (
            entry
            for entry in load_json(paths[j])
            for j in range(i, math.min(len(paths), i + self.block_stride))
        )

    def _load_batch_from(self, i, batch_size):
        return [
            entry
            for j in range(i, i + batch_size, self.block_stride)
            for block in self._load_block_from(j)
            for entry in block
        ]

    def train_generator(self, batch_size):
        def generator():
            generated = 0
            to_generate = math.floor(self.split * self.n)

            while generated < to_generate:
                i = generated // ENTRIES_PER_PATH

                # Load a block
                runs = self._load_batch_from(i, batch_size)
                    
                # Format the runs as arrays
                X, y = format_runs(runs)

                # Truncate if we've generated too much
                if generated + X.shape[0] > to_generate:
                    X = X[:to_generate - generated]
                    y = y[:to_generate - generated]

                # Create scalers if we don't have any
                if self.X_scaler is None:
                    self.X_scaler = create_scaler(X)
                if self.y_scaler is None:
                    self.y_scaler = create_scaler(y)

                # Scale 
                X = self.X_scaler.transform(X)
                y = self.y_scaler.transform(y)
                generated = generated + X.shape[0]

                yield (X, y)

        return Dataset.from_generator(
            generator,
            output_signature = (
                TensorSpec(shape=(), dtype=float32), #TODO: shape
                TensorSpec(shape=(), dtype=float32)
            )
        )

    def test_generator(self, batch_size):

        def generator():
            generated = math.floor(self.split * self.n)
            to_generate = self.n - generated
            while generated < to_generate:
                # find out where we left off
                i = generated // ENTRIES_PER_PATH

                # Load a block
                runs = self._load_batch_from(i, batch_size)

                # Format the runs as arrays
                X, y = format_runs(runs)

                # Truncate any training data
                if generated % ENTRIES_PER_PATH > 0:
                    X = X[generated % ENTRIES_PER_PATH:]
                    y = y[generated % ENTRIES_PER_PATH:]

                # Create scalers if we don't have any
                if self.X_scaler is None:
                    self.X_scaler = create_scaler(X)
                if self.y_scaler is None:
                    self.y_scaler = create_scaler(y)

                # Scale 
                X = self.X_scaler.transform(X)
                y = self.y_scaler.transform(y)
                generated = generated + X.shape[0]

                yield (X, y)

        return Dataset.from_generator(
            generator,
            output_signature = (
                TensorSpec(shape=(), dtype=float32),
                TensorSpec(shape=(), dtype=float32)
            )
        )
