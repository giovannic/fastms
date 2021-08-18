import os.path
import argparse
import math
import numpy as np
import pandas as pd
from .log import setup_log, logging
from .loading import create_training_generator
from .model import create_model, train_model
from .evaluate import model_error
from .hyperparameters import DEFAULT_PARAMS

# take I/O from cmdline
parser = argparse.ArgumentParser(description='Check convergence on the magic')
parser.add_argument('sample_dir', type=str, default='./tp_ibm_interventions')
parser.add_argument('n', type=int, default=100)
parser.add_argument('split', type=float, default=.8)
parser.add_argument('outdir', type=str, default='./')
parser.add_argument('epochs', type=int, default=100)
parser.add_argument('seed', type=int, default=42)
parser.add_argument('--log', type=str, default='WARNING')
args = parser.parse_args()

setup_log(args.log)

def subsample_error(samples, n_subsample):
    logging.info(f"training {n_subsample}")
    params = DEFAULT_PARAMS
    model = create_model(**params)
    train_model(
        model,
        samples.train_generator(params['batch_size'], n_subsample),
        args.epochs,
        args.seed,
        verbose=False
    )
    return model_error(model, samples.test_generator(params['batch_size']))

def run_convergence():
    logging.info(f"seed set at {args.seed}")

    n_train = math.floor(args.n * args.split)
    out_path = os.path.join(args.outdir, 'convergence.csv')
    n_points = 20
    logging.info(f"convergence on {n_train} samples from {args.sample_dir}")
    samples = create_training_generator(
        args.sample_dir,
        args.n,
        args.split,
        args.seed
    )

    errors = pd.DataFrame(
        data=[
            {
                'samples': n_subsample,
                'mse': subsample_error(samples, math.floor(n_subsample))
            }
            for n_subsample in np.linspace(10, n_train, n_points)
        ]
    ).to_csv(out_path, index=False)

    logging.info("done")

if __name__ == "__main__":
    run_convergence()
