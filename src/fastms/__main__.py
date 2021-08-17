import os.path
import argparse
import logging
from .loading import create_sample_generator
from .model import create_model, train_model
from .evaluate import convergence_stats, test_model
from .export import save_model, save_scaler

# take I/O from cmdline
parser = argparse.ArgumentParser(description='Do some magic')
parser.add_argument('indir', type=str, default='./tp_ibm_interventions')
parser.add_argument('n', type=int, default=100)
parser.add_argument('block_size', type=int, default=20)
parser.add_argument('split', type=float, default=.8)
parser.add_argument('outdir', type=str, default='./')
parser.add_argument('epochs', type=int, default=100)
parser.add_argument('seed', type=int, default=42)
parser.add_argument('--log', type=str, default='WARNING')
args = parser.parse_args()

numeric_level = getattr(logging, args.log.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)

logging.info(f"seed set at {args.seed}")

logging.info(f"loading {args.n} samples from {args.indir}")
samples = create_sample_generator(
    args.indir,
    args.n,
    args.split,
    args.seed
)

# logging.info("hyperparameter optimisation")
# TODO

params = {
    'optimiser': 'adam',
    'n_layer': [
        365 + 5, # n_features
        365      # n_outputs
    ],
    'dropout': .1,
    'loss': 'log_cosh',
    'batch_size': 100
}
logging.info(f"evaluating params {params}")
model = create_model(**params)
train_model(
    model,
    samples.train_generator(params['batch_size']),
    args.epochs,
    args.seed
)
test_model(model, samples.test_generator(params['batch_size']))

# logging.info("calculating convergence stats")
# convergence_stats(samples)
# TODO

logging.info("saving outputs")
save_model(model, os.path.join(args.outdir, 'model'))
save_scaler(samples.X_scaler, os.path.join(args.outdir, 'X_scaler'))
save_scaler(samples.y_scaler, os.path.join(args.outdir, 'y_scaler'))

logging.info("done")
