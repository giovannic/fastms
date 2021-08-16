import argparse
import logging
from .loading import create_train_generator, create_test_generator
from .model import train_model
from .evaluate import convergence_stats, test_model
from .export import save_model, save_scaler

# take I/O from cmdline
parser = argparse.ArgumentParser(description='Do some magic')
parser.add_argument('indir', type=str, default='./tp_ibm_interventions')
parser.add_argument('n_nodes', type=int, default=-1)
parser.add_argument('outdir', type=str, default='./')
parser.add_argument('seed', type=int, default=42)
args = parser.parse_args()
outdir = args.outdir
seed = args.seed

logging.info(f"seed set at {seed}")

logging.info(f"loading dataset from {args.indir}")
train_generator, X_scaler, y_scaler = create_train_generator(
    args.indir,
    n_nodes,
    seed
)

params = {}
logging.info(f"evaluating params {params}")

logging.info("calculating convergence stats")
convergence_stats(train_generator)

logging.info("evaluating on unseen parameters")
model = train_model(params, data_generator)
test_generator = create_test_generator(dataset, n_nodes, seed)
test_model(model, test_generator)

logging.info("saving outputs")
export(model, X_scaler, y_scaler)

logging.info("done")
