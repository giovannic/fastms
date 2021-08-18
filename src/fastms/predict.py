import os.path
import argparse
import json
from tensorflow.keras.models import load_model
from .loading import create_evaluating_generator
from .export import load_scaler
from .log import setup_log, logging
from .model import model_predict

# take I/O from cmdline
parser = argparse.ArgumentParser(description='Show some magic')
parser.add_argument('sample_dir', type=str, default='./tp_ibm_interventions')
parser.add_argument('model_dir', type=str, default='./outputs')
parser.add_argument('n', type=int, default=100)
parser.add_argument('split', type=float, default=.8)
parser.add_argument('outdir', type=str, default='./outputs')
parser.add_argument('seed', type=int, default=42)
parser.add_argument('--log', type=str, default='WARNING')
args = parser.parse_args()

setup_log(args.log)

def run():
    logging.info(f"seed set at {args.seed}")

    logging.info(f"loading {args.n} samples from {args.sample_dir}")
    samples = create_evaluating_generator(
        args.sample_dir,
        args.n,
        args.split,
        args.seed,
        load_scaler(os.path.join(args.model_dir, 'X_scaler'))
    )

    logging.info(f"loading model")
    model = load_model(os.path.join(args.model_dir, 'model'))

    logging.info("predicting")
    y_scaler = load_scaler(os.path.join(args.model_dir, 'y_scaler'))
    predictions = model_predict(model, samples.evaluating_generator(), y_scaler)
    truth = samples.truth()

    logging.info("saving outputs")
    results = [
        { 'prediction': predictions[i].tolist(), 'truth': truth[i].tolist() }
        for i in range(predictions.shape[0])
    ]
    with open(os.path.join(args.outdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    logging.info("done")

if __name__ == '__main__':
    run()
