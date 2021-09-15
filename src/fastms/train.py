import os.path
import argparse
import json
from tensorflow.keras.layers import GRU
from .log import setup_log, logging
from .loading import create_training_generator
from .model import create_model, create_ed_model, train_model, model_predict
from .evaluate import test_model
from .export import save_model, save_scaler
from .hyperparameters import default_params, default_ed_params

# take I/O from cmdline
parser = argparse.ArgumentParser(description='Do some magic')
parser.add_argument('sample_dir', type=str, default='./tp_ibm_interventions')
parser.add_argument('n', type=int, default=100)
parser.add_argument('split', type=float, default=.8)
parser.add_argument('outdir', type=str, default='./')
parser.add_argument('epochs', type=int, default=100)
parser.add_argument('seed', type=int, default=42)
parser.add_argument('--log', type=str, default='WARNING')
parser.add_argument('--multigpu', type=bool, default=False)
parser.add_argument('--GRU', type=bool, default=False)
parser.add_argument('--truncate', type=int, default=-1)
parser.add_argument('--ed', type=bool, default=False)
parser.add_argument('--ed_hidden', type=int, default=100)
args = parser.parse_args()

setup_log(args.log)

def train():
    logging.info(f"seed set at {args.seed}")

    logging.info(f"loading {args.n} samples from {args.sample_dir}")
    samples = create_training_generator(
        args.sample_dir,
        args.n,
        args.split,
        args.seed,
        args.truncate
    )

    if (args.ed):
        params = default_ed_params(samples.n_outputs, args.ed_hidden)
    else:
        params = default_params(samples.n_features, samples.n_outputs)

    params['multigpu'] = args.multigpu
    if (args.GRU):
        params['rnn_layer'] = GRU

    logging.info(f"evaluating params {params}")

    if (args.ed):
        model = create_model(**params)
    else:
        model = create_ed_model(**params)

    train_model(
        model,
        samples.train_generator(params['batch_size']),
        args.epochs,
        args.seed
    )

    logging.info("predicting")
    test_model(model, samples.test_generator(params['batch_size']))
    predictions = model_predict(
        model,
        samples.test_generator(params['batch_size']),
        samples.y_scaler
    )
    truth = samples.truth()
    results = [
        { 'prediction': predictions[i].tolist(), 'truth': truth[i].tolist() }
        for i in range(predictions.shape[0])
    ]

    logging.info("saving outputs")
    with open(os.path.join(args.outdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    save_model(model, os.path.join(args.outdir, 'model'))
    save_scaler(samples.X_scaler, os.path.join(args.outdir, 'X_scaler'))
    save_scaler(samples.y_scaler, os.path.join(args.outdir, 'y_scaler'))

    logging.info("done")

if __name__ == "__main__":
    train()
