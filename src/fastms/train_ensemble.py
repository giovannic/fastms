import os.path
import argparse
import json
from tensorflow.keras.layers import GRU
from .log import setup_log, logging
from .loading import create_training_generator
from .model import train_model
from .prob_model import create_prob_model, create_ensemble, prob_model_predict
from .evaluate import test_prob_model
from .export import save_model, save_scaler
from .hyperparameters import default_params

def train(args):
    logging.info(f"seed set at {args.seed}")

    logging.info(f"loading {args.n} samples from {args.sample_dir}")
    samples = create_training_generator(
        args.sample_dir,
        args.n,
        args.split,
        args.seed,
        args.truncate
    )

    params = default_params(
        samples.n_static_features,
        samples.n_seq_features,
        samples.n_outputs
    )

    params['multigpu'] = args.multigpu
    if (args.GRU):
        params['rnn_layer'] = GRU

    logging.info(f"evaluating params {params}")

    model_creation = create_prob_model(**params)
    models = [create_prob_model(**params) for _ in range(args.n_models)]

    for i, model in enumerate(models):
        seed = args.seed + i
        train_model(
            model,
            samples.train_generator(params['batch_size'], seed=seed),
            args.epochs,
            seed,
            log=args.fit_log
        )

    ensemble = create_ensemble(
        models,
        params['n_static_features'],
        params['n_seq_features']
    )

    logging.info("predicting")
    test_prob_model(
        ensemble,
        samples.X_test,
        samples.X_seq_test,
        samples.y_test,
        samples.y_scaler,
        args.n_models
    )
    predictions = prob_model_predict(
        model,
        samples.X_test,
        samples.X_seq_test,
        samples.y_scaler,
        args.n_models
    )
    truth = samples.truth()
    results = [
        { 'prediction': predictions[i].tolist(), 'truth': truth[i].tolist() }
        for i in range(predictions.shape[0])
    ]

    logging.info("saving outputs")
    with open(os.path.join(args.outdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    save_model(ensemble, os.path.join(args.outdir, 'model'))
    save_scaler(samples.X_scaler, os.path.join(args.outdir, 'X_scaler'))
    save_scaler(samples.X_seq_scaler, os.path.join(args.outdir, 'X_seq_scaler'))
    save_scaler(samples.y_scaler, os.path.join(args.outdir, 'y_scaler'))

    logging.info("done")

if __name__ == "__main__":
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
    parser.add_argument('--fit_log', type=str, default=False)
    parser.add_argument('--n_models', type=int, default=5)
    args = parser.parse_args()

    setup_log(args.log)
    train(args)
