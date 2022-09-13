import os.path
import argparse
import json
from tensorflow.keras.layers import GRU
from .log import setup_log, logging
from .loading import create_training_generator
from .model import (
    create_model,
    create_ed_model,
    create_attention_model,
    train_model,
    model_predict
)
from .prob_model import create_prob_model
from .calibration import calibrate_model
from .evaluate import test_model, test_prob_model
from .export import save_model, save_scaler, save_calibrator
from .hyperparameters import (
    default_params,
    default_ed_params,
    default_attention_params,
    default_prob_params
)

def train(args):
    logging.info(f"seed set at {args.seed}")

    logging.info(f"loading {args.n} samples from {args.sample_dir}")
    
    if (args.prob and not args.no_scale_y):
        logging.warn('you probably want to set no_scale_y == True when prob == True')
    samples = create_training_generator(
        args.sample_dir,
        args.n,
        args.split,
        args.seed,
        args.truncate,
        not args.no_scale_y
    )

    if (args.ed):
        params = default_ed_params(samples.n_outputs)
    elif (args.attention):
        params = default_attention_params(samples.n_features, samples.n_outputs)
    elif (args.prob):
        params = default_prob_params(
            samples.n_static_features,
            samples.n_seq_features,
            samples.n_outputs,
            samples.n_samples,
            samples.n_years
        )
    else:
        params = default_params(
            samples.n_static_features,
            samples.n_seq_features,
            samples.n_outputs
        )

    params['multigpu'] = args.multigpu
    if (args.GRU):
        params['rnn_layer'] = GRU

    logging.info(f"evaluating params {params}")

    if (args.ed):
        model = create_ed_model(n_timesteps = samples.n_timesteps, **params)
    elif (args.attention):
        model = create_attention_model(**params)
    elif (args.prob):
        model = create_prob_model(**params)
    else:
        model = create_model(**params)

    train_model(
        model,
        samples.train_generator(params['batch_size']),
        args.epochs,
        args.seed,
        log=args.fit_log
    )

    logging.info("predicting")
    
    if args.prob:
        test_prob_model(
            model,
            samples.X_test,
            samples.X_seq_test,
            samples.y_test,
            samples.y_scaler,
            args.outdir
        )
    else:
        test_model(
            model,
            samples.X_test,
            samples.X_seq_test,
            samples.y_test,
            samples.y_scaler
        )
    predictions = model_predict(
        model,
        samples.X_test,
        samples.X_seq_test,
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
    save_scaler(samples.X_seq_scaler, os.path.join(args.outdir, 'X_seq_scaler'))
    save_scaler(samples.y_scaler, os.path.join(args.outdir, 'y_scaler'))

    if (args.prob and not args.no_calibrate):
        logging.info(f"Calibrating")
        calibrator = calibrate_model(
            model,
            samples.X_test,
            samples.X_seq_test,
            samples.y_test,
            samples.y_scaler,
            args.outdir,
            args.calibration_split,
            args.seed
        )
        save_calibrator(calibrator, os.path.join(args.outdir, 'calibrator'))

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
    parser.add_argument('--ed', type=bool, default=False)
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--fit_log', type=str, default=False)
    parser.add_argument('--prob', type=bool, default=False)
    parser.add_argument('--no_scale_y', type=bool, default=False)
    parser.add_argument('--calibration_split', type=float, default=.5)
    parser.add_argument('--no_calibrate', type=bool, default=False)
    args = parser.parse_args()

    setup_log(args.log)
    train(args)
