import argparse
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from .loading import create_training_generator
from .model import create_model, create_ed_model, create_attention_model
from .log import setup_log, logging

def default_params(n_static_features, n_seq_features, n_outputs):
    return {
        'optimiser': 'adam',
        'n_static_features': n_static_features,
        'n_seq_features': n_seq_features,
        'n_layer': [n_seq_features + n_static_features, n_outputs],
        'n_dense_layer': [n_outputs],
        'n_outputs': n_outputs,
        'dense_activation': ['linear'],
        'dense_initialiser': ['glorot_normal'],
        'dropout': .0,
        'loss': 'log_cosh',
        'batch_size': 100,
        'rnn_layer': LSTM
    }

def default_prob_params(n_static_features, n_seq_features, n_outputs):
    return {
        'optimiser': 'adam',
        'n_static_features': n_static_features,
        'n_seq_features': n_seq_features,
        'n_layer': [n_seq_features + n_static_features, n_outputs],
        'n_dense_layer': [n_outputs * 2],
        'n_outputs': n_outputs,
        'dense_activation': ['linear'],
        'dense_initialiser': ['glorot_normal'],
        'dropout': .0,
        'loss': 'log_cosh',
        'batch_size': 100,
        'rnn_layer': LSTM
    }



def default_ed_params(n_outputs, n_latent = 100):
    return {
        'optimiser': 'adam',
        'n_outputs': n_outputs,
        'n_latent': n_latent,
        'dropout': .0,
        'loss': 'log_cosh',
        'batch_size': 100,
        'rnn_layer': LSTM
    }
    
def default_attention_params(n_features, n_outputs, n_latent = 100):
    return {
        'optimiser': 'adam',
        'n_latent': n_latent,
        'dense_activation': ['linear'],
        'dense_initialiser': ['glorot_uniform'],
        'n_features': n_features,
        'n_outputs': n_outputs,
        'dropout': .0,
        'loss': 'log_cosh',
        'batch_size': 100,
        'rnn_layer': LSTM
    }

def hyperparameters(args):
    samples = create_training_generator(
        args.sample_dir,
        args.n,
        args.split,
        args.seed,
        -1
    )

    if args.ed:
        model = KerasRegressor(create_ed_model)
    elif args.attention:
        model = KerasRegressor(create_attention_model)
    else:
        model = KerasRegressor(create_model)

    optimisers = ( 'adam', )
    losses = ( 'log_cosh', )
    batches = ( 100, )
    dropout = ( 0., .1 )
    # rnn_layer = [LSTM, GRU]
    rnn_layer = ( LSTM, )
    n_latent = ( 10, 50, 100 )
    epochs = ( args.epochs, )

    if args.ed:
        param_grid = dict(
            optimiser=optimisers,
            epochs=epochs,
            loss=losses,
            batch_size=batches,
            rnn_layer=rnn_layer,
            n_latent=n_latent,
            dropout=dropout,
            n_timesteps = ( samples.n_timesteps, ),
            n_outputs = ( samples.n_outputs, ),
            verbose = ( False, )
        )
    elif args.attention:
        param_grid = dict(
            optimiser=optimisers,
            epochs=epochs,
            loss=losses,
            batch_size=batches,
            rnn_layer=rnn_layer,
            n_latent=n_latent,
            dropout=dropout,
            n_features = ( samples.n_features, ),
            n_outputs = ( samples.n_outputs, ),
            verbose = ( False, )
        )
    else:
        param_grid = dict(
            optimiser = optimisers,
            epochs = epochs,
            loss = losses,
            batch_size = batches,
            rnn_layer = rnn_layer,
            dropout = dropout,
            n_layer = ( [samples.n_features, samples.n_outputs], ),
            verbose = ( False, )
        )

    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(samples.X_train, samples.y_train)
    # summarize results
    logging.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        logging.info("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do some magic')
    parser.add_argument('sample_dir', type=str, default='./tp_ibm_interventions')
    parser.add_argument('n', type=int, default=100)
    parser.add_argument('split', type=float, default=.8)
    parser.add_argument('epochs', type=int, default=100)
    parser.add_argument('seed', type=int, default=42)
    parser.add_argument('--log', type=str, default='INFO')
    parser.add_argument('--multigpu', type=bool, default=False)
    parser.add_argument('--ed', type=bool, default=False)
    parser.add_argument('--attention', type=bool, default=False)
    args = parser.parse_args()

    setup_log(args.log)
    hyperparameters(args)
