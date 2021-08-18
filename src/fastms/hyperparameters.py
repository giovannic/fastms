from .preprocessing import N_FEATURES, N_OUTPUTS
from sklearn.model_selection import GridSearchCV

DEFAULT_PARAMS = {
    'optimiser': 'adam',
    'n_layer': [
        N_FEATURES,
        N_OUTPUTS
    ],
    'dropout': .1,
    'loss': 'log_cosh',
    'batch_size': 100
}

def hyperparameters(model):
    optimisers = ['adam', 'rmsprop']
    losses = ['mse', 'log_cosh']
    batches = [50, 100]
    dropout = [0., .1]
    epochs = [100]

    param_grid = dict(
        optimiser=optimisers,
        epochs=epochs,
        loss=losses,
        batch_size=batches,
        dropout=dropout
    )

    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result.best_params_
