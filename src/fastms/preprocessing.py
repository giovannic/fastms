import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def process_runs(runs, seed):
    eir = np.stack([entry['eir'] for entry in runs])
    actual_eir = np.mean(eir[:,-365:], axis=1)

    s_inputs = [
        'seasonal_a0',
        'seasonal_a1', 'seasonal_b1',
        'seasonal_a2', 'seasonal_b2',
        'seasonal_a3', 'seasonal_b3'
    ]

    inputs = ['average_age', 'Q0'] + s_inputs

    X = np.concatenate([
        np.stack([np.array([entry[i] for i in inputs]) for entry in runs ]),
        actual_eir[:, None]
    ], axis = 1)
        

    y = np.stack([entry['prev'] for entry in runs])[:,-365:]

    idx_train, idx_test = train_test_split(
        np.arange(y.shape[0]),
        test_size=0.2,
        random_state=seed
    )

    X_test = scaler.transform(X[idx_test])
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    return {
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "X_train": X_scaler.fit_transform(X[idx_train]),
        "X_test": X_scaler.transform(X[idx_test]),
        "y_train" = y_scaler.fit_transform(y[idx_train]),
        'y_test' = y[idx_test]
    }
