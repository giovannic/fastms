import os
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import numpy as np
from .log import logging

def calibrate_model(
    model,
    X_test,
    X_seq_test,
    y_test,
    y_scaler,
    outpath,
    split,
    seed
    ):
    (
        X_cal_train,
        X_cal_test,
        X_seq_cal_train,
        X_seq_cal_test,
        y_cal_train,
        y_cal_test
    ) = train_test_split(
        X_test,
        X_seq_test,
        y_test,
        train_size=split,
        random_state=seed
    )

    # calibrate the model on training data
    p = np.linspace(0.001, 1, endpoint=False)
    dist = model((X_cal_train, X_seq_cal_train))
    cdf = dist.cdf(y_cal_train).numpy()
    p_hat = np.array([np.sum(cdf < pj) for pj in p]) / cdf.size
    calibrator = IsotonicRegression(
        y_min=0,
        y_max=1,
        out_of_bounds='clip'
    ).fit(p_hat, p)

    # test the calibration
    dist = model((X_cal_test, X_seq_cal_test))
    cdf = dist.cdf(y_cal_test).numpy()
    p_hat = np.array([np.sum(cdf < pj) for pj in p]) / cdf.size
    p_hat_cal = calibrator.predict(p_hat)
    pre_calibration_error = np.sum(np.square(p - p_hat))
    post_calibration_error = np.sum(np.square(p - p_hat_cal))

    logging.info(f'calibration error (pre): {pre_calibration_error}')
    logging.info(f'calibration error (post): {post_calibration_error}')

    plt.plot(p_hat, p, linestyle = '-', marker = 'o', label='pre-calibration')
    plt.plot(p_hat_cal, p, linestyle = '-', marker = 'o', label='post-calibration')
    plt.xlabel('observed confidence level')
    plt.ylabel('actual confidence level')
    plt.title('Calibration plot')
    plt.legend()
    plt.savefig(os.path.join(outpath, 'calibration.png'))

    try:
        shar = np.mean(dist.stddev())
    except NotImplementedError:
        try:
            shar = np.mean(dist.stddev_approx())
            logging.info(f'sharpness: {shar}')
        except AttributeError:
            logging.warn(
                'Tensorflow probability needs to be upgraded to get sharpness info'
            )
