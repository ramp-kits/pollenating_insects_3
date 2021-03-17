import os
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

problem_title = 'Pollenating insect classification (403 classes)'
_target_column_name = 'class'
_prediction_label_names = range(0, 403)
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.Accuracy(name='accuracy', precision=3),
    rw.score_types.NegativeLogLikelihood(name='nll', precision=3),
    rw.score_types.F1Above(name='f170', threshold=0.7, precision=3),
]


def get_cv(X, y):
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=57)
    else:
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    df = pd.read_csv(os.path.join(path, 'data', f_name))
    X = df['id'].values
    y = df['class'].values

    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        X, y = X[::100], y[::100]

    folder = os.path.join(path, 'data', 'imgs')
    X = np.array(list(map(lambda x: f"{folder}/{x}", X)))
    return X, y


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)
