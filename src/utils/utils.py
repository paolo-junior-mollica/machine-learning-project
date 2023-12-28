import os
import numpy as np
from sklearn.metrics import r2_score


def save_plot(plot, folder, filename, format='png'):
    if not filename.endswith(format):
        filename = filename.split('.')[0] + '.' + format

    fig_path = os.path.join(folder, filename)
    plot.savefig(fig_path, format=format)


def multidim_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return r2_score(y_true, y_pred, multioutput='uniform_average')


def mean_euclidean_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.linalg.norm(y_true - y_pred, axis=1))


def root_mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
