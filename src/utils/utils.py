        
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler #TODO possiamo anche provare con StandardScaler e MinMaxScaler e paragonare i risultati
from sklearn.metrics import r2_score



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



