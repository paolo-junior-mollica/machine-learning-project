import os
import numpy as np
from sklearn.metrics import r2_score
import pickle
import random
import tensorflow as tf


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

def load_random_state():
    # Ricaricare lo stato di Python
    with open('python_random_state.pkl', 'rb') as f:
        python_random_state = pickle.load(f)
    random.setstate(python_random_state)

    # Ricaricare lo stato di TensorFlow
    with open('tf_random_state.pkl', 'rb') as f:
        tf_random_state = pickle.load(f)
    tf_random_generator = tf.random.Generator.from_state(*tf_random_state)
    tf.random.set_global_generator(tf_random_generator)
    
def save_random_state() : 
    # Python random state
    python_random_state = random.getstate()

    # Salvare lo stato di Python
    with open('python_random_state.pkl', 'wb') as f:
        pickle.dump(python_random_state, f)

    # TensorFlow random state
    tf_random_generator = tf.random.get_global_generator()
    tf_random_state = (tf_random_generator.state, tf_random_generator.algorithm)

    # Salvare lo stato di TensorFlow
    with open('tf_random_state.pkl', 'wb') as f:
        pickle.dump(tf_random_state, f)

def set_random_state(seed : int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    
    

