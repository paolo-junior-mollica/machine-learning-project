import os
import numpy as np
from sklearn.metrics import r2_score
import pickle
import random
import tensorflow as tf
from joblib import load
import optuna


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

def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


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
    
def calculate_ensemble_prediction(y_test,X_test, model_folder, n_models=5, metric='MEE'):
    
    final_model_ = []
    for i in range(n_models):
        model_path = os.path.join(model_folder, f'NN_model_grid_NN_{metric}_model{i}.joblib')
        final_model_[i] = load(model_path)
    y_pred_ensemble_final = np.zeros_like(y_test)

    for model in final_model_:
        y_pred = model.predict(X_test)
        y_pred_ensemble_final += y_pred

    y_pred_ensemble_final /= len(final_model_)
    return y_pred_ensemble_final


    
def get_hyperparameter_values_from_dataframe(df):
    hyperparameters_values_list = []
    param_names = [col for col in df.columns if col.startswith('params_')]

    # Filtra per righe con stato 'COMPLETE'
    df_complete = df[df['state'] == 'COMPLETE']

    # Itera attraverso le righe filtrate per raccogliere i valori degli iperparametri
    for _, row in df_complete.iterrows():
        hyperparameters_values = [row[param] for param in param_names]
        hyperparameters_values_list.append(hyperparameters_values)

    return hyperparameters_values_list, param_names


def load_optuna_study(model_name,MODEL_FOLDER):        
    with open(os.path.join(MODEL_FOLDER, model_name), 'rb') as f:
        study = pickle.load(f)
    return study

def save_optuna_study(study, model_name, MODEL_FOLDER):
    with open(os.path.join(MODEL_FOLDER, model_name), 'wb') as f:
        pickle.dump(study, f)
        
        



