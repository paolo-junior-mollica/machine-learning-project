#!../venv/bin/python3
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.model_selection import StratifiedKFold
from utils import save_plot, mean_euclidean_error, root_mean_squared_error, multidim_r2
from keras.initializers import he_normal




mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)







import matplotlib.pyplot as plt

class NN():
    def __init__(self, architecture=None, activation= "relu", num_epochs=None, learning_rate=None, momentum=None, use_nesterov=None, batch_size=None, weight_decay=0, input_dimension=None, output_dimension=None, dropout_input_rate=None, dropout_hidden_rate=None):
        self.NetworArchitecture = architecture
        self.activation = activation
        self.epochs = num_epochs
        self.eta = learning_rate
        self.momentum = momentum
        self.nesterov = use_nesterov
        self.batch_size = batch_size
        self.decay = weight_decay
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.dropout_input_rate = dropout_input_rate
        self.dropout_hidden_rate = dropout_hidden_rate
        """
        usiamo RELU come funzione di attivazione per i layer nascosti e linear per l'output layer
        Relu ci permette di avere un apprendimento Non lineare
        inoltre aiuta a mitigare il problema del vanishing gradient
        grazie alla sua natura lineare nei valori positivi aiuta a migliorare la velocità di convergenza rispetto a tanh e sigmoid
        !! nelle reti Relu possono accadere i cosi detti ' neuroni morti' ovvero neuroni che non si attivano mai, per questo motivo si usa la tecnica del dropout
        in alternativa (TODO) si puo usare leaky relu
        inoltre usiamo SGD come ottimizzatore, poiche é un problema di regressione
        
        
        Per l'inizializzazione dei pesi, usiamo la 'he normal', in alternativa si puo usare la 'glorot_uniform' che é la default per le reti con attivazione relu
        he_normal in Keras è un metodo di inizializzazione del kernel (pesi) di un layer, sviluppato da He et al. 
        in uno studio del 2015.
        È specificamente progettato per layer con funzioni di attivazione ReLU (Rectified Linear Unit) e le sue varianti. L'idea di base è di adottare un'inizializzazione dei pesi 
        che mantiene la varianza attraverso i layer in reti molto profonde, 
        aiutando a mitigare il problema del gradiente sparito che spesso si verifica in tali reti.
        """
    def createModel(self):
        model = Sequential()
        model.add(Dense(units=self.input_dimension, activation=self.activation, input_dim=self.input_dimension))  # input_dim = 10, nei layer successivi non vi é bisogno di specificare l'input_dim, 
                                                                                                                        #poiche keras inferisce da solo l'input_dim, che a sua volta sará la dimensione dell'output del layer precedente 
        model.add(Dropout(self.dropout_input_rate))
        model.add(Dense(kernel_initializer= he_normal(), kernel_constraint=maxnorm(3),
                                units=NetworArchitecture[0], activation=self.activation))
        model.add(Dropout(self.dropout_hidden_rate[1]))     
        model.add(Dense(units=self.NetworArchitecture[2], activation=self.activation))
        model.add(Dropout(self.dropout_hidden_rate[1]))  # Use the second element of dropout_hidden_rate
        model.add(Dense(units=3, activation="linear")) # output layer, activation = linear, poiche é un problema di regressione
        model.compile(loss=mee_scorer,
                      optimizer=keras.optimizers.SGD(lr=self.eta, momentum=self.momentum, nesterov=self.nesterov, decay=self.decay))
        return model
    
    def plotArchitecture(self):
        model = self.createModel()
        keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True)
        img = plt.imread('model_architecture.png')
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    





