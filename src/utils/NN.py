#!../venv/bin/python3
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras
import matplotlib.pyplot as plt
from keras.src.constraints import MaxNorm
from keras.src.initializers.initializers import HeNormal
from keras.src.optimizers import SGD
import keras.backend as kb


class NeuralNetwork:
    def __init__(self, input_dimension, output_dimension, architecture, activation, dropout_input_rate,
                 dropout_hidden_rate, learning_rate, momentum, weight_decay, use_nesterov):
        self.built_model = None
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.architecture = architecture
        self.activation = activation
        self.dropout_input_rate = dropout_input_rate
        self.dropout_hidden_rate = dropout_hidden_rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov

        '''
        We utilize the ReLU activation function for the hidden layers and a linear activation for the output layer. 
        ReLU allows for non-linear learning, and it helps mitigate the vanishing gradient problem. Due to its linear 
        nature for positive values, ReLU can also improve convergence speed compared to tanh and sigmoid functions. 
        However, it's worth noting that 'dead neurons' can occur in networks using ReLU, where neurons never activate. 
        To address this, dropout techniques are often employed. Alternatively, one might consider using leaky ReLU.
        
        For the weight initialization, we opt for 'He normal' over the 'Glorot uniform,' which is the default for 
        networks with ReLU activation. The 'He normal' initializer in Keras, developed by He et al. in a 2015 study, 
        is tailored for layers with ReLU activations and its variants. The core principle is to adopt a weight 
        initialization strategy that maintains variance across layers in deep networks, thereby alleviating the 
        vanishing gradient problem often encountered in such architectures.
    
        As for the optimizer, we use Stochastic Gradient Descent (SGD) since it's a regression problem.
        '''

    @staticmethod
    def mean_euclidean_error(y_true, y_pred):
        # Has to work with tensors
        return kb.mean(kb.sqrt(kb.sum(kb.square(y_true - y_pred), axis=-1)))

    def build_model(self):
        model = Sequential()
        model.add(Dense(units=self.architecture[0], activation=self.activation, input_dim=self.input_dimension))
        model.add(Dropout(self.dropout_input_rate))

        self.architecture = self.architecture[1:]  # Remove first element from list
        for units, dropout_rate in zip(self.architecture, self.dropout_hidden_rate):
            model.add(Dense(units=units, activation=self.activation,
                            kernel_initializer=HeNormal(), kernel_constraint=MaxNorm(3)))
            model.add(Dropout(dropout_rate))

        model.add(Dense(units=self.output_dimension, activation='linear'))  # Linear activation for regression output

        optimizer = SGD(learning_rate=self.learning_rate, momentum=self.momentum,
                        nesterov=self.use_nesterov, decay=self.weight_decay)
        model.compile(loss=self.mean_euclidean_error, optimizer=optimizer)

        self.built_model = model

        return self.built_model

    def plot_architecture(self):
        keras.utils.plot_model(self.built_model, to_file='model_architecture.png', show_shapes=True)
        img = plt.imread('model_architecture.png')
        plt.imshow(img)
        plt.axis('off')
        plt.show()

# %%
