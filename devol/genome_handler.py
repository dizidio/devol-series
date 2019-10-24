import numpy as np
import random as rand
import math
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from pyDOE import *

##################################
# Genomes are represented as fixed-with lists of integers corresponding
# to sequential layers and properties. A model with 2 convolutional layers
# and 1 dense layer would look like:
#
# [<conv layer><conv layer><dense layer><optimizer>]
#
# The makeup of the convolutional layers and dense layers is defined in the
# GenomeHandler below under self.convolutional_layer_shape and
# self.dense_layer_shape. <optimizer> consists of just one property.
###################################

class GenomeHandler:
    """
    Defines the configuration and handles the conversion and mutation of
    individual genomes. Should be created and passed to a `DEvol` instance.

    ---
    Genomes are represented as fixed-with lists of integers corresponding
    to sequential layers and properties. A model with 2 convolutional layers
    and 1 dense layer would look like:

    [<conv layer><conv layer><dense layer><optimizer>]

    The makeup of the convolutional layers and dense layers is defined in the
    GenomeHandler below under self.convolutional_layer_shape and
    self.dense_layer_shape. <optimizer> consists of just one property.
    """

    def __init__(self, max_dense_layers,
                 max_dense_nodes, max_input_size,
                 batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None):
        """
        Creates a GenomeHandler according 

        Args:
            
        """
        if max_dense_layers < 1:
            raise ValueError(
                "At least one dense layer is required"
            ) 
        self.optimizer = optimizers or [
            'sgd',
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.activation = activations or [
            'relu',
            'sigmoid',
            'tanh',
            'linear'
        ]
        self.dense_layer_shape = [
            "active",
            "num nodes",
            "batch normalization",
            "activation",
            "dropout"
        ]
        self.layer_params = {
            "active": [0, 1],
            "num nodes": [2*i for i in range(0, max_dense_nodes//2)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "dropout": [(5*i if dropout else 0) for i in range(16)]
        }

        self.dense_layers = max_dense_layers - 1 # this doesn't include the softmax layer, so -1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.max_input_size = max_input_size

    def denseParam(self, i):
        key = self.dense_layer_shape[i]
        return self.layer_params[key]

    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))
            if index < self.max_input_size:
                genome[index] = np.random.randint(2);
            if index < self.dense_layer_size * self.dense_layers:
                offset = self.max_input_size
                new_index = (index - offset)
                present_index = new_index - new_index % self.dense_layer_size
                if genome[present_index + offset]:
                    range_index = new_index % self.dense_layer_size
                    choice_range = self.denseParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01:
                    genome[present_index + offset] = 1
            elif index == len(genome) - 1:
                genome[index] = np.random.choice(list(range(len(self.optimizer))))
        return genome

    def decode(self, genome):
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")
        model = Sequential()
        
        input_size = np.sum(genome[:self.max_input_size]);
        
        offset = self.max_input_size
        
        input_layer = True
        
        for i in range(self.dense_layers):
            if genome[offset]:
                dense = None
                if input_layer:
                    dense = Dense(genome[offset + 1], input_shape=(input_size,))
                    input_layer = False
                else:
                    dense = Dense(genome[offset + 1])
                model.add(dense)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 3]]))
                model.add(Dropout(float(genome[offset + 4] / 10.0)))
            offset += self.dense_layer_size
        
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
            optimizer=self.optimizer[genome[offset]],
            metrics=["mse"])
        return model

    def genome_representation(self):
        encoding = []
        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                encoding.append("Dense" + str(i) + " " + key)
        encoding.append("Optimizer")
        return encoding

    def generate(self):
        genome = []
        for i in range(self.max_input_size):
            genome.append(np.random.randint(2));
        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome[0] = 1
        return genome        
        

    def is_compatible_genome(self, genome):
        expected_len = self.max_input_size + self.dense_layers * self.dense_layer_size + 1
        if len(genome) != expected_len:
            return False
        ind = 0
        for i in range(self.max_input_size):
            if genome[ind] != 0 and genome[ind]!=1:
                return False
            ind+=1
        for i in range(self.dense_layers):
            for j in range(self.dense_layer_size):
                if genome[ind + j] not in self.denseParam(j):
                    return False
            ind += self.dense_layer_size
        if genome[ind] not in range(len(self.optimizer)):
            return False
        return True

    def best_genome(self, csv_path, metric="accuracy", include_metrics=True):
        best = max if metric is "accuracy" else min
        col = -1 if metric is "accuracy" else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    def decode_best(self, csv_path, metric="accuracy"):
        return self.decode(self.best_genome(csv_path, metric, False))
