#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from sklearn.datasets import make_moons


# TODO: Cost Function/Gradient Descent
# TODO: Back Propagation
# TODO: Logs

class NeuralNetwork:
    '''
    Define the number of:
    - Input nodes
    - Hidden nodes
    - Output nodes

    To make a neural network
    '''
    def __init__(self, n_inp, n_hidden, n_out):
        # number of nodes in each layer
        self.n_inp = n_inp
        self.n_hidden = n_hidden
        self.n_out = n_out

        # number of layers
        self.n_layers = 2

        #initial weights for the output
        self.W = np.random.rand(self.n_hidden, self.n_inp)

    '''
    Sigmoid Function -> Smoothed out perceptron.
    Output = 0 < R < 1
    '''
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def train(self):
        pass

    def __back_propagate__(self):
        pass

    def __forward_propagate__(self, X):
        # Z = W.I
        # O = sigmoid(Z)
        print(X)
        for i in range(self.n_layers):
            Z = np.dot(self.W, X)
            O = self.sigmoid(Z)

            # next input
            X = O
            print('Layer', i)
            print(X)
            
            # next layer weights
            self.W = np.random.rand(self.n_out, self.n_hidden)



    # def log(self):
    #     print('-----|Input|-----\n')
    #     print(self.I, '\n')
    #     print('-----|Weights|-----\n')
    #     print(self.W, '\n')
    #     print('-----|Pre-Sigmoid|-----\n')
    #     print(self.Z, '\n')
    #     print('-----|Post-Sigmoid|-----\n')
    #     print(self.O, '\n')



if __name__ == '__main__':
    data, labels = make_moons(n_samples=200, noise=0.04, random_state=0)
    print(data.shape, labels.shape)
    print(data)
    color_map = mc.LinearSegmentedColormap.from_list("", ['red', 'yellow'])
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=color_map)

    nn = NeuralNetwork(400, 4, 2)
    nn.__forward_propagate__(data.reshape(-1, 1))
    plt.show()
    
    #nn.log()