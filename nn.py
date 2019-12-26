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

        # delays the triggering of the activation function
        self.bias = 0.5

        # number of layers
        self.n_layers = 2

    '''
    Activation function
    Sigmoid Function -> Smoothed out perceptron.
    Output = 0 < R < 1
    '''
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    '''
    Activation function
    Perceptron -> Simplest artificial neuron
    Output = 1 or 0
    '''
    def perceptron(self, X):
        for i in range(len(X)):
            if X[i] >= 1:
                X[i] = 1
            else:
                X[i] = 0
        return X

    def train(self, X, y):
        out = self.__forward_propagate(X)
        return out

    def __back_propagate(self):
        pass

    def __forward_propagate(self, X):
        n_weights = [self.n_inp, self.n_hidden, self.n_out]
        # Z = W.I
        # O = sigmoid(Z)
        for i in range(self.n_layers):
            W = np.random.rand(n_weights[i+1], n_weights[i])
            Z = np.dot(W, X) + self.bias
            O = self.sigmoid(Z)

            # next input
            X = O

        return X
            
    def cost(self):
        pass

    def test(self, X):
        print('[Forward Propagation]')
        out = self.__forward_propagate(X)
        return out

if __name__ == '__main__':
    # data, labels = make_moons(n_samples=200, noise=0.04, random_state=0)
    # print(data.shape, labels.shape)
    # print(data)
    # color_map = mc.LinearSegmentedColormap.from_list("", ['red', 'yellow'])
    # plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=color_map)

    # nn = NeuralNetwork(400, 4, 2)
    # nn.__forward_propagate__(data.reshape(-1, 1))
    # plt.show()
    
    #nn.log()

    nn = NeuralNetwork(2, 3, 1)
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    
    y = np.array([0, 1, 1, 0])
    out = nn.test(X[3])
    print(out)