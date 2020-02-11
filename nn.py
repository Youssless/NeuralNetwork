import numpy as np
from numpy import random

'''
Class to model and train a neural network
 - Optimisation algorithms
    - gradient descent (Done)
    - stochastic gradient descent (TBC)
 - Activation functions
    - sigmoid (Done)
    - ReLu (TBC)
'''
class NeuralNetwork:
    # model = the architecture of the network [2, 2, 2] = 2 neurons each layer
    def __init__(self, model):
        try:
            self.model = model
            self.layers = len(model)
            if len(model) < 2:
                raise Exception("model length must be > 1, a network must contain more")
            self.bias = np.ones(shape=(self.layers-1, 1))
            # each column is is a single node connection to next layer, each row is a connection from prev layer to one node in the next layer
            self.weights = [random.random(size=(self.model[i+1], self.model[i])) for i in range(self.layers-1)]
            
        except Exception as e:
            print(e)
            if type(model) is not list:
                print("model is not a list.")
                print("model must be in the format equivalent to [2, 2, 2, etc..] which defines the number of neurons in each layer")

            

    def train(self, X, y, alpha=0.03, epochs=1000):
        self.back_propagate(X, y, alpha, epochs)
        return self.output(X)

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def cost(self, y_pred, y_actual):
        return (1/self.layers)*np.sum((y_pred-y_actual)**2)

    def forward_propagate(self, X):
        next_layer = None
        nodes = [X]
        for i in range(self.layers-1):
            next_layer = np.dot(self.weights[i], X) + self.bias[i]
            nodes.insert(i+1, next_layer)
            X = next_layer
        return np.array(nodes)

    def output(self, X):
        return self.forward_propagate(X)[self.layers-1]

    def back_propagate(self, X, y, alpha, epochs):
        for e in range(epochs):
            nodes = self.forward_propagate(X)
            print("epoch: {}".format(e))
            print("cost = {}".format(self.cost(y, nodes[self.layers-1])))
            curr_error = np.array(y - nodes[self.layers-1])
            layer_error = None
            errors = []

            # calculate the error for each node
            for i in range(self.layers-2, -1, -1):
                errors.insert(i, curr_error)
                layer_error = np.dot(self.weights[i].T, curr_error)
                curr_error = layer_error
                
            errors.insert(0, curr_error)
            E = np.array(errors)

            # perform gradient descent
            self.gradient_descent(nodes, alpha, E)
        

    def gradient_descent(self, nodes, alpha, E):
        for i in range(self.layers-1, 0, -1):
            a = self.sigmoid(nodes[i])
            # k = current layer , j = previous layer
            # partial_C / partial_W = -[(error)*(sigmoid(Zk)*(1-sigmoid(Zk)) . Oj.Transpose]
            delta_w = -(alpha*np.dot((E[i]*(a*(1-a))).reshape(-1, 1), nodes[i-1].reshape(-1, 1).T))

            # update weights
            self.weights[i-1] = self.weights[i-1] - delta_w


if __name__ == "__main__":
    model = [2, 3, 1]
    X = np.array([1, 1])
    y = np.array([1])
    nn = NeuralNetwork(model)

    output = nn.train(X, y, alpha=0.03, epochs=1000)
    print(output)
