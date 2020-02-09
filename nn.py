import numpy as np
from numpy import random


class NeuralNetwork:

    def __init__(self, model):
        try:
            self.model = model
            self.layers = len(model)
            self.bias = np.ones(shape=(self.layers-1, 1))
            # each column is is a single node connection to next layer, each row is a connection from prev layer to one node in the next layer
            self.weights = [random.random(size=(self.model[i+1], self.model[i])) for i in range(self.layers-1)]
            
            
        except Exception as e:
            print(e)
            if type(model) is not list:
                print("model is not a list.")
                print("model must be in the format equivalent to [2, 2, 2, etc..] which defines the number of neurons in each layer")

    def train():
        pass

    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def cost(self, y_pred, y_actual):
        return (1/self.layers)*np.sum((y_pred-y_actual)**2)

    def forward_propagate(self, X):
        next_layer = 0
        for i in range(self.layers-1):
            next_layer = np.dot(self.weights[i], X) + self.bias[i]
            X = next_layer
        return self.sigmoid(next_layer)

    def back_propagate():
        pass

if __name__ == "__main__":
    model = [2, 3, 1]
    X = np.array([1, 1])
    nn = NeuralNetwork(model)

    print("sigmoid test: " + str(nn.sigmoid(X)))
    print("cost test: " + str(nn.cost(0, nn.forward_propagate(X))))
    print("bias test: \n" + str(nn.bias))
    print("weights test: \n" + str(nn.weights))
    print("output test: \n" + str(nn.forward_propagate(X)))

