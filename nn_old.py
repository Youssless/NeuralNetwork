import numpy as np
from numpy import random
import time

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
            self.bias = np.ones(shape=(self.layers-1, 1))
            # each column is is a single node connection to next layer, each row is a connection from prev layer to one node in the next layer
            self.weights = [random.random(size=(self.model[i+1], self.model[i])) for i in range(self.layers-1)]
            
        except Exception as e:
            print(e)
            if type(model) is not list:
                print("model is not a list.")
                print("model must be in the format equivalent to [2, 2, 2, etc..] which defines the number of neurons in each layer")

        finally:
            if self.layers == 1:
                raise Exception("model length must be > 1, a network must contain more than 1 layer")


    def train(self, X, y, alpha=0.03, epochs=100000):
        self.__back_propagate(X, y, alpha, epochs)
        return self.output(X)

    # activation funtion real values between 0 and 1
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def cost(self, y_pred, y_actual):
        return (1/self.layers)*np.sum((y_pred-y_actual)**2)

    def output(self, X):
        return self.__forward_propagate(X)[self.layers-1]

    # calculates the next node value based on the previous layer
    def __forward_propagate(self, X):
        next_layer = None
        nodes = [X]
        for i in range(self.layers-1):
            next_layer = np.dot(self.weights[i], X) + self.bias[i]
            nodes.insert(i+1, next_layer)
            X = next_layer
        return np.array(nodes)

    # back propagates to get the errors then uses gradient descent to calculate change in weights
    def __back_propagate(self, X, y, alpha, epochs):
        for e in range(epochs):
            nodes = self.__forward_propagate(X)
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
            self.__gradient_descent(nodes, alpha, E)
        
    # algorithm that changes weights to give the correct output based on the error
    def __gradient_descent(self, nodes, alpha, E):
        for i in range(self.layers-1, 0, -1):
            a = self.sigmoid(nodes[i])
            # k = current layer , j = previous layer
            # partial_C / partial_W = -[(error)*(sigmoid(Zk)*(1-sigmoid(Zk)) . Oj.Transpose]
            delta_w = -(alpha*np.dot((E[i]*(a*(1-a))).reshape(-1, 1), nodes[i-1].reshape(-1, 1).T))

            # update weights
            self.weights[i-1] = self.weights[i-1] - delta_w


if __name__ == "__main__":
    model = [2, 3, 1]
    model = [2, 3, 1]
    nn = NeuralNetwork([2, 3, 1])
    nn2 = NeuralNetwork([2, 3, 1])
    nn3 = NeuralNetwork([2, 3, 1])
    nn4 = NeuralNetwork([2, 3, 1])

    X = np.array([0, 1])
    y_target = np.array([1])
    

    X2 = np.array([0, 1])
    y_target2 = np.array([1])

    X3 = np.array([1, 0])
    y_target3 = np.array([1])

    X4 = np.array([1, 1])
    y_target4 = np.array([0])

    nn.train(X, y_target, epochs=1000)
    nn2.train(X2, y_target2, epochs=1000)
    nn3.train(X3, y_target3, epochs=1000)
    nn4.train(X4, y_target4, epochs=1000)

    print("Completed in {}".format(time.perf_counter()))
