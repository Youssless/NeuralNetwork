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
            print(self.weights.shape)
            
        except Exception as e:
            print(e)
            if type(model) is not list:
                print("model is not a list.")
                print("model must be in the format equivalent to [2, 2, 2, etc..] which defines the number of neurons in each layer")

    def train(self, X, y, alpha=0.03, epochs=1000):
        pass

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

    def back_propagate(self, X, y, alpha=0.03, epochs=1000):
        # we need to update the weights
        # to update the weights we need the cost and the errors of all the nodes in the network
        # formula output and hidden layers pE/pW => -(target - output_curr_layer).sigmoid(Z_prev_layer).(1-sigmoid(Z_prev_layer)).output_prev_layer

        # error back propagation
        alpha = 0.03
        nodes = self.forward_propagate(X)
        print("cost_before = {}".format(self.cost(0, nodes[self.layers-1])))
        curr_error = np.array(y - nodes[self.layers-1])
        layer_error = None
        errors = []
        for i in range(self.layers-2, -1, -1):
            errors.insert(i, curr_error)
            layer_error = np.dot(self.weights[i].T, curr_error)
            curr_error = layer_error
            
        errors.insert(0, curr_error)
        E = np.array(errors)

        for i in range(self.layers-1, 0, -1):
            a = self.sigmoid(nodes[i])
            delta_w = -(alpha*np.dot((E[i]*(a*(1-a))).reshape(-1, 1), nodes[i-1].reshape(-1, 1).T))
            print(delta_w)
            
            self.weights[i-1] = self.weights[i-1] - delta_w

        nodes = self.forward_propagate(X)
        print("cost_after = {}".format(self.cost(0, nodes[self.layers-1])))
        

    def gradient_descent(self, nodes, alpha, E):
        for i in range(self.layers-1, 0, -1):
            a = self.sigmoid(nodes[i])
            delta_w = -(alpha*np.dot((E[i]*(a*(1-a))).reshape(-1, 1), nodes[i-1].reshape(-1, 1).T))
            print(delta_w)
            
            self.weights[i-1] = self.weights[i-1] + delta_w


if __name__ == "__main__":
    model = [2, 3, 1]
    X = np.array([1, 1])
    y = np.array([0])
    nn = NeuralNetwork(model)

    # print("num layers: " + str(nn.layers))
    # print("sigmoid test: " + str(nn.sigmoid(X)))
    # print("cost test: " + str(nn.cost(0, nn.forward_propagate(X))))
    # print("bias test: \n" + str(nn.bias))
    # print("weights test: \n" + str(nn.weights))
    # print("output test: \n" + str(nn.forward_propagate(X)))
    # arr = nn.forward_propagate(X)
    # [print(arr[i]) for i in range(3)]
    # print("back propagate test: \n" + str(nn.back_propagate(X, np.array(y))))
    # nn.gradient_descent(X, y)

    nn.back_propagate(X, y)
