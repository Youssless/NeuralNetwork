import numpy as np
from numpy import random

'''
'''
class NeuralNetwork:

    def __init__(self, model):
        try:
            if type(model) is not list:
                raise Exception("model must be a list. [2, 3, 1, ...] defines network layout.")

            if model is list:
                if len(model) <= 1:
                    raise Exception("model length must be larger than 1. The length of a model defines the number of layers")
        
            self.model = model
            self.layers = len(model)-1
            self.weights = [random.random(size=(model[i+1], model[i])) for i in range(self.layers)]
            self.biases = [np.ones(shape=(model[i], 1)) for i in range(1, self.layers+1)]

        except Exception as e:
            print(e)

    # Z = sigmoid(W.X)
    def sigmoid(self, O):
        return 1 / (1 + np.exp(-O))


    def cost(self, y_target, y_output):
        return (1/(self.layers+1)) * (y_target - y_output)**2

    def train(self, x_train, y_train, alpha=0.03, epochs=10000):
        self.back_propagate(x_train, y_train, alpha, epochs)
        print(nn.forward_propagate(X))


    def forward_propagate(self, X):
        O = None
        for i in range(self.layers):
            O = np.dot(self.weights[i], self.sigmoid(X)).reshape(-1, 1) + self.biases[i] 
            X = O

        return self.sigmoid(O)

    def back_propagate(self, X, y, alpha=0.03, epochs=10000):
        # back propagate
        for i in range(epochs):
            # 1. input x
            inp = X
            activations = [self.sigmoid(X)]
            weighted_sums = [X]
            # 2. feed_forward
            for x in range(self.layers):
                z = np.dot(self.weights[x], self.sigmoid(inp)).reshape(-1, 1) + self.biases[x]
                inp = z
                activations.append(self.sigmoid(z))
                weighted_sums.append(z)

            # 3. calculate the errors
            cost_gradient = []
            weight_delta = []

            # outp error
            cost_gradient.append(y - activations[self.layers])
            curr_error = np.multiply(cost_gradient, self.sigmoid_prime(weighted_sums[self.layers])).reshape(-1, 1)
            
            for e in range(self.layers, 0, -1):
                # hidden error
                self.biases[e-1] = self.biases[e-1] + curr_error
                self.weights[e-1] = self.weights[e-1] + alpha*(np.dot(curr_error, activations[e-1].reshape(-1, 1).T))
                curr_error = np.dot(self.weights[e-1].T, curr_error)

            print("epoch {0}: cost => {1}".format(i, self.cost(y, self.forward_propagate(X))))
            
            
    def sigmoid_prime(self, Z):
        return self.sigmoid(Z)*(1-self.sigmoid(Z))





if __name__ == '__main__':
    nn = NeuralNetwork([2, 3, 9])
    X = np.array([1, 1])
    y_target = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
    nn.train(X, y_target, epochs=20000)