import numpy as np

class NeuralNetwork:
    def __init__(self, num_of_layers, nodes_per_layer, alpha):
        self.num_of_layers = num_of_layers
        self.nodes_per_layer = nodes_per_layer
        self.alpha = alpha

        self.initial_weights = np.ones(shape=(self.num_of_layers, self.nodes_per_layer), dtype='double')
        #print(self.initial_weights)

        I = np.array([0.5, 0.9, 3.5], dtype='double')
        #print(self.sigmoid(0.99260846))
        X = np.zeros(shape=(1, self.nodes_per_layer))
        for i in range(self.num_of_layers):
            X = np.dot(self.initial_weights, I.T)
            I = X
            O = self.sigmoid(X)
            print(O)

    def sigmoid(self, X):
        #print(-X)
        return 1 / (1 + np.exp(-X))

    def train(self):
        pass

    def __back_propagate__(self):
        pass

    def __forward_propagate__(self):
        pass


if __name__ == '__main__':
    nn = NeuralNetwork(3, 3, 0.01)
