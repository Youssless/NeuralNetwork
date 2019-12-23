import numpy as np

class NeuralNetwork:
    def __init__(self, inp):
        self.nodes_per_layer = np.shape(inp)[0]
        self.num_of_layers = np.shape(inp)[1]

        # Z = W.I , W = Weights matrix, I = Input matrix
        # O = sigmoid(Z) , O = Sigmoid matrix
        self.W = np.random.random((self.nodes_per_layer, self.nodes_per_layer))
        self.I = inp
        self.Z = None
        self.O = None

        # initial output
        self.__forward_propagate__()

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def train(self):
        pass

    def __back_propagate__(self):
        pass

    def __forward_propagate__(self):
        # Z = W.I
        # O = sigmoid(Z)
        self.Z = np.dot(self.W, self.I)
        self.O = self.sigmoid(self.Z)

    def log(self):
        print('-----|Input|-----\n')
        print(self.I, '\n')
        print('-----|Weights|-----\n')
        print(self.W, '\n')
        print('-----|Pre-Sigmoid|-----\n')
        print(self.Z, '\n')
        print('-----|Post-Sigmoid|-----\n')
        print(self.O, '\n')



if __name__ == '__main__':
    inp = np.array([[0.5, 3, 0.2], [0.11, 0.54, 1.3], [0.003, 0.34, 12.54]])
    nn = NeuralNetwork(inp)
    nn.log()