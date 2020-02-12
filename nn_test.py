import numpy as np
import nn

def main():
    binary_network(np.array([1, 1]), np.array([1]))

def binary_network(X, y):
    model = [2, 3, 1]
    binary_nn = nn.NeuralNetwork(model)

    output = binary_nn.train(X, y)
    print(output)

if __name__ == "__main__":
    main()