import numpy as np
import nn
#from keras.datasets import mnist

def main():
    binary_network(np.array([1, 0]), np.array([1]))

def binary_network(X, y):
    model = [2, 3, 1]
    binary_nn = nn.NeuralNetwork(model)

    output = binary_nn.train(X, y)
    print(output)

# def handwritten_digits():
#     model = [28*28, 16, 10]
#     n = nn.NeuralNetwork(model)
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     n.train(x_train[0].reshape(-1, 1), y_train[0])

    
    # print("{0}".format(x_train[0].reshape(-1, 1)))
    # print(y_train[0])


if __name__ == "__main__":
    main()