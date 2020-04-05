import numpy as np
from numpy import random
from numpy.random import RandomState
import threading
import multiprocessing
import time

'''
Artificial Feed Forward Neural Network
- Sigmoid activation
- Gradient Descent optimiseation
- Cost function MSE
'''
class _NeuralNetwork(threading.Thread):
    # model = the architecture of the network [2, 2, 2] = 2 neurons each layer
    def __init__(self, model, name=None):
        super(_NeuralNetwork, self).__init__(name=name)
        try:
            self.start()
            if type(model) is not list:
                raise Exception("model must be a list. [2, 3, 1, ...] defines network layout.")

            if model is list:
                if len(model) <= 1:
                    raise Exception("model length must be larger than 1. The length of a model defines the number of layers")
        
            self.model = model
            self.layers = len(model)-1
            # each column is is a single node connection to next layer, each row is a connection from prev layer to one node in the next layer
            self.weights = [random.random(size=(model[i+1], model[i])) for i in range(self.layers)]
            self.biases = [np.ones(shape=(model[i], 1)) for i in range(1, self.layers+1)]

            self.x_train = None
            self.y_train = None
            self.alpha = None
            self.epochs = None
        except Exception as e:
            print(e)

    def init_training_params(self, x_train, y_train, alpha, epochs):
        self.x_train = x_train
        self.y_train = y_train
        self.alpha = alpha
        self.epochs = epochs
    
    # Z = sigmoid(W.X)
    def sigmoid(self, O):
        return 1 / (1 + np.exp(-O))

    # formula for calculating the cost of the network,
    # cost needs to be minimised as close as possible to 0
    def cost(self, y_target, y_output):
        return (1/(self.layers+1)) * (y_target - y_output)**2


    def run(self):
        self.back_propagate(self.x_train, self.y_train, self.alpha, self.epochs)


    # the initial step before training the network is to calculate the values of the neurons in the next layer
    def forward_propagate(self, X):
        O = None
        for i in range(self.layers):
            O = np.dot(self.weights[i], self.sigmoid(X)).reshape(-1, 1) + self.biases[i] 
            X = O

        return self.sigmoid(O)

    # core algorithm used to train the network
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
            
            # adjusting the weights based on the error of each neuron and thier respective weights
            for e in range(self.layers, 0, -1):
                # hidden error
                self.biases[e-1] = self.biases[e-1] + curr_error
                self.weights[e-1] = self.weights[e-1] + alpha*(np.dot(curr_error, activations[e-1].reshape(-1, 1).T))
                curr_error = np.dot(self.weights[e-1].T, curr_error)

            print("epoch {0}: cost => {1}".format(i, self.cost(y, self.forward_propagate(X))))
            
    # derivative of sigmoid       
    def sigmoid_prime(self, Z):
        return self.sigmoid(Z)*(1-self.sigmoid(Z))

    

class NeuralNetwork :
    #"Thread - {}".format(random.randint(2**32, size=1, dtype='int64')
    def __init__(self, model, input_dim):
        self.input_dim = input_dim
        self.nn_thread = _NeuralNetwork(model=model, name="Thread - {}".format(random.randint(2**32, size=1, dtype='int64')))
        self.output = [] 

    # wrapper function to call backpropagation to train the network
    def train(self, x_train, y_train, alpha=0.03, epochs=10000):
        for i in range(self.input_dim):
            self.nn_thread.init_training_params(x_train[i], y_train[i], alpha, epochs)
            self.nn_thread.run()
            self.output.insert(i, self.nn_thread.forward_propagate(x_train[i]))

    def evaluate(self):
        print(np.array(self.output).reshape(-1, 1))

#testing with NAND gate simulation
if __name__ == '__main__':
    nn = NeuralNetwork([2, 3, 1], 4)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_target = np.array([[1], [1], [1], [0]])

    nn.train(X, y_target, epochs=1000)
    nn.evaluate()
    
    print("Time completed {}".format(time.perf_counter()))
