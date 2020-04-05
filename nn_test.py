import nn
import numpy as np
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import optimizers
from keras.datasets import mnist



def binary_classifyer_keras(x_train, y_train, epochs=1000):
    model = Sequential()
    model.add(Dense(activation='relu', input_dim=2, output_dim=3))
    model.add(Dense(activation='relu', output_dim=3))
    model.add(Dense(activation='sigmoid', output_dim=1))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.SGD(lr=0.03), metrics=[['accuracy', 'mse']])
    model.fit(x_train, y_train, epochs=epochs)

    y_predict = model.predict(x_train)
    return y_predict


def binary_classifyer_original(x_train, y_train, epochs=1000):
    model = [2, 3, 1]
    binary_classifyer = nn.NeuralNetwork(model)
    binary_classifyer.train(x_train, y_train, alpha=0.03, epochs=epochs)

    return binary_classifyer.evaluate(x_train)

def mnist_classifyer_keras():
    # training data (6000, 28, 28)
    # 6000 samples 28 x 28 images
    model = Sequential()
    model.add(Dense(activation='sigmoid', input_dim=28*28, output_dim=16)) # 12544 weights going to the output
    model.add(Dense(activation='sigmoid', output_dim=16)) # 160 weights going to the output
    model.add(Dense(activation='sigmoid', output_dim=10))

    (x_train, y_train) , (x_test, y_test) = mnist.load_data()
    display_handwritten_digit(x_train[1])


def mnist_classifyer_original():
    pass

def display_handwritten_digit(image):
    plt.imshow(image, cmap='Greys')
    plt.show()

if __name__ == '__main__':
    x_train_NAND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train_NAND = np.array([[1], [1], [1], [0]])

    #keras_binary_classifyer_NAND = binary_classifyer_keras(x_train_NAND, y_train_NAND, epochs=10000)
    #keras_binary_classifyer = binary_classifyer_keras(np.array([[1, 1]]), np.array([[0]]))
    #original_binary_classifyer = binary_classifyer_original(np.array([1, 1]), np.array([[0]]))

    #print("KERAS")
    #print(keras_binary_classifyer_NAND)
    #print(keras_binary_classifyer)

    #print("ORIGINAL")
    #print(original_binary_classifyer)

    mnist_classifyer_keras()