import numpy as np
from activation_functions import *


class Perceptron:
    def __init__(self, weights, activation_function):
        if not isinstance(weights, int):
            raise TypeError("Invalid input type, not an int")
        if not isinstance(activation_function, type(ActivationFunction())):
            raise TypeError("Invalid input type, not an activation function")
        self.bias = 0.0
        self.weights = np.random.randn(1, weights)[0]
        self.activation_function = activation_function

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        if not isinstance(weights, np.ndarray):
            raise TypeError("Invalid type, not a numpy.ndarray")
        self.weights = weights

    def getBias(self):
        return self.bias

    def setBias(self, bias):
        if not (isinstance(bias, int) or isinstance(bias, float)):
            raise TypeError("Invalid type, not a number")
        self.bias = bias

    def feed(self, inputs):
        if not isinstance(inputs, np.ndarray):
            raise TypeError("Invalid type, not a numpy.ndarray")
        if len(inputs) != len(self.weights):
            raise ValueError(
                "Number of inputs is different than the number of the weights of the perceptron")
        total = np.dot(self.weights, inputs) + self.bias

        return self.activation_function.apply(total)

    def train(self, inputs, expected):
        pass  # TODO
