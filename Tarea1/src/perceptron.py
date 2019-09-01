import numpy as np
from .activation_functions import ActivationFunction


class Perceptron:
    """Provides a perceptron"""

    def __init__(self, weights, activation_function):
        if not isinstance(weights, int):
            raise TypeError("Invalid input type, not an int")
        if not isinstance(activation_function, type(ActivationFunction())):
            raise TypeError("Invalid input type, not an activation function")
        self.bias = 0.0
        self.weights = np.random.randn(1, weights)[0]
        self.activation_function = activation_function
        self.total = None
        self.delta = None

    def calcDelta(self, delta):
        if not (isinstance(delta, float) or isinstance(delta, int)):
            raise TypeError("Invalid type, not a number")
        self.delta = delta * self.activation_function.derivative(self.total)

    def getDelta(self):
        return self.delta

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        if not isinstance(weights, np.ndarray):
            raise TypeError("Invalid type, not a {}".format(np.ndarray))
        self.weights = weights

    def getBias(self):
        return self.bias

    def setBias(self, bias):
        if not (isinstance(bias, int) or isinstance(bias, float)):
            raise TypeError("Invalid type, not a number")
        self.bias = bias

    def feed(self, inputs):
        """Predicts a class based in the inputs"""
        if not isinstance(inputs, np.ndarray):
            raise TypeError("Invalid type, not a {}".format(np.ndarray))
        if len(inputs) != len(self.weights):
            raise ValueError(
                "Number of inputs is different than the number of the weights of the perceptron")
        total = np.dot(self.weights, inputs) + self.bias

        self.total = total

        return self.activation_function.apply(self.total)

    def train(self, inputs, expected, lr=0.1):
        """
        For a single learning perceptron, do not
        use in neural network
        """
        if not isinstance(inputs, list):
            raise ValueError("Invalid inputs type")
        elif not isinstance(expected, int):
            raise ValueError("Invalid desiredOutput type")
        elif not (isinstance(lr, float) or isinstance(lr, int)):
            raise ValueError("Invalid lr type")

        diff = expected - self.calculate(inputs)
        nweights = self.weights.copy()
        for i in range(len(nweights)):
            nweights[i] += lr * inputs[i] * diff * \
                self.activation_function.derivative(self.total)

        self.setWeights(nweights)
        self.setBias(self.bias + lr * diff *
                     self.activation_function.derivative(self.total))

    def calcNewValues(self, inputs, lr):
        """
        Calculates the new weights and bias using the stored delta,
        the inputs received and the given learning rate
        """
        if not isinstance(inputs, np.ndarray):
            raise TypeError("Invalid type, not a {}".format(np.ndarray))
        if len(inputs) != len(self.weights):
            raise ValueError(
                "Number of inputs is different than the number of the weights of the perceptron")
        if not (isinstance(lr, int) or isinstance(lr, float)):
            raise TypeError("Learning rate invalid, not a number")
        if not lr > 0:
            raise ValueError(
                "Learning rate invalid, has to be greater than zero")
        if self.delta is None:
            raise ValueError("Delta is None")

        nWeights = self.weights.copy()
        nbias = self.bias
        for i in range(len(self.weights)):
            nWeights[i] -= lr * self.delta * inputs[i]
        nbias -= lr * self.delta

        self.weights = nWeights.copy()
        self.bias = nbias
