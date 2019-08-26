import numpy as np


class ActivationFunction:
    def apply(self, x):
        pass

    def derivative(self, x):
        pass


class Step(ActivationFunction):
    def __init__(self, umbral):
        self.umbral = umbral

    def apply(self, x):
        if x < self.umbral:
            return 0
        else:
            return 1

    def derivative(self, x):
        if x == 0:
            raise ValueError("Derivative in zero is infinite")
        return 0


class Sigmoid(ActivationFunction):
    def apply(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.apply(x) * (1 - self.apply(x))


class Tanh(ActivationFunction):
    def apply(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x)**2
