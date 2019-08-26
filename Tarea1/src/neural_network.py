import numpy as np
from activation_functions import ActivationFunction
from perceptron import Perceptron


class NeuralNetwork():
    def __init__(self, properties, activation_function):
        if not isinstance(activation_function, type(ActivationFunction())):
            raise TypeError("Invalid input type, not an activation function")
        if not (
                isinstance(properties["epoch"], int) or
                isinstance(properties["hidden_layers"], int) or
                isinstance(properties["input_length"], int) or
                isinstance(properties["neurons_per_layer"], int) or
                isinstance(properties["number_of_classes"], int)):
            raise TypeError("Invalid input type, not an int")

        if not (
                properties["epoch"] > 0 or
                properties["hidden_layers"] >= 0 or
                properties["input_length"] > 0 or
                properties["neurons_per_layer"] > 0 or
                properties["number_of_classes"] > 0):
            raise ValueError("invalid input")

        hidden_layers = properties["hidden_layers"]
        input_lenght = properties["input_length"]
        neurons_per_layer = properties["neurons_per_layer"]
        number_of_classes = properties["number_of_classes"]

        layers = []
        for i in range(hidden_layers):
            layer = []
            if i == 1:
                for _ in range(neurons_per_layer):
                    layer.append(Perceptron(input_lenght, activation_function))

            else:
                for _ in range(neurons_per_layer):
                    layer.append(Perceptron(
                        neurons_per_layer, activation_function))

            layers.append(layer)

        final_layer = []
        for _ in range(number_of_classes):
            final_layer.append(Perceptron)

        layers.append(final_layer)
        self.layers = layers
        self.epoch = properties["epoch"]
        self.input_lenght = input_lenght

    def get_epoch(self):
        return self.epoch

    def get_layers(self):
        return self.layers

    def feed(self, input):
        if input.dtype.type is not np.float64:
            raise TypeError("Invalid input type, not {}".format(np.float64))
        if len(input) != self.input_lenght:
            raise ValueError(
                "Number of inputs is different than the defined number: {}".format(self.input_lenght))
        # TODO
