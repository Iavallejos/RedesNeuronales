import numpy as np
from .activation_functions import ActivationFunction
from .perceptron import Perceptron
from .utils import calc_with_threshold


class NeuralNetwork():
    def __init__(self, properties, activation_function):
        """ Creates a new NeuralNetwork
        it uses a properties dictionary (in main.py)
        and an activation function (in activation_functions.py)
        it creates a neural network with the specified number of hidden
        layers and whose perceptron's activation function correspond to
        the specified in the arguments, and a final layer of perceptrons
        that indicates the class and whose activation function is the
        specified in the arguments
        """
        if not isinstance(activation_function, type(ActivationFunction())):
            raise TypeError("Invalid input type, not an activation function")
        if not (
                isinstance(properties["epoch"], int) and
                isinstance(properties["hidden_layers"], int) and
                isinstance(properties["input_length"], int) and
                isinstance(properties["neurons_per_layer"], int) and
                isinstance(properties["number_of_classes"], int)):
            raise TypeError("Invalid input type, not an int")

        if not (
                properties["epoch"] > 0 and
                properties["hidden_layers"] >= 0 and
                properties["input_length"] > 0 and
                properties["neurons_per_layer"] > 0 and
                properties["number_of_classes"] > 0):
            raise ValueError("invalid input")

        hidden_layers = properties["hidden_layers"]
        input_length = properties["input_length"]
        neurons_per_layer = properties["neurons_per_layer"]
        number_of_classes = properties["number_of_classes"]

        layers = []
        for i in range(hidden_layers):
            layer = []
            if i == 0:
                for _ in range(neurons_per_layer):
                    layer.append(Perceptron(input_length, activation_function))

            else:
                for _ in range(neurons_per_layer):
                    layer.append(Perceptron(
                        neurons_per_layer, activation_function))

            layers.append(layer)

        final_layer = []
        for _ in range(number_of_classes):
            final_layer.append(Perceptron(neurons_per_layer, activation_function))

        layers.append(final_layer)
        self.layers = layers
        self.epoch = properties["epoch"]
        self.learning_rate = properties["learning_rate"]
        self.threshold = properties["threshold"]
        self.input_length = input_length
        self.hidden_layers = hidden_layers

    def get_epoch(self):
        return self.epoch

    def get_layers(self):
        return self.layers

    def feed(self, inputs, use_threshold=True):
        if inputs.dtype.type is not np.float64:
            raise TypeError("Invalid input type, not {}".format(np.float64))
        if len(inputs) != self.input_length:
            raise ValueError(
                "Number of inputs is different than the defined number: {}".format(self.input_length))
                
        results = []

        # iterate over the hidden layers and the final layer
        for layer_number in range(self.hidden_layers+1):
            layer_results = []
            if layer_number == 0: # first layer
                for neuron in self.layers[layer_number]:
                    layer_results.append(neuron.feed(inputs))
            else: # the other hidden layers and/or last layer
                for neuron in self.layers[layer_number]:
                    layer_results.append(neuron.feed(results[layer_number-1]))
            
            results.append(np.array(layer_results))
        if use_threshold:
            classes = results[-1]
            for i in range(len(classes)):
                classes[i] = calc_with_threshold(classes[i], self.threshold)
            results[-1] = classes
        return np.array(results)
    
    def train(self, inputs, expected):
        results = self.feed(inputs, use_threshold=False)
        predicted = results[-1]
        
        # Calculating deltas for each neuron
        for i in reversed(range(self.hidden_layers+1)):
            layer = self.layers[i]
            if i == self.hidden_layers: # final layer
                for j in range(len(layer)):
                    neuron = layer[j]
                    neuron.calcDelta(predicted[j] - expected[j])
            else:
                for j in range(len(layer)): # inner layers
                    neuron = layer[j]
                    total = 0
                    for nextLayerNeuron in self.layers[i+1]:
                        weights = nextLayerNeuron.getWeights()
                        total += nextLayerNeuron.getDelta() * weights[j]
                    neuron.calcDelta(total)

        # Deltas calculated
        # Change values of weights and bias of every neuron from left to right

        for i in range(len(self.layers)):
            if i == 0:
                layer_input = inputs
            else:
                layer_input = results[i-1]
            for neuron in self.layers[i]:
                neuron.calcNewValues(layer_input, self.learning_rate)

        

                
        
        

