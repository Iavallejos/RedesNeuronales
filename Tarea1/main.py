from src.normalization import normalize
from src.activation_functions import *
from src.neural_network import NeuralNetwork
import numpy as np

if __name__ == "__main__":
    properties = {
        'hidden_layers': 2,
        'neurons_per_layer': 4,
        'input_length': 4,
        'number_of_classes': 3,
        'epoch': 5000,
        'learning_rate': 0.4,
        'data': 'Data/Iris/data.npy',
        'dataset': 'Data/Iris/iris.data',
        'classes': 'Data/Iris/mapping.json'
    }
    #normalize(properties)
    print("Loading data in: {}".format(properties["data"]))
    data = np.load(properties["data"], allow_pickle=True)
    #print(data)
    
    input_data = data[0][0]
    expected = data[0][1]
    print("input data: {} -  class: {}".format(input_data, expected))
    neural_network = NeuralNetwork(properties, Sigmoid())

    prediction = neural_network.feed(input_data)
    print("initial prediction: {} - initial last layer inputs: {}".format(prediction[-1], prediction[-2]))
    for i in range(1,properties["epoch"]+1):
        neural_network.train(input_data, expected)
        prediction = neural_network.feed(input_data)
        if i%200 == 0:
            print("{0} prediction: {1} -  last layer inputs: {2}".format(i, prediction[-1], prediction[-2]))

    
