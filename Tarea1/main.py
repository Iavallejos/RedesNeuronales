from src.normalization import normalize
from src.activation_functions import *
from src.neural_network import NeuralNetwork
from src.utils import get_class, calculate_cost, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

properties = {
    'hidden_layers': 2,
    'neurons_per_layer': 4,
    'input_length': 4,
    'number_of_classes': 3,
    'epoch': 1000,
    'learning_rate': 0.1,
    'threshold': 0.9,
    'data': 'Data/Iris/data.npy',
    'dataset': 'Data/Iris/iris.data',
    'classes': 'Data/Iris/mapping.json'
}


def run_network(properties, train_data, test_data):
    # recall_score(y_true, y_pred, average=None)
    # precision_score(y_true, y_pred, average=None)
    # f1_score(y_true, y_pred, average=None)
    neural_network = NeuralNetwork(properties, Sigmoid())

    precision = []
    recall = []
    f1 = []
    cost = []
    epoch = []

    expected, predictions = predict(neural_network, test_data)
    print("Expected:\n{}".format(expected))
    print("Predictions:\n{}\n\n\n".format(predictions))

    
    precision.append(precision_score(expected, predictions, average=None))
    recall.append(recall_score(expected, predictions, average=None))
    f1.append(f1_score(expected, predictions, average=None))
    cost.append(calculate_cost(expected, predictions))
    epoch.append(0)

    for i in range(1, properties["epoch"]+1):
        if i % 50 == 0:
            print("Epoch {:4}:".format(i))
            expected, predictions = predict(neural_network, test_data)
            epoch_precision = precision_score(
                expected, predictions, average=None)
            epoch_recall = recall_score(expected, predictions, average=None)
            epoch_f1 = f1_score(expected, predictions, average=None)
            epoch_cost = calculate_cost(expected, predictions)

            print("\tPrecision: {}".format(epoch_precision))
            print("\tCost: {:.4}".format(epoch_cost))

            precision.append(epoch_precision)
            recall.append(epoch_recall)
            f1.append(epoch_f1)
            cost.append(epoch_cost)
            epoch.append(i)

        train(neural_network, train_data)
    print("Finished")
    print("Getting confusion matrix")
    classes=np.array(["setosa", "versicolor", "virginica"])
    print("expected len: {}".format(len(expected)))
    print("predictions len: {}".format(len(predictions)))
    print("expected:\n{}".format(get_class(expected)))
    print("predictions:\n{}".format(get_class(predictions)))
    plot_confusion_matrix(get_class(expected), get_class(predictions), classes)

    plt.show()

def predict(neural_network, test_data):
    predictions = []
    expected = []
    for data in test_data:
        predicted = neural_network.feed(data[0])
        predictions.append(predicted[-1])
        expected.append(data[1])

    predictions = np.array(predictions)
    expected = np.array(expected)
    return (expected, predictions)


def train(neural_network, train_data):
    for data in train_data:
        neural_network.train(data[0], data[1])


if __name__ == "__main__":
    normalize(properties)
    print("Loading data in: {}".format(properties["data"]))
    data = np.load(properties["data"], allow_pickle=True)

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        run_network(properties, train_data, test_data)

    '''
    #classes = np.array(["setosa", "versicolor", "virginica"])
    train_data = data[0]
    test_data = data[0]

    print("class: {} -  input data: {}".format(test_data[1], test_data[0]))
    neural_network = NeuralNetwork(properties, Sigmoid())

    prediction = neural_network.feed(test_data[0])
    print("prediction {:4}: {}".format(0, prediction[-1]))
    
    for i in range(1, properties["epoch"]+1):
        neural_network.train(train_data[0], train_data[1])
        if i % 50 == 0:
            prediction = neural_network.feed(test_data[0])
            print("prediction {0:4}: {1}".format(i, prediction[-1]))
    '''
