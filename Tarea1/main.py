from src.normalization import normalize
from src.activation_functions import *
from src.neural_network import NeuralNetwork
from src.utils import calculate_cost, plot_iteraton, calculate_proms
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

properties = {
    'hidden_layers': 2,
    'neurons_per_layer': 4,
    'input_length': 4,
    'number_of_classes': 3,
    'epoch': 200,
    'learning_rate': 0.1,
    'folds': 5,
    'sampling_rate': 5,
    'data': 'Data/Iris/data.npy',
    'dataset': 'Data/Iris/iris.data',
    'classes': 'Data/Iris/mapping.json'
}


def run_network(properties, train_data, test_data, iteration):
    """
    Creates and runs a neural network using the data in properties,
    creates a confusion matrix and returns a dictionary with the metrics
    of the neural network and the las pair of predicted and expected
    classes
    """
    neural_network = NeuralNetwork(properties, Sigmoid())

    precision = []
    recall = []
    f1 = []
    cost = []
    epoch = []

    expected, predictions = predict(neural_network, test_data)

    precision.append(precision_score(
        expected, predictions, average='weighted'))
    recall.append(recall_score(expected, predictions, average='weighted'))
    f1.append(f1_score(expected, predictions, average='weighted'))
    cost.append(calculate_cost(expected, predictions))
    epoch.append(0)

    for i in range(1, properties["epoch"]+1):
        if i % properties["sampling_rate"] == 0:
            print("Epoch {:4}:".format(i))
            expected, predictions = predict(neural_network, test_data)
            epoch_precision = precision_score(
                expected, predictions, average='weighted')
            epoch_recall = recall_score(
                expected, predictions, average='weighted')
            epoch_f1 = f1_score(expected, predictions, average='weighted')
            epoch_cost = calculate_cost(expected, predictions)

            print("{:>14} {:.4}".format('Precision:', epoch_precision))
            print("{:>14} {:.4}".format('Recall:', epoch_recall))
            print("{:>14} {:.4}".format('F1:', epoch_cost))
            print("{:>14} {:.4}".format('Cost:', epoch_cost))

            precision.append(epoch_precision)
            recall.append(epoch_recall)
            f1.append(epoch_f1)
            cost.append(epoch_cost)
            epoch.append(i)

        train(neural_network, train_data)
    print("Finished")
    print("Calculating confusion matrix")
    classes = np.array(["setosa", "versicolor", "virginica"])

    title = "Iteration {}".format(iteration)
   
    plot_iteraton(
        title,
        (precision, recall, f1, cost, epoch),
        (expected, predictions, classes))

    metrics = {
        "iteration:": iteration,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost": cost,
        "last_expected": expected,
        "last_predicted": predictions
    }

    return metrics


def predict(neural_network, test_data):
    """
    Uses the given neural network to predict
    a class for every data in test_data, returns
    a tuple with 2 arrays, the expected classes and
    the predicted classes
    """
    predictions = []
    expected = []
    for data in test_data:
        predicted = neural_network.feed(data[0])
        predictions.append(predicted)
        expected.append(np.argmax(data[1]))

    predictions = np.array(predictions)
    expected = np.array(expected)
    return (expected, predictions)


def train(neural_network, train_data):
    """
    Trains the given neural network using all
    the data in train_data
    """
    for data in train_data:
        neural_network.train(data[0], data[1])


if __name__ == "__main__":
    normalize(properties)
    print("Loading data in: {}".format(properties["data"]))
    data = np.load(properties["data"], allow_pickle=True)

    kf = KFold(n_splits=properties["folds"], shuffle=True)
    cont = 1
    print("{:-^50}".format('Iniciando simulación'))
    print("Serán {} iteraciones".format(kf.get_n_splits()))
    metrics = []
    for train_index, test_index in kf.split(data):
        print("{:-^50}".format('Iniciando iteración {}'.format(cont)))
        train_data, test_data = data[train_index], data[test_index]
        it_metrics = run_network(
            properties, train_data, test_data, cont)
        metrics.append(it_metrics)
        cont += 1
        print("{:-^50}".format('Iteración terminada'))
    print("{:-^50}".format('Simulación terminada'))
    proms = calculate_proms(metrics)
    print("Métricas finales promediadas:")
    print("{:>14} {:.4}".format('Precision:', proms["precision"]))
    print("{:>14} {:.4}".format('Recall:', proms["recall"]))
    print("{:>14} {:.4}".format('F1:', proms["f1"]))
    print("{:>14} {:.4}".format('Cost:', proms["cost"]))
    print("{:-^50}".format('Mostrando métricas de las iteraciones'))

    plt.show()
