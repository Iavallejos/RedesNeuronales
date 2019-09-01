import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def calculate_cost(expected, results):
    cost = np.sum((results - expected)**2) / len(expected)

    return cost


def calculate_proms(params):
    precision = 0
    recall = 0
    f1 = 0
    cost = 0

    for iteration in params:
        precision += iteration["precision"][-1]
        recall += iteration["recall"][-1]
        f1 += iteration["f1"][-1]
        cost += iteration["cost"][-1]

    proms = {
        "precision": precision/len(params),
        "recall": recall/len(params),
        "f1": f1/len(params),
        "cost": cost/len(params)
    }

    return proms


def plot_iteraton(title, metrics_data, matrix_data):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)

    precision, recall, f1, cost, epoch = metrics_data
    plot_metrics(ax[0], precision, recall, f1, cost,
                 epoch, title="Network Metrics")

    expected, predictions, classes = matrix_data    
    plot_confusion_matrix(ax[1], expected, predictions,
                          classes, title="Confusion Matrix")
    fig.set_size_inches(12, 4.5, forward=True)                          
    fig.tight_layout()


def plot_metrics(ax, precision, recall, f1, cost, epoch, title=None):
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.plot(epoch, precision, label="Precision")
    ax.plot(epoch, recall, label="Recall")
    ax.plot(epoch, f1, label="F1")
    ax.plot(epoch, cost, label="Cost")
    ax.legend(loc='best')
    return ax


def plot_confusion_matrix(ax, y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    #fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Expected label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax
