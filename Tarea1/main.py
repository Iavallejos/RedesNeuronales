from src.normalization import normalize
import numpy as np

if __name__ == "__main__":
    properties = {
        'hidden_layers': 2,
        'neurons_per_layer': 4,
        'input_length': 4,
        'number_of_classes': 3,
        'epoch': 100,
        'data': 'Data/Iris/data.npy',
        'dataset': 'Data/Iris/iris.data',
        'classes': 'Data/Iris/mapping.json'
    }
    normalize(properties)
    print("Loading data in: {}".format(properties["data"]))
    data = np.load(properties["data"], allow_pickle=True)
    print(data)
