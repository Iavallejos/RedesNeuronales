from src.normalization import normalize
import numpy as np

if __name__ == "__main__":
    properties = {
        'layers': 2,
        'inputs': 4,
        'epoch': 100,
        'data': 'Data/Iris/data.npy',
        'dataset': 'Data/Iris/iris.data',
        'classes': 'Data/Iris/mapping.json'
    }
    normalize(properties)

    data = np.load(properties["data"])

