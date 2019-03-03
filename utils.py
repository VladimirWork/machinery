import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for (i, sequence) in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def plotter(history_dict):
    frame = pd.DataFrame.from_dict(history_dict)
    frame.plot(subplots=True, cmap='cool')
    plt.show()


def normalize(train_data, test_data):
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data -= mean
    train_data /= std
    test_data -= mean
    test_data /= test_data
    return train_data, test_data


def build_model():
    pass
