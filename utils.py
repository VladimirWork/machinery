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
