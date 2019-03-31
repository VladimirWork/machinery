import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models, layers
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array


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
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(13,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


def sample(preds, temperature=1.0):
    preds = np.array(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def preprocess_image(image_path):
    width, height = load_img(image_path).size
    img_height = 400
    img_width = int(width * img_height / height)
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    # some dark magic for image deprocessing
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
