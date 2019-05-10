from sklearn.datasets.samples_generator import make_blobs
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from matplotlib import pyplot
from os import makedirs


def fit_model(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(64, input_dim=2, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=500, verbose=0)
    return model


def load_models(n):
    all_models = []
    for i in range(n):
        filename = 'models/model_' + str(i + 1) + '.h5'
        model = load_model(filename)
        all_models.append(model)
        print('>>> Loaded %s' % filename)
    return all_models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = np.dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit stacked model
    # model = DecisionTreeClassifier()  # Stacked model accuracy: 1.000
    # model = ExtraTreeClassifier()  # Stacked model accuracy: 1.000
    # model = ExtraTreesClassifier()  # Stacked model accuracy: 1.000
    model = RandomForestClassifier()  # Stacked model accuracy: 0.980 / 0.970
    model.fit(stackedX, inputy)
    return model


# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat


if __name__ == '__main__':
    # generate 2d classification dataset
    X, y = make_blobs(n_samples=1100, centers=5, n_features=2, cluster_std=2, random_state=2)
    # one hot encode output variable
    y = to_categorical(y)
    # split into train and test
    n_train = 1000
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    print(trainX.shape, testX.shape)

    n_members = 10
    try:
        makedirs('models')
    except FileExistsError:
        pass
    for i in range(n_members):
        # fit model
        model = fit_model(trainX, trainy)
        # save model
        filename = 'models/model_' + str(i + 1) + '.h5'
        model.save(filename)
        print('>>> Saved %s' % filename)
    # evaluate standalone models on test dataset
    members = load_models(n_members)
    # evaluate standalone models on test dataset
    for model in members:
        _, acc = model.evaluate(testX, testy, verbose=0)
        print('Single model accuracy: %.3f' % acc)
    # fit stacked model using the ensemble
    model = fit_stacked_model(members, testX, testy)
    # evaluate model on test set
    yhat = stacked_prediction(members, model, testX)
    acc = accuracy_score(testy, yhat)
    print('Stacked model accuracy: %.3f' % acc)
