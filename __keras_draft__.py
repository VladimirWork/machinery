from sklearn.datasets.samples_generator import make_blobs  # raw data generation
from sklearn.preprocessing import normalize, scale  # normalize -> [0, 1] / scale -> [-1, 1]
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from utils import plotter


def get_simple_mlp(x, y, dimensions, classes, epochs):
    model = Sequential()
    model.add(Dense(256, input_dim=dimensions, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x, y, batch_size=64, epochs=epochs, verbose=0)
    return model, history


if __name__ == '__main__':
    # x samples defined by 2 parameters owned by 3 classes (0, 1, 2)
    x, y = make_blobs(n_samples=10000, centers=3, n_features=2, random_state=0)
    # 0 -> (1, 0, 0) / 1 -> (0, 1, 0) / 2 -> (0, 0, 1)
    y = to_categorical(y)
    # 75% pairs of x,y for train / 25% pairs of x,y for test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=0)
    n = 10
    classes = 3
    y_pred = None
    for i in range(n):
        model, history = get_simple_mlp(x_train, y_train, dimensions=2, classes=3, epochs=250)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print('Accuracy %s: %.3f' % (i, acc))
        print('Loss %s: %.3f' % (i, loss))
        model.save('models/model_{}.h5'.format(i))
        if y_pred is None:
            y_pred = model.predict(x_test, batch_size=32, verbose=0)
        else:
            y_pred = np.dstack((y_pred, model.predict(x_test, batch_size=32, verbose=0)))
    stacked_x = y_pred.reshape((y_pred.shape[0], y_pred.shape[1]*y_pred.shape[2]))
    # split stacked samples to train and test
    stacked_x_train, stacked_x_test,\
        stacked_y_train, stacked_y_test = train_test_split(stacked_x, y_test, train_size=0.75, random_state=0)
    stacked_model, stacked_history = get_simple_mlp(stacked_x_train, stacked_y_train,
                                                    dimensions=n*classes, classes=classes, epochs=250)
    stacked_loss, stacked_acc = stacked_model.evaluate(stacked_x_test, stacked_y_test, verbose=0)
    print('Stacked accuracy: %.3f' % stacked_acc)
    print('Stacked loss: %.3f' % stacked_loss)
    plotter(stacked_history.history)
