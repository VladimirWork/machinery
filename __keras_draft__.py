from sklearn.datasets.samples_generator import make_blobs  # raw data generation
from sklearn.preprocessing import normalize, scale  # normalize -> [0, 1] / scale -> [-1, 1]
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from utils import plotter


def get_simple_mlp(x, y, val_x, val_y, classes, epochs):
    model = Sequential()
    model.add(Dense(64, input_dim=2, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x, y, epochs=epochs, validation_data=(val_x, val_y), verbose=0)
    return model, history


if __name__ == '__main__':
    # 1000 samples defined by 2 values owned by 3 classes (0, 1, 2)
    x, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)
    # 0 -> (1, 0, 0) / 1 -> (0, 1, 0) / 2 -> (0, 0, 1)
    y = to_categorical(y)
    # 75% pairs of x,y for train / 25% pairs of x,y for test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=0)
    _, history = get_simple_mlp(x_train, y_train, x_test, y_test, classes=3, epochs=100)
    plotter(history.history)
