from keras.datasets import boston_housing
from keras import models
from keras import layers


(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
