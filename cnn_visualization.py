from keras import models
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


model = load_model('cats_and_dogs_2.h5')
print(model.summary())

img_path = 'C:\\Users\\admin\\Downloads\\cnn_base_dir\\test\\cats\\cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)
plt.imshow(img_tensor[0])

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
forth_layer_activation = activations[3]
seventh_layer_activation = activations[6]
print(first_layer_activation.shape)
cmap = 'cool'
plt.matshow(first_layer_activation[0, :, :, 4], cmap=cmap)
plt.matshow(forth_layer_activation[0, :, :, 4], cmap=cmap)
plt.matshow(seventh_layer_activation[0, :, :, 4], cmap=cmap)

plt.show()
