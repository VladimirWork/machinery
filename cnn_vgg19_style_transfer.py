from keras.preprocessing.image import load_img
from keras.applications import vgg19
from keras import backend as K
from utils import preprocess_image, deprocess_image
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time
import numpy as np


# target_image_path = 'C:\\Users\\admin\\Downloads\\blagoveshhenskij-sobor-voronezh-min-e1498909085781.jpg'
target_image_path = 'C:\\Users\\admin\\Downloads\\56R_GovTRFo.jpg'
# style_image_path = 'C:\\Users\\admin\\Downloads\\1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
# style_image_path = 'C:\\Users\\admin\\Downloads\\Claude_Monet,_Impression,_soleil_levant.jpg'
# style_image_path = 'C:\\Users\\admin\\Downloads\\00-edvard-munch-the-scream-1893.jpg'
# style_image_path = 'C:\\Users\\admin\\Downloads\\flowers.jpg'
# style_image_path = 'C:\\Users\\admin\\Downloads\\utopia_yellow_style.jpg'
style_image_path = 'C:\\Users\\admin\\Downloads\\magenta_astronaut_style.jpg'

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

# 1 target image for style applying
target_image = K.constant(preprocess_image(target_image_path, img_height, img_width))
# 2 style reference image
style_image = K.constant(preprocess_image(style_image_path, img_height, img_width))
# 3 placeholder tensor for result image
combination_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([target_image, style_image, combination_image], axis=0)
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
print('Model has been loaded.')


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])


class Evaluator:
    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

result_prefix = 'figures/magenta_vs_photo'
iterations = 10
x = preprocess_image(target_image_path, img_height, img_width)
x = x.flatten()

# style transfer cycle
for i in range(iterations):
    print('Start of iteration {}'.format(i))
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Current loss value: {}'.format(min_val))
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_{}.png'.format(i)
    imsave(fname, img)
    print('Image saved as {}'.format(fname))
    end_time = time.time()
    print('Iteration {} completed in {}'.format(i, end_time - start_time))
