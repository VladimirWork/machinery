from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
from utils import preprocess_image, deprocess_image


target_image_path = ''
style_image_path = ''

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

# 1 target image for style applying
target_image = K.constant(preprocess_image(target_image_path))
# 2 style reference image
style_image = K.constant(preprocess_image(style_image_path))
# 3 placeholder tensor for result image
combination_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([target_image, style_image, combination_image], axis=0)
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
print('Model has been loaded.')
