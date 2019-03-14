from keras.applications import VGG16


conv_base = VGG16(weights='imagenet',  # pre-trained on ImageNet
                  include_top=False,  # without default dense classifier since custom one will be used
                  input_shape=(150, 150, 3))
