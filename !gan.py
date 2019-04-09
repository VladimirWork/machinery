import keras
from keras import layers
import numpy as np
import os
from keras.preprocessing import image


latent_dim = 32
width = 32
height = 32
channels = 3

# GENERATOR
generator_input = keras.Input(shape=(latent_dim,))

# feature map [16 x 16] x 128 channels
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

# rise definition to [32 x 32]
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

generator = keras.models.Model(generator_input, x)
generator.summary()
# ---

# DISCRIMINATOR
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)  # dropout 0.4 can be fine-tuned
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008,
                                                   clipvalue=1.0,
                                                   decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy')
# ---

discriminator.trainable = False  # freeze discriminator's weights

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004,
                                         clipvalue=1.0,
                                         decay=1e-8)
gan.compile(optimizer=gan_optimizer,
            loss='binary_crossentropy')

(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()  # CIFAR10: 50k [32 x 32] RGB images with 10 classes
x_train = x_train[y_train.flatten() == 6]  # 6 is FROG class
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.  # normalize

iterations = 10000
batch_size = 20
save_dir = 'figures'
start = 0

for step in range(iterations):
    # select random vectors from latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # turn them into fake images
    generated_images = generator.predict(random_latent_vectors)
    # mix fake and real images
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])
    # gather labels for fake and real images
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    # hint: add noise to labels
    labels += 0.05 * np.random.random(labels.shape)
    # train discriminator
    d_loss = discriminator.train_on_batch(combined_images, labels)
    # select random vectors from latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # gather false labels
    misleading_targets = np.zeros((batch_size, 1))
    # train generator
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    # save images every 100 steps
    if step % 100 == 0:
        gan.save_weights('gan.h5')
        print('discriminator loss: {}'.format(d_loss))
        print('adversarial loss: {}'.format(a_loss))
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))
