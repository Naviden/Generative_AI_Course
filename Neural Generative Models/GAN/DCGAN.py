# In this script, we first define the generator and discriminator models
# using Keras. The generator takes a noise vector as input and produces
# a 28x28 image, while the discriminator takes a 28x28 image as input and
# outputs a single scalar representing the probability that the input image is real.

# We then define an adversarial model that chains the generator and discriminator
# together, and compile the models using the Adam optimizer.

# Finally, we train the DCGAN using a loop where we generate fake images with
# the generator, mix them with real images, and train the discriminator to tell
#  them apart. We then generate a new batch of fake images and train the generator
#  via the adversarial model, with the goal of fooling the discriminator.

# Every 10 epochs, the script prints the loss of the discriminator and generator
# , and shows 10 generated images.

# Please note that training DCGANs can take a long time, especially without a GPU.
# This script is a basic example to help you get started, and there are many ways
# to improve and customize it to achieve better results.

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare the dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Create the generator model
def make_generator_model():
    model = Sequential([
        Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
        Reshape((7, 7, 256)),
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        LeakyReLU(),
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        LeakyReLU(),
        Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

generator = make_generator_model()

# Create the discriminator model
def make_discriminator_model():
    model = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        Flatten(),
        Dense(1)
    ])
    return model

discriminator = make_discriminator_model()

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# Compile the adversarial model
adversarial_model = Sequential([generator, discriminator])
discriminator.trainable = False
adversarial_model.compile(loss='binary_crossentropy', optimizer='adam')

# Training the DCGAN
epochs = 10000
batch_size = 256

for epoch in range(epochs):
    for _ in range(batch_size):
        noise = np.random.normal(0, 1, size=[batch_size, 100])
        generated_images = generator.predict(noise)
        image_batch = train_images[np.random.randint(low=0, high=train_images.shape[0], size=batch_size)]
        
        X = np.concatenate([image_batch, generated_images])
        y_dis = np.zeros(2 * batch_size)
        y_dis[:batch_size] = 0.9
        
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y_dis)
        
        noise = np.random.normal(0, 1, size=[batch_size, 100])
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        g_loss = adversarial_model.train_on_batch(noise, y_gen)
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
        samples = generator.predict(np.random.normal(0, 1, size=[10, 100]))
        plt.figure(figsize=(10, 1))
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(samples[i, :, :, 0], cmap='gray_r')
            plt.axis('off')
        plt.show()

# Show final generated images
samples = generator.predict(np.random.normal(0, 1, size=[10, 100]))
plt.figure(figsize=(10, 1))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(samples[i, :, :, 0], cmap='gray_r')
    plt.axis('off')
plt.show()
