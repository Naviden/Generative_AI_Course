# This code sets up and trains a simple GAN on the MNIST dataset for handwritten
# digit generation. The generator network takes a random noise vector as input and 
# produces a 28x28 image, while the discriminator network tries to distinguish between
# real MNIST images and images produced by the generator.

# After training for 50 epochs (you can adjust the number of epochs for better results),
#  the script generates a single image using the trained generator and displays it.

# Note: Training GANs can be tricky, and they may require careful tuning and experimentation
# to get good results. This is a basic example to get you started.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential

# Load MNIST data
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Build the generator
def make_generator_model():
    model = Sequential([
        Dense(128, use_bias=True, activation='relu', input_shape=(100,)),
        Dense(784, use_bias=True, activation='tanh'),
        Reshape((28, 28))
    ])
    return model

# Build the discriminator
def make_discriminator_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, use_bias=True, activation='relu'),
        Dense(1)
    ])
    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

# Define the loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Train the GAN
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
            
        # Produce images for the GIF as we go
        print(f"Epoch {epoch + 1} completed")

EPOCHS = 50
train(train_dataset, EPOCHS)

# Generate an image using the trained generator
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :], cmap='gray')
plt.show()
