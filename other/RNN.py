# In this code, we first load the IMDB dataset, which consists of movie reviews
# labeled as positive (1) or negative (0). We then preprocess the data by padding
# the sequences to ensure they have the same length.

# The RNN model is defined using the Keras Sequential API. It consists of an Embedding
# layer, a Simple RNN layer, and a Dense output layer with a sigmoid activation function.

# We compile the model using the RMSprop optimizer and binary crossentropy as the loss
# function, since this is a binary classification problem.

# After training the model for 10 epochs, we evaluate it on the test data to check its accuracy.

# Please note that RNNs can be quite slow to train, and there are more advanced types of
# recurrent layers available in Keras, such as LSTM and GRU, which can lead to better
# performance on tasks like this one.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load the IMDB dataset
max_features = 10000  # Number of words to consider as features
maxlen = 500  # Cut texts after this number of words (among top max_features most common words)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Define the RNN model
model = models.Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Train the RNN model
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
