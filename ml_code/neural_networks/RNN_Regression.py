import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np
from datetime import datetime as dt
import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from flask import Flask

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import feature_column, data
from tensorflow.keras import Model, layers, regularizers

#%%
'''
                    RNN Regression
single-step and multi-step model for a recurrent neural network
'''
#%%
print("Tensorflow regression:")
print("TF-version:", tf.__version__)
bank_df = df_encoder(rng=4)
dataset = bank_df.copy()
print(dataset.head())
sns.pairplot(bank_df[['amount', 'amount_mean_lag7', 'amount_std_lag7']])

# NO SPLIT UP FOR RNN HERE
# setting label and features (the df itself here)
#model_label = dataset.pop('amount_mean_lag7')
#model_label.astype('int64')

TRAIN_SPLIT = round(0.6 * len(dataset))
# normalize the training set; but not yet split up
train_dataset = dataset[:TRAIN_SPLIT]
dataset_norm = tf.keras.utils.normalize(train_dataset)
#%%
# feed the whole dataset into the function; split into targets/features happens there
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):

    '''

    Parameters
    ----------
    dataset : pandas dataframe
        Standard pandas dataframe.
    target : slice, series or array
        pass a slice, series or array as label.
    start_index : int
        DESCRIPTION.
    end_index : int
        DESCRIPTION.
    history_size : int
        DESCRIPTION.
    target_size : int
        DESCRIPTION.
    step : int
        DESCRIPTION.
    single_step : bool, optional
        specify the size of each data batch. The default is False.

    Returns
    -------
    Array - features; Array - label.

    '''

    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

    if single_step:
        labels.append(target[i+target_size])
    else:
        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)
#%%
ds_label = dataset_norm.pop('amount_mean_lag7')

# train dataset is already shortened and normalized
y_train_multi = ds_label
X_train_multi = dataset_norm[:TRAIN_SPLIT]
# referring to previous dataset; second slice becomes validation data until end of the data
y_val_multi = dataset.pop('amount_mean_lag7').iloc[TRAIN_SPLIT:]
X_val_multi = dataset.iloc[TRAIN_SPLIT:]


print("Shape y_train:", y_train_multi.shape)
print("Shape X_train:", X_train_multi.shape)
print("Shape y_val:", y_val_multi.shape)
print("Shape X_train:", X_val_multi.shape)

# pass as tuples to convert to tensor slices
# buffer_size can be equivalent to the entire length of the df; that way all of it is being shuffled
BUFFER_SIZE = len(train_dataset)
# Batch refers to the chunk of the dataset that is used for validating the predicitions
BATCH_SIZE = 64
# size of data chunk that is fed per time period
timestep = 1

# training dataframe
train_data_multi = tf.data.Dataset.from_tensor_slices((X_train_multi.values, y_train_multi.values))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
# validation dataframe
val_data_multi = tf.data.Dataset.from_tensor_slices((X_val_multi.values, y_val_multi.values))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

#######
# dimension required for a correct batch
# format shape (X= time steps, Y=Batch size(no. of examples, Z=Features))
#train_data_3d = np.reshape(X_train_multi, (timestep, BATCH_SIZE, X_train_multi.shape[1]))
#val_data_3d = np.reshape(X_val_multi, (timestep, BATCH_SIZE, X_val_multi.shape[1]))
#######
#%%
'''
                Recurring Neural Network
-LSTM cell in sequential network
'''
# Test of a RNN multi-step
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=X_train_multi.shape[-3:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
# we want to predict one single weekly average; dimensionality will be reduced
multi_step_model.add(tf.keras.layers.Flatten())
multi_step_model.add(tf.keras.layers.Dense(1))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001,
                                        clipvalue=1.0,
                                        name='RMSprop')

multi_step_model.compile(optimizer=optimizer,
                         loss='mse',
                         # needs to be a list (also with 1 arg)
                         metrics=['mae'])

print("Training data shape")
for x, y in train_data_multi.take(1):
    print (multi_step_model.predict(x).shape)


EPOCHS = 250
# steps within one epoch to validate with val_data
EVALUATION_INTERVAL = 200
multi_step_history = multi_step_model.fit(train_data_multi,
                                          epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50,
                                          verbose=2)
# retrieve weights and variables
#model.weights
#model.variables

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
#%%
'''
        SIMPLE RNN
'''

model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(layers.GRU(256, return_sequences=True))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(layers.SimpleRNN(128))

model.add(layers.Dense(10))

model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(dataset,
          batch_size=BATCH_SIZE,
          epochs=5,
          verbose=2)
print("Tensorflow regression finished...")

#%%
encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None, ))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)

# Return states in addition to output
output, state_h, state_c = layers.LSTM(64,
                                       return_state=True,
                                       name='encoder')(encoder_embedded)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None, ))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab,
                                    output_dim=64)(decoder_input)

# Pass the 2 states to a new LSTM layer, as initial state
decoder_output = layers.LSTM(64,
                             name='decoder')(decoder_embedded,
                                             initial_state=encoder_state)
output = layers.Dense(10)(decoder_output)

model = tf.keras.Model([encoder_input, decoder_input], output)
model.summary()

lstm_layer = layers.LSTM(64, stateful=True)

for s in sub_sequences:
    output = lstm_layer(s)
#%%
# Model resets weights of layers
paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)
output = lstm_layer(paragraph3)

# reset_states() will reset the cached state to the original initial_state.
# If no initial_state was provided, zero-states will be used by default.
lstm_layer.reset_states()
#%%
# Model reuses states/ weights
paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)

existing_state = lstm_layer.states

new_lstm_layer = layers.LSTM(64)
new_output = new_lstm_layer(paragraph3, initial_state=existing_state)
#%%
batch_size = 64
# Each data batch is a tensor of shape (batch_size, num_feat, num_feat)
#                                       (batch_size, 21, 21)
# Each input sequence will be of size (21, 21) (height is treated like time).
input_dim = 21

units = 64
output_size = 10  # labels are from 0 to 9

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
    # The LSTM layer with default options uses CuDNN.
        lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping an LSTMCell in an RNN layer will not use CuDNN.
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(units),
            input_shape=(None, input_dim))

    model = tf.keras.models.Sequential([
        lstm_layer,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(output_size)])

    return model
#%%
model = build_model(allow_cudnn_kernel=False)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(dataset,
          batch_size=BATCH_SIZE,
          epochs=5,
          verbose=2)
print("Tensorflow regression finished...")
