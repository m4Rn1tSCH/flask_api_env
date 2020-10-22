'''
RNN with Layer State Refeed
'''
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
print("Tensorflow regression:")
print("TF-version:", tf.__version__)
bank_df = df_encoder(rng=4)
dataset = bank_df.copy()
print(dataset.head())
# sns.pairplot(bank_df[['amount', 'amount_mean_lag7', 'amount_std_lag7']])

TRAIN_SPLIT = 750

# normalize the training set; but not yet split up
train_dataset = dataset[:TRAIN_SPLIT]
train_ds_norm = tf.keras.utils.normalize(train_dataset)
val_ds_norm = tf.keras.utils.normalize(dataset[TRAIN_SPLIT:])

# train dataset is already shortened and normalized
y_train_multi = train_ds_norm.pop('amount_mean_lag7')
X_train_multi = train_ds_norm[:TRAIN_SPLIT]
# referring to previous dataset; second slice becomes validation data until end of the data
y_val_multi = val_ds_norm.pop('amount_mean_lag7')
X_val_multi = val_ds_norm

print("Shape y_training:", y_train_multi.shape)
print("Shape X_training:", X_train_multi.shape)
print("Shape y_validation:", y_val_multi.shape)
print("Shape X_validation:", X_val_multi.shape)

# buffer_size can be equivalent to the entire length of the df; that way all of it is being shuffled
BUFFER_SIZE = len(train_dataset)

# Batch refers to the chunk of the dataset that is used for validating the predicitions
BATCH_SIZE = 21

# size of data chunk that is fed per time period
# weekly expenses are the label; one week's sexpenses are fed to the layer
timestep = 7

# pass as tuples to convert to tensor slices
#   if pandas dfs fed --> .values to retain rectangular shape and avoid ragged tensors
#   if 2 separate df slices (X/y) fed --> no .values and reshaping needed

# turn the variables into arrays; convert to:
# (X= batch_size(examples), Y=timesteps, Z=features)

# training dataframe
X_train_multi = np.array(X_train_multi)
X_train_multi = np.reshape(X_train_multi, (X_train_multi.shape[0], 1, X_train_multi.shape[1]))
train_data_multi = tf.data.Dataset.from_tensor_slices((X_train_multi, y_train_multi))
# generation of batches
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=False).repeat()

# validation dataframe
X_val_multi = np.array(X_val_multi)
X_val_multi = np.reshape(X_val_multi, (X_val_multi.shape[0], 1, X_val_multi.shape[1]))
val_data_multi = tf.data.Dataset.from_tensor_slices((X_val_multi, y_val_multi))
# generation of batches
val_data_multi = val_data_multi.batch(BATCH_SIZE, drop_remainder=False).repeat()
#%%
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (None, 1, 21) (height is treated like time).
# stateful=True to reuse weights of one step
input_dim = 28

units = 128
output_size = 1

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = tf.keras.layers.LSTM(units, stateful=True, input_shape=(None, X_train_multi.shape[2]))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(units, stateful=True),
        input_shape=(None, input_dim))

    model = tf.keras.models.Sequential([
        lstm_layer,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(output_size)
        ])
    return model
#%%
model = build_model(allow_cudnn_kernel=True)

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.RMSprop(0.001),
              metrics=['mae'])
# fix divisibility problem (sample size to steps per epoch)
# model.fit(X_train_multi,y_train_multi,
#           validation_data=(X_val_multi, y_val_multi),
#           epochs=250,
#           # evaluation steps need to consume all samples without remainder
#           steps_per_epoch=125,
#           validation_steps=250)

model.fit(train_data_multi, epochs=250,
          steps_per_epoch=125,
          # evaluation steps need to consume all samples without remainder
          validation_data=val_data_multi,
          validation_steps=125)




