import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np
from datetime import datetime as dt
import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter
#import seaborn as sns
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
                Tensorflow Regression
'''
#app = Flask('nn_reg')
#@app.route('/nn_tf')
# python automatically passes self to a function; can be cause for 1 arg given even though function takes 0 arguments
# def nn_regression(self):

print("tensorflow regression running...")
print("Tensorflow version:", tf.__version__)
bank_df = df_encoder(rng=4, plots=False)
dataset = bank_df.copy()
print(dataset.head())
#%%
# setting label and features (the df itself here)
model_label = dataset.pop('amount_mean_lag7')
model_label.astype('int64')

# EAGER EXECUTION NEEDS TO BE ENABLED HERE
# features and model labels passed as tuple
tensor_ds = tf.data.Dataset.from_tensor_slices((dataset.values, model_label.values))
for feat, targ in tensor_ds.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))

train_dataset = tensor_ds.shuffle(len(bank_df)).batch(2)
#%%
#OPTIONAL WAY TO FEED DATA
##########################################
# ALTERNATIVE TO FEED FEATURES TO THE MODEL AS DICTIONARY CONSISTING OF DF KEYS
# inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
# x = tf.stack(list(inputs.values()), axis=-1)

# x = tf.keras.layers.Dense(10, activation='relu')(x)
# output = tf.keras.layers.Dense(1)(x)

# model_func = tf.keras.Model(inputs=inputs, outputs=output)

# model_func.compile(optimizer='adam',
#                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                     metrics=['mse', 'mae'])

# dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)
# for dict_slice in dict_slices.take(1):
#   print (dict_slice)
# model_func.fit(dict_slices, epochs=15)
###########################################
#%%
# if numpy arrays are given the first layers needs to be layers.Flatten and specify the quadratic input shape
# REGULARIZATION: add dropout or regularizers in each layer
# combined reg + dropout produces best results
# Dropout layers sets variables to zero in randomized patterns
# when adding weight relugarizer - monitor bin cross entropy directly
def compile_model():
    model = tf.keras.Sequential([
    # initial layer is input layers
    tf.keras.layers.Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.1),
    # final layers is output layers
    tf.keras.layers.Dense(1)
    ])
    # gradients / running average; optimizes stochastic gradient descent
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])
    return model
#%%
# connect to flask decorator

model = compile_model()
#model.fit(train_dataset, epochs=15)

EPOCHS = 10
# when dataset given; only ds needed (no y needed)
# when separate tensors given; arg x and y needed
history = model.fit(train_dataset, epochs=EPOCHS, verbose=2)

# regularizer_histories['reg_and_dropout'] = compile_model(combined_model, "regularizers/combined")
# plt.plot(history)


