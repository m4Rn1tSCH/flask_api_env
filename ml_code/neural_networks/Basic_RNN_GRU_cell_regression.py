import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import feature_column, data
from tensorflow.keras import Model, layers, regularizers

#%%
'''
            Multivariate Regression
-multivariate regression with single Gradient Reduction Unit
'''
print("Tensorflow regression:")
print("TF-version:", tf.__version__)
bank_df = df_encoder(rng=1, plots=False)
dataset = bank_df.copy()
print(dataset.head())
# sns.pairplot(bank_df[['amount', 'amount_mean_lag7', 'amount_std_lag7']])

TRAIN_SPLIT = 750

# normalize the training set; but not yet split up
train_dataset = dataset[:TRAIN_SPLIT]
train_ds_norm = tf.keras.utils.normalize(train_dataset)
val_ds_norm = tf.keras.utils.normalize(dataset[TRAIN_SPLIT:])

# train dataset is already shortened and normalized
y_train_raw = train_ds_norm.pop('amount_mean_lag7')
X_train_raw = train_ds_norm[:TRAIN_SPLIT]
# referring to previous dataset; second slice becomes validation data until end of the data
y_val_raw = val_ds_norm.pop('amount_mean_lag7')
X_val_raw = val_ds_norm

print("Shape y_training:", y_train_raw.shape)
print("Shape X_training:", X_train_raw.shape)
print("Shape y_validation:", y_val_raw.shape)
print("Shape X_validation:", X_val_raw.shape)

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

train_data_multi = tf.data.Dataset.from_tensor_slices((X_train_raw.values, y_train_raw.values))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
# validation dataframe
val_data_multi = tf.data.Dataset.from_tensor_slices((X_val_raw.values, y_val_raw.values))
val_data_multi = val_data_multi.batch(BATCH_SIZE, drop_remainder=True).repeat()
#%%

# training dataframe
# X_train_raw = np.array(X_train_raw)
# y_train_raw = np.array(y_train_raw)
# X_train_multi = np.reshape(X_train_raw, (X_train_raw.shape[0], 1, X_train_raw.shape[1]))
# train_data_multi = tf.data.Dataset.from_tensor_slices((X_train_multi, y_train_multi))
# # generation of batches
# train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=False).repeat()

# # validation dataframe
# X_val_raw = np.array(X_val_raw)
# y_val_raw = np.array(y_val_raw)
# X_val_multi = np.reshape(X_val_raw, (X_val_raw.shape[0], 1, X_val_raw.shape[1]))
# val_data_multi = tf.data.Dataset.from_tensor_slices((X_val_multi, y_val_multi))
# # generation of batches
# val_data_multi = val_data_multi.batch(BATCH_SIZE, drop_remainder=False).repeat()

model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(layers.GRU(256, return_sequences=True))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(layers.SimpleRNN(128))
model.add(layers.Dropout(rate=0.2, seed=True))
model.add(layers.Dense(1))

model.summary()

#%%
# compile model and optimize with mean absolute error
model.compile(loss='mse',
              optimizer=tf.keras.optimizers.RMSprop(0.001),
              metrics=['mae'])

model.fit(train_data_multi, epochs=5,
          steps_per_epoch=125,
          # evaluation steps need to consume all samples without remainder
          validation_data=val_data_multi,
          validation_steps=125)
