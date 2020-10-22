import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import feature_column, data
from tensorflow.keras import Model, layers, regularizers

#%%
'''
            Multivariate Regression
-multivariate regression with Long Short-Term Memory cells
'''
print("Tensorflow regression:")
print("TF-version:", tf.__version__)
bank_df = df_encoder(rng=4)
dataset = bank_df.copy()
print(dataset.head(3))


# split and normalize data
TRAIN_SPLIT = 250
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset_norm = (dataset-data_mean)/data_std
#%%
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
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
# shuffle data and create batches to be fed per intervall
# DF NEEDS TO BE ARRAY HERE
# slicing df .iloc[:, 17]
# slicing array [:, 17]

BUFFER_SIZE = len(dataset)
past_history = 42
future_target = 7
STEP = 1

X_train_single, y_train_single = multivariate_data(np.array(dataset), np.array(dataset)[:, 17],
                                                   0, TRAIN_SPLIT,
                                                   past_history, future_target,
                                                   STEP,
                                                   single_step=True)
X_val_single, y_val_single = multivariate_data(np.array(dataset), np.array(dataset)[:, 17],
                                               TRAIN_SPLIT, None,
                                               past_history, future_target,
                                               STEP,
                                               single_step=True)


# tensor created from separate variablesp assed as tuple
train_data_single = tf.data.Dataset.from_tensor_slices((X_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((X_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE, drop_remainder=True).repeat()
#%%
single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=X_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                          loss='mse',
                          metrics=['mae'])


for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)

EPOCHS = 150
EVALUATION_INTERVAL = 200

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)

def plot_train_history(history, title):
    loss = history.history[['mae', 'mse']]
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()

plot_train_history(single_step_history,
                   'Single Step Training and validation loss')

for x, y in val_data_single.take(3):
  plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                    single_step_model.predict(x)[0]], 12,
                   'Single Step Prediction')
  plot.show()