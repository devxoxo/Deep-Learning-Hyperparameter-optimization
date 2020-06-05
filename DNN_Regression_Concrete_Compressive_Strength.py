import tensorflow as tf
import numpy as np
import argparse
import csv
from datetime import datetime, timezone

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to run trainer.')
parser.add_argument('--batch_size', type=int, default=50,
                    help='Number of samples per gradient update.')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate')
parser.add_argument('--activation_func', type=str, default='relu',
                    help='activation function',
                    choices=['tanh', 'relu', 'elu'])
args = parser.parse_args()

# data load
train_data_path = 'Concrete_Data_train.csv'

feature = np.zeros((927, 8))
y = np.zeros((927, 1))
with open(train_data_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for i, row in enumerate(reader):
        y[i, :] = row[-1]
        feature[i, :] = row[:8]

nrow, ncol = feature.shape

# reshape input feature
feature = tf.reshape(feature, (-1, 8))

# train, validation data split (9:1)
full_dataset = tf.data.Dataset.from_tensor_slices((feature, y))

train_size = int(0.9 * nrow)
val_size = nrow - train_size
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size)

train_dataset = train_dataset.shuffle(train_size).batch(args.batch_size).repeat()
val_dataset = val_dataset.batch(args.batch_size)

# create DNN model
model = tf.keras.Sequential()
model.add(Dense(256, activation=args.activation_func, input_shape=(8,)))
model.add(Dense(128, activation=args.activation_func))
model.add(Dense(32, activation=args.activation_func))
model.add(Dense(1, activation=args.activation_func))

opt = optimizers.Adam(lr=args.learning_rate, clipvalue=0.5)
model.compile(optimizer=opt,
              loss='mse')
model.summary()

# print metrics according to the format {{metricsName}}={{metricsValue}}
class MetricHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        local_time = datetime.now(timezone.utc).astimezone().isoformat()
        print("\nEpoch {}".format(epoch + 1))
        print("{} mse={:.4f}".format(local_time, logs['loss']))
        print("{} Validation-mse={:.4f}".format(local_time, logs['val_loss']))

history = MetricHistory()

# train model
model.fit(train_dataset,
          epochs=args.epochs,
          steps_per_epoch=train_size // args.batch_size,
          validation_data=val_dataset,
          callbacks=[history])