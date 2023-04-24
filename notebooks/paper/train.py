import json
import numpy as np
from optparse import OptionParser

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import GRU, Dense, RepeatVector, \
    TimeDistributed, Input, BatchNormalization, \
    concatenate, Activation, dot
from keras.callbacks import EarlyStopping

# Parse options
parser = OptionParser(usage="%prog")

# Configuration
parser.add_option("-r", "--rand", dest="rand", help="Rand rate", metavar="rand", default=0.0, type="float")

(options, args) = parser.parse_args()
rand_rate = options.rand

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

with np.load('data/dataset.npz', allow_pickle = True) as data:
    encoder_input_data = data['encoder_input_data']
    decoder_input_data = data['decoder_input_data']
    token_index = data['token_index'].tolist()

X_data = encoder_input_data #[:1000]
Y_data = decoder_input_data #[:1000]

## Construct model

n_hidden = 24
n_length = 41
n_splits = X_data.shape[1] // n_length

input_train = Input(name='parent_input', shape=(X_data.shape[1], X_data.shape[2]))
input_rand_rate = Input(name='random_input', shape=())
output_train = Input(name='child_input', shape=(Y_data.shape[1], Y_data.shape[2]))

splits = tf.split(input_train, num_or_size_splits=n_splits, axis=1)
concat = tf.concat(splits, axis=0)

encoder_stack_h, encoder_last_h = GRU(n_hidden, dropout=0.01, recurrent_dropout=0.01,
                                      return_sequences=True, return_state=True)(concat)

encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
decoder_input = RepeatVector(concat.shape[1])(encoder_last_h)

decoder_stack_h = GRU(n_hidden, dropout=0.01, recurrent_dropout=0.01,
                      return_state=False, return_sequences=True)(decoder_input, initial_state=encoder_last_h)

attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
attention = Activation('softmax')(attention)


class RandomDropout(keras.layers.Layer):

    def call(self, inputs, rate):
        return tf.nn.dropout(inputs, rate=rate[0])


context = dot([attention, encoder_stack_h], axes=[2, 1])
context = BatchNormalization(momentum=0.6)(context)
context = RandomDropout()(context, input_rand_rate)

decoder_combined_context = concatenate([context, decoder_stack_h])

out = TimeDistributed(Dense(concat.shape[2], activation="softmax"))(decoder_combined_context)

back_splits = tf.split(out, num_or_size_splits=n_splits, axis=0)
back_concat = tf.concat(back_splits, axis=1)

model = Model(inputs=[input_train, input_rand_rate], outputs=back_concat)
opt = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

batch_size = 64      # Batch size for training.
patience = 25        # Patience for EarlyStop
epochs = 300         # Number of epochs to train for.
if rand_rate >= 0.0:
    rand_rates = np.full((len(X_data)), rand_rate)
else:
    rand_rates = np.random.rand(len(X_data)) * np.abs(rand_rate)

es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True)
history = model.fit([X_data, rand_rates], Y_data,
                    validation_split=0.2,
                    epochs=epochs, verbose=1, callbacks=[es],
                    batch_size=batch_size)

hist = history.history

file_name = f"model_gru_24-41-r{rand_rate:0.3f}"
model.save(f"data/{file_name}")

with open(f"data/{file_name}_history.json", "w") as f:
    json.dump(hist, f)
