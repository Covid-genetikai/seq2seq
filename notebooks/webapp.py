import os
import keras
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)


LATENT_DIM = 256
SEQ_LEN = 3822
TOKEN_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def encode_seq(seqs: list):
    lens = np.vectorize(len)(seqs)
    vecs = np.zeros((len(seqs), np.max(lens), len(TOKEN_INDEX)), dtype="float32")
    for i, input_text in enumerate(seqs):
        for t, char in enumerate(input_text):
            vecs[i, t, TOKEN_INDEX[char]] = 1.0
    return vecs


def decode_seq(vecs: list):
    reverse_char_index = dict((i, char) for char, i in TOKEN_INDEX.items())
    return [''.join([reverse_char_index[i] for i in s]) for s in np.argmax(vecs, axis=2)]


def load_model(model_path: str):

    keras.backend.clear_session()
    model = keras.models.load_model(model_path)

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h1_enc, state_h2_enc = model.layers[2].output  # gru
    encoder_model = keras.Model(encoder_inputs, [ state_h1_enc, state_h2_enc ])

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h1 = keras.Input(shape=(LATENT_DIM,))
    decoder_states_inputs = [decoder_state_input_h1]

    decoder_gru = model.layers[3]
    decoder_outputs, state_h1_dec = decoder_gru(
        decoder_inputs, initial_state=decoder_states_inputs
    )

    decoder_states = [state_h1_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    print(f"Loaded model from {model_path}.")

    return encoder_model, decoder_model


def decode_sequence_batch(input_seqs, encoder_model, decoder_model, progress=None, label=None):
    n_seqs = input_seqs.shape[0]

    output_seqs = np.zeros_like(input_seqs)

    # Encode the input as state vectors.
    state_h1, state_h2 = encoder_model.predict(input_seqs)
    states_value = [state_h1]

    # Generate empty target sequence of length 1.
    target_seqs = np.zeros((n_seqs, 1, len(TOKEN_INDEX)))

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    for i in range(SEQ_LEN):

        if progress is not None:
            progress.value = i + 1

        if label is not None:
            label.value = f'{i + 1}/{SEQ_LEN}'

        output_token, state_h1 = decoder_model.predict([target_seqs] + states_value)

        # Sample a token
        sampled_token_indexes = np.squeeze(np.argmax(output_token, axis=2))
        output_seqs[:, i, sampled_token_indexes] = 1.0

        # Update the target sequence (of length 1).
        target_seqs = np.zeros((n_seqs, 1, len(TOKEN_INDEX)))
        target_seqs[:, 0, sampled_token_indexes] = 1.0

        states_value = [state_h1]

    return output_seqs


