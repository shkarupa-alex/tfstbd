from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tfmiss.keras.layers import TemporalConvNet, ToDense
from .layer import Reduction


def build_model(h_params, ngram_keys):
    word_tokens = tf.keras.layers.Input(shape=(None,), name='word_tokens', dtype=tf.string, ragged=True)
    word_ngrams = tf.keras.layers.Input(shape=(None, None), name='word_ngrams', dtype=tf.string, ragged=True)
    word_feats = tf.keras.layers.Input(shape=(None, 6), name='word_feats', dtype=tf.float32, ragged=True)

    outputs = StringLookup(vocabulary=ngram_keys, name='ngrams_lookup')(word_ngrams)
    outputs = tf.keras.layers.Embedding(
        len(ngram_keys) + 1, h_params.ngram_dim, name='ngram_embedding')(outputs)
    outputs = Reduction(h_params.ngram_comb, name='word_embedding')(outputs)
    outputs = tf.keras.layers.concatenate([outputs, word_feats], name='word_features')
    outputs = ToDense(pad_value=0., mask=True, name='dense_logits')(outputs)

    if 'lstm' == h_params.seq_core:
        for i, units in enumerate(h_params.lstm_units):
            outputs = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=True),
                name='sequence_features_{}'.format(i))(outputs)
    elif 'causal' == h_params.tcn_padding:
        outputs = tf.keras.layers.Bidirectional(
            TemporalConvNet(h_params.tcn_filters, h_params.tcn_ksize, h_params.tcn_drop))(outputs)
    else:
        outputs = TemporalConvNet(h_params.tcn_filters, h_params.tcn_ksize, h_params.tcn_drop, 'same')(outputs)


    # TODO https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold/

    outputs_space = tf.keras.layers.Dense(1)(outputs)
    outputs_space = tf.keras.layers.Activation('sigmoid', dtype='float32', name='space')(outputs_space)

    outputs_token = tf.keras.layers.Dense(1)(outputs)
    outputs_token = tf.keras.layers.Activation('sigmoid', dtype='float32', name='token')(outputs_token)

    outputs_sentence = tf.keras.layers.Dense(1)(outputs)
    outputs_sentence = tf.keras.layers.Activation('sigmoid', dtype='float32', name='sentence')(outputs_sentence)

    model = tf.keras.Model(
        inputs=[word_tokens, word_ngrams, word_feats],
        outputs=[word_tokens, outputs_space, outputs_token, outputs_sentence]
    )

    return model
