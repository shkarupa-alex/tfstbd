from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization as _TextVectorization
from tensorflow.python.keras.layers.preprocessing.reduction import Reduction as _Reduction
from tensorflow.python.keras.backend import convert_inputs_if_ragged, maybe_convert_to_ragged


class TextVectorization(_TextVectorization):
    def __init__(self, vocabulary, **kwargs):
        if not isinstance(vocabulary, list):
            raise ValueError('Vocabulary should be a list')

        super(TextVectorization, self).__init__(
            max_tokens=len(vocabulary) + 1,  # + [UNK]
            standardize=None,
            split=None,
            pad_to_max_tokens=False,
            **kwargs)

        self.set_vocabulary(vocabulary)
        self._vocabulary = vocabulary

    def call(self, inputs):
        inputs, row_lengths = convert_inputs_if_ragged(inputs)
        is_ragged_input = (row_lengths is not None)

        shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, [1, -1])  # TODO https://github.com/tensorflow/tensorflow/issues/39504

        outputs = super(TextVectorization, self).call(inputs)

        outputs = tf.reshape(outputs, shape)
        outputs = maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
        outputs = outputs - 1  # We don't need [PAD] token

        return outputs

    def get_config(self):
        base_config = super(TextVectorization, self).get_config()
        config = {'vocabulary': self._vocabulary}

        return dict(list(base_config.items()) + list(config.items()))


class Reduction(_Reduction):
    def get_config(self):
        base_config = super(Reduction, self).get_config()
        config = {
            'reduction': self.reduction,
            'axis': self.axis,
        }

        return dict(list(base_config.items()) + list(config.items()))
