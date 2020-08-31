# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from nlpvocab import Vocabulary
from ..hparam import build_hparams
from ..input import train_dataset
from ..model import build_model


class TestBuildModel(tf.test.TestCase):
    def test_lstm(self):
        wildcard = os.path.join(os.path.dirname(__file__), 'data', '*.tfrecords.gz')
        params = build_hparams({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 2,
            'ngram_maxn': 6,
            'ngram_freq': 2,
            'lstm_units': [1]
        })
        dataset = train_dataset(wildcard, params)
        ngrams = Vocabulary({
            '< >': 100, '<кто>': 10, '<знает>': 2, '<зна': 4, 'ает>': 4, '<,>': 20, '<что>': 15, '<он>': 10,
            '<там>': 5, '<думал>': 1, '!..': 1
        })
        model = build_model(params, ngrams)

        model.compile(
            optimizer='Adam',
            loss={
                'space': 'binary_crossentropy',
                'token': 'binary_crossentropy',
                'sentence': 'binary_crossentropy',
            },
            run_eagerly=False,
        )
        model.summary()
        history = model.fit(dataset, epochs=1, steps_per_epoch=2)
        self.assertGreater(len(history.history['loss']), 0)


if __name__ == "__main__":
    tf.test.main()