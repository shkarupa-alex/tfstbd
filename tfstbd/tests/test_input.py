# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ..hparam import build_hparams
from ..input import train_dataset, vocab_dataset


class TestTrainDataset(tf.test.TestCase):
    def test_normal(self):
        wildcard = os.path.join(os.path.dirname(__file__), 'data', '*.tfrecords.gz')
        params = build_hparams({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 1,
            'ngram_maxn': 1,
            'ngram_freq': 6,
            'lstm_units': [1]
        })
        dataset = train_dataset(wildcard, params)

        checked = False
        for inputs in dataset.take(1):
            checked = True
            self.assertEqual(tuple, type(inputs))
            self.assertEqual(3, len(inputs))

            features, labels, weights = self.evaluate(inputs)

            self.assertEqual(dict, type(weights))
            self.assertEqual(['token'], sorted(weights.keys()))

            self.assertEqual(dict, type(labels))
            self.assertEqual(['sentence', 'space', 'token'], sorted(labels.keys()))

            self.assertEqual(dict, type(features))
            self.assertEqual(['documents'], sorted(features.keys()))

        self.assertTrue(checked)


class TestVocabDataset(tf.test.TestCase):
    def test_normal(self):
        wildcard = os.path.join(os.path.dirname(__file__), 'data', '*.tfrecords.gz')
        params = build_hparams({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 1,
            'ngram_maxn': 1,
            'ngram_freq': 6,
            'lstm_units': [1]
        })
        dataset = vocab_dataset(wildcard, params)

        checked = False
        for inputs in dataset.take(1):
            checked = True
            self.assertEqual(tf.RaggedTensor, type(inputs))

        self.assertTrue(checked)


if __name__ == "__main__":
    tf.test.main()
