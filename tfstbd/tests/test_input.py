# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ..hparam import build_hparams
from ..input import train_dataset


class TestTrainInput(tf.test.TestCase):
    def test_normal(self):
        wildcard = os.path.join(os.path.dirname(__file__), 'data', '*.tfrecords.gz')
        params = build_hparams({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'word_mean': 1.,
            'word_std': 1.,
            'ngram_minn': 1,
            'ngram_maxn': 1,
            'ngram_freq': 6,
            'lstm_units': [1]
        })
        dataset = train_dataset(wildcard, params)

        for inputs in dataset.take(1):
            self.assertEqual(tuple, type(inputs))
            self.assertEqual(3, len(inputs))

            features, labels, weights = self.evaluate(inputs)

            self.assertEqual(dict, type(weights))
            self.assertEqual(['token'], sorted(weights.keys()))

            self.assertEqual(dict, type(labels))
            self.assertEqual(['sentence', 'space', 'token'], sorted(labels.keys()))

            self.assertEqual(dict, type(features))
            self.assertEqual([
                'word_feats',
                'word_ngrams',
                'word_tokens',
            ], sorted(features.keys()))

            self.assertTupleEqual(features['word_tokens'].shape, (2, None))
            self.assertTupleEqual(features['word_ngrams'].shape, (2, None, None))
            self.assertTupleEqual(features['word_feats'].shape, (2, None, 6))


if __name__ == "__main__":
    tf.test.main()
