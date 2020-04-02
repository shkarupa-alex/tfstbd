# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ..hparam import build_hparams
from ..input import train_input_fn


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
        })
        dataset = train_input_fn(wildcard, params)

        for inputs in dataset.take(1):
            self.assertEqual(tuple, type(inputs))
            self.assertEqual(3, len(inputs))

            features, labels, weights = inputs

            self.assertEqual(dict, type(weights))
            self.assertEqual(['token'], sorted(weights.keys()))

            self.assertEqual(dict, type(labels))
            self.assertEqual(['sentence', 'space', 'token'], sorted(labels.keys()))

            self.assertEqual(dict, type(features))
            self.assertEqual([
                'document',
                'length',
                'word_length',
                'word_lower',
                'word_mixed',
                'word_ngrams',
                'word_nocase',
                'word_title',
                'word_upper',
                'words',
            ], sorted(features.keys()))

            features, labels, weights = self.evaluate([features, labels, weights])
            # self.assertEqual(10, len(features['document']))

            # self.assertEqual(3, len(labels['token'].shape))
            # self.assertEqual(10, labels['sentences'].shape[0])

            # self.assertAllEqual(labels['tokens'].shape, features['words'].shape)
            # self.assertAllEqual(labels['sentences'].shape, features['word_length'].shape)


if __name__ == "__main__":
    tf.test.main()
