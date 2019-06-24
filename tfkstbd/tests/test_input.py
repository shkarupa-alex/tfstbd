# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ..input import train_input


class TestTrainInput(tf.test.TestCase):
    def testNormal(self):
        batch_size = 2
        wildcard = os.path.join(os.path.dirname(__file__), 'data', '*.tfrecords.gz')
        dataset = train_input(wildcard, [], [batch_size], 1.0, 1.0, 1, 1)

        for features, labels in dataset.take(1):
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
                # 'words',
            ], sorted(features.keys()))

            del features['word_ngrams']  # breaks self.evaluate
            features, labels = self.evaluate([features, labels])
            self.assertEqual(batch_size, len(features['document']))

            self.assertEqual(dict, type(labels))
            self.assertEqual(['sentences', 'tokens'], sorted(labels.keys()))

            self.assertEqual(2, len(labels['tokens'].shape))
            self.assertEqual(batch_size, labels['sentences'].shape[0])

            # self.assertAllEqual(labels['tokens'].shape, features['words'].shape)
            self.assertAllEqual(labels['sentences'].shape, features['word_length'].shape)


if __name__ == "__main__":
    tf.test.main()
