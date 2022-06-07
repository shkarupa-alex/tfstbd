import os
import numpy as np
import tensorflow as tf
from ..config import build_config
from ..input import train_dataset, vocab_dataset


class TestTrainDataset(tf.test.TestCase):
    def test_normal(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
        config = build_config({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 1,
            'ngram_maxn': 1,
            'unit_freq': 6,
            'drop_reminder': False,
        })
        dataset = train_dataset(data_path, 'train', config)

        has_examples = False
        for inputs in dataset.take(1):
            has_examples = True
            self.assertIsInstance(inputs, tuple)
            self.assertEqual(3, len(inputs))

            features, labels, weights = self.evaluate(inputs)

            self.assertIsInstance(features, dict)
            self.assertEqual(['document'], sorted(features.keys()))

            self.assertIsInstance(labels, np.ndarray)

            self.assertIsInstance(weights, np.ndarray)

        self.assertTrue(has_examples)


class TestVocabDataset(tf.test.TestCase):
    def test_normal(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
        config = build_config({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 1,
            'ngram_maxn': 1,
            'unit_freq': 6,
            'drop_reminder': False,
        })
        dataset = vocab_dataset(data_path, config)

        has_examples = False
        for inputs in dataset.take(1):
            self.assertIsInstance(inputs, tuple)
            self.assertEqual(3, len(inputs))

            has_examples = True
            self.assertEqual(tf.RaggedTensor, type(inputs[0]))
            self.assertEqual(tf.RaggedTensor, type(inputs[1]))

        self.assertTrue(has_examples)


if __name__ == "__main__":
    tf.test.main()
