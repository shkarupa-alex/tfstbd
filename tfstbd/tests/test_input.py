import os
import tensorflow as tf
from ..hparam import build_hparams
from ..input import train_dataset, vocab_dataset


class TestTrainDataset(tf.test.TestCase):
    def test_normal(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
        params = build_hparams({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 1,
            'ngram_maxn': 1,
            'ngram_freq': 6,
            'lstm_units': [1],
            'rdw_loss': True
        })
        dataset = train_dataset(data_path, 'train', params)

        has_examples = False
        for inputs in dataset.take(1):
            has_examples = True
            self.assertIsInstance(inputs, tuple)
            self.assertEqual(3, len(inputs))

            features, labels, weights = self.evaluate(inputs)

            self.assertIsInstance(features, dict)
            self.assertEqual(['document', 'repdivwrap'], sorted(features.keys()))

            self.assertIsInstance(labels, dict)
            self.assertEqual(['sentence', 'space', 'token'], sorted(labels.keys()))

            self.assertIsInstance(weights, dict)
            self.assertEqual(['sentence', 'space', 'token'], sorted(weights.keys()))

        self.assertTrue(has_examples)


class TestVocabDataset(tf.test.TestCase):
    def test_normal(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
        params = build_hparams({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 1,
            'ngram_maxn': 1,
            'ngram_freq': 6,
            'lstm_units': [1]
        })
        dataset = vocab_dataset(data_path, params)

        has_examples = False
        for inputs in dataset.take(1):
            has_examples = True
            self.assertEqual(tf.RaggedTensor, type(inputs))

        self.assertTrue(has_examples)


if __name__ == "__main__":
    tf.test.main()
