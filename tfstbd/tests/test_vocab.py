import os
import tensorflow as tf
from ..hparam import build_hparams
from ..vocab import extract_vocab


class TestExtractVocab(tf.test.TestCase):
    def test_normal(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
        params = build_hparams({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 2,
            'ngram_maxn': 6,
            'ngram_freq': 2,
            'lstm_units': [1]
        })
        extract_vocab(data_path, params)


if __name__ == "__main__":
    tf.test.main()
