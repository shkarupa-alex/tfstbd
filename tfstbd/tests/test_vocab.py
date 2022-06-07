import os
import tensorflow as tf
from ..config import build_config
from ..vocab import extract_vocab


class TestExtractVocab(tf.test.TestCase):
    def test_ngram(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
        config = build_config({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 2,
            'ngram_maxn': 6,
            'unit_freq': 1,
            'drop_reminder': False,
        })
        tokens_vocab, spaces_vocab = extract_vocab(data_path, config)
        self.assertListEqual(
            ['[UNK]', '.>', '<.', '<.>', 'о>', 'то', 'то>', '!>', ',>', '<!'], tokens_vocab.tokens()[:10])
        self.assertListEqual(['[UNK]', ' >', '< ', '< >', '<>'], spaces_vocab.tokens()[:10])

    def test_cnn(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
        config = build_config({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'input_unit': 'CNN',
            'unit_freq': 2,
            'drop_reminder': False,
        })
        tokens_vocab, spaces_vocab = extract_vocab(data_path, config)
        self.assertListEqual(['[UNK]', '[BOW]', '[EOW]', 'т', 'а', 'о', '.', 'м', 'н', '!'], tokens_vocab.tokens()[:10])
        self.assertListEqual(['[UNK]', '< >', '<>'], spaces_vocab.tokens()[:10])



if __name__ == "__main__":
    tf.test.main()
