import os
import tensorflow as tf
from ..config import build_config, SequenceType


class TestHParam(tf.test.TestCase):
    def test_lstm_units(self):
        config = build_config({'seq_units': [5]})
        self.assertListEqual([5], list(config.seq_units))

    def test_addons_optimizer(self):
        config = build_config({'train_optim': 'Addons>RectifiedAdam', 'seq_units': [1]})
        self.assertEqual('Addons>RectifiedAdam', config.train_optim)

    def test_from_file(self):
        h_path = os.path.join(os.path.dirname(__file__), 'data', 'hparams.yaml')
        config = build_config(h_path)
        self.assertListEqual([257, 265], list(config.bucket_bounds))
        self.assertEqual(SequenceType.TCN, config.seq_type)
        self.assertListEqual([256, 128, 64, 32], list(config.seq_units))


if __name__ == "__main__":
    tf.test.main()
