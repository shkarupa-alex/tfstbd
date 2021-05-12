import os
import tensorflow as tf
from ..hparam import build_hparams


class TestHParam(tf.test.TestCase):
    def test_empty_overrides(self):
        with self.assertRaises(AssertionError):
            build_hparams({})

    def test_lstm_units(self):
        h_params = build_hparams({'lstm_units': [5]})
        self.assertListEqual([5], h_params.lstm_units)

    def test_addons_optimizer(self):
        h_params = build_hparams({'train_optim': 'Addons>RectifiedAdam', 'lstm_units': [1]})
        self.assertEqual('Addons>RectifiedAdam', h_params.train_optim)

    def test_from_file(self):
        h_path = os.path.join(os.path.dirname(__file__), 'data', 'hparams.json')
        h_params = build_hparams(h_path)
        self.assertListEqual([257, 265], h_params.bucket_bounds)
        self.assertEqual('tcn', h_params.seq_core)
        self.assertListEqual([256, 128, 64, 32], h_params.tcn_filters)


if __name__ == "__main__":
    tf.test.main()
