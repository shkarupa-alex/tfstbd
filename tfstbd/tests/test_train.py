import os
import shutil
import tempfile
import tensorflow as tf
from ..config import build_config
from ..train import train_model


class TestTrainModel(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fit(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
        config = build_config({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 2,
            'ngram_maxn': 5,
            'unit_freq': 2,
            'seq_units': [1],
            'num_epochs': 100,
            'drop_reminder': False,
        })
        train_model(data_dir, config, self.temp_dir)


if __name__ == "__main__":
    tf.test.main()
