# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import tensorflow as tf
from ..hparam import build_hparams
from ..train import train_model


class TestTrainModel(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lstm(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        params = build_hparams({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 2,
            'ngram_maxn': 6,
            'ngram_freq': 2,
            'lstm_units': [1]
        })
        train_model(data_dir, params, self.temp_dir)


if __name__ == "__main__":
    tf.test.main()
