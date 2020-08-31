# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ..hparam import build_hparams
from ..vocab import extract_vocab


class TestExtractVocab(tf.test.TestCase):
    def test_lstm(self):
        params = build_hparams({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 2,
            'ngram_maxn': 6,
            'ngram_freq': 2,
            'lstm_units': [1]
        })
        extract_vocab(os.path.join(os.path.dirname(__file__), 'data'), params)


if __name__ == "__main__":
    tf.test.main()
