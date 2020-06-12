# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ..hparam import build_hparams


class TestHParam(tf.test.TestCase):
    def test_empty_overrides(self):
        with self.assertRaises(ValueError):
            build_hparams({})

    def test_lstm_units(self):
        build_hparams({'lstm_units': [1]})

    def test_addons_optimizer(self):
        build_hparams({'train_optim': 'Addons>RectifiedAdam', 'lstm_units': [1]})


if __name__ == "__main__":
    tf.test.main()
