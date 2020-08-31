# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ..check import check_dataset


class TestCheckDataset(tf.test.TestCase):
    def test_normal(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        check_dataset(data_path)


if __name__ == "__main__":
    tf.test.main()
