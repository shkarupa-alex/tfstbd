from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import tensorflow as tf
from logging import INFO
from ray import tune
from .hparam import build_hparams
from .train import train_model


def tune_model(data_dir, h_params, model_dir):
    h_params.num_epochs = 2
    h_params.seq_core = 'lstm'

    search_space = {
        'ngram_minn': tune.choice([2, 3, 4, 5]),
        'ngram_maxn': tune.choice([2, 3, 4, 5, 6]),
        'ngram_freq': tune.choice([10, 20, 40, 80, 160]),
        'ngram_dim': tune.choice([64, 128, 192, 256, 320]),
        'ngram_self': tune.choice(['always', 'alone']),
        'ngram_comb': tune.choice(['mean', 'sum', 'min', 'max']),
        # 'seq_core': tune.choice(['lstm', 'tcn']),
        'lstm_units': tune.choice([
            [64],
            [128],
            [256],
            [64, 64],
            [128, 64],
            [128, 128],
            [256, 64],
            [256, 128],
            [256, 256]
        ]),
        # 'tcn_filters': tune.choice([
        #     [256, 128, 64, 32]
        # ]),
        # 'tcn_ksize': tune.choice([2, 3, 4]),
        # 'tcn_drop': tune.choice([0.0, 0.1]),
        'train_optim': tune.choice(['Adam', 'Addons>RectifiedAdam', 'Ranger']),
        'learn_rate': tune.choice([1e-2, 1e-3, 1e-4]),
    }

    def trainable(config):
        params = build_hparams(h_params.values())
        params.override_from_dict(config)
        history = train_model(data_dir, params, model_dir, verbose=0)
        print(history)
        for score in history['val_sentence_f1']:
            tune.report(score=score)

    tune.run(trainable, num_samples=1)


def main():
    parser = argparse.ArgumentParser(description='Tune hyperparameters')
    parser.add_argument(
        'params_path',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparameters file')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory with TFRecord files for training')
    parser.add_argument(
        'model_dir',
        type=str,
        help='Directory to store models')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.data_dir) and os.path.isdir(argv.data_dir)
    assert not os.path.exists(argv.model_dir) or os.path.isdir(argv.model_dir)

    params_path = argv.params_path.name
    argv.params_path.close()

    tf.get_logger().setLevel(INFO)

    with open(params_path, 'r') as f:
        h_params = build_hparams(json.loads(f.read()))

    tune_model(argv.data_dir, h_params, argv.model_dir)
