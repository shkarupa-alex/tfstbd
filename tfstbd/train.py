from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import tensorflow as tf
from logging import INFO
from nlpvocab import Vocabulary
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from tfmiss.keras.callbacks import LRFinder
from tfmiss.keras.metrics import F1Binary
from .input import train_dataset
from .hparam import build_hparams
from .model import build_model


def train_model(data_dir, params_path, model_dir, findlr_steps=0):
    with open(params_path, 'r') as f:
        h_params = build_hparams(json.loads(f.read()))

    ngram_vocab = Vocabulary.load(os.path.join(data_dir, 'vocab.pkl'), format=Vocabulary.FORMAT_BINARY_PICKLE)
    ngram_top, _ = ngram_vocab.split_by_frequency(h_params.ngram_freq)
    ngram_keys = ngram_top.tokens()

    model = build_model(h_params, ngram_keys)

    if 'ranger' == h_params.train_optim.lower():
        optimizer = Lookahead(RectifiedAdam(h_params.learn_rate))
    else:
        optimizer = tf.keras.optimizers.get(h_params.train_optim)
        tf.keras.backend.set_value(optimizer.lr, h_params.learn_rate)

    model.compile(
        optimizer=optimizer,
        loss={
            'space': 'binary_crossentropy',
            'token': 'binary_crossentropy',
            'sentence': 'binary_crossentropy',
        },
        loss_weights={
            'space': h_params.space_weight,
            'token': h_params.token_weight,
            'sentence': h_params.sentence_weight,
        },
        metrics={
            'space': [tf.keras.metrics.Accuracy(name='space/accuracy'), F1Binary(name='space/f1')],
            'token': [tf.keras.metrics.Accuracy(name='token/accuracy'), F1Binary(name='token/f1')],
            'sentence': [tf.keras.metrics.Accuracy(name='sentence/accuracy'), F1Binary(name='sentence/f1')],
        },
        # sample_weight_mode='temporal',
        run_eagerly=False,
    )
    model.summary()

    train_ds = train_dataset(os.path.join(data_dir, 'train-*.tfrecords.gz'), h_params)
    valid_ds = train_dataset(os.path.join(data_dir, 'test-*.tfrecords.gz'), h_params)

    lr_finder = None if not findlr_steps else LRFinder(findlr_steps)
    callbacks = [
        tf.keras.callbacks.TensorBoard(os.path.join(model_dir, 'logs'), update_freq=100, profile_batch='20, 30'),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'train'), monitor='loss', verbose=True)
    ]
    if lr_finder:
        callbacks.append(lr_finder)

    model.fit(
        train_ds,
        epochs=1 if lr_finder else h_params.num_epochs,
        callbacks=callbacks,
        steps_per_epoch=findlr_steps if lr_finder else None,
        validation_data=valid_ds
    )

    if lr_finder:
        tf.get_logger().info('Best lr should be near: {}'.format(lr_finder.find()))
        tf.get_logger().info('Best lr graph with average=10: {}'.format(lr_finder.plot(10)))
    else:
        tf.saved_model.save(model, os.path.join(model_dir, 'export'))


def main():
    parser = argparse.ArgumentParser(description='Train, evaluate and export tfdstbd model')
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
        help='Directory to store model checkpoints')
    parser.add_argument(
        '--findlr_steps',
        type=int,
        default=0,
        help='Run model with LRFinder callback')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.data_dir) and os.path.isdir(argv.data_dir)
    assert not os.path.exists(argv.model_dir) or os.path.isdir(argv.model_dir)

    params_path = argv.params_path.name
    argv.params_path.close()

    tf.get_logger().setLevel(INFO)

    train_model(argv.data_dir, params_path, argv.model_dir, argv.findlr_steps)
