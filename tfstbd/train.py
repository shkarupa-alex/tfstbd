import argparse
import os
import tensorflow as tf
from logging import INFO
from nlpvocab import Vocabulary
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from tfmiss.keras.callbacks import LRFinder
from tfmiss.keras.metrics import F1Binary
from tfmiss.keras.optimizers.schedules import WarmHoldCosineCoolAnnihilateScheduler
from .input import train_dataset
from .hparam import build_hparams, HParams
from .input import train_dataset
from .model import build_model


def train_model(data_dir: str, h_params: HParams, model_dir: str, findlr_steps: int = 0, verbose: int = 1) -> dict:
    ngram_vocab = Vocabulary.load(os.path.join(data_dir, 'vocab.pkl'), format=Vocabulary.FORMAT_BINARY_PICKLE)
    model = build_model(h_params, ngram_vocab)

    train_ds = train_dataset(data_dir, 'train', h_params)
    valid_ds = train_dataset(data_dir, 'test', h_params)

    learn_rate = h_params.learn_rate
    if not findlr_steps:
        epoch_steps = 0
        for _ in train_ds:
            epoch_steps += 1
        learn_rate = WarmHoldCosineCoolAnnihilateScheduler(
            min_lr=h_params.learn_rate / 50,
            max_lr=h_params.learn_rate,
            warm_steps=epoch_steps * 0.5,
            hold_steps=(h_params.num_epochs - 1) * epoch_steps * 0.3,
            cool_steps=(h_params.num_epochs - 1) * epoch_steps * 0.7,
            cosine_cycles=5,
            cosine_height=0.75,
            annih_steps=epoch_steps * 0.5)

    if 'ranger' == h_params.train_optim.lower():
        optimizer = Lookahead(RectifiedAdam(learn_rate))
    else:
        optimizer = tf.keras.optimizers.get(h_params.train_optim)
        optimizer._set_hyper('learning_rate', learn_rate)

    model.compile(
        optimizer=optimizer,
        loss={
            'space': 'binary_crossentropy',
            'token': 'binary_crossentropy',
            'sentence': 'binary_crossentropy',
        },
        weighted_metrics={
            'space': [tf.keras.metrics.BinaryAccuracy(name='accuracy'), F1Binary(name='f1')],
            'token': [tf.keras.metrics.BinaryAccuracy(name='accuracy'), F1Binary(name='f1')],
            'sentence': [tf.keras.metrics.BinaryAccuracy(name='accuracy'), F1Binary(name='f1')],
        },
        run_eagerly=False,
    )
    if verbose > 0:
        model.summary()

    if findlr_steps:
        lr_finder = LRFinder(findlr_steps)
        history = model.fit(
            train_ds,
            callbacks=[lr_finder],
            epochs=1,
            steps_per_epoch=findlr_steps,
            verbose=verbose
        )
    else:
        lr_finder = None
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    os.path.join(model_dir, 'logs'),
                    update_freq=100,
                    profile_batch=0),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(model_dir, 'train'),
                    save_weights_only=True,
                    verbose=True)
            ],
            epochs=h_params.num_epochs,
            verbose=verbose
        )

    if findlr_steps:
        best_lr, loss_graph = lr_finder.plot()
        tf.get_logger().info('Best lr should be near: {}'.format(best_lr))
        tf.get_logger().info('Best lr graph saved to: {}'.format(loss_graph))
    else:
        save_options = tf.saved_model.SaveOptions(namespace_whitelist=['Miss'])
        model.save(os.path.join(model_dir, 'last'), options=save_options)
        export = tf.keras.Model(inputs=model.inputs[:1], outputs=model.outputs)
        tf.saved_model.save(export, os.path.join(model_dir, 'export'), options=save_options)

    return history.history


def main():
    parser = argparse.ArgumentParser(description='Train, evaluate and export tfdstbd model')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparameters file')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to dataset')
    parser.add_argument(
        'model_dir',
        type=str,
        help='Directory to save model')
    parser.add_argument(
        '--findlr_steps',
        type=int,
        default=0,
        help='Run model with LRFinder callback')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.data_dir) and os.path.isdir(argv.data_dir)
    assert not os.path.exists(argv.model_dir) or os.path.isdir(argv.model_dir)

    tf.get_logger().setLevel(INFO)

    params_path = argv.hyper_params.name
    argv.hyper_params.close()
    params = build_hparams(params_path)

    train_model(argv.data_dir, params, argv.model_dir, argv.findlr_steps)
