import argparse
import os
import tensorflow as tf
from logging import INFO
from keras import callbacks, metrics, models, optimizers
from nlpvocab import Vocabulary
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from tfmiss.keras.callbacks import LRFinder
from tfmiss.keras.metrics import F1Binary
from tfmiss.keras.optimizers.schedules import WarmHoldCosineCoolAnnihilateScheduler
from .input import train_dataset
from .config import build_config, Config
from .input import train_dataset
from .model import build_model
from .vocab import _vocab_names


def train_model(data_dir: str, config: Config, model_dir: str, findlr_steps: int = 0, verbose: int = 1) -> dict:
    if config.use_jit and not findlr_steps:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        tf.config.optimizer.set_jit('autoclustering')

    if config.mixed_fp16:
        mixed_precision.set_global_policy('mixed_float16')

    token_path, space_path = _vocab_names(data_dir, config, fmt=Vocabulary.FORMAT_BINARY_PICKLE)
    token_vocab = Vocabulary.load(token_path, format=Vocabulary.FORMAT_BINARY_PICKLE)
    space_vocab = Vocabulary.load(space_path, format=Vocabulary.FORMAT_BINARY_PICKLE)
    model = build_model(config, token_vocab, space_vocab)

    train_ds = train_dataset(data_dir, 'train', config)
    valid_ds = train_dataset(data_dir, 'test', config)

    learn_rate = config.learn_rate
    if not findlr_steps:
        epoch_steps = 0
        for _ in train_ds:
            epoch_steps += 1
        learn_rate = WarmHoldCosineCoolAnnihilateScheduler(
            min_lr=config.learn_rate / 50,
            max_lr=config.learn_rate,
            warm_steps=epoch_steps * 0.5,
            hold_steps=(config.num_epochs - 1) * epoch_steps * 0.3,
            cool_steps=(config.num_epochs - 1) * epoch_steps * 0.7,
            cosine_cycles=5,
            cosine_height=0.75,
            annih_steps=epoch_steps * 0.5)

    if 'ranger' == config.train_optim.lower():
        optimizer = Lookahead(RectifiedAdam(learn_rate))
    else:
        optimizer = optimizers.get(config.train_optim)
        optimizer._set_hyper('learning_rate', learn_rate)

    model.compile(
        optimizer=optimizer,
        loss=[None, None, 'sparse_categorical_crossentropy'],
        weighted_metrics=[None, None, [metrics.SparseCategoricalAccuracy(name='accuracy')]],
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
                callbacks.TensorBoard(
                    os.path.join(model_dir, 'logs'),
                    update_freq=100,
                    profile_batch=(100, 105)),
                callbacks.ModelCheckpoint(
                    os.path.join(model_dir, 'train'),
                    save_weights_only=True,
                    verbose=True)
            ],
            epochs=config.num_epochs,
            verbose=verbose
        )

    if findlr_steps:
        best_lr, loss_graph = lr_finder.plot(skip_start=50, skip_end=25)
        tf.get_logger().info('Best lr should be near: {}'.format(best_lr))
        tf.get_logger().info('Best lr graph saved to: {}'.format(loss_graph))
    else:
        save_options = tf.saved_model.SaveOptions(namespace_whitelist=['Miss'])
        model.save(os.path.join(model_dir, 'last'), options=save_options)

        export = models.Model(inputs=model.inputs[:1], outputs=model.outputs)
        export.save(os.path.join(model_dir, 'export'), options=save_options, include_optimizer=False)

    return history.history


def main():
    parser = argparse.ArgumentParser(description='Train, evaluate and export tfdstbd model')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='YAML-encoded model hyperparameters file')
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
    config = build_config(params_path)

    train_model(argv.data_dir, config, argv.model_dir, argv.findlr_steps)
