from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from .feature import feature_columns
from .input import train_input
from .hparam import build_hparams


# def train_eval_export(ngram_vocab, custom_params, model_path, train_data,
#                       eval_data=None, export_path=None, eval_first=True, threads_count=12, cycle_length=3):
#     # Prepare hyperparameters
#     params = build_hparams(custom_params)
#
#     # Prepare sequence estimator
#     sequence_feature_columns = input_feature_columns(
#         ngram_vocab=ngram_vocab,
#         ngram_dimension=params.ngram_dimension,
#         ngram_oov=params.ngram_oov,
#         ngram_combiner=params.ngram_combiner,
#     )
#     estimator = SentenceTokenEstimator(
#         label_vocabulary=['N', 'B'],  # Not a boundary, Boundary
#         loss_reduction=params.loss_reduction,
#         sequence_columns=sequence_feature_columns,
#         sequence_dropout=params.sequence_dropout,
#         rnn_type=params.rnn_type,
#         rnn_layers=params.rnn_layers,
#         rnn_dropout=params.rnn_dropout,
#         dense_layers=params.dense_layers,
#         dense_activation=params.dense_activation,
#         dense_dropout=params.dense_dropout,
#         dense_norm=False,
#         train_optimizer=params.train_optimizer,
#         learning_rate=params.learning_rate,
#         model_dir=model_path,
#     )
#
#     # Add F1 metrics
#     def custom_metrics(features, labels, predictions):
#         weights_ = tf.to_float(tf.not_equal(features['words'], ''))
#
#         token_labels = tf.math.not_equal(
#             labels['tokens'],
#             tf.fill(tf.shape(labels['tokens']), 'N')
#         )
#         token_predictions = tf.math.not_equal(
#             predictions[('tokens', 'classes')],
#             tf.fill(tf.shape(predictions[('tokens', 'classes')]), 'N')
#         )
#
#         sentence_labels = tf.math.not_equal(
#             labels['sentences'],
#             tf.fill(tf.shape(labels['sentences']), 'N')
#         )
#         sentence_predictions = tf.math.not_equal(
#             predictions[('sentences', 'classes')],
#             tf.fill(tf.shape(predictions[('sentences', 'classes')]), 'N')
#         )
#
#         return {
#             'f1/tokens': f1_binary(labels=token_labels, predictions=token_predictions, weights=weights_),
#             'f1/sentences': f1_binary(labels=sentence_labels, predictions=sentence_predictions, weights=weights_),
#         }
#
#     estimator = tf.contrib.estimator.add_metrics(estimator, custom_metrics)
#
#     # Forward splitted words
#     estimator = tf.contrib.estimator.forward_features(estimator, 'words')
#
#     # # Run training
#     # train_wildcard = os.path.join(train_data, '*.tfrecords.gz')
#     # train_steps = 1 if eval_first and not os.path.exists(model_path) else None  # Make evaluation after first step
#     # estimator.train(input_fn=lambda: train_input_fn(
#     #     wild_card=train_wildcard,
#     #     batch_size=params.batch_size,
#     #     ngram_minn=params.ngram_minn,
#     #     ngram_maxn=params.ngram_maxn,
#     #     threads_count=threads_count,
#     #     cycle_length=cycle_length
#     # ), steps=train_steps)
#     #
#     # # Save vocabulary for TensorBoard
#     # ngram_vocab.update(['<UNK_{}>'.format(i) for i in range(params.ngram_oov)])
#     # ngram_vocab.save(os.path.join(model_path, 'tensorboard.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)
#     #
#     # # Run evaluation
#     # metrics = None
#     # if eval_data is not None:
#     #     eval_wildcard = os.path.join(eval_data, '*.tfrecords.gz')
#     #     metrics = estimator.evaluate(input_fn=lambda: train_input_fn(
#     #         wild_card=eval_wildcard,
#     #         batch_size=params.batch_size,
#     #         ngram_minn=params.ngram_minn,
#     #         ngram_maxn=params.ngram_maxn
#     #     ))
#
#     if export_path is not None:
#         estimator.export_savedmodel(
#             export_path, serve_input_fn(
#                 ngram_minn=params.ngram_minn,
#                 ngram_maxn=params.ngram_maxn
#             ),
#             strip_default_attrs=True
#         )
#
#     # return metrics


def main():
    # parser = argparse.ArgumentParser(description='Train, evaluate and export tfdstbd model')
    # parser.add_argument(
    #     'train_data',
    #     type=str,
    #     help='Directory with TFRecord files for training')
    # parser.add_argument(
    #     'ngram_vocab',
    #     type=str,
    #     help='Pickle-encoded ngram vocabulary file')
    # parser.add_argument(
    #     'hyper_params',
    #     type=argparse.FileType('rb'),
    #     help='JSON-encoded model hyperparameters file')
    # parser.add_argument(
    #     'model_path',
    #     type=str,
    #     help='Path to store model checkpoints')
    # parser.add_argument(
    #     '--eval_data',
    #     type=str,
    #     default=None,
    #     help='Directory with TFRecord files for evaluation')
    # parser.add_argument(
    #     '--export_path',
    #     type=str,
    #     default=None,
    #     help='Path to store exported model')
    # parser.add_argument(
    #     '--threads_count',
    #     type=int,
    #     default=12,
    #     help='Number of threads for data preprocessing')
    # parser.add_argument(
    #     '--cycle_length',
    #     type=int,
    #     default=0,
    #     help='Number of input files to process in parallel. By default 1/4 of threads count')
    #
    # argv, _ = parser.parse_known_args()
    # assert os.path.exists(argv.train_data) and os.path.isdir(argv.train_data)
    # assert os.path.exists(argv.ngram_vocab) and os.path.isfile(argv.ngram_vocab)
    # assert not os.path.exists(argv.model_path) or os.path.isdir(argv.model_path)
    # assert argv.eval_data is None or os.path.exists(argv.eval_data) and os.path.isdir(argv.eval_data)
    # assert argv.export_path is None or not os.path.exists(argv.export_path) or os.path.isdir(argv.export_path)
    # assert argv.threads_count >= 0
    # if argv.cycle_length == 0:
    #     argv.cycle_length = max(1, argv.threads_count // 4)

    with open('/Users/alex/HDD/Develop/semtech/tfstbd/config/default.json') as f:
        custom_params = json.loads(f.read())
        current_params = build_hparams(custom_params)

    train_dataset = lambda: train_input(
        wild_card='/Users/alex/HDD/Develop/semtech/tfstbd/data/dataset/train*.tfrecords.gz',
        buck_bounds=current_params.bucket_bounds,
        batch_sizes=current_params.batch_sizes,
        word_mean=current_params.word_mean,
        word_std=current_params.word_std,
        ngram_minn=current_params.ngram_minn,
        ngram_maxn=current_params.ngram_maxn,
    )
    valid_dataset = lambda: train_input(
        wild_card='/Users/alex/HDD/Develop/semtech/tfstbd/data/dataset/valid*.tfrecords.gz',
        buck_bounds=current_params.bucket_bounds,
        batch_sizes=current_params.batch_sizes,
        word_mean=current_params.word_mean,
        word_std=current_params.word_std,
        ngram_minn=current_params.ngram_minn,
        ngram_maxn=current_params.ngram_maxn,
    )

    ngram_vocab = Vocabulary.load('/Users/alex/HDD/Develop/semtech/tfstbd/data/dataset/vocab.pkl',
                                  format=Vocabulary.FORMAT_BINARY_PICKLE)
    sequence_columns = feature_columns(
        ngram_vocab=ngram_vocab.tokens(),
        ngram_dim=current_params.ngram_dim,
        ngram_oov=current_params.ngram_oov,
        ngram_comb=current_params.ngram_comb,
    )

    class STBDModel(tf.keras.Model):
        def __init__(self):
            super(STBDModel, self).__init__()
            self.features = tf.keras.experimental.SequenceFeatures(sequence_columns)
            self.lstm = tf.keras.layers.LSTM(128, return_sequences=True)
            # self.tokens = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))
            self.sentences = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))

        def call(self, inputs, training=None, mask=None):
            outputs, length = self.features(inputs)
            outputs = self.lstm(outputs)

            # tokens = self.tokens(outputs)
            sentences = self.sentences(outputs)

            return sentences
            # return [tokens, sentences]

    model = STBDModel()
    # assert not model.run_eagerly
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.05),
        loss='binary_crossentropy',
        # loss=['binary_crossentropy', 'binary_crossentropy'],
        metrics=['accuracy'],
    )
    # model.run_eagerly = True
    assert not model.run_eagerly
    # model.build()
    # model.summary()

    # tf.python.keras.backend
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]
    # model.fit(train_dataset(), epochs=30, steps_per_epoch=10, callbacks=callbacks, validation_data=valid_dataset(), validation_steps=5)
    model.fit_generator(train_dataset(), use_multiprocessing=True)
