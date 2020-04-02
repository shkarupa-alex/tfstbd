from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from .input import train_input_fn
from .hparam import build_hparams


def extract_vocab(dest_path, params):
    if not tf.executing_eagerly():
        raise EnvironmentError('Only TensorFlow with eager mode enabled by default supported')

    wildcard = os.path.join(dest_path, 'train*.tfrecords.gz')
    dataset = train_input_fn(wildcard, params)

    vocab = Vocabulary()
    for features, _, _ in dataset:
        ngrams = features['word_ngrams'].flat_values.numpy()
        ngrams = [n.decode('utf-8') for n in ngrams if n not in {b'', b'<>'}]

        # # only non-alpha, including suffixes, postfixes and other interesting parts
        # ngrams = [n for n in ngrams if not n.isalpha()] # TODO
        vocab.update(ngrams)

    vocab.trim(2)  # at least 2 occurrences

    return vocab


def main():
    parser = argparse.ArgumentParser(
        description='Extract ngram vocabulary from dataset')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparams file')
    parser.add_argument(
        'src_path',
        type=str,
        help='Path with train TFRecord files')

    argv, unparsed = parser.parse_known_args()
    if not os.path.exists(argv.src_path) or not os.path.isdir(argv.src_path):
        raise ValueError('Wrong source path')

    params = build_hparams(json.loads(argv.hyper_params.read()))

    print('Processing training vocabulary with min freq {}'.format(params.ngram_freq))
    vocab = extract_vocab(argv.src_path, params)

    vocab.save(os.path.join(argv.src_path, 'vocab.pkl'), Vocabulary.FORMAT_BINARY_PICKLE)

    vocab.update(['<UNK_{}>'.format(i) for i in range(params.ngram_oov)])
    vocab.save(os.path.join(argv.src_path, 'vocab.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)

    freq1000 = vocab.most_common(1001)[-1][1] - 1
    vocab.trim(freq1000)
    vocab.save(os.path.join(argv.src_path, 'vocab_tensorboard.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)
