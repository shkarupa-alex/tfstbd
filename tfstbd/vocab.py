from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from tfmiss.keras.layers import CharNgams
from .input import vocab_dataset
from .hparam import build_hparams


def extract_vocab(dest_path, h_params):
    wildcard = os.path.join(dest_path, 'train*.tfrecords.gz')
    dataset = vocab_dataset(wildcard, h_params)

    token_vocab = Vocabulary()
    for tokens in dataset:
        tokens = np.char.decode(tokens.flat_values.numpy().astype('S'), 'utf-8')
        token_vocab.update(tokens)

    ngram_vocab = Vocabulary()
    tokens = tf.constant(token_vocab.tokens(), dtype=tf.string)
    ngrams = CharNgams(h_params.ngram_minn, h_params.ngram_maxn, h_params.ngram_self)(tokens)
    for token, ngram in zip(token_vocab.tokens(), ngrams):
        ngram = np.char.decode(ngram.numpy().astype('S'), 'utf-8').reshape([-1])
        for n in ngram:
            ngram_vocab[n] += token_vocab[token]

    ngram_vocab, _ = ngram_vocab.split_by_frequency(2)  # at least 2 occurrences

    return ngram_vocab


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

    vocab['[UNK]'] = vocab[vocab.tokens()[0]] + 1
    vocab.save(os.path.join(argv.src_path, 'vocab.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)

    vocab, _ = vocab.split_by_size(1000)
    vocab.save(os.path.join(argv.src_path, 'vocab_tensorboard.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)
