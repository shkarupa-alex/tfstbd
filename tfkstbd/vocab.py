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


def extract_vocab(dest_path, ngram_minn, ngram_maxn, min_freq, bucket_bounds, batch_sizes):
    assert tf.executing_eagerly()

    wildcard = os.path.join(dest_path, 'train*.tfrecords.gz')
    dataset = train_input_fn(wildcard, bucket_bounds, batch_sizes, ngram_minn, ngram_maxn)

    vocab = Vocabulary()
    for features, _ in dataset:
        ngrams = features['word_ngrams'].values.numpy()
        ngrams = [n.decode('utf-8') for n in ngrams if n not in {b'', b'<>'}]

        # # only non-alpha, including suffixes, postfixes and other interesting parts
        ngrams = [n for n in ngrams if not n.isalpha()] # TODO
        vocab.update(ngrams)

    vocab.trim(min_freq)

    return vocab


def main():
    parser = argparse.ArgumentParser(
        description='Extract ngram vocabulary from dataset')
    parser.add_argument(
        'src_path',
        type=str,
        help='Path with train TFRecord files')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparams file')
    parser.add_argument(
        'ngram_vocab',
        type=str,
        help='Output vocabulary file')

    argv, unparsed = parser.parse_known_args()
    assert os.path.exists(argv.src_path) and os.path.isdir(argv.src_path)

    params = build_hparams(json.loads(argv.hyper_params.read()))

    print('Processing training vocabulary with min freq {}'.format(params.ngram_freq))
    vocab = extract_vocab(
        argv.src_path,
        params.ngram_minn,
        params.ngram_maxn,
        params.ngram_freq,
        params.bucket_bounds,
        params.batch_sizes
    )

    vocab.save(argv.ngram_vocab, Vocabulary.FORMAT_BINARY_PICKLE)
    vocab.save(os.path.splitext(argv.ngram_vocab)[0] + '.tsv', Vocabulary.FORMAT_TSV_WITH_HEADERS)
    # TODO: save top 1000?
