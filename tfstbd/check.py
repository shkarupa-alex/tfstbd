from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf
from collections import Counter
from tfmiss.text import split_words
from tfmiss.training import estimate_bucket_boundaries
from .input import train_dataset
from .hparam import build_hparams


def check_dataset(data_path):
    if not tf.executing_eagerly():
        raise EnvironmentError('Only TensorFlow with eager mode enabled by default supported')

    wildcard = os.path.join(data_path, '*.tfrecords.gz')
    params = build_hparams({
        'bucket_bounds': [],
        'mean_samples': 1,
        'samples_mult': 1,
        'ngram_minn': 999,
        'ngram_maxn': 999,
        'lstm_units': [1]
    })
    dataset = train_dataset(wildcard, params)

    sent_len = Counter()
    sps_class, tok_class, sent_class = Counter(), Counter(), Counter()
    samples = []
    for features, labels, weights in dataset:
        documents, spaces, tokens, sentences, repdivwrap, token_weights = \
            features['documents'], \
            labels['space'].numpy().reshape([-1]), \
            labels['token'].numpy().reshape([-1]), \
            labels['sentence'].numpy().reshape([-1]), \
            labels['repdivwrap'].numpy().reshape([-1]), \
            weights['token'].numpy().reshape([-1])

        words = split_words(documents, extended=True)
        words = np.char.decode(words.flat_values.numpy().reshape([-1]).astype('S'), 'utf-8')
        sent_len.update([len(words)])

        sps_class.update(spaces)
        tok_class.update(tokens)
        sent_class.update(sentences)

        if len(words) != len(spaces) or \
                len(spaces) != len(tokens) or \
                len(tokens) != len(sentences) or \
                len(sentences) != len(repdivwrap) or \
                len(repdivwrap) != len(token_weights):
            print('Found error in inputs shapes: {} vs {} vs {} vs {} vs {} vs {}'.format(
                len(words), len(spaces), len(tokens), len(sentences), len(repdivwrap), len(token_weights)))
            print(u'documents: {}'.format(np.char.decode(documents.numpy().reshape([-1]).astype('S'), 'utf-8')))
            print(u'words ({}): {}'.format(len(words), words))
            print(u'tokens ({}): {}'.format(len(tokens), tokens))
            print(u'sentences ({}): {}'.format(len(sentences), sentences))
            print(u'repdivwrap ({}): {}'.format(len(repdivwrap), repdivwrap))
            raise Exception('Dataset check failed')

        if len(samples) < 10:
            samples.append(u''.join(words))

        del words, tokens, sentences

    return sent_len, sps_class, tok_class, sent_class, samples


def mean_from_counter(cntr):
    tot = sum([l * f for l, f in cntr.most_common()])
    cnt = sum([f for _, f in cntr.most_common()])

    return tot / cnt


def std_from_counter(cntr, mean):
    cnt = sum([f for _, f in cntr.most_common()])
    dev = sum([f * (l - mean) ** 2 for l, f in cntr.most_common()])

    return (dev / cnt) ** 0.5


def weighted_class_balance(cntr):
    i_classes = sorted([k for k, _ in cntr.most_common()])
    n_samples = np.sum([f for _, f in cntr.most_common()])
    n_classes = len(cntr)
    bincount = np.array([cntr[i] for i in i_classes])

    return n_samples / (n_classes * bincount)


def main():
    parser = argparse.ArgumentParser(description='Check dataset for correct shapes')
    parser.add_argument(
        'data_path',
        type=str,
        help='Directory with TFRecord files')

    argv, _ = parser.parse_known_args()
    if not os.path.exists(argv.data_path) or not os.path.isdir(argv.data_path):
        raise ValueError('Wrong data path')

    sent_len, sps_class, tok_class, sent_class, samples = check_dataset(argv.data_path)
    print('Dataset checked successfully')

    print('Samples from dataset:')
    for s in samples:
        print('-' * 80)
        print(s)

    buck_bounds = estimate_bucket_boundaries(sent_len)
    print('\nFor better performance use these value in your config:')
    print('bucket_bounds: {}'.format(buck_bounds))

    sent_mean = int(mean_from_counter(sent_len))
    sent_max = max(list(sent_len.values()))
    print('\nKeep in mind when choosing number of samples in batch:')
    print('Mean samples per example: {}'.format(sent_mean))
    print('Max samples per example: {}'.format(sent_max))

    sps_weights = weighted_class_balance(sps_class)
    tok_weights = weighted_class_balance(tok_class)
    sent_weights = weighted_class_balance(sent_class)
    print('\nFor better metrics use these value in your config:')
    print('space_weight: {}'.format(sps_weights))
    print('token_weight: {}'.format(tok_weights))
    print('sentence_weight: {}'.format(sent_weights))
