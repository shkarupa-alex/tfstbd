from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf
from collections import Counter
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
    })
    dataset = train_dataset(wildcard, params)

    sent_len, word_len = Counter(), Counter()
    sps_class, tok_class, sent_class = Counter(), Counter(), Counter()
    samples = []
    for features, labels, weights in dataset:
        words, spaces, tokens, sentences = \
            features['words'].numpy(), labels['space'].numpy(), labels['token'].numpy(), labels['sentence'].numpy()
        sent_len.update([len(words[0])])
        word_len.update([len(w) for w in words[0]])

        sps_class.update([c for c in spaces[0]])
        tok_class.update([c for c in tokens[0]])
        sent_class.update([c for c in sentences[0]])

        if len(words[0]) != len(spaces[0]) or \
                len(spaces[0]) != len(tokens[0]) or \
                len(tokens[0]) != len(sentences[0]) or \
                len(sentences[0]) != len(weights['token'][0]):
            print('Found error in inputs shapes: {} vs {} vs {} vs {} vs {}'.format(
                len(words[0]), len(spaces[0]), len(tokens[0]), len(sentences[0]), len(weights['token'][0])))
            print(u'words ({}):'.format(len(words[0])), [w.decode('utf-8') for w in words[0]])
            print(u'tokens ({}):'.format(len(tokens[0])), [w.decode('utf-8') for w in tokens[0]])
            print(u'sentences ({}):'.format(len(sentences[0])), [w.decode('utf-8') for w in sentences[0]])
            raise Exception('Dataset check failed')

        if len(samples) < 5:
            samples.append(u''.join([w.decode('utf-8') for w in words[0]]))

        del words, tokens, sentences

    return sent_len, word_len, sps_class, tok_class, sent_class, samples


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

    try:
        sent_len, word_len, sps_class, tok_class, sent_class, samples = check_dataset(argv.data_path)
        print('Dataset checked successfully')

        print('Samples from dataset:')
        for s in samples:
            print('-' * 80)
            print(s)

        word_mean = mean_from_counter(word_len)
        word_std = std_from_counter(word_len, word_mean)
        print('\nFor better word length scaling use these values in your config:')
        print('word_mean: {}'.format(word_mean))
        print('word_std: {}'.format(word_std))

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

    except Exception as e:
        print(e)
