from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf
from collections import Counter
from math import floor
from .input import train_input


def mean_waste(sent_len):
    if not len(sent_len):
        return 0.0

    max_len = max([l for l, _ in sent_len.most_common()])
    zero_len = sum([(max_len - l) * f for l, f in sent_len.most_common()])

    total_freq = sum([f for _, f in sent_len.most_common()])
    total_len = total_freq * max_len

    return zero_len / total_len


def estimate_buckets(sent_len):
    total_freq = sum([f for _, f in sent_len.most_common()])
    sorted_len = sorted([v for v, _ in sent_len.most_common()])

    buck_bounds = []
    left_bound = 0
    for curr_len in sorted_len:
        curr_cnt = Counter(dict([(l, f) for l, f in sent_len.most_common() if left_bound <= l < curr_len]))
        curr_aggr = sum([f for _, f in curr_cnt.most_common()]) / total_freq
        curr_waste = mean_waste(curr_cnt)

        if curr_waste > 0.1 or curr_aggr > 0.01 and curr_waste > 0.01:
            # better: at least 1% of corpora and 1% of paddings
            # worse: 5% of paddings
            buck_bounds.append(curr_len)
            left_bound = curr_len

    return buck_bounds


def estimate_batches(sent_len, buck_bounds, batch_size):
    if len(buck_bounds) < 2:
        return [batch_size] * (len(buck_bounds) + 1)

    _buck_bounds = list(buck_bounds)
    _buck_bounds.append(buck_bounds[-1] * 2 - buck_bounds[-2])

    total_freq = sum([f for _, f in sent_len.most_common()])
    mean_len = sum([l * f / total_freq for l, f in sent_len.most_common()])
    expect_size = mean_len * batch_size

    batch_sizes = []
    for upper_len in _buck_bounds:
        curr_batch = floor(expect_size / (upper_len - 1))
        batch_sizes.append(max(curr_batch, 1.0))

    min_size = min(batch_sizes)
    batch_sizes = [round(bs / min_size, 3) for bs in batch_sizes]

    return batch_sizes


def check_dataset(data_path):
    assert tf.executing_eagerly()

    wildcard = os.path.join(data_path, '*.tfrecords.gz')
    dataset = train_input(
        wild_card=wildcard,
        buck_bounds=[],
        batch_sizes=[1],
        word_mean=1.0,  # will be estimated below
        word_std=1.0,  # will be estimated below
        ngram_minn=999,
        ngram_maxn=999,
    )

    sent_len, word_len = Counter(), Counter()
    for features, labels in dataset:
        words, tokens, sentences = features['words'].numpy(), labels['tokens'].numpy(), labels['sentences'].numpy()
        sent_len.update([len(words[0])])
        word_len.update([len(w) for w in words[0]])

        if len(words[0]) != len(tokens[0]) or len(tokens[0]) != len(sentences[0]):
            print('Found error in inputs shapes: {} vs {} vs {}'.format(
                len(words[0]), len(tokens[0]), len(sentences[0])))
            print('words ({}):'.format(len(words[0])), [w.decode('utf-8') for w in words[0]])
            print('tokens ({}):'.format(len(tokens[0])), [w.decode('utf-8') for w in tokens[0]])
            print('sentences ({}):'.format(len(sentences[0])), [w.decode('utf-8') for w in sentences[0]])
            raise Exception('Dataset check failed')

    return sent_len, word_len


def main():
    parser = argparse.ArgumentParser(description='Check dataset for correct shapes')
    parser.add_argument(
        'data_path',
        type=str,
        help='Directory with TFRecord files')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8192,
        help='Default batch size')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.data_path) and os.path.isdir(argv.data_path)
    assert argv.batch_size > 0

    try:
        sent_len, word_len = check_dataset(argv.data_path)
        print('Dataset checked successfully')

        word_mean, word_std = np.mean(list(word_len.elements())), np.std(list(word_len.elements()))
        print('For better word length scaling use these values in your config:')
        print('word_mean: {}'.format(word_mean))
        print('word_std: {}'.format(word_std))

        buck_bounds = estimate_buckets(sent_len)
        batch_sizes = estimate_batches(sent_len, buck_bounds, argv.batch_size)
        print('For better performance use these values in your config:')
        print('bucket_bounds: {}'.format(buck_bounds))
        print('batch_sizes: {}'.format(batch_sizes))
    except Exception as e:
        print(e)
