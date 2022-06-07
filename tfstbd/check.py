import argparse
import os
import numpy as np
from collections import Counter
from tfmiss.training import estimate_bucket_boundaries
from typing import List, Tuple
from .input import raw_dataset, parse_documents


def check_dataset(data_path: str) -> Tuple[Counter, List[str]]:
    dataset = raw_dataset(data_path, 'train')
    dataset = dataset.concatenate(raw_dataset(data_path, 'test'))
    dataset = dataset.shuffle(1000)

    samples = []
    sent_len = Counter()
    has_examples = False
    for row in dataset:
        if len(samples) < 10:
            samples.append(row['document'].numpy().decode('utf-8'))

        has_examples = True
        words, spaces, _ = parse_documents(row['document'][..., None])
        words, spaces = words[0], spaces[0]
        words = np.char.decode(words.numpy().reshape([-1]).astype('S'), 'utf-8')
        spaces = np.char.decode(spaces.numpy().reshape([-1]).astype('S'), 'utf-8')

        length = row['length'].numpy().item()
        token = row['token'].numpy().decode('utf-8')
        sentence = row['sentence'].numpy().decode('utf-8')

        sent_len.update([length])
        sizes = {len(words), len(spaces), length, len(token), len(sentence)}
        if len(sizes) > 1:
            print('Found different input shapes: precomputed {} vs estimated {}'.format(length, sizes))
            print('document: {}'.format(row['document'].numpy().decode('utf-8')))
            print('words ({}): {}'.format(len(words), words))
            print('spaces ({}): {}'.format(len(spaces), spaces))
            print('tokens ({}): {}'.format(len(token), token))
            print('sentences ({}): {}'.format(len(sentence), sentence))
            raise AssertionError('Dataset check failed')
    assert has_examples, 'Empty dataset'

    return sent_len, samples


def mean_from_counter(cntr: Counter) -> float:
    tot = sum([l * f for l, f in cntr.most_common()])
    cnt = sum([f for _, f in cntr.most_common()])

    return tot / cnt


def main():
    parser = argparse.ArgumentParser(description='Check dataset for correct shapes')
    parser.add_argument(
        'data_path',
        type=str,
        help='Directory with TFRecord files')

    argv, _ = parser.parse_known_args()
    assert not os.path.exists(argv.data_path) or os.path.isdir(argv.data_path), 'Wrong data path'

    sent_len, samples = check_dataset(argv.data_path)
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
