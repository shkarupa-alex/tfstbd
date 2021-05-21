import argparse
import os
import numpy as np
from collections import Counter
from tfmiss.text import split_words
from tfmiss.training import estimate_bucket_boundaries
from typing import List, Tuple
from .input import raw_dataset


def check_dataset(data_path: str) -> Tuple[Counter, Counter, int, List[str]]:
    dataset = raw_dataset(data_path, 'train')
    dataset = dataset.concatenate(raw_dataset(data_path, 'test'))

    sent_len, class_len, total = Counter(), Counter(), 0
    samples = []

    has_examples = False
    for row in dataset:
        has_examples = True
        words = split_words(row['document'], extended=True)
        words = np.char.decode(words.numpy().reshape([-1]).astype('S'), 'utf-8')
        if len(samples) < 10:
            samples.append(''.join(words))

        length = row['length'].numpy().item()
        space = row['space'].numpy().decode('utf-8')
        token = row['token'].numpy().decode('utf-8')
        weight = row['weight'].numpy()
        repdivwrap = row['repdivwrap'].numpy().decode('utf-8')
        sentence = row['sentence'].numpy().decode('utf-8')

        sent_len.update([length])
        mean_weight = np.mean(weight).item()
        class_len['space_T'] += space.count('T')
        class_len['space_S'] += space.count('S')
        class_len['token_B'] += token.count('B') * mean_weight
        class_len['token_I'] += token.count('I') * mean_weight
        class_len['sentence_B'] += sentence.count('B')
        class_len['sentence_I'] += sentence.count('I')
        total += length

        sizes = {len(words), length, len(space), len(token), len(weight), len(repdivwrap), len(sentence)}
        if len(sizes) > 1:
            print('Found different input shapes: precomputed {} vs estimated {}'.format(length, sizes))
            print('document: {}'.format(row['document'].numpy().decode('utf-8')))
            print('words ({}): {}'.format(len(words), words))
            print('spaces ({}): {}'.format(len(space), space))
            print('tokens ({}): {}'.format(len(token), token))
            print('weights ({}): {}'.format(len(weight), weight))
            print('repdivwrap ({}): {}'.format(len(repdivwrap), repdivwrap))
            print('sentences ({}): {}'.format(len(sentence), sentence))
            raise AssertionError('Dataset check failed')
    assert has_examples, 'Empty dataset'

    return sent_len, class_len, total, samples


def mean_from_counter(cntr: Counter) -> float:
    tot = sum([l * f for l, f in cntr.most_common()])
    cnt = sum([f for _, f in cntr.most_common()])

    return tot / cnt


# def std_from_counter(cntr: Counter, mean: float) -> float:
#     cnt = sum([f for _, f in cntr.most_common()])
#     dev = sum([f * (l - mean) ** 2 for l, f in cntr.most_common()])
#
#     return (dev / cnt) ** 0.5


def main():
    parser = argparse.ArgumentParser(description='Check dataset for correct shapes')
    parser.add_argument(
        'data_path',
        type=str,
        help='Directory with TFRecord files')

    argv, _ = parser.parse_known_args()
    assert not os.path.exists(argv.data_path) or os.path.isdir(argv.data_path), 'Wrong data path'

    sent_len, class_len, total, samples = check_dataset(argv.data_path)
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

    sizes = [class_len[key] for key in ['space_T', 'space_S', 'token_I', 'token_B', 'sentence_I', 'sentence_B']]
    balanced = (total / (len(sizes) * np.array(sizes))).tolist()
    print('\nFor better metrics use these value in your config:')
    print('space_weight: {}'.format(balanced[0: 2]))
    print('token_weight: {}'.format(balanced[2: 4]))
    print('sent_weight: {}'.format(balanced[4: 6]))
