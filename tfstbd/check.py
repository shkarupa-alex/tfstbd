import argparse
import os
import numpy as np
from collections import Counter
from tfmiss.text import split_words
from tfmiss.training import estimate_bucket_boundaries
from typing import List, Tuple
from .input import raw_dataset


def check_dataset(data_path: str) -> Tuple[Counter, Counter, Counter, Counter, List[str]]:
    dataset = raw_dataset(data_path, 'train')
    dataset = dataset.concatenate(raw_dataset(data_path, 'test'))

    sent_len, sps_class, tok_class, sent_class = Counter(), Counter(), Counter(), Counter()
    samples = []

    has_examples = False
    for row in dataset:
        has_examples = True
        words = split_words(row['document'], extended=True)
        words = np.char.decode(words.numpy().reshape([-1]).astype('S'), 'utf-8')
        if len(samples) < 10:
            samples.append(''.join(words))

        length = row['length'].numpy().item()
        space = row['space'].numpy()
        token = row['token'].numpy()
        weight = row['weight'].numpy()
        repdivwrap = row['repdivwrap'].numpy()
        sentence = row['sentence'].numpy()

        sent_len.update([length])
        sps_class.update(space.decode('utf-8'))
        tok_class.update(token.decode('utf-8'))
        sent_class.update(sentence.decode('utf-8'))

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

    return sent_len, sps_class, tok_class, sent_class, samples


def mean_from_counter(cntr: Counter) -> float:
    tot = sum([l * f for l, f in cntr.most_common()])
    cnt = sum([f for _, f in cntr.most_common()])

    return tot / cnt


# def std_from_counter(cntr: Counter, mean: float) -> float:
#     cnt = sum([f for _, f in cntr.most_common()])
#     dev = sum([f * (l - mean) ** 2 for l, f in cntr.most_common()])
#
#     return (dev / cnt) ** 0.5


def weighted_class_balance(bincount: List[int]) -> List[float]:
    bincount = np.array(bincount)
    balanced = np.sum(bincount) / (len(bincount) * bincount)

    return balanced.tolist()


def main():
    parser = argparse.ArgumentParser(description='Check dataset for correct shapes')
    parser.add_argument(
        'data_path',
        type=str,
        help='Directory with TFRecord files')

    argv, _ = parser.parse_known_args()
    assert not os.path.exists(argv.data_path) or os.path.isdir(argv.data_path), 'Wrong data path'

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

    sps_weights = weighted_class_balance([sps_class['T'], sps_class['S']])
    tok_weights = weighted_class_balance([tok_class['I'], tok_class['B']])
    sent_weights = weighted_class_balance([sent_class['I'], sent_class['B']])
    print('\nFor better metrics use these value in your config:')
    print('space_weight: {}'.format(sps_weights))
    print('token_weight: {}'.format(tok_weights))
    print('sentence_weight: {}'.format(sent_weights))
