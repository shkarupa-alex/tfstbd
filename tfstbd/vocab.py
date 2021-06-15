import argparse
import numpy as np
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from tfmiss.keras.layers import CharNgams
from typing import Tuple
from .input import vocab_dataset
from .hparam import HParams, build_hparams


def extract_vocab(dest_path: str, h_params: HParams) -> Tuple[Vocabulary, Vocabulary]:
    dataset = vocab_dataset(dest_path, h_params)

    token_vocab = Vocabulary()
    space_vocab = Vocabulary()

    has_examples = False
    for tokens, spaces in dataset:
        has_examples = True
        tokens = np.char.decode(tokens.flat_values.numpy().astype('S'), 'utf-8')
        spaces = np.char.decode(spaces.flat_values.numpy().astype('S'), 'utf-8')
        token_vocab.update(tokens)
        space_vocab.update(spaces)
    assert has_examples, 'Empty dataset'
    assert 0 == token_vocab['']
    del space_vocab['']

    token_ngrams = Vocabulary()
    tokens = tf.constant(token_vocab.tokens(), dtype=tf.string)
    ngrams = CharNgams(h_params.ngram_minn, h_params.ngram_maxn, h_params.ngram_self)(tokens)
    for token, ngram in zip(token_vocab.tokens(), ngrams):
        ngram = np.char.decode(ngram.numpy().astype('S'), 'utf-8').reshape([-1])
        for n in ngram:
            token_ngrams[n] += token_vocab[token]
    token_ngrams, _ = token_ngrams.split_by_frequency(2)  # at least 2 occurrences

    space_ngrams = Vocabulary()
    spaces = tf.constant(space_vocab.tokens(), dtype=tf.string)
    ngrams = CharNgams(h_params.ngram_minn, h_params.ngram_maxn, h_params.ngram_self)(spaces)
    for space, ngram in zip(space_vocab.tokens(), ngrams):
        ngram = np.char.decode(ngram.numpy().astype('S'), 'utf-8').reshape([-1])
        for n in ngram:
            space_ngrams[n] += space_vocab[space]
    space_ngrams, _ = space_ngrams.split_by_frequency(2)  # at least 2 occurrences

    return token_ngrams, space_ngrams


def main():
    parser = argparse.ArgumentParser(
        description='Extract token and space ngram vocabularies from dataset')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparameters file')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to dataset')

    argv, unparsed = parser.parse_known_args()
    assert os.path.exists(argv.data_path) and os.path.isdir(argv.data_path), 'Wrong dataset path'

    hyper_params = argv.hyper_params.name
    argv.hyper_params.close()
    params = build_hparams(hyper_params)

    print('Estimating ngram vocabulary')
    token_ngram, space_ngram = extract_vocab(argv.data_path, params)

    token_ngram.save(os.path.join(argv.data_path, 'token_vocab.pkl'), Vocabulary.FORMAT_BINARY_PICKLE)
    token_ngram['[UNK]'] = token_ngram[token_ngram.tokens()[0]] + 1
    token_ngram.save(os.path.join(argv.data_path, 'token_vocab.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)

    space_ngram.save(os.path.join(argv.data_path, 'space_vocab.pkl'), Vocabulary.FORMAT_BINARY_PICKLE)
    space_ngram['[UNK]'] = space_ngram[space_ngram.tokens()[0]] + 1
    space_ngram.save(os.path.join(argv.data_path, 'space_vocab.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)
