import argparse
import numpy as np
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from tfmiss.keras.layers import CharNgams
from .input import vocab_dataset
from .hparam import HParams, build_hparams


def extract_vocab(dest_path: str, h_params: HParams) -> Vocabulary:
    dataset = vocab_dataset(dest_path, h_params)

    token_vocab = Vocabulary()
    has_examples = False
    for tokens in dataset:
        has_examples = True
        tokens = np.char.decode(tokens.flat_values.numpy().astype('S'), 'utf-8')
        token_vocab.update(tokens)
    assert has_examples, 'Empty dataset'

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
    vocab = extract_vocab(argv.data_path, params)

    vocab.save(os.path.join(argv.data_path, 'vocab.pkl'), Vocabulary.FORMAT_BINARY_PICKLE)

    vocab['[UNK]'] = vocab[vocab.tokens()[0]] + 1
    vocab.save(os.path.join(argv.data_path, 'vocab.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)

    vocab, _ = vocab.split_by_size(1000)
    vocab.save(os.path.join(argv.data_path, 'vocab_tensorboard.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)
