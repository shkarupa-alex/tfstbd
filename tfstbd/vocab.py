import argparse
import numpy as np
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from tfmiss.keras.layers import CharNgramEmbedding
from typing import Tuple
from .input import vocab_dataset, RESERVED
from .config import Config, build_config
from .model import _token_embed, _space_embed


def extract_vocab(dest_path: str, config: Config) -> Tuple[Vocabulary, Vocabulary]:
    dataset = vocab_dataset(dest_path, config)

    token_vocab = Vocabulary()
    space_vocab = Vocabulary()

    has_examples = False
    for tokens, spaces, _ in dataset:
        has_examples = True
        token_vocab.update(tokens.flat_values.numpy())
        space_vocab.update(spaces.flat_values.numpy())

    assert has_examples, 'Empty dataset'
    assert 0 == token_vocab[b'']

    token_vocab = Vocabulary({w.decode('utf-8'): f for w, f in token_vocab.most_common()})
    space_vocab = Vocabulary({w.decode('utf-8'): f for w, f in space_vocab.most_common()})

    token_vocab = _token_embed(config, RESERVED).vocab(token_vocab)
    space_vocab = _space_embed(config, RESERVED).vocab(space_vocab)

    token_vocab['[UNK]'] = token_vocab[token_vocab.tokens()[0]] + 1
    space_vocab['[UNK]'] = space_vocab[space_vocab.tokens()[0]] + 1

    # char_cats = Vocabulary()
    # char_category(inputs)
    # char_category(inputs, first=False)

    return token_vocab, space_vocab


def _vocab_names(data_path, config, fmt=Vocabulary.FORMAT_BINARY_PICKLE):
    ext = 'pkl' if Vocabulary.FORMAT_BINARY_PICKLE == fmt else 'tsv'

    token_vocab = 'vocab_{}_token.{}'.format(config.input_unit.value, ext)
    space_vocab = 'vocab_{}_space.{}'.format(config.input_unit.value, ext)

    return os.path.join(data_path, token_vocab), os.path.join(data_path, space_vocab)


def main():
    parser = argparse.ArgumentParser(
        description='Extract token and space ngram vocabularies from dataset')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='YAML-encoded model hyperparameters file')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to dataset')

    argv, unparsed = parser.parse_known_args()
    assert os.path.exists(argv.data_path) and os.path.isdir(argv.data_path), 'Wrong dataset path'

    hyper_params = argv.hyper_params.name
    argv.hyper_params.close()
    config = build_config(hyper_params)

    print('Estimating {} vocabulary'.format(config.input_unit.value))
    token_vocab, space_vocab = extract_vocab(argv.data_path, config)

    token_path, space_path = _vocab_names(argv.data_path, config, Vocabulary.FORMAT_BINARY_PICKLE)
    token_vocab.save(token_path, Vocabulary.FORMAT_BINARY_PICKLE)
    space_vocab.save(space_path, Vocabulary.FORMAT_BINARY_PICKLE)

    token_path, space_path = _vocab_names(argv.data_path, config, Vocabulary.FORMAT_TSV_WITH_HEADERS)
    token_vocab.save(token_path, Vocabulary.FORMAT_TSV_WITH_HEADERS)
    space_vocab.save(space_path, Vocabulary.FORMAT_TSV_WITH_HEADERS)
