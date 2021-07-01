import argparse
import numpy as np
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from tfmiss.keras.layers import CharNgramEmbedding
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
        token_vocab.update(tokens.flat_values.numpy())
        space_vocab.update(spaces.flat_values.numpy())
    token_vocab = Vocabulary({w.decode('utf-8'): f for w, f in token_vocab.most_common()})
    space_vocab = Vocabulary({w.decode('utf-8'): f for w, f in space_vocab.most_common()})

    assert has_examples, 'Empty dataset'
    assert 0 == token_vocab['']
    del space_vocab['']

    embedder = CharNgramEmbedding(
        vocabulary=[], output_dim=h_params.ngram_dim, minn=h_params.ngram_minn, maxn=h_params.ngram_maxn,
        itself=h_params.ngram_self, reduction=h_params.ngram_comb, show_warning=False)
    token_ngrams = embedder.vocab(token_vocab)
    space_ngrams = embedder.vocab(space_vocab)

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
