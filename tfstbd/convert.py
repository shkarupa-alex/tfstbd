import argparse
import hashlib
import numpy as np
import os
from collections import OrderedDict
from conllu import parse, TokenList
from typing import List
from ufal.udpipe import Model, Pipeline
from .conllu import repair_spaces


def tokenize_sentence(sentence: str, tokenizer: Pipeline, newdoc: str) -> TokenList:
    # Tokenize single sentence

    sentence = sentence.replace('\u240A', '\n')  # "LF"
    processed = tokenizer.process(sentence).strip()
    parsed = parse(processed)
    assert len(parsed), 'Wrong "parsed" size'

    head = parsed[0]
    for tail in parsed[1:]:
        head.extend(tail)
    if head[-1]['misc'] is not None and not sentence.endswith('\n'):
        head[-1]['misc'] = None

    for i, token in enumerate(head):
        token['id'] = i + 1
        token['lemma'] = None
        token['upos'] = None
        token['xpos'] = None
        token['feats'] = None
        head[i] = token

    metadata = [('newdoc', newdoc)] if newdoc else []
    metadata += [
        ('sent_id', hashlib.md5(sentence.encode('utf-8')).hexdigest()),
        ('text', sentence.replace('\n', ' '))
    ]
    head.metadata = OrderedDict(metadata)

    return repair_spaces(head)


def tokenize_paragraphs(paragraphs: List[str], tokenizer: Pipeline, document_name: str) -> List[str]:
    # Tokenize multiple paragraphs

    result = []

    paragraphs = map(lambda p: p.strip(), paragraphs)
    paragraphs = filter(len, paragraphs)
    paragraphs = map(str, paragraphs)

    for i, paragraph in enumerate(paragraphs):
        sentences = map(lambda s: s.strip(), paragraph.split('\n'))
        sentences = filter(len, sentences)
        sentences = map(str, sentences)

        for j, sentence in enumerate(sentences):
            newdoc = '{}_{}'.format(document_name, i) if 0 == j else ''
            processed = tokenize_sentence(sentence, tokenizer, newdoc)
            result.append(processed.serialize())

    return result


def split_tokenize(source_file: str, udpipe_model: str, destination_path: str, test_frac: float) -> None:
    # Convert file with paragraph markup to train/test CoNLL-U datasets (only tokens, no POS and etc.)

    with open(source_file, 'r') as sf:
        train_paragraphs = sf.read().strip().split('\n\n')

    ud_model = Model.load(udpipe_model)
    tokenizer = Pipeline(ud_model, 'tokenize', Pipeline.NONE, Pipeline.NONE, 'conllu')

    np.random.shuffle(train_paragraphs)
    test_size = int(len(train_paragraphs) * test_frac)
    test_paragraphs, train_paragraphs = train_paragraphs[:test_size], train_paragraphs[test_size:]

    basename = os.path.basename(source_file)

    if len(train_paragraphs):
        train_parsed = tokenize_paragraphs(train_paragraphs, tokenizer, '{}-train'.format(basename))

        train_file = os.path.join(destination_path, '___{}-train.conllu'.format(basename))
        with open(train_file, 'w') as f:
            f.write(''.join(train_parsed))

    if len(test_paragraphs):
        test_parsed = tokenize_paragraphs(test_paragraphs, tokenizer, '{}-test'.format(basename))

        test_file = os.path.join(destination_path, '___{}-test.conllu'.format(basename))
        with open(test_file, 'w') as f:
            f.write(''.join(test_parsed))


def main():
    parser = argparse.ArgumentParser(description='Convert sentences with paragraph markup to CoNLL-U')
    parser.add_argument(
        'udpipe_model',
        type=argparse.FileType('rb'),
        help='UDPipe tokenizer model')
    parser.add_argument(
        'src_file',
        type=argparse.FileType('rb'),
        help='Text file with paragraphs divided by double \\n, sentences by single one')
    parser.add_argument(
        'dest_path',
        type=str,
        help='Directory to store .conllu files')
    parser.add_argument(
        '--test_frac',
        type=float,
        default=0.05,
        help='Proportion of data to include in test dataset')

    np.random.seed(123)

    argv, _ = parser.parse_known_args()

    src_file = argv.src_file.name
    argv.src_file.close()

    assert os.path.exists(argv.dest_path) and os.path.isdir(argv.dest_path), 'Wrong destination path'

    udpipe_model = argv.udpipe_model.name
    argv.udpipe_model.close()

    assert 0.0 <= argv.test_frac <= 1.0, 'Wrong test size'

    split_tokenize(
        source_file=src_file,
        udpipe_model=udpipe_model,
        destination_path=argv.dest_path,
        test_frac=argv.test_frac
    )
