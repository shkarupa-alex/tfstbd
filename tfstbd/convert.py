from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import math
import numpy as np
import os
from collections import OrderedDict
from conllu import parse
from ufal.udpipe import Model, Pipeline
from .conllu import repair_spaces


def parse_paragraphs(paragraphs, tokenizer_model, document_name):
    result = []
    if not len(paragraphs):
        return result

    if not os.path.exists(tokenizer_model) or not os.path.isfile(tokenizer_model):
        raise IOError('Wrong UDPipe tokenizer model')

    model = Model.load(tokenizer_model)
    pipeline = Pipeline(model, 'tokenize', Pipeline.NONE, Pipeline.NONE, 'conllu')

    for pi, p in enumerate(paragraphs):
        p = p.strip()
        if not len(p):
            continue

        sentences = p.split('\n')
        sentences = [s.strip() for s in sentences if len(s.strip())]
        if not len(sentences):
            continue

        for si, s in enumerate(sentences):
            s = s.replace(u'\u240A', '\n')  # "LF"
            processed = pipeline.process(s).strip()
            parsed = parse(processed)
            if not len(parsed):
                raise AssertionError('Wrong "parsed" size')

            head = parsed[0]
            for tail in parsed[1:]:
                head.extend(tail)
            for ti in range(len(head)):
                head[ti]['id'] = ti + 1
            if head[-1]['misc'] is not None and not s.endswith('\n'):
                head[-1]['misc'] = None

            metadata = [
                ('sent_id', hashlib.md5(s.encode('utf-8')).hexdigest()),
                ('text', s.replace('\n', ' '))
            ]
            if 0 == si:
                metadata.insert(0, ('newdoc', '{}_{}'.format(document_name, pi)))
            head.metadata = OrderedDict(metadata)
            for i in range(len(head)):
                head[i]['lemma'] = None
                head[i]['upos'] = None
                head[i]['xpos'] = None
                head[i]['feats'] = None

            head = repair_spaces(head)
            result.append(head.serialize())

    return result


def split_convert(source_file, tokenizer_model, destination_path, test_size):
    with open(source_file, 'rb') as sf:
        train_paragraphs = sf.read().decode('utf-8').strip().split('\n\n')
        np.random.shuffle(train_paragraphs)

        test_count = int(math.floor(len(train_paragraphs) * test_size))
        test_paragraphs, train_paragraphs = train_paragraphs[:test_count], train_paragraphs[test_count:]

    basename = os.path.basename(source_file)

    if len(train_paragraphs):
        train_parsed = parse_paragraphs(train_paragraphs, tokenizer_model, '{}-train'.format(basename))

        train_file = os.path.join(destination_path, '___{}-train.conllu'.format(basename))
        with open(train_file, 'wb') as f:
            f.write(''.join(train_parsed).encode('utf-8'))

    if len(test_paragraphs):
        test_parsed = parse_paragraphs(test_paragraphs, tokenizer_model, '{}-test'.format(basename))

        test_file = os.path.join(destination_path, '___{}-test.conllu'.format(basename))
        with open(test_file, 'wb') as f:
            f.write(''.join(test_parsed).encode('utf-8'))


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
        '--test_size',
        type=float,
        default=0.05,
        help='Proportion of data to include in test dataset')

    np.random.seed(123)

    argv, _ = parser.parse_known_args()

    src_file = argv.src_file.name
    argv.src_file.close()

    if not os.path.exists(argv.dest_path) or not os.path.isdir(argv.dest_path):
        raise IOError('Wrong destination path')

    udpipe_model = argv.udpipe_model.name
    argv.udpipe_model.close()

    if not 0.0 <= argv.test_size <= 1.0:
        raise ValueError('Wrong test size')

    split_convert(
        source_file=src_file,
        tokenizer_model=udpipe_model,
        destination_path=argv.dest_path,
        test_size=argv.test_size
    )
