from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import numpy as np
import re
import six
import tensorflow as tf
import unicodedata
from conllu import parse
from itertools import cycle
from tfmiss.text.unicode_expand import split_words
from .conllu import repair_spaces, extract_tokens, extract_text, split_sent


def parse_paragraphs(file_name, token_weight):
    with open(file_name, 'rb') as f:
        content = '\n' + f.read().decode('utf-8')

    content = content.replace('\n# newdoc\n', '\n# newdoc id = 0\n')
    content = content.replace('\n# newpar\n', '\n# newpar id = 0\n')
    has_groups = False

    paragraphs = []
    paragraph = []
    for block in parse(content):
        meta = ' '.join(six.iterkeys(block.metadata))

        start_group = 'newdoc id' in meta or 'newpar id' in meta
        has_groups = has_groups or start_group

        if len(paragraph) and (not has_groups or start_group):
            paragraphs.append((paragraph, token_weight))
            paragraph = []

        block = repair_spaces(block)
        sentence = extract_text(block)  # validate
        if not len(sentence):
            continue

        sent_parts = split_sent(block)
        sent_texts = [extract_tokens(p, last_space=False) for p in sent_parts]
        paragraph.extend(sent_texts)

    if len(paragraph):
        paragraphs.append((paragraph, token_weight))

    result = []
    duplicates = set()
    for p in paragraphs:
        if len(p[0]) > 1:
            result.append(p)
            continue

        sent = ''.join(itertools.chain(*p[0][0]))
        if sent in duplicates:
            continue

        duplicates.add(sent)
        result.append(p)

    return result


def random_glue(space=1, tab=0, newline=0, reserve=0):
    assert space > 0

    max_spaces0 = int((space + 1) * 0.995 * reserve)
    num_spaces0 = np.random.randint(0, max_spaces0) if max_spaces0 > 0 else 0

    max_spaces1 = int((space + 1) * 0.005 * reserve)
    num_spaces1 = np.random.randint(0, max_spaces1) if max_spaces1 > 0 else 0

    max_tabs = int((tab + 1) * reserve)
    num_tabs = np.random.randint(0, max_tabs) if max_tabs > 0 else 0

    max_newlines0 = int((newline + 1) * 0.95 * reserve)
    num_newlines0 = np.random.randint(0, max_newlines0) if max_newlines0 > 0 else 0

    max_newlines1 = int((newline + 1) * 0.05 * reserve)
    num_newlines1 = np.random.randint(0, max_newlines1) if max_newlines1 > 0 else 0

    glue_values = [' '] * num_spaces0 + \
                  [u'\u00A0'] * num_spaces1 + \
                  ['\t'] * num_tabs + \
                  ['\n'] * num_newlines0 + \
                  ['\r\n'] * num_newlines1
    np.random.shuffle(glue_values)
    glue_sizes = np.random.exponential(0.5, len(glue_values))

    result = [[' ']]  # At least one space should exist
    si, vi = 0, 0
    while len(glue_values) > vi and len(glue_sizes) > si:
        size = 1 + int(glue_sizes[si])
        value = glue_values[vi:vi + size]
        si, vi = si + 1, vi + size
        result.append(value)

    return result


def augment_paragraphs(source):
    sentence_len = [len(s) for p, _ in source for s in p]
    reserve_inner = min(100000, max(100, sum(sentence_len)))
    inner_glue = cycle(random_glue(space=500, tab=1, newline=1, reserve=reserve_inner))
    outer_glue = cycle(random_glue(space=500, tab=1, newline=15, reserve=max(10, reserve_inner // 5)))
    extra_glue = cycle(random_glue(space=500, tab=5, newline=150, reserve=max(10, reserve_inner // 10)))

    result = []
    for paragraph, token_weight in source:
        _sentences = []
        for sentence in paragraph:
            spaces = [i for i in range(len(sentence)) if ' ' == sentence[i][1]]
            for space in spaces:
                word_glue = next(inner_glue)

                if ''.join(word_glue) == ' ':
                    continue
                if ''.join(word_glue) == '' and \
                        sentence[space][0][-1].isalnum() and \
                        sentence[space + 1][0][0].isalnum():
                    continue

                sentence[space] = (sentence[space][0], ''.join(word_glue))

            sentence_glue = next(extra_glue) if sentence[-1][0].isalnum() else next(outer_glue)
            sentence[-1] = (sentence[-1][0], ''.join(sentence_glue))
            _sentences.append(sentence)
        result.append((_sentences, token_weight))

    return result


def label_spaces(source, target):
    target_labels = ''.join(['T' * len(p[0]) + 'S' * len(p[1]) for p in target])

    source_len = [len(w) for w in source]
    source_acc = [sum(source_len[:i]) for i in range(len(source_len))]

    # Use first character label from source
    labels = [target_labels[i] for i in source_acc]
    assert len(source) == len(labels)

    return labels


def label_tokens(source, target):
    assert ''.join(target) == ''.join(source)

    source_len = [len(w) for w in source]
    source_acc = [sum(source_len[:i + 1]) for i in range(len(source_len))]
    target_len = [len(w) for w in target]
    target_acc = [sum(target_len[:i + 1]) for i in range(len(target_len))]
    same_split = set(source_acc).intersection(target_acc)

    # Break label if same break in target and source at the same time
    labels = ['D' if sum(source_len[:i + 1]) in same_split else 'C' for i in range(len(source_len))]
    assert len(source) == len(labels)

    return labels


def label_paragraphs(source_paragraphs, batch_size=1024):
    assert tf.executing_eagerly()

    # Sort by tokens count for lower memory consumption
    source_paragraphs = sorted(source_paragraphs, key=lambda p: sum([len(s) for s in p[0]]), reverse=True)

    _batch_size = 1
    result_paragraphs = []
    while len(source_paragraphs):
        # Smaller batch size for longer sentences
        _batch_size = min(_batch_size + 1, batch_size)

        pipeline_todo, source_paragraphs = source_paragraphs[:_batch_size], source_paragraphs[_batch_size:]
        pipeline_input = [[''.join(itertools.chain(*s)) for s in p] for p, _ in pipeline_todo]
        pipeline_done = split_words(tf.ragged.constant(pipeline_input), extended=True).to_list()
        assert len(pipeline_done) == len(pipeline_input)

        for done_prgr, (src_prgr, src_wght) in zip(pipeline_done, pipeline_todo):
            paragraph = []
            for done_sent, src_sent in zip(done_prgr, src_prgr):
                done_sent = [w.decode('utf-8') for w in done_sent if len(w)]
                token_labels = label_tokens(done_sent, list(itertools.chain(*src_sent)))
                space_labels = label_spaces(done_sent, src_sent)
                paragraph.append((done_sent, space_labels, token_labels))

            assert len(paragraph) > 0
            result_paragraphs.append((paragraph, src_wght))

    return result_paragraphs


def make_documents(paragraphs, doc_size):
    assert tf.executing_eagerly()

    documents = []
    spaces = []
    tokens = []
    weights = []
    sentences = []

    while len(paragraphs) > 0:
        sample_words = []
        sample_spaces = []
        sample_tokens = []
        sample_weights = []
        sample_sentences = []

        while len(sample_words) < doc_size and len(paragraphs) > 0:
            curr_prgr, weight = paragraphs.pop()

            for sent_words, sent_spaces, sent_tokens in curr_prgr:
                if len(sample_words) and not sent_words[0][0].isalnum():
                    prev_word = sample_words[-1]
                    next_word = sent_words[0]
                    split_size = tf.size(split_words(
                        prev_word + next_word,
                        extended=True
                    )).numpy()
                    if split_size != 2:
                        assert 1 == split_size
                        sample_spaces[-1] = ''
                        sample_tokens[-1] = ''
                        sample_sentences[-1] = ''

                sample_words.extend(sent_words)
                sample_spaces.extend(sent_spaces)
                sample_tokens.extend(sent_tokens)
                sample_weights.extend([weight] * len(sent_tokens))

                word_print = [any([c.isprintable() and not c.isspace() for c in w]) for w in sent_words]
                last_printable = len(word_print) - 1 - word_print[::-1].index(True)
                sent_breaks = ['J'] * len(sent_words)
                sent_breaks[last_printable] = 'B'
                sample_sentences.extend(sent_breaks)

        last_in_doc = max([i for i, w in enumerate(sample_words) if len(w.strip())])
        sample_words = sample_words[:last_in_doc + 1]
        sample_spaces = sample_spaces[:last_in_doc + 1]
        sample_tokens = sample_tokens[:last_in_doc + 1]
        sample_weights = sample_weights[:last_in_doc + 1]
        sample_sentences = sample_sentences[:last_in_doc + 1]
        assert len(sample_words) == len(sample_tokens) == len(sample_spaces) == len(sample_weights) == len(
            sample_sentences)

        if len(sample_sentences):
            sample_sentences[-1] = 'B'

        documents.append(sample_words)
        spaces.append(sample_spaces)
        tokens.append(sample_tokens)
        weights.append(sample_weights)
        sentences.append(sample_sentences)

    assert len(spaces) == len(tokens) == len(weights) == len(sentences)
    dataset = list(zip(documents, spaces, tokens, weights, sentences))

    return dataset


def _serialize_example(document, space_labels, token_labels, token_weights, sentence_labels):
    _doc = ''.join(document)

    space_labels = [s for s in space_labels if len(s)]
    token_weights = [w for w, s in zip(token_weights, token_labels) if len(s)]
    token_labels = [s for s in token_labels if len(s)]
    assert len(token_weights) == len(token_labels)
    sentence_labels = [s for s in sentence_labels if len(s)]

    _sps = ','.join(space_labels)
    _tok = ','.join(token_labels)
    _sent = ','.join(sentence_labels)

    _spsi = [0 if 'T' == s else 1 for s in space_labels]  # 0 == TOKEN, 1 == SPACE
    _toki = [0 if 'D' == s else 1 for s in token_labels]  # 0 == DIVIDE, 1 == COMBINE
    _senti = [0 if 'J' == s else 1 for s in sentence_labels]  # 0 == JOIN, 1 == BREAK

    return tf.train.Example(features=tf.train.Features(feature={
        'document': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(_doc)])),
        'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(space_labels)])),
        'space_labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(_sps)])),
        'token_labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(_tok)])),
        'token_weights': tf.train.Feature(float_list=tf.train.FloatList(value=token_weights)),
        'sentence_labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(_sent)])),
        'space_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=_spsi)),
        'token_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=_toki)),
        'sentence_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=_senti)),
    })).SerializeToString()


def write_dataset(dest_path, base_name, examples_batch):
    try:
        os.makedirs(dest_path)
    except:
        pass

    exist_pattern = base_name + r'-(\d+).tfrecords.gz'
    exist_records = [f for f in os.listdir(dest_path) if re.match(exist_pattern, f)]
    exist_records = [f.replace(base_name + '-', '').replace('.tfrecords.gz', '') for f in exist_records]

    next_index = max([int(head) if head.isdigit() else -1 for head in exist_records] + [-1]) + 1
    file_name = os.path.join(dest_path, '{}-{}.tfrecords.gz'.format(base_name, next_index))

    writer_options = tf.io.TFRecordOptions(compression_type='GZIP')
    with tf.io.TFRecordWriter(file_name, options=writer_options) as writer:
        for doc, sps_lab, tok_lab, tok_wght, sent_lab in examples_batch:
            writer.write(_serialize_example(doc, sps_lab, tok_lab, tok_wght, sent_lab))


def process_split(paragraphs, doc_size, rec_size, num_repeats, dest_path, base_name):
    if num_repeats > 1:
        num_breaks = num_repeats // 3
        _breaks = []
        if num_breaks > 0:
            print('Breaking...')
            _breaks = [([s], w) for p, w in paragraphs for s in p] * num_breaks

        print('Repeating...')
        _repeats = paragraphs * (num_repeats - num_breaks)

        paragraphs = _repeats + _breaks
        del _repeats, _breaks

    print('Shuffling...')
    np.random.shuffle(paragraphs)

    print('Augmenting...')
    paragraphs = augment_paragraphs(paragraphs)

    print('Shuffling...')
    np.random.shuffle(paragraphs)

    print('Labeling...')
    paragraphs = label_paragraphs(paragraphs)

    print('Baking...')
    documents = make_documents(paragraphs, doc_size)

    print('Writing...')
    while len(documents):
        todo, documents = documents[:rec_size], documents[rec_size:]
        write_dataset(dest_path, base_name, todo)


def create_dataset(src_path, dest_path, doc_size, rec_size, num_repeats, token_weight):
    print('Reading source files from {}'.format(src_path))
    source_paragraphs = []
    for file_name in os.listdir(src_path):
        if not file_name.endswith('.conllu'):
            continue

        print('Parsing file {}'.format(file_name))
        file_path = os.path.join(src_path, file_name)

        _token_weight = token_weight if file_name.startswith('_') else 1.0
        current_paragraphs = parse_paragraphs(file_path, _token_weight)
        print('Found {}K paragraphs in {}'.format(len(current_paragraphs) // 1000, file_name))

        source_paragraphs.extend(current_paragraphs)
    print('Finished reading. Found {}K paragraphs'.format(len(source_paragraphs) // 1000))

    print('Shuffling and splitting')
    np.random.shuffle(source_paragraphs)
    total_size = len(source_paragraphs)
    train_paragraphs = source_paragraphs[:int(total_size * 0.9)]
    test_paragraphs = source_paragraphs[int(total_size * 0.9):int(total_size * 0.95)]
    valid_paragraphs = source_paragraphs[int(total_size * 0.95):]
    del source_paragraphs

    print('Processing train dataset')
    process_split(train_paragraphs, doc_size, rec_size, num_repeats, dest_path, 'train')
    del train_paragraphs

    print('Processing test dataset')
    process_split(test_paragraphs, doc_size, rec_size, num_repeats, dest_path, 'test')
    del test_paragraphs

    print('Processing valid dataset')
    process_split(valid_paragraphs, doc_size, rec_size, num_repeats, dest_path, 'valid')
    del valid_paragraphs


def main():
    parser = argparse.ArgumentParser(
        description='Create dataset from files with CoNLL-U markup')
    parser.add_argument(
        'src_path',
        type=str,
        help='Directory with source CoNLL-U files')
    parser.add_argument(
        'dest_path',
        type=str,
        help='Directory to store TFRecord files')
    parser.add_argument(
        '--doc_size',
        type=int,
        default=256,
        help='Target words count per document')
    parser.add_argument(
        '--rec_size',
        type=int,
        default=100000,
        help='Maximum documents count per TFRecord file')
    parser.add_argument(
        '--num_repeats',
        type=int,
        default=5,
        help='How many times repeat source data (useful due paragraphs shuffling and random glue)')
    parser.add_argument(
        '--token_weight',
        type=float,
        default=0.0,
        help='Weight of token for files starting with underscore')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.src_path) and os.path.isdir(argv.src_path)
    assert not os.path.exists(argv.dest_path) or os.path.isdir(argv.dest_path)
    assert argv.doc_size > 0
    assert argv.rec_size > 0
    assert argv.num_repeats > 0

    create_dataset(argv.src_path, argv.dest_path, argv.doc_size, argv.rec_size, argv.num_repeats, argv.token_weight)