from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import re
import six
import tensorflow as tf
from conllu import parse
from itertools import cycle
from tfmiss.text.unicode_expand import split_words
from .conllu import repair_spaces, extract_text, split_sent


def parse_paragraphs(file_name):
    with open(file_name, 'rb') as f:
        content = '\n' + f.read().decode('utf-8')

    content = content.replace('\n# newdoc\n', '\n# newdoc id = 0\n')
    content = content.replace('\n# newpar\n', '\n# newpar id = 0\n')
    has_groups = False

    result = []
    paragraph = []
    for block in parse(content):
        meta = ' '.join(six.iterkeys(block.metadata))

        start_group = 'newdoc id' in meta or 'newpar id' in meta
        has_groups = has_groups or start_group

        if len(paragraph) and (not has_groups or start_group):
            result.append(paragraph)
            paragraph = []

        block = repair_spaces(block)
        sentence = extract_text(block)
        # TODO: clean begin & end
        # TODO: len == 1
        # TODO: contains alpha > 1
        if len(sentence):
            sent_parts = split_sent(block)
            sent_texts = [extract_text(p, validate=False) for p in sent_parts]
            paragraph.extend(sent_texts)

    if len(paragraph):
        result.append(paragraph)

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
    sentence_len = [len(s) for p in source for s in p]
    reserve_inner = min(100000, max(100, sum(sentence_len)))
    inner_glue = cycle(random_glue(space=298, tab=1, newline=1, reserve=reserve_inner))
    outer_glue = cycle(random_glue(space=285, tab=5, newline=10, reserve=max(10, reserve_inner // 5)))
    extra_glue = cycle(random_glue(space=150, tab=25, newline=125, reserve=max(10, reserve_inner // 10)))

    result = []
    for paragraph in source:
        _sentences = []
        for sentence in paragraph:
            spaces = [i for i in range(len(sentence)) if ' ' == sentence[i]]
            for space in reversed(spaces):
                word_glue = next(inner_glue)

                if ''.join(word_glue) == ' ':
                    continue
                if ''.join(word_glue) == '' and \
                        sentence[space - 1][-1].isalnum() and \
                        sentence[space + 1][0].isalnum():
                    continue

                sentence[space] = ''.join(word_glue)

            sentence_glue = next(extra_glue) if sentence[-1].isalnum() else next(outer_glue)
            _sentences.append(sentence + sentence_glue)
        result.append(_sentences)

    del inner_glue, outer_glue, extra_glue

    return result


def label_tokens(source, target):
    assert ''.join(target) == ''.join(source)
    assert len(source) and len(target)

    source_len = [len(w) for w in source]
    source_acc = [sum(source_len[:i + 1]) for i in range(len(source_len))]
    target_len = [len(w) for w in target]
    target_acc = [sum(target_len[:i + 1]) for i in range(len(target_len))]
    same_split = set(source_acc).intersection(target_acc)

    # Break label if same break in target and source at the same time
    labels = ['B' if sum(source_len[:i + 1]) in same_split else 'J' for i in range(len(source_len))]
    assert len(source) == len(labels)

    return labels


def label_paragraphs(source_paragraphs, batch_size=1024):
    assert tf.executing_eagerly()

    # Sort by tokens count for lower memory consumption
    source_paragraphs = sorted(source_paragraphs, key=lambda p: sum([len(s) for s in p]), reverse=True)

    _batch_size = 1
    result_paragraphs = []
    while len(source_paragraphs):
        # Smaller batch size for longer sentences
        _batch_size = min(_batch_size + 1, batch_size)

        pipeline_todo, source_paragraphs = source_paragraphs[:_batch_size], source_paragraphs[_batch_size:]
        # Join tokens into sentences
        pipeline_input = [[''.join(s) for s in p] for p in pipeline_todo]

        # Align paragraphs length
        max_len = max([len(p) for p in pipeline_todo])
        pipeline_input = [p + [''] * (max_len - len(p)) for p in pipeline_input]

        pipeline_done = split_words(pipeline_input, stop=True).to_tensor(default_value='')
        assert len(pipeline_done) == len(pipeline_input)

        for done_prgr, src_prgr in zip(pipeline_done, pipeline_todo):
            done_prgr = np.char.decode(done_prgr.numpy().astype('S'), encoding='utf-8').tolist()
            done_prgr = [done_sent for done_sent in done_prgr if len(''.join(done_sent)) > 0]
            assert len(done_prgr) == len(src_prgr)

            paragraph = []
            for done_sent, src_sent in zip(done_prgr, src_prgr):
                done_sent = [w for w in done_sent if len(w)]
                assert len(done_sent) > 0

                labels = label_tokens(done_sent, src_sent)
                paragraph.append((done_sent, labels))

            assert len(paragraph) > 0
            result_paragraphs.append(paragraph)

    return result_paragraphs


def make_documents(paragraphs, doc_size):
    assert tf.executing_eagerly()

    documents = []
    tokens = []
    sentences = []

    while len(paragraphs) > 0:
        sample_words = []
        token_labels = []
        sentence_labels = []

        while len(sample_words) < doc_size and len(paragraphs) > 0:
            curr_prgr = paragraphs.pop()

            for words, breaks in curr_prgr:
                if len(sample_words) and not words[0][0].isalnum():
                    prev_word = sample_words[-1]
                    next_word = words[0]
                    rest_words = split_words(
                        prev_word + next_word,
                        stop=True
                    ).numpy()
                    if len(rest_words) != 2:
                        assert len(rest_words) == 1
                        token_labels[-1] = ''
                        sentence_labels[-1] = ''

                last_in_sent = max([i for i, w in enumerate(words) if len(w.strip())])
                sample_words.extend(words)
                token_labels.extend(breaks)
                sentence_labels.extend(['J'] * last_in_sent)
                sentence_labels.extend(['B'] * (len(words) - last_in_sent))

        last_in_doc = max([i for i, w in enumerate(sample_words) if len(w.strip())])
        sample_words = sample_words[:last_in_doc + 1]
        token_labels = token_labels[:last_in_doc + 1]
        sentence_labels = sentence_labels[:last_in_doc + 1]
        assert len(sample_words) == len(token_labels) == len(sentence_labels)

        documents.append(sample_words)
        tokens.append(token_labels)
        sentences.append(sentence_labels)

    assert len(tokens) == len(sentences)
    dataset = list(zip(documents, tokens, sentences))

    return dataset


def write_dataset(dest_path, base_name, examples_batch):
    def create_example(document, tokens, sentences):
        _src = '|'.join(document)
        _doc = ''.join(document)
        _tok = ','.join([t for t in tokens if len(t)])
        _sent = ','.join([s for s in sentences if len(s)])

        return tf.train.Example(features=tf.train.Features(feature={
            'document': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(_doc)])),
            'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(tokens)])),
            'tokens': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(_tok)])),
            'sentences': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(_sent)])),
        })).SerializeToString()

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
        for doc, tok, sent in examples_batch:
            writer.write(create_example(doc, tok, sent))


def _process_split(paragraphs, doc_size, dest_path, base_name, rec_size, num_repeats=1):
    print('Breaking...')
    _paragraphs = list(paragraphs)  # clone
    _paragraphs.extend([[s] for p in paragraphs for s in p])  # break
    paragraphs = list(_paragraphs)
    del _paragraphs

    if num_repeats > 1:
        print('Shuffling...')
        np.random.shuffle(paragraphs)

        print('Repeating...')
        paragraphs = paragraphs * num_repeats

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
        help='Maximum words count per document')
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

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.src_path) and os.path.isdir(argv.src_path)
    assert not os.path.exists(argv.dest_path) or os.path.isdir(argv.dest_path)
    assert argv.doc_size > 0
    assert argv.rec_size > 0
    assert argv.num_repeats > 0

    print('Reading source files from {}'.format(argv.src_path))
    source_paragraphs = []
    for file_name in os.listdir(argv.src_path):
        if not file_name.endswith('.conllu'):
            continue

        print('Parsing file {}'.format(file_name))
        file_path = os.path.join(argv.src_path, file_name)

        _paragraphs = parse_paragraphs(file_path)
        print('Found {}K paragraphs in {}'.format(len(_paragraphs) // 1000, file_name))

        source_paragraphs.extend(_paragraphs)
    print('Finished reading. Found {}K paragraphs'.format(len(source_paragraphs) // 1000))

    print('Shuffling and splitting')
    np.random.shuffle(source_paragraphs)
    total_size = len(source_paragraphs)
    train_paragraphs = source_paragraphs[:int(total_size * 0.9)]
    test_paragraphs = source_paragraphs[int(total_size * 0.9):int(total_size * 0.95)]
    valid_paragraphs = source_paragraphs[int(total_size * 0.95):]
    del source_paragraphs

    print('Processing train dataset')
    _process_split(train_paragraphs, argv.doc_size, argv.dest_path, 'train', argv.rec_size, argv.num_repeats)
    del train_paragraphs

    print('Processing test dataset')
    _process_split(test_paragraphs, argv.doc_size, argv.dest_path, 'test', argv.rec_size, argv.num_repeats)
    del test_paragraphs

    print('Processing valid dataset')
    _process_split(valid_paragraphs, argv.doc_size, argv.dest_path, 'valid', argv.rec_size, argv.num_repeats)
    del valid_paragraphs
