import argparse
import copy
import itertools
import os
import numpy as np
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from conllu import parse
from itertools import cycle
from tfmiss.text.unicode_expand import split_words
from tfmiss.preprocessing import spaces_after
from typing import List, Tuple, Union
from .conllu import repair_spaces, extract_tokens, extract_text, split_sents

Paragraphs = List[List[List[Tuple[str, str]]]]
LabledParagraphs = List[List[Tuple[List[str], List[str], List[str]]]]
LabeledDocuments = List[Tuple[List[str], List[str], List[str]]]

KNOWN_SPACES = {
    '\t', '\n', '\x0b', '\x0c', '\r', '\x1c', '\x1d', '\x1e', '\x1f', ' ', '\x85', '\xa0', '\u1680', '\u2000', '\u2001',
    '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a', '\u2028', '\u2029',
    '\u200b', '\u202f', '\u205f', '\u2060', '\u2061', '\u2800', '\u3000', '\ufeff'}


def parse_paragraphs(file_name: str) -> Paragraphs:
    # Parse CoNLL-U file and separate paragraphs

    with open(file_name, 'rb') as f:
        content = '\n' + f.read().decode('utf-8')

    content = content.replace('\n# newdoc\n', '\n# newdoc id = 0\n')
    content = content.replace('\n# newpar\n', '\n# newpar id = 0\n')
    has_groups = False

    paragraphs, current = [], []
    for block in parse(content):
        meta = ' '.join(block.metadata.keys())
        start_group = 'newdoc id' in meta or 'newpar id' in meta
        has_groups = has_groups or start_group

        if len(current) and (not has_groups or start_group):
            paragraphs.append(current)
            current = []

        block = repair_spaces(block)
        text = extract_text(block, validate=True, last_space=True)
        if not len(text):
            continue

        sent_parts = split_sents(block)
        sent_texts = [extract_tokens(p, last_space=True) for p in sent_parts]
        for text in sent_texts:
            for tt, _ in text:
                assert tt[0] not in KNOWN_SPACES, \
                    'Token should not starts with space: {} / {} / {}'.format(file_name, block.metadata['sent_id'], tt)
                assert tt[-1] not in KNOWN_SPACES, \
                    'Token should not ends with space: {} / {} / {}'.format(file_name, block.metadata['sent_id'], tt)

        current.extend(sent_texts)

    if len(current):
        paragraphs.append(current)

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


def good_paragraphs(paragraphs: Paragraphs) -> Paragraphs:
    # Drop bad-without-context sentences like "? hello !" and "|"

    def _bad(par):
        if len(par) > 1:
            return False

        sent = par[0]
        if len(sent) < 2:
            return True

        text = ''.join(itertools.chain(*sent)).strip()

        if not re.match(r'.*\w{2,}.*', text):
            return True

        good_stoppers = set('.!?…')
        if text[0].isupper() and text[-1] in good_stoppers:
            return False

        amb_starters = set('-—"№«([―.•↑*@#–+©〈¶·‣►●◦')
        if text[0] in amb_starters and text[-1] in good_stoppers:
            return False

        bad_starters = set('.!?,:;/_…>‼~%|\\»)]')
        bad_stoppers = set(',•©¶·')
        if text[0] in bad_starters or text[-1] in bad_stoppers:
            return True

        return False

    return [p for p in paragraphs if not _bad(p)]


def random_glue(space: int = 1, tab: int = 0, newline: int = 0, empty: int = 0, reserve: int = 0) -> List[List[str]]:
    # Generate random spaces with random size

    assert space > 0, 'Number of spaces should be positive'

    max_spaces0 = int((space + 1) * 0.995 * reserve)
    num_spaces0 = np.random.randint(0, max_spaces0) if space > 0 and max_spaces0 > 0 else 0

    max_spaces1 = int((space + 1) * 0.005 * reserve)
    num_spaces1 = np.random.randint(0, max_spaces1) if space > 0 and max_spaces1 > 0 else 0

    max_tabs = int((tab + 1) * reserve)
    num_tabs = np.random.randint(0, max_tabs) if tab > 0 and max_tabs > 0 else 0

    max_newlines0 = int((newline + 1) * 0.95 * reserve)
    num_newlines0 = np.random.randint(0, max_newlines0) if newline > 0 and max_newlines0 > 0 else 0

    max_newlines1 = int((newline + 1) * 0.05 * reserve)
    num_newlines1 = np.random.randint(0, max_newlines1) if newline > 0 and max_newlines1 > 0 else 0

    max_empties = int((empty + 1) * reserve)
    num_empties = np.random.randint(0, max_empties) if empty > 0 and max_empties > 0 else 0

    glue_values = [' '] * num_spaces0 + \
                  ['\u00A0'] * num_spaces1 + \
                  ['\t'] * num_tabs + \
                  ['\n'] * num_newlines0 + \
                  ['\r\n'] * num_newlines1 + \
                  [''] * num_empties

    np.random.shuffle(glue_values)
    glue_sizes = np.random.exponential(0.25, len(glue_values))

    result = [[' ']]  # At least one space should exist
    si, vi = 0, 0
    while len(glue_values) > vi and len(glue_sizes) > si:
        size = 1 + int(glue_sizes[si])
        value = glue_values[vi:vi + size]
        si, vi = si + 1, vi + size
        result.append(value)

    return result


def augment_paragraphs(raw_paragraphs: Paragraphs) -> Paragraphs:
    # Randomly replace spaces between tokens and sentences

    sentence_len = sum([len(s) for p in raw_paragraphs for s in p])
    reserve_inner = min(100000, max(100, sentence_len))
    inner_glue = cycle(random_glue(space=1000, tab=1, newline=1, reserve=reserve_inner))
    outer_glue = cycle(random_glue(space=500, tab=1, newline=5, empty=5, reserve=max(10, reserve_inner // 5)))
    extra_glue = cycle(random_glue(space=500, tab=5, newline=100, reserve=max(10, reserve_inner // 10)))
    bad_stoppers = set(',:;•©¶·')

    aug_paragraphs = []
    for paragraph in raw_paragraphs:
        paragraph_ = []
        for si, sentence in enumerate(paragraph):
            spaces = [k for k, (_, s) in enumerate(sentence[:-1]) if ' ' == s]
            for k in spaces:
                word_glue = ''.join(next(inner_glue))
                if ' ' == word_glue:
                    continue

                curr_token = sentence[k][0]
                next_token = sentence[k + 1][0]
                if '' == word_glue and curr_token[-1].isalnum() and next_token[0].isalnum():
                    continue

                sentence[k] = (curr_token, word_glue)

            last_token, last_space = sentence[-1]
            if '\n' in last_space:
                sent_glue = last_space
            elif last_token[-1].isalnum() or last_token[-1] in bad_stoppers:
                sent_glue = ''.join(next(extra_glue))
            else:
                sent_glue = ''.join(next(outer_glue))
                next_token = ' ' if si == len(paragraph) - 1 else paragraph[si + 1][0][0]
                if '' == sent_glue and (last_token[-1] not in {'.', '!', '?'} or not next_token[0].isalnum()):
                    sent_glue = ' '
            sentence[-1] = (last_token, sent_glue)
            paragraph_.append(sentence)
        aug_paragraphs.append(paragraph_)

    return aug_paragraphs


def label_tokens(tokens: List[str], spaces: List[str], target: List[Tuple[str, str]]) -> List[str]:
    # Estimate token labels

    for tt, _ in target:
        assert tt[0] not in KNOWN_SPACES, 'Token should not starts with space'
        assert tt[-1] not in KNOWN_SPACES, 'Token should not ends with space'

    flat_target = list(itertools.chain(*target))
    flat_src = [(tokens[i], spaces[i]) for i in range(len(tokens))]
    flat_src = [w for w in itertools.chain(*flat_src) if len(w)]
    flat_src = list(itertools.chain(*flat_src))
    assert ''.join(flat_target) == ''.join(flat_src), 'Joined sources and target tokens should be equal'

    if not len(flat_target):
        return []

    source_len = [len(w) for w in tokens]
    source_acc = [sum(source_len[:i + 1]) for i in range(len(source_len))]
    target_len = [len(w[0]) for w in target]
    target_acc = [sum(target_len[:i + 1]) for i in range(len(target_len))]
    same_split = set(source_acc).intersection(target_acc)

    sure_split = [sum(source_len[:i + 1]) for i in range(len(source_len)) if len(spaces[i])]

    # Break label if same break in target and source at the same time or if space placed after token
    full_split = same_split.union(sure_split)
    labels = ['B' if sum(source_len[:i + 1]) in full_split else 'I' for i in range(len(source_len))]
    assert len(tokens) == len(labels), 'Joined sources and labels should be equal'

    return ['B'] + labels[:-1]  # shift right


def label_paragraphs(source: Paragraphs, batch_size: int = 1024) -> LabledParagraphs:
    # Estimate labels for paragraphs
    result = []
    for paragraph in source:
        flat = [''.join(itertools.chain(*s)) for s in paragraph]
        # flat = tf.ragged.constant()
        split = split_words(flat, extended=True)
        tokens, spaces = spaces_after(split)
        tokens, spaces = tokens.to_list(), spaces.to_list()
        assert len(tokens) == len(spaces) == len(flat), 'Sizes of paragraphs before and after split should be equal'

        labeled = []
        for done_tok, done_sps, src_sent in zip(tokens, spaces, paragraph):
            done_tok = [w.decode('utf-8') for w in done_tok if len(w)]
            done_sps = [w.decode('utf-8') for w in done_sps][:len(done_tok)]
            token_labels = label_tokens(done_tok, done_sps, src_sent)
            labeled.append((done_tok, done_sps, token_labels))

        assert len(labeled), 'Paragraph could not be empty'
        result.append(labeled)

    return result


def make_documents(paragraphs: LabledParagraphs, doc_size: int) -> LabeledDocuments:
    # Combine labeled paragraphs into labeled documents

    documents = []
    while len(paragraphs):
        sample_words = []
        sample_tokens = []
        sample_sentences = []

        while len(sample_words) < doc_size and len(paragraphs):
            curr_par = paragraphs.pop()

            for sent_words, sent_spaces, sent_tokens in curr_par:
                assert len(sent_words), 'Sentence could not be empty'
                assert len(sent_words) == len(sent_spaces) == len(sent_tokens)

                word_print = [c.isprintable() and not c in KNOWN_SPACES for c in ''.join(sent_words)]
                if not any(word_print):
                    continue

                if len(sample_words):
                    prev_word = sample_words[-1]
                    next_word = sent_words[0]
                    can_split = (tf.size(split_words(prev_word + next_word, extended=True)) == \
                                 tf.size(split_words(prev_word, extended=True)) + \
                                 tf.size(split_words(next_word, extended=True))).numpy().item()

                    if not can_split:
                        sample_words[-1] += ' '
                        prev_word = sample_words[-1]
                        can_split = (tf.size(split_words(prev_word + next_word, extended=True)) == \
                                     tf.size(split_words(prev_word, extended=True)) + \
                                     tf.size(split_words(next_word, extended=True))).numpy().item()

                    assert can_split, 'Words can\'t be splitted: {} & {}'.format(prev_word, next_word)

                for word, space in zip(sent_words, sent_spaces):
                    sample_words.append(word + space)
                sample_tokens.extend(sent_tokens)

                sent_breaks = ['B'] + ['I'] * (len(sent_words) - 1)
                sample_sentences.extend(sent_breaks)

        assert len(sample_words) == len(sample_tokens) == len(sample_sentences), \
            'Sequence and it\'s labels should have same size'

        documents.append((sample_words, sample_tokens, sample_sentences))

    return documents


class STBDDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}

    def __init__(self, *, source_dirs, data_dir, doc_size, num_repeats, test_re, config=None,
                 version=None):
        super().__init__(data_dir=data_dir, config=config, version=version)

        if isinstance(source_dirs, str):
            source_dirs = [source_dirs]
        assert isinstance(source_dirs, list), 'A list expected for source directories'
        source_dirs = [os.fspath(s) for s in source_dirs]

        bad = [s for s in source_dirs if not os.path.isdir(s)]
        assert not bad, 'Some of source directories do not exist: {}'.format(bad)

        self.source_dirs = source_dirs
        self.doc_size = doc_size
        self.num_repeats = num_repeats
        self.test_re = test_re

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description='Sentence & token boundary detection dataset',
            features=tfds.features.FeaturesDict({
                'document': tfds.features.Text(),
                'length': tfds.features.Tensor(shape=(), dtype=tf.int32),
                'token': tfds.features.Text(),
                'sentence': tfds.features.Text()
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'test': self._generate_examples(False),
            'train': self._generate_examples(True),
        }

    def _generate_examples(self, training):
        for source_dir in self.source_dirs:
            print('Processing directory {}'.format(source_dir))
            backup = []

            for file_path in self._iterate_source(source_dir, training):
                print('Parsing file {}'.format(file_path))

                parsed = parse_paragraphs(file_path)
                print('Found {}K paragraphs in {}'.format(len(parsed) // 1000, file_path))
                backup.extend(parsed)

            for i in range(self.num_repeats):
                paragraphs = copy.deepcopy(backup)

                breaking = 1 == i % 2
                print('Iteration: {} of {}. Breaking: {}'.format(i + 1, self.num_repeats, breaking))

                if breaking:
                    paragraphs = [[s] for p in paragraphs for s in p]

                print('Filtering...')
                paragraphs = good_paragraphs(paragraphs)

                print('Augmenting...')
                paragraphs = augment_paragraphs(paragraphs)

                print('Labeling...')
                paragraphs = label_paragraphs(paragraphs)

                print('Shuffling...')
                np.random.shuffle(paragraphs)

                print('Baking...')
                doc_size = 1 if breaking and 1 == i % 4 else self.doc_size
                documents = make_documents(paragraphs, doc_size)

                for j, (words, toks, sents) in enumerate(documents):
                    key = '{}_{}_{}_{}'.format(source_dir, training, i, j)

                    labels = [s + t for s, t in zip(sents, toks)]
                    assert 0 == len(set(labels) - {'BB', 'IB', 'II'})

                    yield key, {
                        'document': ''.join(words),
                        'length': len(toks),
                        'token': ''.join(toks),
                        'sentence': ''.join(sents)
                    }

    def _iterate_source(self, source_dir, training):
        for dirpath, _, filenames in os.walk(source_dir):
            for file in filenames:
                if training == bool(re.search(self.test_re, os.path.join('/', dirpath, file))):
                    continue
                if not file.endswith('.conllu'):
                    continue

                yield os.path.join(dirpath, file)


def create_dataset(
        source_dirs: Union[str, List[str]], data_dir: str, doc_size: int, num_repeats: int,
        test_re: str) -> STBDDataset:
    tfds.disable_progress_bar()

    builder = STBDDataset(
        source_dirs=source_dirs, data_dir=data_dir, doc_size=doc_size, num_repeats=num_repeats, test_re=test_re)
    builder.download_and_prepare()

    return builder


def main():
    parser = argparse.ArgumentParser(
        description='Create dataset from files with CoNLL-U markup')
    parser.add_argument(
        'src_path',
        type=str,
        nargs='+',
        help='Directory with languages directories followed by CoNLL-U files')
    parser.add_argument(
        'dest_path',
        type=str,
        help='Directory to store composed dataset')
    parser.add_argument(
        '--doc_size',
        type=int,
        default=256,
        help='Target words count per document')
    parser.add_argument(
        '--num_repeats',
        type=int,
        default=5,
        help='How many times repeat source data (useful due paragraphs shuffling and random glue)')

    argv, _ = parser.parse_known_args()
    for src_path in argv.src_path:
        assert os.path.exists(src_path) and os.path.isdir(src_path), 'Wrong source path'
    assert not os.path.exists(argv.dest_path) or os.path.isdir(argv.dest_path), 'Wrong destination path'
    assert 0 < argv.doc_size, 'Wrong document size'
    assert 0 < argv.num_repeats, 'Wrong number of repeats'

    create_dataset(argv.src_path, argv.dest_path, argv.doc_size, argv.num_repeats, test_re='[-_]test\\.')
