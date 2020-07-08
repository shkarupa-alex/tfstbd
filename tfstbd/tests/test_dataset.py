# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import tensorflow as tf
import tempfile
import unittest
from tfmiss.text.unicode_expand import split_words
from ..dataset import parse_paragraphs, random_glue, augment_paragraphs, label_spaces, label_tokens
from ..dataset import label_paragraphs, make_documents, write_dataset
from ..hparam import build_hparams
from ..input import _parse_examples


class TestParseConllu(unittest.TestCase):
    def test_parse_paragraphs(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_paragraphs.conllu'), 1.0)
        self.assertEqual([
            ([
                 [('«', ''), ('Если', ' '), ('передача', ' '), ('цифровых', ' '), ('технологий', ' '), ('сегодня', ' '),
                  ('в', ' '), ('США', ' '), ('происходит', ' '), ('впервые', ''), (',', ' '), ('то', ' '), ('о', ' '),
                  ('мирной', ' '), ('передаче', ' '), ('власти', ' '), ('такого', ' '), ('не', ' '), ('скажешь', ''),
                  ('»', ''), (',', ' '), ('–', ' '), ('написала', ' '), ('Кори', ' '), ('Шульман', ''), (',', ' '),
                  ('специальный', ' '), ('помощник', ' '), ('президента', ' '), ('Обамы', ' '), ('в', ' '),
                  ('своем', ' '), ('блоге', ' '), ('в', ' '), ('понедельник', ''), ('.', '')],
                 [('Для', ' '), ('тех', ''), (',', ' '), ('кто', ' '), ('следит', ' '), ('за', ' '), ('передачей', ' '),
                  ('всех', ' '), ('материалов', ''), (',', ' '), ('появившихся', ' '), ('в', ' '), ('социальных', ' '),
                  ('сетях', ' '), ('о', ' '), ('Конгрессе', ''), (',', ' '), ('это', ' '), ('будет', ' '),
                  ('происходить', ' '), ('несколько', ' '), ('по-другому', ''), ('.', '')]
             ], 1.0),
            ([
                 [('Но', ' '), ('в', ' '), ('отступлении', ' '), ('от', ' '), ('риторики', ' '), ('прошлого', ' '),
                  ('о', ' '), ('сокращении', ' '), ('иммиграции', ' '), ('кандидат', ' '), ('Республиканской', ' '),
                  ('партии', ' '), ('заявил', ''), (',', ' '), ('что', ' '), ('в', ' '), ('качестве', ' '),
                  ('президента', ' '), ('он', ' '), ('позволил', ' '), ('бы', ' '), ('въезд', ' '), ('«', ''),
                  ('огромного', ' '), ('количества', ''), ('»', ' '), ('легальных', ' '), ('мигрантов', ' '),
                  ('на', ' '), ('основе', ' '), ('«', ''), ('системы', ' '), ('заслуг', ''), ('»', ''), ('.', '')]
             ], 1.0)
        ], result)

    def test_parse_plain(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_plain.conllu'), 1.0)
        self.assertEqual([
            ([
                 [(u'Ранее', u' '), (u'часто', u' '), (u'писали', u' '), (u'"', u''), (u'алгорифм', u''), (u'"', u''),
                  (u',', u' '), (u'сейчас', u' '), (u'такое', u' '), (u'написание', u' '), (u'используется', u' '),
                  (u'редко', u''), (u',', u' '), (u'но', u''), (u',', u' '), (u'тем', u' '), (u'не', u' '),
                  (u'менее', u''), (u',', u' '), (u'имеет', u' '), (u'место', u' '), (u'(', u''), (u'например', u''),
                  (u',', u' '), (u'Нормальный', u' '), (u'алгорифм', u' '), (u'Маркова', u''), (u')', u''),
                  (u'.', u'')],
             ], 1.0),
            ([
                 [(u'Кто', u' '), (u'знает', u''), (u',', u' '), (u'что', u' '), (u'он', u' '), (u'там', u' '),
                  (u'думал', u''), (u'!', u''), (u'.', u''), (u'.', '')],
             ], 1.0),
        ], result)

    def test_parse_udpipe(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_udpipe.conllu'), 1.0)
        self.assertEqual([
            ([
                 [(u'Порву', u''), (u'!', u'')],
             ], 1.0),
            ([
                 [(u'Порву', u''), (u'!', u''), (u'"', u''), (u'...', u''), (u'©', u'')],
                 [(u'Порву', u''), (u'!', u'')]
             ], 1.0),
            ([
                 [(u'Ребят', u''), (u',', u' '), (u'я', u' '), (u'никому', u' '), (u'не', u' '), (u'звонила', u''),
                  (u'?', u''), (u'?', u''), (u'?', u' \t '), (u')))', u'')],
                 [(u'Вот', u' '), (u'это', u' '), (u'был', u' '), (u'номер', u''), (u'...', u''), (u')', ''),
                  (u'))', u'')],
             ], 1.0),
        ], result)

    def test_parse_collapse(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_collapse.conllu'), 1.0)
        self.assertEqual([
            ([
                 [(u'Er', u' '), (u'arbeitet', u' '), (u'fürs', u' '), (u'FBI', u' '), (u'(', u''), (u'deutsch', u' '),
                  (u'etwa', u''), (u':', u' '), (u'„', u''), (u'Bundesamt', u' '), (u'für', u' '), (u'Ermittlung', u''),
                  (u'“', u''), (u')', u''), (u'.', u'')],
             ], 1.0),
        ], result)

    def test_parse_space(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_space.conllu'), 1.0)
        self.assertEqual([
            ([
                 [(u'Таким образом', u' '), (u',', u' '), (u'если', u' '), (u'приговор', u'')]
             ], 1.0),
            ([
                 [(u'Она', u' '), (u'заболела', u' '), (u'потому, что', u' '), (u'мало', u' '), (u'ела', u' '),
                  (u'.', u'')]
             ], 1.0)
        ], result)

    def test_parse_duplicates(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_duplicates.conllu'), 1.0)
        self.assertEqual([([[(u'Ранее', u'')]], 1.0)], result)

    def test_parse_weigted(self):
        result = parse_paragraphs(
            os.path.join(os.path.dirname(__file__), 'data', 'dataset__tw_10.01__weighted.conllu'), 10.01)
        self.assertEqual([
            ([[('Ранее', '')]], 10.01),
            ([[('Позднее', '')]], 10.01)
        ], result)


class TestRandomGlue(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)

    def test_empty(self):
        self.assertEqual([[' ']], random_glue())

    def test_shape(self):
        result = random_glue(1, 1, 1, 1)
        self.assertEqual([[' ']], result)

    def test_normal(self):
        result = random_glue(space=10, tab=1, newline=1, reserve=1)
        self.assertEqual([[' '], [' '], [' '], [' '], [' '], [' '], [' '], [' '], [' '], ['\t']], result)

    def test_extra(self):
        result = random_glue(space=30, tab=5, newline=25, reserve=1)
        self.assertEqual({' \n', ' ', '\t', '\n'}, set([''.join(s) for s in result]))


class TestAugmentParagraphs(unittest.TestCase):
    def test_empty(self):
        result = augment_paragraphs([])
        self.assertEqual([], result)

    def test_normal(self):
        np.random.seed(36)
        source = [
            ([[('First', ' '), ('sentence', ' '), ('in', ' '), ('paragraph', ''), ('.', '')],
              [('Second', ' '), ('"', ''), ('sentence', ''), ('"', ' '), ('in', ' '), ('paragraph', ''), ('.', '')]],
             1.0),
            ([[('Single', ' '), ('-', ' '), ('sentence', ''), ('.', '')]], 1.0),
        ]
        expected = [
            ([[('First', ' '), ('sentence', ' '), ('in', u'\xa0'), ('paragraph', ''), ('.', ' ')],
              [('Second', '  '), ('"', ''), ('sentence', ''), ('"', '  '), ('in', ' '), ('paragraph', ''), ('.', ' ')]],
             1.0),
            ([[('Single', ' '), ('-', ' '), ('sentence', ''), ('.', ' ')]], 1.0),
        ]
        result = augment_paragraphs(source)
        self.assertEqual(expected, result)

    def test_spaces(self):
        np.random.seed(1)
        source = [
            ([[('Single', ' '), ('sentence', '')],
              [('Next', ' '), ('single', ' '), ('sentence', '')]], 1.1)
        ]
        expected = [
            ([[('Single', ' '), ('sentence', ' ')],
              [('Next', ' '), ('single', ' '), ('sentence', '\n')]], 1.1)
        ]
        result = augment_paragraphs(source)
        self.assertEqual(expected, result)


class TestLabelSpaces(unittest.TestCase):
    def test_empty(self):
        result = label_spaces([], [])
        self.assertEqual([], result)

    def test_same(self):
        result = label_spaces([
            'word', ' ', ' ', ' ', 'word', '-', 'word'
        ], [
            ('word', '   '), ('word', ''), ('-', ''), ('word', '')
        ])
        self.assertEqual(['T', 'S', 'S', 'S', 'T', 'T', 'T'], result)

    def test_joined_source(self):
        result = label_spaces([
            'word', ' ', 'word-word'
        ], [
            ('word', ' '), ('word', ''), ('-', ''), ('word', '')
        ])
        self.assertEqual(['T', 'S', 'T'], result)

    def test_joined_target(self):
        result = label_spaces([
            'word', ' ', 'word', '-', 'word'
        ], [
            ('word', ' '), ('word-word', '')
        ])
        self.assertEqual(['T', 'S', 'T', 'T', 'T'], result)


class TestLabelTokens(unittest.TestCase):
    def test_empty(self):
        result = label_tokens([], [])
        self.assertEqual([], result)

    def test_same(self):
        result = label_tokens([
            'word', '+', 'word', '-', 'word'
        ], [
            'word', '+', 'word', '-', 'word'
        ])
        self.assertEqual(['D', 'D', 'D', 'D', 'D'], result)

    def test_normal_join(self):
        result = label_tokens([
            'word', '+', 'word', '-', 'word', '_', 'word'
        ], [
            'word', '+', 'word-word', '_word'
        ])
        self.assertEqual(['D', 'D', 'C', 'C', 'D', 'C', 'D'], result)

    def test_joined_source(self):
        result = label_tokens([
            'word+word-word'
        ], [
            'word', '+', 'word', '-', 'word'
        ])
        self.assertEqual(['D'], result)

    def test_joined_target(self):
        result = label_tokens([
            'word', '+', 'word', '-', 'word'
        ], [
            'word+word-word'
        ])
        self.assertEqual(['C', 'C', 'C', 'C', 'D'], result)

    def test_different_join(self):
        result = label_tokens([
            'word+word', '-', 'word'
        ], [
            'word', '+', 'word-word'
        ])
        self.assertEqual(['C', 'C', 'D'], result)


class TestLabelParagraphs(tf.test.TestCase):
    def test_empty(self):
        source = []
        result = label_paragraphs(source)
        self.assertEqual([], result)

    def test_normal(self):
        source = [
            ([[('Single', ' '), ('-', ' '), ('sentence.', '\t')]], 1.0),
            ([[('First', ' '), ('sentence', '   '), ('in', ' '), ('paragraph', ''), ('.', '\n')],
              [('Second', ' '), ('"', ''), ('sentence', ''), ('"', ' '), ('in', u'\u00A0'), ('paragraph', ''),
               ('.', ' ')]], 1.1),
        ]
        expected = [
            ([
                 ([u'First', u' ', u'sentence', u'   ', u'in', u' ', u'paragraph', u'.', u'\n'],
                  ['T', 'S', 'T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']),
                 ([u'Second', u' ', u'"', u'sentence', u'"', u' ', u'in', u'\u00A0', u'paragraph', u'.', u' '],
                  ['T', 'S', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'])
             ], 1.1),
            ([
                 ([u'Single', u' ', u'-', u' ', u'sentence', u'.', u'\t'],
                  ['T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['D', 'D', 'D', 'D', 'C', 'D', 'D'])
             ], 1.0),
        ]
        result = label_paragraphs(source)
        self.assertEqual(expected, result)

    def test_compex(self):
        source = [
            ([[('', ' '), ('test@test.com', ' ')],
              [('', ' '), ('www.test.com', ' ')],
              [('', ' '), ('word..word', ' ')],
              [('', ' '), ('word+word-word', ' ')],
              [('', ' '), ('word\\word/word#word', ' ')]], 1.0),
            ([[('', ' '), ('test', ''), ('@', ''), ('test', ''), ('.', ''), ('com', ' ')],
              [('', ' '), ('www', ''), ('.', ''), ('test', ''), ('.', ''), ('com', ' ')],
              [('', ' '), ('word', ''), ('..', ''), ('word', ' ')],
              [('', ' '), ('word', ''), ('+', ''), ('word', ''), ('-', ''), ('word', ' ')],
              [('', ' '), ('word', ''), ('\\', ''), ('word', ''), ('/', ''), ('word', ''), ('#', ''), ('word', ' ')]],
             1.1),
        ]
        expected = [
            ([
                 ([' ', 'test', '@', 'test', '.', 'com', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'D', 'D', 'D', 'D', 'D', 'D']),

                 ([' ', 'www', '.', 'test', '.', 'com', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'D', 'D', 'D', 'D', 'D', 'D']),

                 ([' ', 'word', '.', '.', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'D', 'C', 'D', 'D', 'D']),

                 ([' ', 'word', '+', 'word', '-', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'D', 'D', 'D', 'D', 'D', 'D']),

                 ([' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']),
             ], 1.1),
            ([
                 ([' ', 'test', '@', 'test', '.', 'com', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'C', 'C', 'C', 'C', 'D', 'D']),

                 ([' ', 'www', '.', 'test', '.', 'com', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'C', 'C', 'C', 'C', 'D', 'D']),

                 ([' ', 'word', '.', '.', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'C', 'C', 'C', 'D', 'D']),

                 ([' ', 'word', '+', 'word', '-', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'C', 'C', 'C', 'C', 'D', 'D']),

                 ([' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'D']),
             ], 1.0),
        ]

        result = label_paragraphs(source)
        self.assertEqual(expected, result)


class TestMakeDocuments(tf.test.TestCase):
    def test_empty(self):
        source = []
        result = make_documents(source, 2)
        self.assertEqual([], result)

    def test_normal(self):
        source = [
            ([
                 (['Single', ' ', '-', ' ', 'sentence', '.', '\t'],
                  ['T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['D', 'D', 'D', 'D', 'D', 'D', 'D'])
             ], 1.0),
            ([
                 ([' ', 'test', '@', 'test', '.', 'com', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['D', 'D', 'C', 'C', 'C', 'C', 'D']),
             ], 1.1),
            ([
                 (['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n'],
                  ['T', 'S', 'T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']),
                 (['Second', ' ', '"', 'sentence', '"', ' ', 'in', u'\u00A0', 'paragraph', '.', ' '],
                  ['T', 'S', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'])
             ], 1.2),
        ]

        expected_documents = [
            ['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n', 'Second', ' ', '"', 'sentence', '"',
             ' ', 'in', u'\xa0', 'paragraph', '.', ' '],
            [' ', 'test', '@', 'test', '.', 'com', ' ', 'Single', ' ', '-', ' ', 'sentence', '.', '\t']
        ]
        expected_spaces = [
            ['T', 'S', 'T', 'S', 'T', 'S', 'T', 'T', 'S', 'T', 'S', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'T', 'S'],
            ['S', 'T', 'T', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'S', 'T', 'T', 'S']
        ]
        expected_tokens = [
            ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
            ['D', 'D', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
        ]
        expected_weights = [
            [1.2] * 20,
            [1.1] * 7 + [1.0] * 7
        ]
        expected_labels = [
            ['J', 'J', 'J', 'J', 'J', 'J', 'J', 'B', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'B', 'J'],
            ['J', 'J', 'J', 'J', 'J', 'B', 'J', 'J', 'J', 'J', 'J', 'J', 'B', 'J']
        ]

        result = make_documents(source, doc_size=15)
        result_documents, result_spaces, result_tokens, result_weights, result_labels = zip(*result)
        result_documents, result_spaces, result_tokens, result_weights, result_labels = \
            list(result_documents), list(result_spaces), list(result_tokens), list(result_weights), list(result_labels)

        self.assertEqual(expected_documents, result_documents)
        self.assertEqual(expected_spaces, result_spaces)
        self.assertEqual(expected_tokens, result_tokens)
        self.assertEqual(expected_weights, result_weights)
        self.assertEqual(expected_labels, result_labels)

    def test_spaces(self):
        source = [([([' ', '\n', ' '])], 1.0)]

        expected_documents = [[]]
        expected_spaces = [[]]
        expected_tokens = [[]]
        expected_weights = [[]]
        expected_labels = [[]]

        result = make_documents(source, doc_size=15)
        result_documents, result_spaces, result_tokens, result_weights, result_labels = zip(*result)
        result_documents, result_spaces, result_tokens, result_weights, result_labels = \
            list(result_documents), list(result_spaces), list(result_tokens), list(result_weights), list(result_labels)

        self.assertEqual(expected_documents, result_documents)
        self.assertEqual(expected_spaces, result_spaces)
        self.assertEqual(expected_tokens, result_tokens)
        self.assertEqual(expected_weights, result_weights)
        self.assertEqual(expected_labels, result_labels)

    def test_complex(self):
        source = [
            ([[('.', ' ')]], 1.0),
            ([[(u'\u00adTest1', ''), ('.', ' ')]], 1.1),
            ([[(u'\u200eTest2', ''), ('.', ' ')]], 1.2),
            ([[(u'\u200fTest3', ''), ('.', ' ')]], 1.3),
            ([[(u'\ufe0fTest4', ''), ('.', ' ')]], 1.4),
            ([[(u'\ufeffTest5', ''), ('.', ' ')]], 1.5),
            ([[(u'\u1f3fbTest6', ''), ('.', ' ')]], 1.6),
            ([[('.', ''), ('.', ''), ('.', ''), ('.', ''), ('.', ''), ('.', ' ')]], 1.7),
        ]

        for words, spses, toks, wghts, sents in make_documents(label_paragraphs(source), 1):
            self.assertEqual(len(words), len(spses))
            self.assertEqual(len(spses), len(toks))
            self.assertEqual(len(toks), len(wghts))
            self.assertEqual(len(wghts), len(sents))

            repaired = self.evaluate(split_words(u''.join(words), extended=True))
            repaired = [w.decode('utf-8') for w in repaired.tolist()]
            self.assertListEqual(words, repaired)
            self.assertEqual(len(repaired), len([s for s in sents if len(s)]))
            self.assertEqual(len(repaired), len([t for t in toks if len(t)]))

        for words, spses, toks, wghts, sents in make_documents(label_paragraphs(source), 9999):
            self.assertEqual(len(words), len(spses))
            self.assertEqual(len(spses), len(toks))
            self.assertEqual(len(toks), len(wghts))
            self.assertEqual(len(wghts), len(sents))

            repaired = self.evaluate(split_words(u''.join(words), extended=True))
            repaired = [w.decode('utf-8') for w in repaired.tolist()]
            self.assertEqual(len(repaired), len([s for s in sents if len(s)]))
            self.assertEqual(len(repaired), len([t for t in toks if len(t)]))


class TestWriteDataset(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normal(self):
        source = [
            (
                ['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '', 'Second', ' ', '"', 'sentence',
                 '"', ' ', 'in', u'\xa0', 'paragraph', '.'],
                ['T', 'S', 'T', 'S', 'T', 'S', 'T', 'T', '', 'T', 'S', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'T'],
                ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', '', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
                [1.0] * 19,
                ['J', 'J', 'J', 'J', 'J', 'J', 'J', 'B', '', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'B'],
            ),
            (
                [' ', 'test', '@', 'test', '.', 'com', ' ', 'Single', ' ', '-', ' ', 'sentence', '.'],
                ['S', 'T', 'T', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'S', 'T', 'T'],
                ['D', 'D', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
                [0.5] * 13,
                ['J', 'J', 'J', 'J', 'J', 'B', 'B', 'J', 'J', 'J', 'J', 'J', 'B'],
            )
        ]

        expected_document = ''.join(source[0][0])
        expected_spaces = ','.join(source[0][1]).replace(',,', ',')
        expected_tokens = ','.join(source[0][2]).replace(',,', ',')
        expected_sentences = ','.join(source[0][4]).replace(',,', ',')
        expected_ispaces = list(map(int, expected_spaces.replace('T', '0').replace('S', '1').split(',')))
        expected_itokens = list(map(int, expected_tokens.replace('D', '0').replace('C', '1').split(',')))
        expected_isentences = list(map(int, expected_sentences.replace('J', '0').replace('B', '1').split(',')))

        write_dataset(self.temp_dir, 'buffer', source)

        wildcard = os.path.join(self.temp_dir, '*.tfrecords.gz')
        files = tf.data.Dataset.list_files(wildcard, shuffle=False)
        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
        dataset = dataset.batch(1)

        params = build_hparams({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'word_mean': 1.,
            'word_std': 1.,
            'ngram_minn': 1,
            'ngram_maxn': 1,
            'ngram_freq': 6,
            'lstm_units': [1]
        })
        dataset = dataset.map(lambda protos: _parse_examples(protos, params))

        for features, labels, weights in dataset.take(1):
            actual_document = ''.join(w.numpy().decode('utf-8') for w in features['word_tokens'].flat_values)
            self.assertEqual(actual_document, expected_document)
            self.assertListEqual(labels['space'].numpy().reshape(-1).tolist(), expected_ispaces)
            self.assertListEqual(labels['token'].numpy().reshape(-1).tolist(), expected_itokens)
            self.assertListEqual(labels['sentence'].numpy().reshape(-1).tolist(), expected_isentences)
            self.assertListEqual(weights['token'].numpy().reshape(-1).tolist(), [1.0] * 18)
