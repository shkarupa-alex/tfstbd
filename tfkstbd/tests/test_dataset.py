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
from ..dataset import parse_paragraphs, random_glue, augment_paragraphs, label_tokens, label_paragraphs
from ..dataset import make_documents, write_dataset
from ..input import train_input


class TestParseConllu(unittest.TestCase):
    def testParseParagraphs(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_paragraphs.conllu'))
        self.assertEqual([
            [
                [u'«', u'Если', u' ', u'передача', u' ', u'цифровых', u' ', u'технологий', u' ', u'сегодня', u' ', u'в',
                 u' ', u'США', u' ', u'происходит', u' ', u'впервые', u',', u' ', u'то', u' ', u'о', u' ', u'мирной',
                 u' ', u'передаче', u' ', u'власти', u' ', u'такого', u' ', u'не', u' ', u'скажешь', u'»', u',', u' ',
                 u'–', u' ', u'написала', u' ', u'Кори', u' ', u'Шульман', u',', u' ', u'специальный', u' ',
                 u'помощник', u' ', u'президента', u' ', u'Обамы', u' ', u'в', u' ', u'своем', u' ', u'блоге', u' ',
                 u'в', u' ', u'понедельник', u'.'],

                [u'Для', u' ', u'тех', u',', u' ', u'кто', u' ', u'следит', u' ', u'за', u' ', u'передачей', u' ',
                 u'всех', u' ', u'материалов', u',', u' ', u'появившихся', u' ', u'в', u' ', u'социальных', u' ',
                 u'сетях', u' ', u'о', u' ', u'Конгрессе', u',', u' ', u'это', u' ', u'будет', u' ', u'происходить',
                 u' ', u'несколько', u' ', u'по-другому', u'.'],
            ],
            [
                [u'Но', u' ', u'в', u' ', u'отступлении', u' ', u'от', u' ', u'риторики', u' ', u'прошлого', u' ', u'о',
                 u' ', u'сокращении', u' ', u'иммиграции', u' ', u'кандидат', u' ', u'Республиканской', u' ', u'партии',
                 u' ', u'заявил', u',', u' ', u'что', u' ', u'в', u' ', u'качестве', u' ', u'президента', u' ', u'он',
                 u' ', u'позволил', u' ', u'бы', u' ', u'въезд', u' ', u'«', u'огромного', u' ', u'количества', u'»',
                 u' ', u'легальных', u' ', u'мигрантов', u' ', u'на', u' ', u'основе', u' ', u'«', u'системы', u' ',
                 u'заслуг', u'»', u'.'],

            ],
        ], result)

    def testParsePlain(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', u'dataset_plain.conllu'))
        self.assertEqual([
            [
                [u'Ранее', u' ', u'часто', u' ', u'писали', u' ', u'"', u'алгорифм', u'"', u',', u' ', u'сейчас', u' ',
                 u'такое', u' ', u'написание', u' ', u'используется', u' ', u'редко', u',', u' ', u'но', u',', u' ',
                 u'тем', u' ', u'не', u' ', u'менее', u',', u' ', u'имеет', u' ', u'место', u' ', u'(', u'например',
                 u',', u' ', u'Нормальный', u' ', u'алгорифм', u' ', u'Маркова', u')', u'.'],
            ],
            [
                [u'Кто', u' ', u'знает', u',', u' ', u'что', u' ', u'он', u' ', u'там', u' ', u'думал', u'!', u'.',
                 u'.'],
            ],
        ], result)

    def testParseUdpipe(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', u'dataset_udpipe.conllu'))
        self.assertEqual([
            [
                [u'Порву', u'!'],
            ],
            [
                [u'Порву', u'!', u'"', u'...', u'©'],
                [u'Порву', u'!']
            ],
            [
                [u'Ребят', u',', u' ', u'я', u' ', u'никому', u' ', u'не', u' ', u'звонила', u'?', u'?', u'?', u' \t ',
                 u')))'],
                [u'Вот', u' ', u'это', u' ', u'был', u' ', u'номер', u'...', u')', u'))'],
            ],
        ], result)

    def testParseCollapse(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', u'dataset_collapse.conllu'))
        self.assertEqual([
            [
                [u'Er', u' ', u'arbeitet', u' ', u'fürs', u' ', u'FBI', u' ', u'(', u'deutsch', u' ', u'etwa', u':',
                 u' ', u'„', u'Bundesamt', u' ', u'für', u' ', u'Ermittlung', u'“', u')', u'.'],
            ],
        ], result)

    def testParseSpace(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', u'dataset_space.conllu'))
        self.assertEqual([
            [
                [u'Таким образом', u' ', u',', u' ', u'если', u' ', u'приговор']
            ],
            [
                [u'Она', u' ', u'заболела', u' ', u'потому, что', u' ', u'мало', u' ', u'ела', u' ',
                 u'.']
            ]
        ], result)


class TestRandomGlue(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)

    def testEmpty(self):
        self.assertEqual([[' ']], random_glue())

    def testShape(self):
        result = random_glue(1, 1, 1, 1)
        self.assertEqual([[' ']], result)

    def testNormal(self):
        result = random_glue(space=10, tab=1, newline=1, reserve=1)
        self.assertEqual([[' '], [' '], [' '], [' '], [' '], [' '], [' '], [' '], [' '], ['\t']], result)

    def testExtra(self):
        result = random_glue(space=30, tab=5, newline=25, reserve=1)
        self.assertEqual({' \n', ' ', '\t', '\n'}, set([''.join(s) for s in result]))


class TestAugmentParagraphs(unittest.TestCase):
    def testEmpty(self):
        result = augment_paragraphs([])
        self.assertEqual([], result)

    def testNormal(self):
        np.random.seed(5)
        source = [
            [['First', ' ', 'sentence', ' ', 'in', ' ', 'paragraph', '.'],
             ['Second', ' ', '"', 'sentence', '"', ' ', 'in', ' ', 'paragraph', '.']],
            [['Single', ' ', '-', ' ', 'sentence', '.']],
        ]
        expected = [
            [['First', u'\xa0', 'sentence', ' ', 'in', ' ', 'paragraph', '.', ' '],
             ['Second', ' ', '"', 'sentence', '"', ' ', 'in', u'\xa0', 'paragraph', '.', ' ']],
            [['Single', ' ', '-', ' ', 'sentence', '.', ' ']]
        ]
        result = augment_paragraphs(source)
        self.assertEqual(expected, result)

    def testSpaces(self):
        np.random.seed(2)
        source = [
            [['Single', ' ', 'sentence'],
             ['Next', ' ', 'single', ' ', 'sentence']]
        ]
        expected = [
            [['Single', ' ', 'sentence', ' '],
             ['Next', ' ', 'single', ' ', 'sentence', '\n']]
        ]
        result = augment_paragraphs(source)
        self.assertEqual(expected, result)


class TestLabelTokens(unittest.TestCase):
    def testEmpty(self):
        with self.assertRaises(AssertionError):
            label_tokens([], [])

    def testSame(self):
        result = label_tokens([
            'word', '+', 'word', '-', 'word'
        ], [
            'word', '+', 'word', '-', 'word'
        ])
        self.assertEqual(['B', 'B', 'B', 'B', 'B'], result)

    def testNormalJoin(self):
        result = label_tokens([
            'word', '+', 'word', '-', 'word', '_', 'word'
        ], [
            'word', '+', 'word-word', '_word'
        ])
        self.assertEqual(['B', 'B', 'J', 'J', 'B', 'J', 'B'], result)

    def testJoinedSource(self):
        result = label_tokens([
            'word+word-word'
        ], [
            'word', '+', 'word', '-', 'word'
        ])
        self.assertEqual(['B'], result)

    def testJoinedTarget(self):
        result = label_tokens([
            'word', '+', 'word', '-', 'word'
        ], [
            'word+word-word'
        ])
        self.assertEqual(['J', 'J', 'J', 'J', 'B'], result)

    def testDifferentJoin(self):
        result = label_tokens([
            'word+word', '-', 'word'
        ], [
            'word', '+', 'word-word'
        ])
        self.assertEqual(['J', 'J', 'B'], result)


class TestLabelParagraphs(tf.test.TestCase):
    def testEmpty(self):
        source = []
        result = label_paragraphs(source)
        self.assertEqual([], result)

    def testNormal(self):
        source = [
            [['Single', ' ', '-', ' ', 'sentence.', '\t']],
            [['First', ' ', 'sentence', ' ', ' ', ' ', 'in', ' ', 'paragraph', '.', '\n'],
             ['Second', ' ', '"', 'sentence', '"', ' ', 'in', u'\u00A0', 'paragraph', '.', ' ']],
        ]
        expected = [
            [
                (['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n'],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']),
                (['Second', ' ', '"', 'sentence', '"', ' ', 'in', u'\u00A0', 'paragraph', '.', ' '],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'])
            ],
            [
                (['Single', ' ', '-', ' ', 'sentence', '.', '\t'],
                 ['B', 'B', 'B', 'B', 'J', 'B', 'B'])
            ],
        ]
        result = label_paragraphs(source)
        self.assertEqual(expected, result)

    def testCompex(self):
        source = [
            [[' ', 'test@test.com', ' '],
             [' ', 'www.test.com', ' '],
             [' ', 'word..word', ' '],
             [' ', 'word+word-word', ' '],
             [' ', 'word\\word/word#word', ' ']],
            [[' ', 'test', '@', 'test', '.', 'com', ' '],
             [' ', 'www', '.', 'test', '.', 'com', ' '],
             [' ', 'word', '..', 'word', ' '],
             [' ', 'word', '+', 'word', '-', 'word', ' '],
             [' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' ']],
        ]
        expected = [
            [
                ([' ', 'test', '@', 'test', '.', 'com', ' '],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B']),

                ([' ', 'www', '.', 'test', '.', 'com', ' '],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B']),

                ([' ', 'word', '.', '.', 'word', ' '],
                 ['B', 'B', 'J', 'B', 'B', 'B']),

                ([' ', 'word', '+', 'word', '-', 'word', ' '],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B']),

                ([' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']),
            ],
            [
                ([' ', 'test', '@', 'test', '.', 'com', ' '],
                 ['B', 'J', 'J', 'J', 'J', 'B', 'B']),

                ([' ', 'www', '.', 'test', '.', 'com', ' '],
                 ['B', 'J', 'J', 'J', 'J', 'B', 'B']),

                ([' ', 'word', '.', '.', 'word', ' '],
                 ['B', 'J', 'J', 'J', 'B', 'B']),

                ([' ', 'word', '+', 'word', '-', 'word', ' '],
                 ['B', 'J', 'J', 'J', 'J', 'B', 'B']),

                ([' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '],
                 ['B', 'J', 'J', 'J', 'J', 'J', 'J', 'B', 'B']),
            ],
        ]

        result = label_paragraphs(source)
        self.assertEqual(expected, result)


class TestMakeDocuments(tf.test.TestCase):
    def testEmpty(self):
        source = []
        result = make_documents(source, 2)
        self.assertEqual([], result)

    def testNormal(self):
        source = [
            [
                (['Single', ' ', '-', ' ', 'sentence', '.', '\t'],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B'])
            ],
            [
                ([' ', 'test', '@', 'test', '.', 'com', ' '],
                 ['B', 'B', 'J', 'J', 'J', 'J', 'B']),
            ],
            [
                (['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n'],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']),
                (['Second', ' ', '"', 'sentence', '"', ' ', 'in', u'\u00A0', 'paragraph', '.', ' '],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'])
            ],
        ]

        expected_documents = [
            ['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n', 'Second', ' ', '"', 'sentence', '"',
             ' ', 'in', u'\xa0', 'paragraph', '.'],
            [' ', 'test', '@', 'test', '.', 'com', ' ', 'Single', ' ', '-', ' ', 'sentence', '.']
        ]
        expected_tokens = [
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', 'B', 'J', 'J', 'J', 'J', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
        ]
        expected_labels = [
            ['J', 'J', 'J', 'J', 'J', 'J', 'J', 'B', 'B', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'B'],
            ['J', 'J', 'J', 'J', 'J', 'B', 'B', 'J', 'J', 'J', 'J', 'J', 'B']
        ]

        result = make_documents(source, doc_size=15)
        result_documents, result_tokens, result_labels = zip(*result)
        result_documents, result_tokens, result_labels = \
            list(result_documents), list(result_tokens), list(result_labels)

        self.assertEqual(expected_documents, result_documents)
        self.assertEqual(expected_tokens, result_tokens)
        self.assertEqual(expected_labels, result_labels)

    def testComplex(self):
        source = [
            [['.', ' ']],
            [[u'\u00adTest1', '.', ' ']],
            [[u'\u200eTest2', '.', ' ']],
            [[u'\u200fTest3', '.', ' ']],
            [[u'\ufe0fTest4', '.', ' ']],
            [[u'\ufeffTest5', '.', ' ']],
            [[u'\u1f3fbTest6', '.', ' ']],
            [['.', '.', '.', '.', '.', '.', ' ']],
        ]

        for words, sents, toks in make_documents(label_paragraphs(source), 1):
            self.assertEqual(len(words), len(sents))
            self.assertEqual(len(words), len(toks))

            repaired = self.evaluate(split_words(u''.join(words), stop=True))
            repaired = [w.decode('utf-8') for w in repaired.tolist()]
            self.assertListEqual(words, repaired)
            self.assertEqual(len(repaired), len([s for s in sents if len(s)]))
            self.assertEqual(len(repaired), len([t for t in toks if len(t)]))

        for words, sents, toks in make_documents(label_paragraphs(source), 9999):
            self.assertEqual(len(words), len(sents))
            self.assertEqual(len(words), len(toks))

            repaired = self.evaluate(split_words(u''.join(words), stop=True))
            repaired = [w.decode('utf-8') for w in repaired.tolist()]
            self.assertEqual(len(repaired), len([s for s in sents if len(s)]))
            self.assertEqual(len(repaired), len([t for t in toks if len(t)]))


class TestWriteDataset(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testNormal(self):
        source = [
            (
                ['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n', 'Second', ' ', '"', 'sentence',
                 '"', ' ', 'in', u'\xa0', 'paragraph', '.'],
                ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
                ['J', 'J', 'J', 'J', 'J', 'J', 'J', 'B', 'B', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'B'],
            ),
            (
                [' ', 'test', '@', 'test', '.', 'com', ' ', 'Single', ' ', '-', ' ', 'sentence', '.'],
                ['B', 'B', 'J', 'J', 'J', 'J', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
                ['J', 'J', 'J', 'J', 'J', 'B', 'B', 'J', 'J', 'J', 'J', 'J', 'B'],
            )
        ]

        write_dataset(self.temp_dir, 'buffer', source)

        wildcard = os.path.join(self.temp_dir, '*.tfrecords.gz')
        dataset = train_input(wildcard, [], [1], 1.0, 1.0, 1, 1)

        for i, (features, labels) in enumerate(dataset.take(2)):
            document = features['document']
            length = features['length']
            tokens = labels['tokens']
            sentences = labels['sentences']

            self.assertEqual(''.join(source[i][0]).encode('utf-8'), document[0].numpy())
            self.assertEqual(len(source[i][0]), length.numpy())
            self.assertListEqual([t.encode('utf-8') for t in source[i][1]], tokens[0].numpy().tolist())
            self.assertListEqual([s.encode('utf-8') for s in source[i][2]], sentences[0].numpy().tolist())
