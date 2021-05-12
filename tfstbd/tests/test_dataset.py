import numpy as np
import os
import shutil
import tensorflow as tf
import tempfile
from tfmiss.text.unicode_expand import split_words
from ..dataset import parse_paragraphs, good_paragraphs, random_glue, augment_paragraphs, label_spaces
from ..dataset import label_tokens, label_paragraphs, label_repdivwrap, make_documents, create_dataset


class TestParseParagraphs(tf.test.TestCase):
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
                 [('Ранее', ' '), ('часто', ' '), ('писали', ' '), ('"', ''), ('алгорифм', ''), ('"', ''),
                  (',', ' '), ('сейчас', ' '), ('такое', ' '), ('написание', ' '), ('используется', ' '),
                  ('редко', ''), (',', ' '), ('но', ''), (',', ' '), ('тем', ' '), ('не', ' '),
                  ('менее', ''), (',', ' '), ('имеет', ' '), ('место', ' '), ('(', ''), ('например', ''),
                  (',', ' '), ('Нормальный', ' '), ('алгорифм', ' '), ('Маркова', ''), (')', ''),
                  ('.', '')],
             ], 1.0),
            ([
                 [('Кто', ' '), ('знает', ''), (',', ' '), ('что', ' '), ('он', ' '), ('там', ' '),
                  ('думал', ''), ('!', ''), ('.', ''), ('.', '')],
             ], 1.0),
        ], result)

    def test_parse_udpipe(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_udpipe.conllu'), 1.0)
        self.assertEqual([
            ([
                 [('Порву', ''), ('!', '')],
             ], 1.0),
            ([
                 [('Порву', ''), ('!', ''), ('"', ''), ('...', ''), ('©', '')],
                 [('Порву', ''), ('!', '')]
             ], 1.0),
            ([
                 [('Ребят', ''), (',', ' '), ('я', ' '), ('никому', ' '), ('не', ' '), ('звонила', ''),
                  ('?', ''), ('?', ''), ('?', ' \t '), (')))', '')],
                 [('Вот', ' '), ('это', ' '), ('был', ' '), ('номер', ''), ('...', ''), (')', ''),
                  ('))', '')],
             ], 1.0),
        ], result)

    def test_parse_collapse(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_collapse.conllu'), 1.0)
        self.assertEqual([
            ([
                 [('Er', ' '), ('arbeitet', ' '), ('fürs', ' '), ('FBI', ' '), ('(', ''), ('deutsch', ' '),
                  ('etwa', ''), (':', ' '), ('„', ''), ('Bundesamt', ' '), ('für', ' '), ('Ermittlung', ''),
                  ('“', ''), (')', ''), ('.', '')],
             ], 1.0),
        ], result)

    def test_parse_space(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_space.conllu'), 1.0)
        self.assertEqual([
            ([
                 [('Таким образом', ' '), (',', ' '), ('если', ' '), ('приговор', '')]
             ], 1.0),
            ([
                 [('Она', ' '), ('заболела', ' '), ('потому, что', ' '), ('мало', ' '), ('ела', ' '),
                  ('.', '')]
             ], 1.0)
        ], result)

    def test_parse_duplicates(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_duplicates.conllu'), 1.0)
        self.assertEqual([([[('Ранее', '')]], 1.0)], result)

    def test_parse_weigted(self):
        result = parse_paragraphs(
            os.path.join(os.path.dirname(__file__), 'data', 'dataset_weighted.conllu'), 10.01)
        self.assertEqual([
            ([[('Ранее', '')]], 10.01),
            ([[('Позднее', '')]], 10.01)
        ], result)


class TestGoodParagraphs(tf.test.TestCase):
    def test_multi_sent(self):
        source = [
            ([[('«', ''), ('США', ''), ('»', '')]], 1.0),
            ([[('Но', ' '), ('в', ' '), ('отступлении', ''), ('.', ' ')]], 1.0)
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 2)

    def test_small_sent(self):
        source = [
            ([[('США', '')]], 1.0)
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 0)

    def test_norm_sent(self):
        source = [
            ([[('В', ' '), ('12', ' '), ('часов', ''), ('.', '')]], 1.0)
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 1)

    def test_just_letters(self):
        source = [
            ([[('В', ' '), ('1', ' '), ('2', ' '), ('.', '')]], 1.0)
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 0)

    def test_ambig_start(self):
        source = [
            ([[('-', ' '), ('Я', ' '), ('тоже', ''), ('.', '')]], 1.0)
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 1)

    def test_bad_start(self):
        source = [
            ([[('?', ' '), ('Я', ' '), ('тоже', ''), ('.', '')]], 1.0)
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 0)

    def test_bad_stop(self):
        source = [
            ([[('-', ' '), ('Я', ' '), ('тоже', ''), (',', '')]], 1.0)
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 0)


class TestRandomGlue(tf.test.TestCase):
    def setUp(self):
        np.random.seed(2)

    def test_default(self):
        self.assertEqual([[' ']], random_glue())

    def test_shape(self):
        result = random_glue(1, 1, 1, 0, 1)
        self.assertEqual([[' ']], result)

    def test_normal(self):
        result = random_glue(space=10, tab=1, newline=1, reserve=1)
        self.assertEqual([[' '], [' '], [' '], [' '], [' '], [' '], [' '], [' '], [' '], ['\t']], result)

    def test_empty(self):
        result = random_glue(space=1, tab=1, empty=10, reserve=1)
        self.assertEqual([[' '], [''], [''], [''], [''], [''], [''], [''], ['']], result)

    def test_extra(self):
        result = random_glue(space=30, tab=5, newline=25, reserve=1)
        self.assertEqual({' ', '\t', '\n'}, set([''.join(s) for s in result]))


class TestAugmentParagraphs(tf.test.TestCase):
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
            ([[('First', ' '), ('sentence', ' '), ('in', ' '), ('paragraph', ''), ('.', ' ')],
              [('Second', ' '), ('"', ''), ('sentence', ''), ('"', ' '), ('in', ' '), ('paragraph', ''), ('.', '  ')]],
             1.0),
            ([[('Single', ' '), ('-', ' '), ('sentence', ''), ('.', ' ')]], 1.0),
        ]
        result = augment_paragraphs(source)
        self.assertEqual(expected, result)

    def test_spaces(self):
        np.random.seed(5)
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

    def test_empty_end(self):
        np.random.seed(20)
        source = [
            ([[('A', ' '), ('.', '')],
              [('B', ' '), ('!', ' ')],
              [('C', ' '), ('?', '')],
              [('D', ' '), ('.', ' ')],
              [('E', ' '), ('!', '')],
              [('F', ' '), ('?', ' ')]], 1.1)
        ]
        expected = [
            ([[('A', ' '), ('.', ' ')],
              [('B', ' '), ('!', ' ')],
              [('C', ' '), ('?', '\xa0')],
              [('D', ' '), ('.', ' ')],
              [('E', ' '), ('!', ' ')],
              [('F', ' '), ('?', ' ')]], 1.1)
        ]
        result = augment_paragraphs(source)
        self.assertEqual(expected, result)


class TestLabelSpaces(tf.test.TestCase):
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


class TestLabelTokens(tf.test.TestCase):
    def test_empty(self):
        result = label_tokens([], [])
        self.assertEqual([], result)

    def test_same(self):
        result = label_tokens([
            'word', '+', 'word', '-', 'word'
        ], [
            ('word', ''), ('+', ''), ('word', ''), ('-', ''), ('word', '')
        ])
        self.assertEqual(['B', 'B', 'B', 'B', 'B'], result)

    def test_normal_join(self):
        result = label_tokens([
            'word', '+', 'word', '-', 'word', '_', 'word'
        ], [
            ('word', ''), ('+', ''), ('word-word', ''), ('_word', '')
        ])
        self.assertEqual(['B', 'B', 'B', 'I', 'I', 'B', 'I'], result)

    def test_joined_source(self):
        result = label_tokens([
            'word+word-word'
        ], [
            ('word', ''), ('+', ''), ('word', ''), ('-', ''), ('word', '')
        ])
        self.assertEqual(['B'], result)

    def test_joined_target(self):
        result = label_tokens([
            'word', '+', 'word', '-', 'word'
        ], [
            ('word+word-word', '')
        ])
        self.assertEqual(['B', 'I', 'I', 'I', 'I'], result)

    def test_different_join(self):
        result = label_tokens([
            'word+word', '-', 'word'
        ], [
            ('word', ''), ('+', ''), ('word-word', '')
        ])
        self.assertEqual(['B', 'I', 'I'], result)

    def test_compex(self):
        source = [
            [' ', 'test', '@', 'test', '.', 'com', ' '],
            [' ', 'www', '.', 'test', '.', 'com', ' '],
            [' ', 'word', '.', '.', 'word', ' '],
            [' ', 'word', '+', 'word', '-', 'word', ' '],
            [' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '],
            [' ', 'test', '@', 'test', '.', 'com', ' '],
            [' ', 'www', '.', 'test', '.', 'com', ' '],
            [' ', 'word', '.', '.', 'word', ' '],
            [' ', 'word', '+', 'word', '-', 'word', ' '],
            [' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' ']
        ]
        target = [
            [('', ' '), ('test@test.com', ' ')],
            [('', ' '), ('www.test.com', ' ')],
            [('', ' '), ('word..word', ' ')],
            [('', ' '), ('word+word-word', ' ')],
            [('', ' '), ('word\\word/word#word', ' ')],
            [('', ' '), ('test', ''), ('@', ''), ('test', ''), ('.', ''), ('com', ' ')],
            [('', ' '), ('www', ''), ('.', ''), ('test', ''), ('.', ''), ('com', ' ')],
            [('', ' '), ('word', ''), ('..', ''), ('word', ' ')],
            [('', ' '), ('word', ''), ('+', ''), ('word', ''), ('-', ''), ('word', ' ')],
            [('', ' '), ('word', ''), ('\\', ''), ('word', ''), ('/', ''), ('word', ''), ('#', ''), ('word', ' ')]
        ]
        expected = [
            ['B', 'B', 'I', 'I', 'I', 'I', 'B'],
            ['B', 'B', 'I', 'I', 'I', 'I', 'B'],
            ['B', 'B', 'I', 'I', 'I', 'B'],
            ['B', 'B', 'I', 'I', 'I', 'I', 'B'],
            ['B', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'B'],
            ['B', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', 'B', 'B', 'I', 'B', 'B'],
            ['B', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
        ]

        result = [label_tokens(s, t) for s, t in zip(source, target)]
        self.assertEqual(expected, result)


class TestLabelRepDivWrap(tf.test.TestCase):
    def test_repeaters(self):
        words = ['I', '-', '-', '-', 'I', ' ', 'H', ' ', ')', ')']
        spaces = ['T', 'T', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'T']
        tokens = ['J', 'J', 'J', 'J', 'B', 'B', 'B', 'B', 'J', 'B']
        expected = ['N', 'R', 'R', 'N', 'N', 'N', 'N', 'N', 'N', 'N']

        result = label_repdivwrap(words, spaces, tokens)
        self.assertEqual(expected, result)

    def test_dividers(self):
        words = ['I', '’', 'm', ' ', 'a', ' ', 'd', '-', 'i', '.', ' ', '-', 'I', ' ', 'a', ' ', 'n', '-']
        spaces = ['T', 'T', 'T', 'S', 'T', 'S', 'T', 'T', 'T', 'T', 'S', 'T', 'T', 'S', 'T', 'S', 'T', 'T']
        tokens = ['J', 'J', 'B', 'B', 'B', 'B', 'J', 'J', 'B', 'B', 'B', 'J', 'B', 'B', 'B', 'B', 'J', 'B']
        expected = ['D', 'D', 'N', 'N', 'N', 'N', 'D', 'D', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']

        result = label_repdivwrap(words, spaces, tokens)
        self.assertEqual(expected, result)

    def test_wrappers(self):
        words = ['I', ' ', '[', '?', ']', ')', ')', ' ', '(', 'c', ')']
        spaces = ['T', 'S', 'T', 'T', 'T', 'T', 'T', 'S', 'T', 'T', 'T']
        tokens = ['B', 'B', 'J', 'J', 'B', 'J', 'B', 'B', 'J', 'J', 'B']
        expected = ['N', 'N', 'W', 'W', 'N', 'N', 'N', 'N', 'W', 'W', 'N']

        result = label_repdivwrap(words, spaces, tokens)
        self.assertEqual(expected, result)


class TestLabelParagraphs(tf.test.TestCase):
    def test_empty(self):
        source = []
        result = label_paragraphs(source)
        self.assertEqual([], result)

    def test_normal(self):
        source = [
            ([[('Single', ' '), ('-', ' '), ('sentence.', '\t')]], 1.0),
            ([[('First', ' '), ('sentence', '   '), ('in', ' '), ('paragraph', ''), ('.', '\n')],
              [('Second', ' '), ('"', ''), ('sentence', ''), ('"', ' '), ('in', '\u00A0'), ('paragraph', ''),
               ('.', ' ')]], 1.1),
        ]
        expected = [
            ([
                 (['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n'],
                  ['T', 'S', 'T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
                  ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']),
                 (['Second', ' ', '"', 'sentence', '"', ' ', 'in', '\u00A0', 'paragraph', '.', ' '],
                  ['T', 'S', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
                  ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'])
             ], 1.1),
            ([
                 (['Single', ' ', '-', ' ', 'sentence', '.', '\t'],
                  ['T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['B', 'B', 'B', 'B', 'B', 'I', 'B'],
                  ['N', 'N', 'N', 'N', 'N', 'N', 'N'])
             ], 1.0),
        ]
        result = label_paragraphs(source)
        self.assertEqual(expected, result)

    def test_compex(self):
        source = [
            ([[('', ' '), ('test', ''), ('@', ''), ('test', ''), ('.', ''), ('com', ' ')],
              [('', ' '), ('www', ''), ('.', ''), ('test', ''), ('.', ''), ('com', ' ')],
              [('', ' '), ('word', ''), ('..', ''), ('word', ' ')],
              [('', ' '), ('word', ''), ('+', ''), ('word', ''), ('-', ''), ('word', ' ')],
              [('', ' '), ('word', ''), ('\\', ''), ('word', ''), ('/', ''), ('word', ''), ('#', ''), ('word', ' ')]],
             1.1),
            ([[('', ' '), ('test@test.com', ' ')],
              [('', ' '), ('www.test.com', ' ')],
              [('', ' '), ('word..word', ' ')],
              [('', ' '), ('word+word-word', ' ')],
              [('', ' '), ('word\\word/word#word', ' ')]], 1.0)
        ]
        expected = [
            ([
                 ([' ', 'test', '@', 'test', '.', 'com', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'B', 'B', 'B', 'B', 'B'],
                  ['N', 'D', 'D', 'D', 'D', 'N', 'N']),

                 ([' ', 'www', '.', 'test', '.', 'com', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'B', 'B', 'B', 'B', 'B'],
                  ['N', 'D', 'D', 'D', 'D', 'N', 'N']),

                 ([' ', 'word', '.', '.', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'B', 'I', 'B', 'B'],
                  ['N', 'N', 'N', 'N', 'N', 'N']),

                 ([' ', 'word', '+', 'word', '-', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'B', 'B', 'B', 'B', 'B'],
                  ['N', 'D', 'W', 'D', 'D', 'N', 'N']),

                 ([' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
                  ['N', 'D', 'D', 'D', 'D', 'D', 'D', 'N', 'N']),
             ], 1.1),
            ([
                 ([' ', 'test', '@', 'test', '.', 'com', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'I', 'I', 'I', 'I', 'B'],
                  ['N', 'N', 'N', 'D', 'D', 'N', 'N']),

                 ([' ', 'www', '.', 'test', '.', 'com', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'I', 'I', 'I', 'I', 'B'],
                  ['N', 'N', 'N', 'D', 'D', 'N', 'N']),

                 ([' ', 'word', '.', '.', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'I', 'I', 'I', 'B'],
                  ['N', 'N', 'N', 'N', 'N', 'N']),

                 ([' ', 'word', '+', 'word', '-', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'I', 'I', 'I', 'I', 'B'],
                  ['N', 'N', 'W', 'D', 'D', 'N', 'N']),

                 ([' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'B'],
                  ['N', 'N', 'N', 'D', 'D', 'D', 'D', 'N', 'N']),
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
                  ['B', 'B', 'B', 'B', 'B', 'B', 'B'],
                  ['N', 'N', 'N', 'N', 'N', 'N', 'N'])
             ], 1.0),
            ([
                 ([' ', 'test', '@', 'test', '.', 'com', ' '],
                  ['S', 'T', 'T', 'T', 'T', 'T', 'S'],
                  ['B', 'B', 'I', 'I', 'I', 'I', 'B'],
                  ['N', 'N', 'N', 'D', 'D', 'N', 'N']),
             ], 1.1),
            ([
                 (['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n'],
                  ['T', 'S', 'T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
                  ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']),
                 (['Second', ' ', '"', 'sentence', '"', ' ', 'in', '\u00A0', 'paragraph', '.', ' '],
                  ['T', 'S', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'T', 'S'],
                  ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
                  ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'])
             ], 1.2),
        ]

        expected_documents = [
            ['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n', 'Second', ' ', '"', 'sentence', '"',
             ' ', 'in', '\xa0', 'paragraph', '.', ' '],
            [' ', 'test', '@', 'test', '.', 'com', ' ', 'Single', ' ', '-', ' ', 'sentence', '.', '\t']
        ]
        expected_spaces = [
            ['T', 'S', 'T', 'S', 'T', 'S', 'T', 'T', 'S', 'T', 'S', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'T', 'S'],
            ['S', 'T', 'T', 'T', 'T', 'T', 'S', 'T', 'S', 'T', 'S', 'T', 'T', 'S']
        ]
        expected_tokens = [
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', 'B', 'I', 'I', 'I', 'I', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
        ]
        expected_weights = [
            [1.2] * 20,
            [1.1] * 7 + [1.0] * 7
        ]
        expected_rediwrs = [
            ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'],
            ['N', 'N', 'N', 'D', 'D', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']
        ]
        expected_sentences = [
            ['B', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I'],
            ['B', 'I', 'I', 'I', 'I', 'I', 'I', 'B', 'I', 'I', 'I', 'I', 'I', 'I']
        ]

        result = make_documents(source, doc_size=15)
        result = [list(r) for r in zip(*result)]
        result_documents, result_spaces, result_tokens, result_weights, result_rediwrs, result_sentences = result

        self.assertEqual(expected_documents, result_documents)
        self.assertEqual(expected_spaces, result_spaces)
        self.assertEqual(expected_tokens, result_tokens)
        self.assertEqual(expected_weights, result_weights)
        self.assertEqual(expected_rediwrs, result_rediwrs)
        self.assertEqual(expected_sentences, result_sentences)

    def test_spaces(self):
        source = [([([' ', '\n', ' '], ['S', 'S', 'S'], ['B', 'I', 'I'], ['N', 'N', 'N'])], 1.0)]

        expected = [[]]

        result = make_documents(source, doc_size=15)
        result = [list(r) for r in zip(*result)]
        result_documents, result_spaces, result_tokens, result_weights, result_rediwrs, result_sentences = result

        self.assertEqual(expected, result_documents)
        self.assertEqual(expected, result_spaces)
        self.assertEqual(expected, result_tokens)
        self.assertEqual(expected, result_weights)
        self.assertEqual(expected, result_rediwrs)
        self.assertEqual(expected, result_sentences)

    def test_complex(self):
        source = [
            ([[('.', ' ')]], 1.0),
            ([[('\u00adTest1', ''), ('.', ' ')]], 1.1),
            ([[('\u200eTest2', ''), ('.', ' ')]], 1.2),
            ([[('\u200fTest3', ''), ('.', ' ')]], 1.3),
            ([[('\ufe0fTest4', ''), ('.', ' ')]], 1.4),
            ([[('\ufeffTest5', ''), ('.', ' ')]], 1.5),
            ([[('\u1f3fbTest6', ''), ('.', ' ')]], 1.6),
            ([[('.', ''), ('.', ''), ('.', ''), ('.', ''), ('.', ''), ('.', ' ')]], 1.7),
        ]

        for words, spses, toks, wghts, rdws, sents in make_documents(label_paragraphs(source), 1):
            self.assertEqual(len(words), len(spses))
            self.assertEqual(len(spses), len(toks))
            self.assertEqual(len(toks), len(wghts))
            self.assertEqual(len(wghts), len(rdws))
            self.assertEqual(len(rdws), len(sents))

            repaired = self.evaluate(split_words(''.join(words), extended=True))
            repaired = [w.decode('utf-8') for w in repaired.tolist()]
            self.assertListEqual(words, repaired)
            self.assertEqual(len(repaired), len([s for s in sents if len(s)]))
            self.assertEqual(len(repaired), len([t for t in toks if len(t)]))

        for words, spses, toks, wghts, rdws, sents in make_documents(label_paragraphs(source), 9999):
            self.assertEqual(len(words), len(spses))
            self.assertEqual(len(spses), len(toks))
            self.assertEqual(len(toks), len(wghts))
            self.assertEqual(len(wghts), len(rdws))
            self.assertEqual(len(rdws), len(sents))

            repaired = self.evaluate(split_words(''.join(words), extended=True))
            repaired = [w.decode('utf-8') for w in repaired.tolist()]
            self.assertEqual(len(repaired), len([s for s in sents if len(s)]))
            self.assertEqual(len(repaired), len([t for t in toks if len(t)]))


class TestCreateDataset(tf.test.TestCase):
    def setUp(self):
        super(TestCreateDataset, self).setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super(TestCreateDataset, self).tearDown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normal(self):
        src_dir = os.path.join(self.temp_dir, 'src')
        os.makedirs(src_dir, exist_ok=True)

        dst_dir = os.path.join(self.temp_dir, 'dst')
        os.makedirs(dst_dir, exist_ok=True)

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        shutil.copyfile(os.path.join(data_dir, 'dataset_plain.conllu'), os.path.join(src_dir, 'ru_pl-train.conllu'))
        shutil.copyfile(os.path.join(data_dir, 'dataset_paragraphs.conllu'), os.path.join(src_dir, 'ru_pr-test.conllu'))

        builder = create_dataset(src_dir, dst_dir, 1, 0.1, 1, '-test.')
        dataset = builder.as_dataset(split='train')

        has_examples = False
        for example in dataset:
            has_examples = True
            self.assertSetEqual({
                'document', 'length', 'space', 'token', 'repdivwrap', 'weight', 'sentence'}, set(example.keys()))
            self.assertLen(example['document'].shape, 0)
            self.assertLen(example['length'].shape, 0)
            self.assertLen(example['space'].shape, 0)
            self.assertLen(example['token'].shape, 0)
            self.assertLen(example['repdivwrap'].shape, 0)
            self.assertLen(example['weight'].shape, 1)
            self.assertLen(example['sentence'].shape, 0)

            length = example['length'].numpy()
            self.assertLen(example['space'].numpy(), length)
            self.assertLen(example['token'].numpy(), length)
            self.assertEqual(example['weight'].shape[0], length)
            self.assertLen(example['repdivwrap'].numpy(), length)
            self.assertLen(example['sentence'].numpy(), length)

        self.assertTrue(has_examples)
