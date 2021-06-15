import numpy as np
import os
import shutil
import tensorflow as tf
import tempfile
from tfmiss.text.unicode_expand import split_words
from ..dataset import parse_paragraphs, good_paragraphs, random_glue, augment_paragraphs
from ..dataset import label_tokens, label_paragraphs, make_documents, create_dataset


class TestParseParagraphs(tf.test.TestCase):
    def test_parse_paragraphs(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_paragraphs.conllu'))
        self.assertEqual([
            [
                [('«', ''), ('Если', ' '), ('передача', ' '), ('цифровых', ' '), ('технологий', ' '), ('сегодня', ' '),
                 ('в', ' '), ('США', ' '), ('происходит', ' '), ('впервые', ''), (',', ' '), ('то', ' '), ('о', ' '),
                 ('мирной', ' '), ('передаче', ' '), ('власти', ' '), ('такого', ' '), ('не', ' '), ('скажешь', ''),
                 ('»', ''), (',', ' '), ('–', ' '), ('написала', ' '), ('Кори', ' '), ('Шульман', ''), (',', ' '),
                 ('специальный', ' '), ('помощник', ' '), ('президента', ' '), ('Обамы', ' '), ('в', ' '),
                 ('своем', ' '), ('блоге', ' '), ('в', ' '), ('понедельник', ''), ('.', ' ')],
                [('Для', ' '), ('тех', ''), (',', ' '), ('кто', ' '), ('следит', ' '), ('за', ' '), ('передачей', ' '),
                 ('всех', ' '), ('материалов', ''), (',', ' '), ('появившихся', ' '), ('в', ' '), ('социальных', ' '),
                 ('сетях', ' '), ('о', ' '), ('Конгрессе', ''), (',', ' '), ('это', ' '), ('будет', ' '),
                 ('происходить', ' '), ('несколько', ' '), ('по-другому', ''), ('.', ' ')]
            ],
            [
                [('Но', ' '), ('в', ' '), ('отступлении', ' '), ('от', ' '), ('риторики', ' '), ('прошлого', ' '),
                 ('о', ' '), ('сокращении', ' '), ('иммиграции', ' '), ('кандидат', ' '), ('Республиканской', ' '),
                 ('партии', ' '), ('заявил', ''), (',', ' '), ('что', ' '), ('в', ' '), ('качестве', ' '),
                 ('президента', ' '), ('он', ' '), ('позволил', ' '), ('бы', ' '), ('въезд', ' '), ('«', ''),
                 ('огромного', ' '), ('количества', ''), ('»', ' '), ('легальных', ' '), ('мигрантов', ' '),
                 ('на', ' '), ('основе', ' '), ('«', ''), ('системы', ' '), ('заслуг', ''), ('»', ''), ('.', ' ')]
            ]
        ], result)

    def test_parse_plain(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_plain.conllu'))
        self.assertEqual([
            [
                [('Ранее', ' '), ('часто', ' '), ('писали', ' '), ('"', ''), ('алгорифм', ''), ('"', ''),
                 (',', ' '), ('сейчас', ' '), ('такое', ' '), ('написание', ' '), ('используется', ' '),
                 ('редко', ''), (',', ' '), ('но', ''), (',', ' '), ('тем', ' '), ('не', ' '),
                 ('менее', ''), (',', ' '), ('имеет', ' '), ('место', ' '), ('(', ''), ('например', ''),
                 (',', ' '), ('Нормальный', ' '), ('алгорифм', ' '), ('Маркова', ''), (')', ''),
                 ('.', ' ')],
            ],
            [
                [('Кто', ' '), ('знает', ''), (',', ' '), ('что', ' '), ('он', ' '), ('там', ' '),
                 ('думал', ''), ('!', ''), ('.', ''), ('.', ' ')],
            ],
        ], result)

    def test_parse_udpipe(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_udpipe.conllu'))
        self.assertEqual([
            [
                [('Порву', ''), ('!', ' ')],
            ],
            [
                [('Порву', ''), ('!', ''), ('"', ''), ('...', ''), ('©', ' ')],
                [('Порву', ''), ('!', ' ')]
            ],
            [
                [('Ребят', ''), (',', ' '), ('я', ' '), ('никому', ' '), ('не', ' '), ('звонила', ''),
                 ('?', ''), ('?', ''), ('?', ' \t '), (')))', ' ')],
                [('Вот', ' '), ('это', ' '), ('был', ' '), ('номер', ''), ('...', ''), (')', ''),
                 ('))', ' ')],
            ],
        ], result)

    def test_parse_collapse(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_collapse.conllu'))
        self.assertEqual([
            [
                [('Er', ' '), ('arbeitet', ' '), ('fürs', ' '), ('FBI', ' '), ('(', ''), ('deutsch', ' '),
                 ('etwa', ''), (':', ' '), ('„', ''), ('Bundesamt', ' '), ('für', ' '), ('Ermittlung', ''),
                 ('“', ''), (')', ''), ('.', ' ')],
            ],
        ], result)

    def test_parse_space(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_space.conllu'))
        self.assertEqual([
            [
                [('Таким образом', ' '), (',', ' '), ('если', ' '), ('приговор', '⠀')]
            ],
            [
                [('Она', ' '), ('заболела', ' '), ('потому, что', ' '), ('мало', ' '), ('ела', ' '),
                 ('.', '\ufeff')]
            ],
        ], result)

    def test_parse_duplicates(self):
        result = parse_paragraphs(os.path.join(os.path.dirname(__file__), 'data', 'dataset_duplicates.conllu'))
        self.assertEqual([[[('Ранее', ' ')]]], result)


class TestGoodParagraphs(tf.test.TestCase):
    def test_multi_sent(self):
        source = [
            [[('«', ''), ('США', ''), ('»', '')]],
            [[('Но', ' '), ('в', ' '), ('отступлении', ''), ('.', ' ')]]
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 2)

    def test_small_sent(self):
        source = [
            [[('США', '')]]
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 0)

    def test_norm_sent(self):
        source = [
            [[('В', ' '), ('12', ' '), ('часов', ''), ('.', '')]]
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 1)

    def test_just_letters(self):
        source = [
            [[('В', ' '), ('1', ' '), ('2', ' '), ('.', '')]]
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 0)

    def test_ambig_start(self):
        source = [
            [[('-', ' '), ('Я', ' '), ('тоже', ''), ('.', '')]]
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 1)

    def test_bad_start(self):
        source = [
            [[('?', ' '), ('Я', ' '), ('тоже', ''), ('.', '')]]
        ]
        result = good_paragraphs(source)
        self.assertLen(result, 0)

    def test_bad_stop(self):
        source = [
            [[('-', ' '), ('Я', ' '), ('тоже', ''), (',', '')]]
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
            [[('First', ' '), ('sentence', ' '), ('in', ' '), ('paragraph', ''), ('.', '')],
             [('Second', ' '), ('"', ''), ('sentence', ''), ('"', ' '), ('in', ' '), ('paragraph', ''), ('.', '')]],
            [[('Single', ' '), ('-', ' '), ('sentence', ''), ('.', '')]]
        ]
        expected = [
            [[('First', ' '), ('sentence', ' '), ('in', ' '), ('paragraph', ''), ('.', ' ')],
             [('Second', ' '), ('"', ''), ('sentence', ''), ('"', ' '), ('in', ' '), ('paragraph', ''), ('.', '  ')]],
            [[('Single', ' '), ('-', ' '), ('sentence', ''), ('.', ' ')]]
        ]
        result = augment_paragraphs(source)
        self.assertEqual(expected, result)

    def test_spaces(self):
        np.random.seed(5)
        source = [
            [[('Single', ' '), ('sentence', '')],
             [('Next', ' '), ('single', ' '), ('sentence', '')]]
        ]
        expected = [
            [[('Single', ' '), ('sentence', ' ')],
             [('Next', ' '), ('single', ' '), ('sentence', '\n')]]
        ]
        result = augment_paragraphs(source)
        self.assertEqual(expected, result)

    def test_empty_end(self):
        np.random.seed(20)
        source = [
            [[('A', ' '), ('.', '')],
             [('B', ' '), ('!', ' ')],
             [('C', ' '), ('?', '')],
             [('D', ' '), ('.', ' ')],
             [('E', ' '), ('!', '')],
             [('F', ' '), ('?', ' ')]]
        ]
        expected = [
            [[('A', ' '), ('.', ' ')],
             [('B', ' '), ('!', ' ')],
             [('C', ' '), ('?', '\xa0')],
             [('D', ' '), ('.', ' ')],
             [('E', ' '), ('!', ' ')],
             [('F', ' '), ('?', ' ')]]
        ]
        result = augment_paragraphs(source)
        self.assertEqual(expected, result)


class TestLabelTokens(tf.test.TestCase):
    def test_empty(self):
        result = label_tokens([], [], [])
        self.assertEqual([], result)

    def test_same(self):
        result = label_tokens(
            ['word', '+', 'word', '-', 'word'],
            [''] * 5,
            [('word', ''), ('+', ''), ('word', ''), ('-', ''), ('word', '')]
        )
        self.assertEqual(['B', 'B', 'B', 'B', 'B'], result)

    def test_normal_join(self):
        result = label_tokens(
            ['word', '+', 'word', '-', 'word', '_', 'word'],
            [''] * 7,
            [('word', ''), ('+', ''), ('word-word', ''), ('_word', '')]
        )
        self.assertEqual(['B', 'B', 'B', 'I', 'I', 'B', 'I'], result)

    def test_joined_source(self):
        result = label_tokens(
            ['word+word-word'],
            [''],
            [('word', ''), ('+', ''), ('word', ''), ('-', ''), ('word', '')]
        )
        self.assertEqual(['B'], result)

    def test_joined_target(self):
        result = label_tokens(
            ['word', '+', 'word', '-', 'word'],
            [''] * 5,
            [('word+word-word', '')]
        )
        self.assertEqual(['B', 'I', 'I', 'I', 'I'], result)

    def test_different_join(self):
        result = label_tokens(
            ['word+word', '-', 'word'],
            [''] * 3,
            [('word', ''), ('+', ''), ('word-word', '')]
        )
        self.assertEqual(['B', 'I', 'I'], result)

    def test_compex(self):
        tokens = [
            ['test', '@', 'test', '.', 'com'],
            ['www', '.', 'test', '.', 'com'],
            ['word', '.', '.', 'word'],
            ['word', '+', 'word', '-', 'word'],
            ['word', '\\', 'word', '/', 'word', '#', 'word'],
            ['test', '@', 'test', '.', 'com'],
            ['www', '.', 'test', '.', 'com'],
            ['word', '.', '.', 'word'],
            ['word', '+', 'word', '-', 'word'],
            ['word', '\\', 'word', '/', 'word', '#', 'word']
        ]
        spaces = [[''] * len(e) for e in tokens]
        targets = [
            [('test@test.com', '')],
            [('www.test.com', '')],
            [('word..word', '')],
            [('word+word-word', '')],
            [('word\\word/word#word', '')],
            [('test', ''), ('@', ''), ('test', ''), ('.', ''), ('com', '')],
            [('www', ''), ('.', ''), ('test', ''), ('.', ''), ('com', '')],
            [('word', ''), ('..', ''), ('word', '')],
            [('word', ''), ('+', ''), ('word', ''), ('-', ''), ('word', '')],
            [('word', ''), ('\\', ''), ('word', ''), ('/', ''), ('word', ''), ('#', ''), ('word', '')]
        ]
        expected = [
            ['B', 'I', 'I', 'I', 'I'],
            ['B', 'I', 'I', 'I', 'I'],
            ['B', 'I', 'I', 'I'],
            ['B', 'I', 'I', 'I', 'I'],
            ['B', 'I', 'I', 'I', 'I', 'I', 'I'],
            ['B', 'B', 'B', 'B', 'B'],
            ['B', 'B', 'B', 'B', 'B'],
            ['B', 'B', 'I', 'B'],
            ['B', 'B', 'B', 'B', 'B'],
            ['B', 'B', 'B', 'B', 'B', 'B', 'B']
        ]

        for tok, sps, targ in zip(tokens, spaces, targets):
            print(tok, sps, targ)
            label_tokens(tok, sps, targ)
        result = [label_tokens(tok, sps, targ) for tok, sps, targ in zip(tokens, spaces, targets)]
        self.assertEqual(expected, result)

    def test_complex_space_1(self):
        result = label_tokens(
            ['1', '200', '000'],
            [' ', ' ', ''],
            [('1 200 000', '')]
        )
        self.assertEqual(['B', 'B', 'B'], result)

    def test_complex_space_2(self):
        result = label_tokens(
            ['1', '200', '000'],
            ['\xa0', '\u200b', ' '],
            [('1\xa0200\u200b000', ' ')]
        )
        self.assertEqual(['B', 'B', 'B'], result)

    def test_complex_space_3(self):
        with self.assertRaises(AssertionError):
            result = label_tokens(
                ['^', '1', '.', '2', '$'],
                ['\xa0', '', '\u200b', '', ''],
                [('^', ''), ('\xa01', ''), ('.', ''), ('\u200b2', ''), ('$', '')]
            )
            # self.assertEqual(['B', 'B', 'B', 'B', 'B'], result)


class TestLabelParagraphs(tf.test.TestCase):
    def test_empty(self):
        source = []
        result = label_paragraphs(source)
        self.assertEqual([], result)

    def test_normal(self):
        source = [
            [[('Single', ' '), ('-', ' '), ('sentence.', '\t')]],
            [[('First', ' '), ('sentence', '   '), ('in', ' '), ('paragraph', ''), ('.', '\n')],
             [('Second', ' '), ('"', ''), ('sentence', ''), ('"', ' '), ('in', '\u00A0'), ('paragraph', ''),
              ('.', ' ')]],
        ]
        expected = [
            [
                (['Single', '-', 'sentence', '.'],
                 [' ', ' ', '', '\t'],
                 ['B', 'B', 'B', 'I'])
            ],
            [
                (['First', 'sentence', 'in', 'paragraph', '.'],
                 [' ', '   ', ' ', '', '\n'],
                 ['B', 'B', 'B', 'B', 'B']),
                (['Second', '"', 'sentence', '"', 'in', 'paragraph', '.'],
                 [' ', '', '', ' ', '\u00A0', '', ' '],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B'])
            ]
        ]
        result = label_paragraphs(source)
        self.assertEqual(expected, result)

    def test_compex(self):
        source = [
            [[('test', ''), ('@', ''), ('test', ''), ('.', ''), ('com', ' ')],
             [('www', ''), ('.', ''), ('test', ''), ('.', ''), ('com', ' ')],
             [('word', ''), ('..', ''), ('word', ' ')],
             [('word', ''), ('+', ''), ('word', ''), ('-', ''), ('word', ' ')],
             [('word', ''), ('\\', ''), ('word', ''), ('/', ''), ('word', ''), ('#', ''), ('word', ' ')]],
            [[('test@test.com', ' ')],
             [('www.test.com', ' ')],
             [('word..word', ' ')],
             [('word+word-word', ' ')],
             [('word\\word/word#word', ' ')]]
        ]
        expected = [
            [
                (['test', '@', 'test', '.', 'com'],
                 ['', '', '', '', ' '],
                 ['B', 'B', 'B', 'B', 'B']),

                (['www', '.', 'test', '.', 'com'],
                 ['', '', '', '', ' '],
                 ['B', 'B', 'B', 'B', 'B']),

                (['word', '.', '.', 'word'],
                 ['', '', '', ' '],
                 ['B', 'B', 'I', 'B']),

                (['word', '+', 'word', '-', 'word'],
                 ['', '', '', '', ' '],
                 ['B', 'B', 'B', 'B', 'B']),

                (['word', '\\', 'word', '/', 'word', '#', 'word'],
                 ['', '', '', '', '', '', ' '],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B']),
            ],
            [
                (['test', '@', 'test', '.', 'com'],
                 ['', '', '', '', ' '],
                 ['B', 'I', 'I', 'I', 'I']),

                (['www', '.', 'test', '.', 'com'],
                 ['', '', '', '', ' '],
                 ['B', 'I', 'I', 'I', 'I']),

                (['word', '.', '.', 'word'],
                 ['', '', '', ' '],
                 ['B', 'I', 'I', 'I']),

                (['word', '+', 'word', '-', 'word'],
                 ['', '', '', '', ' '],
                 ['B', 'I', 'I', 'I', 'I']),

                (['word', '\\', 'word', '/', 'word', '#', 'word'],
                 ['', '', '', '', '', '', ' '],
                 ['B', 'I', 'I', 'I', 'I', 'I', 'I']),
            ],
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
            [
                (['Single', '-', 'sentence', '.'],
                 [' ', ' ', '', '\t'],
                 ['B', 'B', 'B', 'B'])
            ],
            [
                (['test', '@', 'test', '.', 'com'],
                 ['', '', '', '', ' '],
                 ['B', 'I', 'I', 'I', 'I']),
            ],
            [
                (['First', 'sentence', 'in', 'paragraph', '.'],
                 [' ', '   ', ' ', '', '\n'],
                 ['B', 'B', 'B', 'B', 'B']),
                (['Second', '"', 'sentence', '"', 'in', 'paragraph', '.'],
                 [' ', '', '', ' ', '\u00A0', '', ' '],
                 ['B', 'B', 'B', 'B', 'B', 'B', 'B'])
            ],
        ]

        expected_documents = [
            ['First ', 'sentence   ', 'in ', 'paragraph', '.\n', 'Second ', '"', 'sentence', '" ', 'in\xa0',
             'paragraph', '. '],
            ['test', '@', 'test', '.', 'com ', 'Single ', '- ', 'sentence', '.\t']
        ]
        expected_tokens = [
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', 'I', 'I', 'I', 'I', 'B', 'B', 'B', 'B']
        ]
        expected_sentences = [
            ['B', 'I', 'I', 'I', 'I', 'B', 'I', 'I', 'I', 'I', 'I', 'I'],
            ['B', 'I', 'I', 'I', 'I', 'B', 'I', 'I', 'I']
        ]

        result = make_documents(source, doc_size=10)
        result = [list(r) for r in zip(*result)]
        result_documents, result_tokens, result_sentences = result

        print(result_documents)
        self.assertEqual(expected_documents, result_documents)
        self.assertEqual(expected_tokens, result_tokens)
        self.assertEqual(expected_sentences, result_sentences)

    def test_complex(self):
        source = [
            [[('.', ' ')]],
            # [[('\u00adTest1', ''), ('.', ' ')]],
            # [[('\u200eTest2', ''), ('.', ' ')]],
            # [[('\u200fTest3', ''), ('.', ' ')]],
            # [[('\ufe0fTest4', ''), ('.', ' ')]],
            # [[('\ufeffTest5', ''), ('.', ' ')]],
            # [[('\u1f3fbTest6', ''), ('.', ' ')]],
            # [[('.', ''), ('.', ''), ('.', ''), ('.', ''), ('.', ''), ('.', ' ')]],
        ]

        for words, toks, sents in make_documents(label_paragraphs(source), 1):
            self.assertEqual(len(words), len(toks))
            self.assertEqual(len(toks), len(sents))

            repaired = self.evaluate(split_words(''.join(words), extended=True))
            repaired = [''.join([w.decode('utf-8') for w in repaired.tolist()])]
            self.assertListEqual(words, repaired)
            self.assertEqual(len(repaired), len([s for s in sents if len(s)]))
            self.assertEqual(len(repaired), len([t for t in toks if len(t)]))

        for words, toks, sents in make_documents(label_paragraphs(source), 9999):
            self.assertEqual(len(words), len(toks))
            self.assertEqual(len(toks), len(sents))

            repaired = self.evaluate(split_words(''.join(words), extended=True))
            repaired = [''.join([w.decode('utf-8') for w in repaired.tolist()])]
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

        builder = create_dataset(src_dir, dst_dir, 1, 1, '-test.')
        dataset = builder.as_dataset(split='train')

        has_examples = False
        for example in dataset:
            has_examples = True
            self.assertSetEqual({'document', 'length', 'token', 'sentence'}, set(example.keys()))
            self.assertLen(example['document'].shape, 0)
            self.assertLen(example['length'].shape, 0)
            self.assertLen(example['token'].shape, 0)
            self.assertLen(example['sentence'].shape, 0)

            length = example['length'].numpy()
            self.assertLen(example['token'].numpy(), length)
            self.assertLen(example['sentence'].numpy(), length)

        self.assertTrue(has_examples)
