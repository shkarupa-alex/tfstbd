import os
import tensorflow as tf
from conllu import parse
from conllu.models import Token
from ..conllu import decode_space, encode_space, extract_space, meaning_tokens
from ..conllu import extract_tokens, extract_text, join_text, split_sents, repair_spaces


class TestDecodeSpace(tf.test.TestCase):
    def test_normal(self):
        self.assertEqual('\n', decode_space('\n'))
        self.assertEqual('\n', decode_space(r'\n'))
        self.assertEqual('\xa0', decode_space('\xa0'))
        self.assertEqual('\xa0', decode_space('\\xa0'))
        self.assertEqual('\xa0', decode_space('\u00A0'))
        self.assertEqual('\xa0', decode_space('\\u00A0'))


class TestEncodeSpace(tf.test.TestCase):
    def test_normal(self):
        self.assertEqual(r'\n', encode_space('\n'))
        self.assertEqual(r'\n', encode_space(r'\n'))
        self.assertEqual(r'\xa0', encode_space('\xa0'))
        self.assertEqual(r'\xa0', encode_space('\\xa0'))
        self.assertEqual(r'\xa0', encode_space('\u00A0'))
        self.assertEqual(r'\xa0', encode_space('\\u00A0'))


class TestExtractSpace(tf.test.TestCase):
    def test_none(self):
        self.assertEqual(' ', extract_space(Token()))
        self.assertEqual(' ', extract_space(Token({'misc': None})))

    def test_space_after(self):
        self.assertEqual('', extract_space(Token({'misc': {'SpaceAfter': 'No'}})))
        self.assertEqual('', extract_space(Token({'misc': {'SpaceAfter': 'No', 'SpacesAfter': '  '}})))

    def test_spaces_after(self):
        self.assertEqual('\n', extract_space(Token({'misc': {'SpacesAfter': '\n'}})))
        self.assertEqual('\n', extract_space(Token({'misc': {'SpacesAfter': '\\n'}})))
        self.assertEqual('\u00A0', extract_space(Token({'misc': {'SpacesAfter': '\u00A0'}})))
        self.assertEqual('\u00A0', extract_space(Token({'misc': {'SpacesAfter': '\\xa0'}})))
        self.assertEqual(' ', extract_space(Token({'misc': {'SpacesAfter': '_'}})))


class TestMeaningTokens(tf.test.TestCase):
    def test_normal(self):
        source = parse('\n'.join([
            '1     Er           er           PRON    …   _',
            '2     arbeitet     arbeiten     VERB    …   _',
            '3.1   fürs         _            _       …   _',
            '3-4   fürs         _            _       …   _',
            '3     für          für          ADP     …   _',
            '4     das          der          DET     …   _',
            '5     FBI          FBI          PROPN   …   _',
        ]))[0]  # first sentence
        self.assertListEqual([0, 1, 3, 6], meaning_tokens(source))


class TestExtractTokens(tf.test.TestCase):
    def test_normal(self):
        content = [
            '# newdoc id = doc1',
            '# newpar id = par1',
            '# sent_id = 1',
            '# text = Результаты (нет',
            '1	Результаты	_	_	_	_	_	_	_	_',
            '2	(	_	_	_	_	_	_	_	SpaceAfter=No',
            '3	нет	_	_	_	_	_	_	_	_',
            '',
            '',
        ]
        sentences = parse('\n'.join(content))
        tokens = extract_tokens(sentences[0])
        expected = [('Результаты', ' '), ('(', ''), ('нет', ' ')]
        self.assertListEqual(expected, tokens)

    def test_complex(self):
        content = [
            '# sent_id = 2011Interviyu_Mariny_Astvatsaturyan.xml_11',
            '# text = Тогда, \xa0как и сейчас, в качестве внештатного сотрудника.',
            '0.1	_	_	_	_	_	_	_	0:exroot	_',
            '1	Тогда	тогда	ADV	_	Degree=Pos	10	orphan	0.1:advmod	SpaceAfter=No',
            '2	,	,	PUNCT	_	_	5	punct	5:punct	SpacesAfter=\\s\\xa0',
            '3	как	как	SCONJ	_	_	5	mark	5:mark	_',
            '4	и	и	PART	_	_	5	advmod	5:advmod	_',
            '5	сейчас	сейчас	ADV	_	Degree=Pos	1	advcl	1:advcl	SpaceAfter=No',
            '6	,	,	PUNCT	_	_	5	punct	5:punct	_',
            '7	в	в	ADP	_	_	10	case	10:case	_',
            '8	качестве	качество	NOUN	_	Animacy=Inan|Case=Loc|Gender=Neut|Number=Sing	7	fixed	7:fixed	_',
            '9	внештатного	внештатный	ADJ	_	Case=Gen|Degree=Pos|Gender=Masc|Number=Sing	10	amod	10:amod	_',
            '10	сотрудника	сотрудник	NOUN	_	Animacy=Anim|Case=Gen|Gender=Masc|Number=Sing	0	root	0:root	SpaceAfter=No',
            '11	.	.	PUNCT	_	_	10	punct	10:punct	_'
        ]
        sentences = parse('\n'.join(content))
        tokens = extract_tokens(sentences[0], last_space=False)
        expected = [
            ('Тогда', ''), (',', ' \xa0'), ('как', ' '), ('и', ' '), ('сейчас', ''), (',', ' '),
            ('в', ' '), ('качестве', ' '), ('внештатного', ' '), ('сотрудника', ''), ('.', '')
        ]
        self.assertEqual(expected, tokens)


class TestExtractText(tf.test.TestCase):
    def test_normal(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.conll')]

        content = ''
        for file_name in data_files:
            with open(file_name, 'rb') as f:
                content += f.read().decode('utf-8') + '\n\n'

        sentences = parse(content)
        for s in sentences:
            extract_text(s)

    def test_complex_1(self):
        content = [
            '# newdoc id = doc1',
            '# newpar id = par1',
            '# sent_id = 1',
            '# text = Результаты \xa0(\xa0 нет',
            '1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s\\xa0',
            '2	(	_	_	_	_	_	_	_	SpacesAfter=\\xa0\\s',
            '3	нет	_	_	_	_	_	_	_	SpaceAfter=No',
            '',
            '',
        ]
        sentences = parse('\n'.join(content))
        tokens = extract_text(sentences[0])
        expected = ['Результаты', ' \xa0', '(', '\xa0 ', 'нет']
        self.assertListEqual(expected, tokens)

    def test_complex_2(self):
        content = [
            '# sent_id = 2011Interviyu_Mariny_Astvatsaturyan.xml_11',
            '# text = Тогда, как и сейчас, в качестве внештатного сотрудника.',
            '0.1	_	_	_	_	_	_	_	0:exroot	_',
            '1	Тогда	тогда	ADV	_	Degree=Pos	10	orphan	0.1:advmod	SpaceAfter=No',
            '2	,	,	PUNCT	_	_	5	punct	5:punct	_',
            '3	как	как	SCONJ	_	_	5	mark	5:mark	_',
            '4	и	и	PART	_	_	5	advmod	5:advmod	_',
            '5	сейчас	сейчас	ADV	_	Degree=Pos	1	advcl	1:advcl	SpaceAfter=No',
            '6	,	,	PUNCT	_	_	5	punct	5:punct	_',
            '7	в	в	ADP	_	_	10	case	10:case	_',
            '8	качестве	качество	NOUN	_	Animacy=Inan|Case=Loc|Gender=Neut|Number=Sing	7	fixed	7:fixed	_',
            '9	внештатного	внештатный	ADJ	_	Case=Gen|Degree=Pos|Gender=Masc|Number=Sing	10	amod	10:amod	_',
            '10	сотрудника	сотрудник	NOUN	_	Animacy=Anim|Case=Gen|Gender=Masc|Number=Sing	0	root	0:root	SpaceAfter=No',
            '11	.	.	PUNCT	_	_	10	punct	10:punct	_'
        ]
        sentences = parse('\n'.join(content))
        tokens = extract_text(sentences[0])
        expected = sentences[0].metadata['text']
        self.assertEqual(expected, ''.join(tokens))

    def test_complex_3(self):
        content = [
            '# sent_id = solarix_883',
            '# text = Она мало ела.﻿',
            '1	Она	_	_	_	_	_	_	_	_',
            '4	мало	_	_	_	_	_	_	_	_',
            '5	ела	_	_	_	_	_	_	_	SpaceAfter=No',
            '6	.	_	_	_	_	_	_	_	SpacesAfter=\ufeff',
            '',
            '',
        ]
        sentences = parse('\n'.join(content))
        tokens = extract_text(sentences[0])
        expected = ['Она', ' ', 'мало', ' ', 'ела', '.']
        self.assertListEqual(expected, tokens)


class TestJoinText(tf.test.TestCase):
    def test_normal(self):
        self.assertEqual('test', join_text([' ', '\u200b', ' ', '\ufeff', '\n', 'test', ' ', '\xa0']))


class TestSplitSents(tf.test.TestCase):
    def test_normal(self):
        content = [
            '# newdoc id = doc1',
            '# newpar id = par1',
            '# sent_id = 1',
            '# text = Результаты \xa0(\xa0 нет',
            '1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s\\xa0',
            '2	(	_	_	_	_	_	_	_	SpacesAfter=\\xa0\\s',
            '3	нет	_	_	_	_	_	_	_	_',
            '',
            '',
        ]
        sentences = parse('\n'.join(content))
        sentences = split_sents(sentences[0])
        result = sentences[0].serialize().split('\n')

        self.assertLen(sentences, 1)
        self.assertEqual(content[3:], result)

    def test_complex_1(self):
        content = [
            '# newdoc id = doc1',
            '# newpar id = par1',
            '# sent_id = 1',
            '# text = Результаты.Выводы',
            '1	Результаты	_	_	_	_	_	_	_	SpaceAfter=No',
            '2	.	_	_	_	_	_	_	_	SentenceBreak=Yes|SpaceAfter=No',
            '3	Выводы	_	_	_	_	_	_	_	SpaceAfter=No',
            '',
            '',
        ]
        expected = [
            [
                '# text = Результаты.',
                '1\tРезультаты\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No',
                '2\t.\t_\t_\t_\t_\t_\t_\t_\t_',
                '',
                ''
            ],
            [
                '# text = Выводы',
                '3\tВыводы\t_\t_\t_\t_\t_\t_\t_\t_',
                '',
                ''
            ]
        ]

        sentences = parse('\n'.join(content))
        sentences = split_sents(sentences[0])
        result = [s.serialize().split('\n') for s in sentences]

        self.assertLen(sentences, 2)
        self.assertEqual(expected, result)

    # Disabled due to \u2800 ambiguity
    # def test_complex_2(self):
    #     content = [
    #         '# text = Зима.⠀Холодно.',
    #         '1	Зима	_	_	_	_	_	_	_	SpaceAfter=No',
    #         '2	.	_	_	_	_	_	_	_	SentenceBreak=Yes|SpacesAfter=\u2800',
    #         '3	Холодно	_	_	_	_	_	_	_	SpaceAfter=No',
    #         '4	.	_	_	_	_	_	_	_	_',
    #         '',
    #         '',
    #     ]
    #     expected = [
    #         [
    #             '# text = Зима.',
    #             '1\tЗима\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No',
    #             '2\t.\t_\t_\t_\t_\t_\t_\t_\tSpacesAfter=\u2800',
    #             '',
    #             ''
    #         ],
    #         [
    #             '# text = Холодно',
    #             '3\tВыводы\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No',
    #             '4\t.\t_\t_\t_\t_\t_\t_\t_\t_',
    #             '',
    #             ''
    #         ]
    #     ]
    #
    #     sentences = parse('\n'.join(content))
    #     sentences = split_sents(sentences[0])
    #     result = [s.serialize().split('\n') for s in sentences]
    #
    #     self.assertLen(sentences, 2)
    #     self.assertEqual(expected, result)


class TestRepairSpaces(tf.test.TestCase):
    def test_normal(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.conll')]

        content = ''
        for file_name in data_files:
            with open(file_name, 'rb') as f:
                content += f.read().decode('utf-8') + '\n\n'

        sentences = parse(content)
        for s in sentences:
            s = repair_spaces(s)
            extract_text(s)

    def test_complex(self):
        content = [
            '# newdoc id = doc1',
            '# newpar id = par1',
            '# sent_id = 1',
            '# text = Результаты \xa0(\xa0 нет',
            '1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s',
            '2	(	_	_	_	_	_	_	_	_',
            '3	нет	_	_	_	_	_	_	_	_',
            '',
            '',
        ]
        sentences = parse('\n'.join(content))
        s = repair_spaces(sentences[0])
        extract_text(s)

        expected = [
            '# newdoc id = doc1',
            '# newpar id = par1',
            '# sent_id = 1',
            '# text = Результаты \xa0(\xa0 нет',
            '1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s\\xa0',
            '2	(	_	_	_	_	_	_	_	SpacesAfter=\\xa0\\s',
            '3	нет	_	_	_	_	_	_	_	_',
            '',
            '',
        ]
        self.assertListEqual(expected, s.serialize().split('\n'))

    def test_lf(self):
        content = [
            '# newdoc id = doc1',
            '# newpar id = par1',
            '# sent_id = 1',
            '# text = Результаты  :нет',
            '1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s\\n',
            '2	:	_	_	_	_	_	_	_	SpaceAfter=No',
            '3	нет	_	_	_	_	_	_	_	_',
            '',
            '',
        ]
        sentences = parse('\n'.join(content))
        s = repair_spaces(sentences[0])
        extract_text(s)

        expected = [
            '# newdoc id = doc1',
            '# newpar id = par1',
            '# sent_id = 1',
            '# text = Результаты  :нет',
            '1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s\\n',
            '2	:	_	_	_	_	_	_	_	SpaceAfter=No',
            '3	нет	_	_	_	_	_	_	_	_',
            '',
            '',
        ]
        self.assertListEqual(expected, s.serialize().split('\n'))
