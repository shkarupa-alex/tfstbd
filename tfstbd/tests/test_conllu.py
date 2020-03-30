# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
from collections import OrderedDict
from conllu import parse
from ..conllu import meaning_tokens, decode_space, encode_space, extract_space
from ..conllu import extract_tokens, extract_text, split_sent, repair_spaces


class TestMeaningTokens(unittest.TestCase):
    def testNormal(self):
        source = parse('\n'.join([
            u'1     Er           er           PRON    …   _',
            u'2     arbeitet     arbeiten     VERB    …   _',
            u'3.1   fürs         _            _       …   _',
            u'3-4   fürs         _            _       …   _',
            u'3     für          für          ADP     …   _',
            u'4     das          der          DET     …   _',
            u'5     FBI          FBI          PROPN   …   _',
        ]))[0]  # first sentence
        self.assertListEqual([0, 1, 3, 6], meaning_tokens(source))


class TestDecodeSpace(unittest.TestCase):
    def testNormal(self):
        self.assertEqual('\n', decode_space('\n'))
        self.assertEqual('\n', decode_space(r'\n'))
        self.assertEqual(u'\xa0', decode_space(u'\xa0'))
        self.assertEqual(u'\xa0', decode_space('\\xa0'))
        self.assertEqual(u'\xa0', decode_space(u'\u00A0'))
        self.assertEqual(u'\xa0', decode_space('\\u00A0'))


class TestEncodeSpace(unittest.TestCase):
    def testNormal(self):
        self.assertEqual(r'\n', encode_space('\n'))
        self.assertEqual(r'\n', encode_space(r'\n'))
        self.assertEqual(r'\xa0', encode_space(u'\xa0'))
        self.assertEqual(r'\xa0', encode_space('\\xa0'))
        self.assertEqual(r'\xa0', encode_space(u'\u00A0'))
        self.assertEqual(r'\xa0', encode_space('\\u00A0'))


class TestExtractSpace(unittest.TestCase):
    def testNone(self):
        self.assertEqual(' ', extract_space(OrderedDict()))
        self.assertEqual(' ', extract_space(OrderedDict({'misc': None})))

    def testSpaceAfter(self):
        self.assertEqual('', extract_space(OrderedDict({'misc': OrderedDict({'SpaceAfter': 'No'})})))
        self.assertEqual('', extract_space(
            OrderedDict({'misc': OrderedDict({'SpaceAfter': 'No', 'SpacesAfter': '  '})})))

    def testSpacesAfter(self):
        self.assertEqual('\n', extract_space(OrderedDict({'misc': OrderedDict({'SpacesAfter': '\n'})})))
        self.assertEqual('\n', extract_space(OrderedDict({'misc': OrderedDict({'SpacesAfter': '\\n'})})))
        self.assertEqual(u'\u00A0', extract_space(OrderedDict({'misc': OrderedDict({'SpacesAfter': '\u00A0'})})))
        self.assertEqual(u'\u00A0', extract_space(OrderedDict({'misc': OrderedDict({'SpacesAfter': '\\xa0'})})))
        self.assertEqual(' ', extract_space(OrderedDict({'misc': OrderedDict({'SpacesAfter': '_'})})))


class TestExtractTokens(unittest.TestCase):
    def testNormal(self):
        content = [
            u'# newdoc id = doc1',
            u'# newpar id = par1',
            u'# sent_id = 1',
            u'# text = Результаты (нет',
            u'1	Результаты	_	_	_	_	_	_	_	_',
            u'2	(	_	_	_	_	_	_	_	SpaceAfter=No',
            u'3	нет	_	_	_	_	_	_	_	_',
            u'',
            u'',
        ]
        sentences = parse('\n'.join(content))
        tokens = extract_tokens(sentences[0])
        expected = [(u'Результаты', u' '), (u'(', u''), (u'нет', u' ')]
        self.assertListEqual(expected, tokens)

    def testComplex(self):
        content = [
            u'# sent_id = 2011Interviyu_Mariny_Astvatsaturyan.xml_11',
            u'# text = Тогда, \xa0как и сейчас, в качестве внештатного сотрудника.',
            u'0.1	_	_	_	_	_	_	_	0:exroot	_',
            u'1	Тогда	тогда	ADV	_	Degree=Pos	10	orphan	0.1:advmod	SpaceAfter=No',
            u'2	,	,	PUNCT	_	_	5	punct	5:punct	SpacesAfter=\\s\\xa0',
            u'3	как	как	SCONJ	_	_	5	mark	5:mark	_',
            u'4	и	и	PART	_	_	5	advmod	5:advmod	_',
            u'5	сейчас	сейчас	ADV	_	Degree=Pos	1	advcl	1:advcl	SpaceAfter=No',
            u'6	,	,	PUNCT	_	_	5	punct	5:punct	_',
            u'7	в	в	ADP	_	_	10	case	10:case	_',
            u'8	качестве	качество	NOUN	_	Animacy=Inan|Case=Loc|Gender=Neut|Number=Sing	7	fixed	7:fixed	_',
            u'9	внештатного	внештатный	ADJ	_	Case=Gen|Degree=Pos|Gender=Masc|Number=Sing	10	amod	10:amod	_',
            u'10	сотрудника	сотрудник	NOUN	_	Animacy=Anim|Case=Gen|Gender=Masc|Number=Sing	0	root	0:root	SpaceAfter=No',
            u'11	.	.	PUNCT	_	_	10	punct	10:punct	_'
        ]
        sentences = parse('\n'.join(content))
        tokens = extract_tokens(sentences[0], last_space=False)
        expected = [
            (u'Тогда', u''), (u',', u' \xa0'), (u'как', u' '), (u'и', u' '), (u'сейчас', u''), (u',', u' '),
            (u'в', u' '), (u'качестве', u' '), (u'внештатного', u' '), (u'сотрудника', u''), (u'.', u'')
        ]
        self.assertEqual(expected, tokens)


class TestExtractText(unittest.TestCase):
    def testNormal(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.conllu')]

        content = ''
        for file_name in data_files:
            with open(file_name, 'rb') as f:
                content += f.read().decode('utf-8') + '\n\n'

        sentences = parse(content)
        for s in sentences:
            if all([extract_space(t) is not None for t in s]):
                extract_text(s)

    def testComplex1(self):
        content = [
            u'# newdoc id = doc1',
            u'# newpar id = par1',
            u'# sent_id = 1',
            u'# text = Результаты \xa0(\xa0 нет',
            u'1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s\\xa0',
            u'2	(	_	_	_	_	_	_	_	SpacesAfter=\\xa0\\s',
            u'3	нет	_	_	_	_	_	_	_	SpaceAfter=No',
            u'',
            u'',
        ]
        sentences = parse('\n'.join(content))
        tokens = extract_text(sentences[0])
        expected = [u'Результаты', u' \xa0', u'(', u'\xa0 ', u'нет']
        self.assertListEqual(expected, tokens)

    def testComplex2(self):
        content = [
            u'# sent_id = 2011Interviyu_Mariny_Astvatsaturyan.xml_11',
            u'# text = Тогда, как и сейчас, в качестве внештатного сотрудника.',
            u'0.1	_	_	_	_	_	_	_	0:exroot	_',
            u'1	Тогда	тогда	ADV	_	Degree=Pos	10	orphan	0.1:advmod	SpaceAfter=No',
            u'2	,	,	PUNCT	_	_	5	punct	5:punct	_',
            u'3	как	как	SCONJ	_	_	5	mark	5:mark	_',
            u'4	и	и	PART	_	_	5	advmod	5:advmod	_',
            u'5	сейчас	сейчас	ADV	_	Degree=Pos	1	advcl	1:advcl	SpaceAfter=No',
            u'6	,	,	PUNCT	_	_	5	punct	5:punct	_',
            u'7	в	в	ADP	_	_	10	case	10:case	_',
            u'8	качестве	качество	NOUN	_	Animacy=Inan|Case=Loc|Gender=Neut|Number=Sing	7	fixed	7:fixed	_',
            u'9	внештатного	внештатный	ADJ	_	Case=Gen|Degree=Pos|Gender=Masc|Number=Sing	10	amod	10:amod	_',
            u'10	сотрудника	сотрудник	NOUN	_	Animacy=Anim|Case=Gen|Gender=Masc|Number=Sing	0	root	0:root	SpaceAfter=No',
            u'11	.	.	PUNCT	_	_	10	punct	10:punct	_'
        ]
        sentences = parse('\n'.join(content))
        tokens = extract_text(sentences[0])
        expected = sentences[0].metadata['text']
        self.assertEqual(expected, ''.join(tokens))


class TestSplitSent(unittest.TestCase):
    def testNormal(self):
        content = [
            u'# newdoc id = doc1',
            u'# newpar id = par1',
            u'# sent_id = 1',
            u'# text = Результаты \xa0(\xa0 нет',
            u'1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s\\xa0',
            u'2	(	_	_	_	_	_	_	_	SpacesAfter=\\xa0\\s',
            u'3	нет	_	_	_	_	_	_	_	SpaceAfter=No',
            u'',
            u'',
        ]
        sentences = parse('\n'.join(content))
        sentences = split_sent(sentences[0])
        result = [extract_text(s, validate=False) for s in sentences]
        expected = [
            [u'Результаты', u' \xa0', u'(', u'\xa0 ', u'нет'],
        ]

        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            self.assertListEqual(e, r)

    def testComplex(self):
        content = [
            u'# newdoc id = doc1',
            u'# newpar id = par1',
            u'# sent_id = 1',
            u'# text = Результаты.Выводы',
            u'1	Результаты	_	_	_	_	_	_	_	SpaceAfter=No',
            u'2	.	_	_	_	_	_	_	_	SentenceBreak=Yes|SpaceAfter=No',
            u'3	Выводы	_	_	_	_	_	_	_	SpaceAfter=No',
            u'',
            u'',
        ]
        sentences = parse('\n'.join(content))
        sentences = split_sent(sentences[0])
        result = [extract_text(s, validate=False) for s in sentences]
        expected = [
            [u'Результаты', u'.'],
            [u'Выводы'],
        ]

        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            self.assertListEqual(e, r)


class TestRepairSpaces(unittest.TestCase):
    def testNormal(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.conllu')]

        content = ''
        for file_name in data_files:
            with open(file_name, 'rb') as f:
                content += f.read().decode('utf-8') + '\n\n'

        sentences = parse(content)
        for s in sentences:
            s = repair_spaces(s)
            extract_text(s)

    def testComplex(self):
        content = [
            u'# newdoc id = doc1',
            u'# newpar id = par1',
            u'# sent_id = 1',
            u'# text = Результаты \xa0(\xa0 нет',
            u'1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s',
            u'2	(	_	_	_	_	_	_	_	_',
            u'3	нет	_	_	_	_	_	_	_	_',
            u'',
            u'',
        ]
        sentences = parse('\n'.join(content))
        s = repair_spaces(sentences[0])
        extract_text(s)

        expected = [
            u'# newdoc id = doc1',
            u'# newpar id = par1',
            u'# sent_id = 1',
            u'# text = Результаты \xa0(\xa0 нет',
            u'1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s\\xa0',
            u'2	(	_	_	_	_	_	_	_	SpacesAfter=\\xa0\\s',
            u'3	нет	_	_	_	_	_	_	_	_',
            u'',
            u'',
        ]
        self.assertListEqual(expected, s.serialize().split('\n'))

    def testLf(self):
        content = [
            u'# newdoc id = doc1',
            u'# newpar id = par1',
            u'# sent_id = 1',
            u'# text = Результаты  :нет',
            u'1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s\\n',
            u'2	:	_	_	_	_	_	_	_	SpaceAfter=No',
            u'3	нет	_	_	_	_	_	_	_	_',
            u'',
            u'',
        ]
        sentences = parse('\n'.join(content))
        s = repair_spaces(sentences[0])
        extract_text(s)

        expected = [
            u'# newdoc id = doc1',
            u'# newpar id = par1',
            u'# sent_id = 1',
            u'# text = Результаты  :нет',
            u'1	Результаты	_	_	_	_	_	_	_	SpacesAfter=\\s\\n',
            u'2	:	_	_	_	_	_	_	_	SpaceAfter=No',
            u'3	нет	_	_	_	_	_	_	_	_',
            u'',
            u'',
        ]
        self.assertListEqual(expected, s.serialize().split('\n'))
