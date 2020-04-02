# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import tempfile
import unittest
from ..convert import parse_paragraphs, split_convert


class TestParseParagraphs(unittest.TestCase):
    def test_normal(self):
        result = parse_paragraphs(
            [
                u'Тест.'
            ],
            os.path.join(os.path.dirname(__file__), 'data', 'udpipe.model'),
            'normal',
        )

        self.assertListEqual([
            u'# newdoc = normal_0',
            u'# sent_id = 35c07b0445bba8d5035d56b98995de32',
            u'# text = Тест.',
            u'1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            u'2	.	_	_	_	_	_	_	_	_',
            u'',
            u''
        ], ''.join(result).split('\n'))

    def test_complex(self):
        result = parse_paragraphs(
            [
                u'Тест, тест. Тест,\tтест',
                u'Тест, \t тест\u00A0.',
                u'Тест, тест.\nТест,\tтест',
                u'',
            ],
            os.path.join(os.path.dirname(__file__), 'data', 'udpipe.model'),
            'complex',
        )

        self.assertListEqual([
            u'# newdoc = complex_0',
            u'# sent_id = 56c0ae4d1f9a145d9f02287337fa02ff',
            u'# text = Тест, тест. Тест,	тест',
            u'1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            u'2	,	_	_	_	_	_	_	_	_',
            u'3	тест	_	_	_	_	_	_	_	SpaceAfter=No',
            u'4	.	_	_	_	_	_	_	_	_',
            u'5	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            u'6	,	_	_	_	_	_	_	_	SpacesAfter=\\t',
            u'7	тест	_	_	_	_	_	_	_	_',
            u'',
            u'# newdoc = complex_1',
            u'# sent_id = 3079e63ab70c2665778c78958479ac38',
            u'# text = Тест, 	 тест\u00A0.',
            u'1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            u'2	,	_	_	_	_	_	_	_	SpacesAfter=\\s\\t\\s',
            u'3	тест	_	_	_	_	_	_	_	SpacesAfter=\\xa0',
            u'4	.	_	_	_	_	_	_	_	_',
            u'',
            u'# newdoc = complex_2',
            u'# sent_id = 6d2c2fe018ea7bb1a3f8e83655aa396e',
            u'# text = Тест, тест.',
            u'1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            u'2	,	_	_	_	_	_	_	_	_',
            u'3	тест	_	_	_	_	_	_	_	SpaceAfter=No',
            u'4	.	_	_	_	_	_	_	_	_',
            u'',
            u'# sent_id = 525ad00a4bd3080e551c57f24db6d0de',
            u'# text = Тест,	тест',
            u'1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            u'2	,	_	_	_	_	_	_	_	SpacesAfter=\\t',
            u'3	тест	_	_	_	_	_	_	_	_',
            u'',
            u'',
        ], ''.join(result).split('\n'))

    def test_lf(self):
        result = parse_paragraphs(
            [
                u'Как на масленой неделе␊Из трубы блины летели!␊\nС пылу, с жару, из печи, ␊Все румяны, горячи!␊'
            ],
            os.path.join(os.path.dirname(__file__), 'data', 'udpipe.model'),
            'lf',
        )

        self.assertListEqual([
            u'# newdoc = lf_0',
            u'# sent_id = 8f036ea975890ac0527b6a2dd4d69eb3',
            u'# text = Как на масленой неделе Из трубы блины летели!',
            u'1	Как	_	_	_	_	_	_	_	_',
            u'2	на	_	_	_	_	_	_	_	_',
            u'3	масленой	_	_	_	_	_	_	_	_',
            u'4	неделе	_	_	_	_	_	_	_	SpacesAfter=\\n',
            u'5	Из	_	_	_	_	_	_	_	_',
            u'6	трубы	_	_	_	_	_	_	_	_',
            u'7	блины	_	_	_	_	_	_	_	_',
            u'8	летели	_	_	_	_	_	_	_	SpaceAfter=No',
            u'9	!	_	_	_	_	_	_	_	SpacesAfter=\\n',
            u'',
            u'# sent_id = cb286095c90b87a255c04f7c2173ef35',
            u'# text = С пылу, с жару, из печи,  Все румяны, горячи!',
            u'1	С	_	_	_	_	_	_	_	_',
            u'2	пылу	_	_	_	_	_	_	_	SpaceAfter=No',
            u'3	,	_	_	_	_	_	_	_	_',
            u'4	с	_	_	_	_	_	_	_	_',
            u'5	жару	_	_	_	_	_	_	_	SpaceAfter=No',
            u'6	,	_	_	_	_	_	_	_	_',
            u'7	из	_	_	_	_	_	_	_	_',
            u'8	печи	_	_	_	_	_	_	_	SpaceAfter=No',
            u'9	,	_	_	_	_	_	_	_	SpacesAfter=\\s\\n',
            u'10	Все	_	_	_	_	_	_	_	_',
            u'11	румяны	_	_	_	_	_	_	_	SpaceAfter=No',
            u'12	,	_	_	_	_	_	_	_	_',
            u'13	горячи	_	_	_	_	_	_	_	SpaceAfter=No',
            u'14	!	_	_	_	_	_	_	_	SpacesAfter=\\n',
            u'',
            u''
        ], ''.join(result).split('\n'))


class TestSplitConvert(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normal(self):
        tokenizer_model = os.path.join(os.path.dirname(__file__), 'data', 'udpipe.model')
        source_file = os.path.join(os.path.dirname(__file__), 'data', 'sentencies_source.txt')
        split_convert(source_file, tokenizer_model, self.temp_dir, 0.5)

        result_files = os.listdir(self.temp_dir)
        self.assertListEqual(
            result_files, ['___sentencies_source.txt-test.conllu', '___sentencies_source.txt-train.conllu'])

        with open(os.path.join(os.path.dirname(__file__), 'data', 'sentencies_test.txt'), 'rt') as f:
            expected = f.read()
        with open(os.path.join(self.temp_dir, '___sentencies_source.txt-test.conllu'), 'rt') as f:
            actual = f.read()

        self.assertEqual(expected, actual)