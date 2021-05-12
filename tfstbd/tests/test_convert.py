import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
from ufal.udpipe import Model, Pipeline
from ..convert import tokenize_sentence, tokenize_paragraphs, split_tokenize


class TestTokenizeSentence(tf.test.TestCase):
    def test_normal(self):
        model_path = os.path.join(os.path.dirname(__file__), 'data', 'udpipe.model')
        ud_model = Model.load(model_path)
        tokenizer = Pipeline(ud_model, 'tokenize', Pipeline.NONE, Pipeline.NONE, 'conllu')
        result = tokenize_sentence('Тест.', tokenizer, 'normal')

        self.assertListEqual([
            '# newdoc = normal',
            '# sent_id = 35c07b0445bba8d5035d56b98995de32',
            '# text = Тест.',
            '1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            '2	.	_	_	_	_	_	_	_	_',
            '',
            ''
        ], result.serialize().split('\n'))


class TestTokenizeParagraphs(tf.test.TestCase):
    def setUp(self):
        super(TestTokenizeParagraphs, self).setUp()
        model_path = os.path.join(os.path.dirname(__file__), 'data', 'udpipe.model')
        self.ud_model = Model.load(model_path)

    def test_normal(self):
        tokenizer = Pipeline(self.ud_model, 'tokenize', Pipeline.NONE, Pipeline.NONE, 'conllu')
        result = tokenize_paragraphs(
            [
                'Тест.'
            ],
            tokenizer,
            'normal',
        )

        self.assertListEqual([
            '# newdoc = normal_0',
            '# sent_id = 35c07b0445bba8d5035d56b98995de32',
            '# text = Тест.',
            '1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            '2	.	_	_	_	_	_	_	_	_',
            '',
            ''
        ], ''.join(result).split('\n'))

    def test_complex(self):
        tokenizer = Pipeline(self.ud_model, 'tokenize', Pipeline.NONE, Pipeline.NONE, 'conllu')
        result = tokenize_paragraphs(
            [
                'Тест, тест. Тест,\tтест',
                'Тест, \t тест\u00A0.',
                'Тест, тест.\nТест,\tтест',
                '',
            ],
            tokenizer,
            'complex',
        )

        self.assertListEqual([
            '# newdoc = complex_0',
            '# sent_id = 56c0ae4d1f9a145d9f02287337fa02ff',
            '# text = Тест, тест. Тест,	тест',
            '1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            '2	,	_	_	_	_	_	_	_	_',
            '3	тест	_	_	_	_	_	_	_	SpaceAfter=No',
            '4	.	_	_	_	_	_	_	_	_',
            '5	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            '6	,	_	_	_	_	_	_	_	SpacesAfter=\\t',
            '7	тест	_	_	_	_	_	_	_	_',
            '',
            '# newdoc = complex_1',
            '# sent_id = 3079e63ab70c2665778c78958479ac38',
            '# text = Тест, 	 тест\u00A0.',
            '1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            '2	,	_	_	_	_	_	_	_	SpacesAfter=\\s\\t\\s',
            '3	тест	_	_	_	_	_	_	_	SpacesAfter=\\xa0',
            '4	.	_	_	_	_	_	_	_	_',
            '',
            '# newdoc = complex_2',
            '# sent_id = 6d2c2fe018ea7bb1a3f8e83655aa396e',
            '# text = Тест, тест.',
            '1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            '2	,	_	_	_	_	_	_	_	_',
            '3	тест	_	_	_	_	_	_	_	SpaceAfter=No',
            '4	.	_	_	_	_	_	_	_	_',
            '',
            '# sent_id = 525ad00a4bd3080e551c57f24db6d0de',
            '# text = Тест,	тест',
            '1	Тест	_	_	_	_	_	_	_	SpaceAfter=No',
            '2	,	_	_	_	_	_	_	_	SpacesAfter=\\t',
            '3	тест	_	_	_	_	_	_	_	_',
            '',
            '',
        ], ''.join(result).split('\n'))

    def test_lf(self):
        tokenizer = Pipeline(self.ud_model, 'tokenize', Pipeline.NONE, Pipeline.NONE, 'conllu')
        result = tokenize_paragraphs(
            [
                'Как на масленой неделе␊Из трубы блины летели!␊\nС пылу, с жару, из печи, ␊Все румяны, горячи!␊'
            ],
            tokenizer,
            'lf',
        )

        self.assertListEqual([
            '# newdoc = lf_0',
            '# sent_id = 8f036ea975890ac0527b6a2dd4d69eb3',
            '# text = Как на масленой неделе Из трубы блины летели!',
            '1	Как	_	_	_	_	_	_	_	_',
            '2	на	_	_	_	_	_	_	_	_',
            '3	масленой	_	_	_	_	_	_	_	_',
            '4	неделе	_	_	_	_	_	_	_	SpacesAfter=\\n',
            '5	Из	_	_	_	_	_	_	_	_',
            '6	трубы	_	_	_	_	_	_	_	_',
            '7	блины	_	_	_	_	_	_	_	_',
            '8	летели	_	_	_	_	_	_	_	SpaceAfter=No',
            '9	!	_	_	_	_	_	_	_	SpacesAfter=\\n',
            '',
            '# sent_id = cb286095c90b87a255c04f7c2173ef35',
            '# text = С пылу, с жару, из печи,  Все румяны, горячи!',
            '1	С	_	_	_	_	_	_	_	_',
            '2	пылу	_	_	_	_	_	_	_	SpaceAfter=No',
            '3	,	_	_	_	_	_	_	_	_',
            '4	с	_	_	_	_	_	_	_	_',
            '5	жару	_	_	_	_	_	_	_	SpaceAfter=No',
            '6	,	_	_	_	_	_	_	_	_',
            '7	из	_	_	_	_	_	_	_	_',
            '8	печи	_	_	_	_	_	_	_	SpaceAfter=No',
            '9	,	_	_	_	_	_	_	_	SpacesAfter=\\s\\n',
            '10	Все	_	_	_	_	_	_	_	_',
            '11	румяны	_	_	_	_	_	_	_	SpaceAfter=No',
            '12	,	_	_	_	_	_	_	_	_',
            '13	горячи	_	_	_	_	_	_	_	SpaceAfter=No',
            '14	!	_	_	_	_	_	_	_	SpacesAfter=\\n',
            '',
            ''
        ], ''.join(result).split('\n'))


class TestSplitTokenize(tf.test.TestCase):
    def setUp(self):
        super(TestSplitTokenize, self).setUp()
        np.random.seed(2)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super(TestSplitTokenize, self).tearDown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normal(self):
        ud_model = os.path.join(os.path.dirname(__file__), 'data', 'udpipe.model')
        source_file = os.path.join(os.path.dirname(__file__), 'data', 'sentencies_source.txt')
        split_tokenize(source_file, ud_model, self.temp_dir, 0.5)

        result_files = os.listdir(self.temp_dir)
        self.assertListEqual(
            result_files, ['___sentencies_source.txt-test.conllu', '___sentencies_source.txt-train.conllu'])

        with open(os.path.join(os.path.dirname(__file__), 'data', 'sentencies_test.txt'), 'rt') as f:
            expected = f.read()
        with open(os.path.join(self.temp_dir, '___sentencies_source.txt-test.conllu'), 'rt') as f:
            actual = f.read()

        self.assertEqual(expected, actual)
