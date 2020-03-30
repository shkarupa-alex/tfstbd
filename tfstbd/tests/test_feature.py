# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ..feature import length_features, case_features, ngram_features


class TestLengthFeatures(tf.test.TestCase):
    def test_features(self):
        source = tf.strings.split([u'123 нижний ВЕРХНИЙ Предложение 34т аC'])
        expected = [[-0.40, 0.06, 0.21, 0.83, -0.40, -0.55]]
        result = self.evaluate(length_features(source, 5.6, 6.5).to_tensor(default_value=0.0))

        self.assertAllClose(expected, result.tolist(), rtol=1e-2, atol=1e-2)


class TestCaseFeatures(tf.test.TestCase):
    def test_features(self):
        source = tf.strings.split([u'123 нижний ВЕРХНИЙ Предложение 34т аC'])
        no_case, lower_case, upper_case, title_case, mixed_case = case_features(source)

        expected_no_case = [[1, 0, 0, 0, 0, 0]]
        expected_lower_case = [[0, 1, 0, 0, 1, 0]]
        expected_upper_case = [[0, 0, 1, 0, 0, 0]]
        expected_title_case = [[0, 0, 0, 1, 0, 0]]
        expected_mixed_case = [[0, 0, 0, 0, 0, 1]]

        result_no_case = self.evaluate(no_case.to_tensor(default_value=-1))
        self.assertAllEqual(expected_no_case, result_no_case.tolist())

        result_lower_case = self.evaluate(lower_case.to_tensor(default_value=-1))
        self.assertAllEqual(expected_lower_case, result_lower_case.tolist())

        result_upper_case = self.evaluate(upper_case.to_tensor(default_value=-1))
        self.assertAllEqual(expected_upper_case, result_upper_case.tolist())

        result_title_case = self.evaluate(title_case.to_tensor(default_value=-1))
        self.assertAllEqual(expected_title_case, result_title_case.tolist())

        result_mixed_case = self.evaluate(mixed_case.to_tensor(default_value=-1))
        self.assertAllEqual(expected_mixed_case, result_mixed_case.tolist())


class TestExtractNgramFeatures(tf.test.TestCase):
    def test_features(self):
        source = tf.strings.split([
            u'1\u00602\u00B43 Тест т\u02CAе\u02CBс\u0300т\u0301 '
            u'\u0060 \u00B4 \u02CA \u02CB \u0300 \u0301'
        ], sep=' ')
        expected = tf.constant([
            u'<000>', u'<тест', u'тест>', u'<тест', u'тест>',
            u'<>', u'<>', u'<>', u'<>', u'<>', u'<>',
        ])
        ngrams = ngram_features(source, 5, 5)
        ngrams, expected = self.evaluate([ngrams.flat_values, expected])

        self.assertAllEqual(expected.tolist(), ngrams.tolist())
