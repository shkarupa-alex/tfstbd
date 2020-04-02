# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfmiss.text.unicode_transform import normalize_unicode, lower_case, upper_case, title_case
from tfmiss.text.unicode_transform import replace_string, zero_digits, wrap_with
from tfmiss.text.unicode_expand import char_ngrams, split_words


def length_features(input_words, word_mean, word_std):
    word_length = tf.strings.length(input_words, unit='UTF8_CHAR')

    return (tf.cast(word_length, tf.float32) - word_mean) / word_std


def case_features(input_words):
    input_words = normalize_unicode(input_words, 'NFKC')
    words_lower = lower_case(input_words)
    words_upper = upper_case(input_words)
    words_title = title_case(input_words)

    has_case = tf.not_equal(words_lower, words_upper)
    no_case = tf.logical_not(has_case)

    is_lower = tf.logical_and(
        has_case,
        tf.equal(input_words, words_lower)
    )

    is_upper = tf.logical_and(
        has_case,
        tf.equal(input_words, words_upper)
    )

    is_title = tf.logical_and(
        has_case,
        tf.equal(input_words, words_title)
    )

    is_mixed = tf.logical_not(tf.logical_or(
        tf.logical_or(no_case, is_lower),
        tf.logical_or(is_upper, is_title)
    ))

    return tf.cast(no_case, tf.int32), \
           tf.cast(is_lower, tf.int32), \
           tf.cast(is_upper, tf.int32), \
           tf.cast(is_title, tf.int32), \
           tf.cast(is_mixed, tf.int32)


def ngram_features(input_words, minn, maxn):
    input_words = normalize_unicode(input_words, 'NFKC')
    input_words = replace_string(  # accentuation
        input_words,
        [u'\u0060', u' \u0301', u'\u02CA', u'\u02CB', u'\u0300', u'\u0301'],
        [''] * 6
    )
    input_words = lower_case(input_words)
    input_words = zero_digits(input_words)
    input_words = wrap_with(input_words, '<', '>')
    word_ngrams = char_ngrams(input_words, minn, maxn, itself='ALONE')

    return word_ngrams


def document_features(documents, word_mean, word_std, ngram_minn, ngram_maxn):
    words = split_words(documents, extended=True)

    length = length_features(words, word_mean, word_std)
    no_case, lower_case, upper_case, title_case, mixed_case = case_features(words)
    ngrams = ngram_features(words, ngram_minn, ngram_maxn)

    return {
        'document': documents,
        'words': words.to_tensor(default_value=''),  # Required to pass in prediction
        'word_ngrams': ngrams,
        'word_length': length,
        'word_nocase': no_case,
        'word_lower': lower_case,
        'word_upper': upper_case,
        'word_title': title_case,
        'word_mixed': mixed_case,
    }
