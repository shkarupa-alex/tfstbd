from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Activation, AdditiveAttention, Attention, Dense, Embedding, Lambda
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tfmiss.keras.layers import CharNgams, Reduction, TemporalConvNet, ToDense, WithRagged, WordShape
from tfmiss.text import lower_case, normalize_unicode, replace_string, split_words, zero_digits


def build_model(h_params, ngram_vocab):
    ngram_top, _ = ngram_vocab.split_by_frequency(h_params.ngram_freq)
    ngram_keys = ngram_top.tokens()

    documents = tf.keras.layers.Input(shape=(), name='documents', dtype=tf.string)
    tokens = Lambda(lambda doc: split_words(doc, extended=True), name='tokens')(documents)
    normals = Lambda(lambda tok: _normalize_tokens(tok), name='normals')(tokens)

    shapes = WordShape(WordShape.SHAPE_ALL, name='shapes')(normals)
    shapes = WithRagged(Dense(8, name='projections'))(shapes)

    ngrams = CharNgams(h_params.ngram_minn, h_params.ngram_maxn, h_params.ngram_self, name='ngrams')(normals)
    lookup = StringLookup(vocabulary=ngram_keys, mask_token=None, name='indexes')
    indexes = lookup(ngrams)
    embeddings = Embedding(lookup.vocab_size(), h_params.ngram_dim, name='embeddings')(indexes)
    embeddings = Reduction(h_params.ngram_comb, name='reduction')(embeddings)

    features = tf.keras.layers.concatenate([embeddings, shapes], name='features')
    features = ToDense(0.0, mask=True)(features)

    if 'lstm' == h_params.seq_core:
        for i, units in enumerate(h_params.lstm_units):
            features = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=True),
                name='lstm_{}'.format(i)
            )(features)
    else:
        features = TemporalConvNet(
            h_params.tcn_filters, h_params.tcn_ksize, h_params.tcn_drop, 'same', name='tcn')(features)

    if 'luong' == h_params.att_core:
        features = Attention(dropout=h_params.att_drop, name='attention')([features, features])
    elif 'bahdanau' == h_params.att_core:
        features = AdditiveAttention(dropout=h_params.att_drop, name='attention')([features, features])

    dense_tokens = ToDense('', mask=False, name='dense_tokens')(tokens)

    space_head = Dense(1, name='space_logits')(features)
    space_head = Activation('sigmoid', dtype='float32', name='space')(space_head)

    token_head = Dense(1, name='token_logits')(features)
    token_head = Activation('sigmoid', dtype='float32', name='token')(token_head)

    sentence_head = Dense(1, name='sentence_logits')(features)
    sentence_head = Activation('sigmoid', dtype='float32', name='sentence')(sentence_head)

    model = tf.keras.Model(
        inputs=documents,
        outputs=[dense_tokens, space_head, token_head, sentence_head]
    )

    return model


def _normalize_tokens(tokens):
    tokens = normalize_unicode(tokens, 'NFKC')
    tokens = replace_string(  # accentuation
        tokens,
        [u'\u0060', u' \u0301', u'\u02CA', u'\u02CB', u'\u0300', u'\u0301'],
        [''] * 6
    )
    tokens = lower_case(tokens)
    tokens = zero_digits(tokens)

    return tokens
