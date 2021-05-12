import tensorflow as tf
from nlpvocab import Vocabulary
from tensorflow.keras.layers import Activation, Dense, Embedding, Lambda
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tfmiss.keras.layers import AdditiveSelfAttention, MultiplicativeSelfAttention
from tfmiss.keras.layers import CharNgams, Reduction, TemporalConvNet, ToDense, WithRagged, WordShape
from tfmiss.text import split_words
from .input import normalize_tokens
from .hparam import HParams


def build_model(h_params: HParams, ngram_vocab: Vocabulary) -> tf.keras.Model:
    ngram_top, _ = ngram_vocab.split_by_frequency(h_params.ngram_freq)
    ngram_keys = ngram_top.tokens()

    documents = tf.keras.layers.Input(shape=(), name='document', dtype=tf.string)
    tokens = Lambda(lambda doc: split_words(doc, extended=True), name='tokens')(documents)
    normals = Lambda(lambda tok: normalize_tokens(tok), name='normals')(tokens)

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

    if 'add' == h_params.att_core:
        features = AdditiveSelfAttention(32, dropout=h_params.att_drop, name='attention')(features)
    elif 'mult' == h_params.att_core:
        features = MultiplicativeSelfAttention(dropout=h_params.att_drop, name='attention')(features)

    space_head = Dense(1, name='space_logits')(features)
    space_head = Activation('sigmoid', dtype='float32', name='space')(space_head)

    token_head = Dense(1, name='token_logits')(features)
    token_head = Activation('sigmoid', dtype='float32', name='token')(token_head)

    sentence_head = Dense(1, name='sentence_logits')(features)
    sentence_head = Activation('sigmoid', dtype='float32', name='sentence')(sentence_head)

    rdw_weight = None
    if h_params.rdw_loss:
        rdw_weight = tf.keras.layers.Input(shape=(None, 1), name='repdivwrap', dtype=tf.float32)

    dense_tokens = ToDense('', mask=False, name='dense_tokens')(tokens)
    model = tf.keras.Model(
        inputs=[documents] + ([rdw_weight] if h_params.rdw_loss else []),
        outputs=[dense_tokens, space_head, token_head, sentence_head]
    )
    if h_params.rdw_loss:
        rdw_loss = tf.keras.layers.Lambda(
            lambda a: tf.keras.losses.MeanAbsoluteError()(tf.zeros_like(a[1]), a[0][:, 1:] - a[0][:, :-1], a[1])
        )([token_head, rdw_weight])
        model.add_loss(rdw_loss)

    return model
