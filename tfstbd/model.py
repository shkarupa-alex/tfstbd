import tensorflow as tf
from nlpvocab import Vocabulary
from tensorflow.keras.layers import Activation, Dense, Embedding, Lambda
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow_addons.layers import CRF
from tensorflow_addons.text import crf_log_likelihood
from tfmiss.keras.layers import AdditiveSelfAttention, MultiplicativeSelfAttention
from tfmiss.keras.layers import CharNgams, Reduction, TemporalConvNet, ToDense, WithRagged, WordShape
from tfmiss.text import split_words
from .input import parse_documents
from .hparam import HParams


def build_model(h_params: HParams, token_vocab: Vocabulary, space_vocab: Vocabulary) -> tf.keras.Model:
    token_keys = token_vocab.split_by_frequency(h_params.ngram_freq)[0].tokens()
    space_keys = space_vocab.split_by_frequency(h_params.ngram_freq)[0].tokens()

    documents = tf.keras.layers.Input(shape=(), name='document', dtype=tf.string)
    tokens, spaces, raws = Lambda(lambda doc: parse_documents(doc, raw_tokens=True), name='parse')(documents)

    token_shapes = WordShape(WordShape.SHAPE_ALL, name='token_shapes')(tokens)
    space_shapes = WordShape(WordShape.SHAPE_LENGTH_NORM, name='space_shapes')(spaces)
    common_shapes = tf.keras.layers.concatenate([token_shapes, space_shapes], name='common_shapes')
    common_shapes = WithRagged(Dense(4, name='shape_projections'))(common_shapes)

    token_ngrams = CharNgams(h_params.ngram_minn, h_params.ngram_maxn, h_params.ngram_self, name='token_ngrams')(tokens)
    token_lookup = StringLookup(vocabulary=token_keys, mask_token=None, name='token_indexes')
    token_indices = token_lookup(token_ngrams)
    token_embeddings = Embedding(
        token_lookup.vocabulary_size(), h_params.ngram_dim, name='token_embeddings')(token_indices)
    token_embeddings = Reduction(h_params.ngram_comb, name='token_reduction')(token_embeddings)


    space_ngrams = CharNgams(h_params.ngram_minn, h_params.ngram_maxn, h_params.ngram_self, name='space_ngrams')(spaces)
    space_lookup = StringLookup(vocabulary=space_keys, mask_token=None, name='space_indexes')
    space_indices = space_lookup(space_ngrams)
    space_embeddings = Embedding(space_lookup.vocabulary_size(), 3, name='space_embeddings')(space_indices)
    space_embeddings = Reduction(h_params.ngram_comb, name='space_reduction')(space_embeddings)

    features = tf.keras.layers.concatenate([common_shapes, token_embeddings, space_embeddings], name='features')
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

    dense_tokens = ToDense('', mask=False, name='dense_tokens')(raws)
    dense_spaces = ToDense('', mask=False, name='dense_spaces')(spaces)

    if h_params.crf_loss:
        labels = tf.keras.layers.Input(shape=(None, 3), name='label', dtype='int32')
        decoded, potentials, length, chain = CRF(3, name='crf')(features)
        crf_loss = Lambda(lambda yplc: -crf_log_likelihood(*yplc)[0], name='crf_loss')([
            labels, potentials, length, chain])

        model = tf.keras.Model(
            inputs=[documents, labels],
            outputs=[dense_tokens, dense_spaces, decoded]
        )
        model.add_loss(crf_loss)  # TODO: reduce?

    else:
        logits = Dense(3, name='logits')(features)
        probs = Activation('softmax', dtype='float32', name='probs')(logits)

        probs_sent = Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1) == 2, 'float32')[..., None])(probs)
        probs_word = Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1) != 0, 'float32')[..., None])(probs)

        model = tf.keras.Model(
            inputs=[documents],
            outputs=[dense_tokens, dense_spaces, probs, probs_sent, probs_word]
        )

    return model
