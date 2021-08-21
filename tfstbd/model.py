import tensorflow as tf
from keras import layers, models
from nlpvocab import Vocabulary
from tfmiss.keras.layers import AdditiveSelfAttention, MultiplicativeSelfAttention
from tfmiss.keras.layers import CharNgramEmbedding, MapFlat, TemporalConvNet, ToDense, WithRagged, WordShape
from .input import parse_documents
from .hparam import HParams


def build_model(h_params: HParams, token_vocab: Vocabulary, space_vocab: Vocabulary) -> models.Model:
    token_keys = token_vocab.split_by_frequency(h_params.ngram_freq)[0].tokens()
    space_keys = space_vocab.split_by_frequency(h_params.ngram_freq)[0].tokens()

    documents = layers.Input(shape=(), name='document', dtype=tf.string)
    tokens, spaces, raws = layers.Lambda(lambda doc: parse_documents(doc, raw_tokens=True), name='parse')(documents)

    token_shapes = WordShape(WordShape.SHAPE_ALL, name='token_shapes')(raws)
    space_shapes = WordShape(WordShape.SHAPE_LENGTH_NORM, name='space_shapes')(spaces)
    common_shapes = layers.concatenate([token_shapes, space_shapes], name='common_shapes')
    common_shapes = WithRagged(layers.Dense(4, name='shape_projections'))(common_shapes)

    token_embeddings = MapFlat(CharNgramEmbedding(
        vocabulary=token_keys, output_dim=h_params.ngram_dim, minn=h_params.ngram_minn, maxn=h_params.ngram_maxn,
        itself=h_params.ngram_self, reduction=h_params.ngram_comb), name='token_embeddings')(tokens)
    space_embeddings = MapFlat(CharNgramEmbedding(
        vocabulary=space_keys, output_dim=4, minn=h_params.ngram_minn, maxn=h_params.ngram_maxn,
        itself=h_params.ngram_self, reduction=h_params.ngram_comb), name='space_embeddings')(spaces)

    features = layers.concatenate([common_shapes, token_embeddings, space_embeddings], name='features')
    features = ToDense(0.0, mask=True)(features)

    if 'lstm' == h_params.seq_core:
        for i, units in enumerate(h_params.lstm_units):
            features = layers.Bidirectional(
                layers.LSTM(units, return_sequences=True),
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

    logits = layers.Dense(3, name='logits')(features)
    probs = layers.Activation('softmax', dtype='float32', name='probs')(logits)

    model = models.Model(
        inputs=[documents],
        outputs=[dense_tokens, dense_spaces, probs]
    )

    return model
