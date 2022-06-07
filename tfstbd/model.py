import tensorflow as tf
from keras import layers, models
from nlpvocab import Vocabulary
from tfmiss.keras.layers import AdditiveSelfAttention, MultiplicativeSelfAttention
from tfmiss.keras.layers import TemporalConvNet, QRNN, ToDense, WithRagged, WordShape
from tfmiss.keras.layers import WordEmbedding, CharNgramEmbedding, CharBpeEmbedding, CharCnnEmbedding, MapFlat
from .input import parse_documents, RESERVED
from .config import Config, AttentionType, InputUnit, SequenceType


def build_model(config: Config, token_vocab: Vocabulary, space_vocab: Vocabulary) -> models.Model:
    token_keys = token_vocab.split_by_frequency(config.unit_freq)[0].tokens()
    space_keys = space_vocab.split_by_frequency(config.unit_freq)[0].tokens()

    documents = layers.Input(shape=(), name='document', dtype=tf.string)
    tokens, spaces, raws = layers.Lambda(parse_documents, name='parse')(documents)

    token_shapes = WordShape(WordShape.SHAPE_ALL, name='token_shapes')(raws)
    space_shapes = WordShape(WordShape.SHAPE_LENGTH_NORM, name='space_shapes')(spaces)
    common_shapes = layers.concatenate([token_shapes, space_shapes], name='common_shapes')
    common_shapes = WithRagged(layers.Dense(4), name='shape_projections')(common_shapes)

    token_embeddings = MapFlat(_token_embed(config, token_keys), name='token_embeddings')(tokens)
    space_embeddings = MapFlat(_space_embed(config, space_keys), name='space_embeddings')(spaces)

    features = layers.concatenate([common_shapes, token_embeddings, space_embeddings], name='features')
    features = ToDense(0.0, mask=True)(features)

    if SequenceType.LSTM == config.seq_type:
        for i, units in enumerate(config.seq_units):
            features = layers.Bidirectional(
                layers.LSTM(units, return_sequences=True),
                name='lstm_{}'.format(i)
            )(features)
    elif SequenceType.GRU == config.seq_type:
        for i, units in enumerate(config.seq_units):
            features = layers.Bidirectional(
                layers.GRU(units, return_sequences=True),
                name='gru_{}'.format(i)
            )(features)
    elif SequenceType.QRNN == config.seq_type:
        for i, units in enumerate(config.seq_units):
            features = layers.Bidirectional(
                QRNN(units, return_sequences=True),
                name='qrnn_{}'.format(i)
            )(features)
    elif SequenceType.TCN == config.seq_type:
        features = TemporalConvNet(
            config.seq_units, config.tcn_ksize, config.tcn_drop, 'same', name='tcn')(features)
    else:
        raise ValueError('Unknown sequence type')

    if AttentionType.Additive == config.att_type:
        features = AdditiveSelfAttention(32, dropout=config.att_drop, name='attention')(features)
    elif AttentionType.Multiplicative == config.att_type:
        features = MultiplicativeSelfAttention(dropout=config.att_drop, name='attention')(features)

    dense_tokens = ToDense('', mask=False, name='dense_tokens')(raws)
    dense_spaces = ToDense('', mask=False, name='dense_spaces')(spaces)

    logits = layers.Dense(3, name='logits')(features)
    probs = layers.Activation('softmax', dtype='float32', name='probs')(logits)

    model = models.Model(
        inputs=[documents],
        outputs=[dense_tokens, dense_spaces, probs]
    )

    return model


def _token_embed(config, vocabulary):
    common_kwargs = {
        'vocabulary': vocabulary,
        'output_dim': config.embed_size,
        'normalize_unicode': 'NFKC',
        'lower_case': True,
        'zero_digits': True,
        'max_len': config.max_len,
        'reserved_words': RESERVED,
        'show_warning': len(vocabulary) > len(RESERVED)
    }

    if InputUnit.NGRAM == config.input_unit:
        return CharNgramEmbedding(
            minn=config.ngram_minn, maxn=config.ngram_maxn, itself=config.ngram_self.value,
            reduction=config.ngram_comb.value, **common_kwargs)

    if InputUnit.BPE == config.input_unit:
        return CharBpeEmbedding(
            reduction=config.ngram_comb.value, vocab_size=config.bpe_size, max_chars=config.bpe_chars, **common_kwargs)

    if InputUnit.CNN == config.input_unit:
        return CharCnnEmbedding(filters=list(config.cnn_filt), kernels=list(config.cnn_kern), **common_kwargs)

    return WordEmbedding(**common_kwargs)


def _space_embed(config, vocabulary):
    common_kwargs = {
        'vocabulary': vocabulary,
        'output_dim': 4,
        'normalize_unicode': None,
        'lower_case': False,
        'zero_digits': False,
        'max_len': config.max_len,
        'reserved_words': RESERVED,
        'show_warning': len(vocabulary) > len(RESERVED)
    }

    return CharNgramEmbedding(
        minn=config.ngram_minn, maxn=config.ngram_maxn, itself=config.ngram_self.value,
        reduction=config.ngram_comb.value, **common_kwargs)
