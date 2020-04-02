from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfmiss.training import HParams


def build_hparams(custom):
    if not isinstance(custom, dict):
        raise ValueError('Bad params format')

    params = HParams(
        bucket_bounds=[1],
        mean_samples=1,
        samples_mult=1,
        word_mean=1.,
        word_std=1.,
        ngram_minn=1,
        ngram_maxn=1,
        ngram_freq=2,
        ngram_dim=1,
        ngram_oov=1,
        ngram_self='always',
        ngram_comb='mean',
        space_weight=[1., 1.],
        token_weight=[1., 1.],
        sentence_weight=[1., 1.],
        train_opt='Adam',
        learn_rate=0.05,
    )
    params.set_hparam('bucket_bounds', [])

    params.override_from_dict(custom)

    if not 0 < params.mean_samples:
        raise ValueError('Bad "mean_samples" value')

    if not 0 < params.samples_mult:
        raise ValueError('Bad "samples_mult" value')

    if not 0. < params.word_mean:
        raise ValueError('Bad "word_mean" value')

    if not 0. < params.word_std:
        raise ValueError('Bad "word_std" value')

    if not 0 < params.ngram_minn <= params.ngram_maxn:
        raise ValueError('Bad "ngram_minn" or "ngram_maxn" value')

    if not 1 < params.ngram_freq:
        raise ValueError('Bad "ngram_freq" value')

    if not 0 < params.ngram_dim:
        raise ValueError('Bad "ngram_dim" value')

    if not 0 < params.ngram_oov:
        raise ValueError('Bad "ngram_oov" value')

    if not params.ngram_self in {'asis', 'never', 'always', 'alone'}:
        raise ValueError('Bad "ngram_self" value')

    if not params.ngram_comb in {'sum', 'mean', 'sqrtn'}:
        raise ValueError('Bad "ngram_comb" value')

    if not 2 == len(params.space_weight) and all(w > 0. for w in params.space_weight):
        raise ValueError('Bad "space_weight" value')

    if not 2 == len(params.token_weight) and all(w > 0. for w in params.token_weight):
        raise ValueError('Bad "token_weight" value')

    if not 2 == len(params.sentence_weight) and all(w > 0. for w in params.sentence_weight):
        raise ValueError('Bad "sentence_weight" value')

    if not 0. < params.learn_rate:
        raise ValueError('Bad "learn_rate" value')

    return params
