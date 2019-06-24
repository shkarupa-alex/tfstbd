from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class HParams(collections.namedtuple('HParams', [
    'bucket_bounds', 'batch_rates', 'batch_mult', 'batch_sizes', 'word_mean', 'word_std', 'ngram_minn', 'ngram_maxn',
    'ngram_itself', 'ngram_freq', 'ngram_dim', 'ngram_oov', 'ngram_comb', 'train_opt', 'learn_rate'])):
    pass


def build_hparams(custom):
    assert isinstance(custom, dict)

    params = HParams(
        bucket_bounds=[0],
        batch_rates=[2.],
        batch_mult=0.0,
        batch_sizes=[2],
        word_mean=0.0,
        word_std=0.0,
        ngram_minn=3,
        ngram_maxn=3,
        ngram_itself='ALWAYS',
        ngram_freq=100,
        ngram_dim=3,
        ngram_oov=1,
        ngram_comb='mean',
        train_opt="Adam",
        learn_rate=0.05,
    )

    params = params._replace(**custom)
    params = params._replace(batch_sizes=[
        max(int(round(s * params.batch_mult)), 1)
        for s in params.batch_rates
    ])

    # params.set_hparam('bucket_bounds', [])
    # params.set_hparam('batch_rates', [])
    # params.override_from_dict(custom)
    # params.set_hparam('batch_sizes', [int(round(s)) for s in params.batch_rates])

    assert len(params.bucket_bounds) + 1 == len(params.batch_sizes)
    assert 0 < params.batch_mult
    assert 0 < params.word_mean
    assert 0 < params.word_std
    assert 0 < params.ngram_minn <= params.ngram_maxn
    # TODO

    return params
