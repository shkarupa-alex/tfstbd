from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfmiss.training import HParams
from tensorflow.keras import optimizers as core_opt
from tensorflow_addons import optimizers as add_opt


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
        ngram_self='always',  # or 'asis' or 'never' or 'alone'
        ngram_comb='mean',  # or 'sum' or 'min' or 'max' or 'prod'
        seq_core='lstm',  # or 'tcn'
        lstm_units=[1],
        tcn_filters=[1],
        tcn_ksize=2,
        tcn_drop=0.1,
        tcn_padding='causal',  # or 'same'
        space_weight=[1., 1.],
        token_weight=[1., 1.],
        sentence_weight=[1., 1.],
        num_epochs=1,
        train_optim='Adam',
        learn_rate=0.05,
    )
    params.bucket_bounds = []
    params.lstm_units = []
    params.tcn_filters = []
    params.override_from_dict(custom)

    params.ngram_self = params.ngram_self.lower()
    params.ngram_comb = params.ngram_comb.lower()
    params.seq_core = params.seq_core.lower()
    params.tcn_padding = params.tcn_padding.lower()

    if not all(0 < b for b in params.bucket_bounds):
        raise ValueError('Bad "bucket_bounds" value')
    if not all(b0 < b1 for b0, b1 in zip(params.bucket_bounds[:-1], params.bucket_bounds[1:])):
        raise ValueError('Bad "bucket_bounds" value')

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

    if params.ngram_self not in {'always', 'asis', 'never', 'alone'}:
        raise ValueError('Bad "ngram_self" value')

    if params.ngram_comb not in {'mean', 'sum', 'min', 'max', 'prod'}:
        raise ValueError('Bad "ngram_comb" value')

    if params.seq_core not in {'lstm', 'tcn'}:
        raise ValueError('Bad "seq_core" value')

    if 'lstm' == params.seq_core:
        if not params.lstm_units:
            raise ValueError('Bad "lstm_units" value')
        if not all(0 < u for u in params.lstm_units):
            raise ValueError('Bad "lstm_units" value')
    else:
        if not params.tcn_filters:
            raise ValueError('Bad "tcn_filters" value')
        if not all(0 < t for t in params.tcn_filters):
            raise ValueError('Bad "tcn_filters" value')

    if not 1 < params.tcn_ksize:
        raise ValueError('Bad "tcn_ksize" value')

    if not 0. <= params.tcn_drop:
        raise ValueError('Bad "tcn_drop" value')

    if params.tcn_padding not in {'causal', 'same'}:
        raise ValueError('Bad "tcn_padding" value')

    if 2 != len(params.space_weight) or any(0. >= w for w in params.space_weight):
        raise ValueError('Bad "space_weight" value')

    if 2 != len(params.token_weight) or any(0. >= w for w in params.token_weight):
        raise ValueError('Bad "token_weight" value')

    if 2 != len(params.sentence_weight) or any(0. >= w for w in params.sentence_weight):
        raise ValueError('Bad "sentence_weight" value')

    if not 0 < params.num_epochs:
        raise ValueError('Bad "num_epochs" value')

    if not len(params.train_optim):
        raise ValueError('Bad "train_optim" value')
    elif 'ranger' != params.train_optim.lower():
        try:
            core_opt.get(params.train_optim)
        except:
            raise ValueError('Unsupported "train_optim" value')

    if not 0. < params.learn_rate:
        raise ValueError('Bad "learn_rate" value')

    return params
