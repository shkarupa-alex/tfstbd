import json
import os
from tfmiss.training import HParams
from tensorflow.keras import optimizers as core_opt
from tensorflow_addons import optimizers as add_opt  # Required to initialize custom optimizers
from typing import Union


def build_hparams(custom: Union[dict, str]) -> HParams:
    # Create hyperparameters with overrides

    if isinstance(custom, str) and custom.endswith('.json') and os.path.isfile(custom):
        with open(custom, 'r') as file:
            custom = json.loads(file.read())

    assert isinstance(custom, dict), 'Bad hyperparameters format'

    params = HParams(
        bucket_bounds=[1],
        mean_samples=1,
        samples_mult=1,
        ngram_minn=1,
        ngram_maxn=1,
        ngram_freq=2,
        ngram_dim=1,
        ngram_self='always',  # or 'alone'
        ngram_comb='mean',  # or 'sum' or 'min' or 'max'
        seq_core='lstm',  # or 'tcn'
        lstm_units=[1],
        tcn_filters=[1],
        tcn_ksize=2,
        tcn_drop=0.1,
        att_core='none',  # or 'add' or 'mult'
        att_drop=0.0,
        num_epochs=1,
        crf_loss=False,
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
    params.att_core = params.att_core.lower()

    assert all(0 < b for b in params.bucket_bounds), 'Bad "bucket_bounds" value'

    bb_pairs = zip(params.bucket_bounds[:-1], params.bucket_bounds[1:])
    assert all(b0 < b1 for b0, b1 in bb_pairs), 'Bad "bucket_bounds" value'

    assert 0 < params.mean_samples, 'Bad "mean_samples" value'
    assert 0 < params.samples_mult, 'Bad "samples_mult" value'
    assert 0 < params.ngram_minn <= params.ngram_maxn, 'Bad "ngram_minn" or "ngram_maxn" value'
    assert 1 < params.ngram_freq, 'Bad "ngram_freq" value'
    assert 0 < params.ngram_dim, 'Bad "ngram_dim" value'
    assert params.ngram_self in {'always', 'alone'}, 'Bad "ngram_self" value'
    assert params.ngram_comb in {'mean', 'sum', 'min', 'max'}, 'Bad "ngram_comb" value'
    assert params.seq_core in {'lstm', 'tcn'}, 'Bad "seq_core" value'

    if 'lstm' == params.seq_core:
        assert params.lstm_units, 'Bad "lstm_units" value'
        assert all(0 < u for u in params.lstm_units), 'Bad "lstm_units" value'
    else:  # tcn
        assert params.tcn_filters, 'Bad "tcn_filters" value'
        assert all(0 < t for t in params.tcn_filters), 'Bad "tcn_filters" value'

    assert 1 < params.tcn_ksize, 'Bad "tcn_ksize" value'
    assert 0. <= params.tcn_drop, 'Bad "tcn_drop" value'
    assert params.att_core in {'none', 'add', 'mult'}, 'Bad "att_core" value'
    assert 0. <= params.att_drop, 'Bad "att_drop" value'
    assert 0 < params.num_epochs, 'Bad "num_epochs" value'
    assert len(params.train_optim), 'Bad "train_optim" value'

    if 'ranger' != params.train_optim.lower():
        try:
            core_opt.get(params.train_optim)
        except:
            assert False, 'Unsupported "train_optim" value'

    assert 0. < params.learn_rate, 'Bad "learn_rate" value'

    return params
