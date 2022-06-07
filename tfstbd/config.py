import os
from dataclasses import dataclass
from enum import Enum
from keras import optimizers as core_opt
from omegaconf import OmegaConf
from typing import Optional, List
from tensorflow_addons import optimizers as add_opt  # Required to initialize custom optimizers


class InputUnit(Enum):
    WORD = 'word'
    NGRAM = 'ngram'
    BPE = 'bpe'
    CNN = 'cnn'


class NgramSelf(Enum):
    ALWAYS = 'always'
    ALONE = 'alone'


class NgramComb(Enum):
    MIN = 'min'
    MAX = 'max'
    MEAN = 'mean'
    SUM = 'sum'
    PROD = 'prod'
    SQRTN = 'sqrtn'


class SequenceType(Enum):
    LSTM = 'lstm'
    GRU = 'gru'
    TCN = 'tcn'
    QRNN = 'qrnn'


class AttentionType(Enum):
    NONE = 'none'
    Additive = 'add'
    Multiplicative = 'mult'


class LossType(Enum):
    CROSSENTROPY = 'ce'
    BITEMPERED = 'bit'
    SOFT_F1 = 'sf1'


@dataclass
class Config:
    # Input
    input_unit: InputUnit = InputUnit.NGRAM
    max_len: int = 32
    unit_freq: int = 25

    ngram_minn: Optional[int] = 3
    ngram_maxn: Optional[int] = 5
    ngram_self: Optional[NgramSelf] = NgramSelf.ALWAYS
    ngram_comb: Optional[NgramComb] = NgramComb.MEAN

    bpe_size: Optional[int] = 32000
    bpe_chars: Optional[int] = 1000

    cnn_filt: Optional[List[int]] = (32, 32, 64, 128, 256, 512, 1024)
    cnn_kern: Optional[List[int]] = (1, 2, 3, 4, 5, 6, 7)

    # Body
    embed_size: int = 256
    seq_type: SequenceType = SequenceType.LSTM
    seq_units: List[int] = (128,)
    tcn_ksize: int = 2
    tcn_drop: float = 0.1
    att_type: AttentionType = AttentionType.NONE
    att_drop: float = 0.0

    # Train
    bucket_bounds: List[int] = (1,)
    mean_samples: int = 1
    samples_mult: int = 64
    drop_reminder: bool = True
    num_epochs: int = 5
    mixed_fp16: bool = False
    use_jit: bool = False
    train_loss: LossType = LossType.CROSSENTROPY
    loss_bit1: float = 1.
    loss_bit2: float = 1.
    train_optim: str = 'Adam'
    learn_rate: float = 0.012


def build_config(custom):
    default = OmegaConf.structured(Config)

    if isinstance(custom, str) and custom.endswith('.yaml') and os.path.isfile(custom):
        custom = OmegaConf.load(custom)
    else:
        assert isinstance(custom, dict), 'Bad custom config format'
        custom = OmegaConf.create(custom)

    merged = OmegaConf.merge(default, custom)
    OmegaConf.set_readonly(merged, True)

    # noinspection PyTypeChecker
    conf: Config = merged

    # Input
    assert conf.max_len is None or 3 < conf.max_len, 'Bad maximum word length'
    assert 0 < conf.unit_freq, 'Bad minimum unit frequency'
    if InputUnit.NGRAM == conf.input_unit:
        assert 0 < conf.ngram_minn <= conf.ngram_maxn, 'Bad min/max ngram sizes'
    if InputUnit.BPE == conf.input_unit:
        assert 0 < conf.bpe_size, 'Bad BPE vocabulary size'
        assert 0 < conf.bpe_chars, 'Bad BPE chars count'
    if InputUnit.CNN == conf.input_unit:
        assert conf.cnn_kern, 'Bad (empty) CNN kernels list'
        assert len(conf.cnn_kern) == len(conf.cnn_filt), 'Bad CNN filters length'

    # Body
    assert 0 < conf.embed_size, 'Bad embedding size'
    assert conf.seq_units, 'Bad sequence units size'
    if SequenceType.TCN == conf.seq_type:
        assert 0 < conf.tcn_ksize, 'Bad TCN kernel size'
        assert 0. <= conf.tcn_drop <= 1., 'Bad TCN dropout rate'
    if AttentionType.NONE != conf.att_type:
        assert 0. <= conf.att_drop <= 1., 'Bad attention dropout rate'

    # Train
    assert conf.bucket_bounds, 'Bad bocket bounds size'
    assert 0 < conf.mean_samples, 'Bad mean samples size'
    assert 0 < conf.samples_mult, 'Bad samples multiplicator size'
    assert 0 < conf.num_epochs, 'Bad number of epochs'
    if LossType.BITEMPERED != conf.train_loss:
        assert 0. <= conf.loss_bit1 <= 1., 'Bad t1 value for Bi-Tempered loss'
        assert 1. <= conf.loss_bit2, 'Bad t2 value for Bi-Tempered loss'
    assert conf.train_optim, 'Bad train optimizer'
    assert 'ranger' == conf.train_optim.lower() or core_opt.get(conf.train_optim), \
        'Unsupported train optimizer'
    assert 0. < conf.learn_rate, 'Bad learning rate'

    return conf
