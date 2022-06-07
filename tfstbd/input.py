import tensorflow as tf
from tensorflow.python.data import Dataset
from tfmiss.text import lower_case, normalize_unicode, replace_regex, split_chars, split_words, zero_digits
from tfmiss.preprocessing import spaces_after
from tfmiss.training import estimate_bucket_pipeline
from typing import Union
from .dataset import create_dataset
from .config import Config

UNK_MARK = '[UNK]'
RESERVED = [UNK_MARK]


def raw_dataset(data_dir: str, phase: str, config: Union[Config, None] = None) -> Dataset:
    builder = create_dataset(source_dirs=[], data_dir=data_dir, doc_size=1, num_repeats=1, test_re='')
    dataset = builder.as_dataset(phase)

    if config is not None and len(config.bucket_bounds) > 1:
        num_samples = config.mean_samples * config.samples_mult
        buck_bounds, batch_sizes, max_bound = estimate_bucket_pipeline(config.bucket_bounds, num_samples)
        dataset = dataset.filter(lambda example: example['length'] < max_bound)
        dataset = dataset.shuffle(max(batch_sizes) * 1000)
        dataset = dataset.bucket_by_sequence_length(
            lambda example: example['length'], buck_bounds, batch_sizes, drop_remainder=config.drop_reminder)

    return dataset


def vocab_dataset(data_dir: str, config: Config) -> Dataset:
    dataset = raw_dataset(data_dir, 'train', config)
    dataset = dataset.map(lambda ex: parse_documents(ex['document']), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def train_dataset(data_dir: str, phase: str, config: Config) -> Dataset:
    def _separate_inputs(examples):
        features = {'document': examples['document']}

        tokens = split_chars(examples['token']) == 'B'
        sentences = split_chars(examples['sentence']) == 'B'
        labels = tf.maximum(tf.cast(sentences, 'int32') * 2, tf.cast(tokens, 'int32'))

        weights = tf.ones_like(labels, dtype='float32').to_tensor()[..., None]
        labels = tf.cast(labels, 'int32').to_tensor()[..., None]

        return features, labels, weights

    dataset = raw_dataset(data_dir, phase, config)
    dataset = dataset.map(_separate_inputs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def parse_documents(documents):
    tokens = split_words(documents, extended=True)
    tokens, spaces = spaces_after(tokens)

    normals = normalize_unicode(tokens, 'NFKC')
    normals = lower_case(normals)
    normals = zero_digits(normals)

    accent = '[\u0060\u0301\u02CA\u02CB\u0300\u0301]+'
    normals = replace_regex(normals, ['(\\S)' + accent, accent + '(\\S)'], ['\\1', '\\1'])

    return normals, spaces, tokens
