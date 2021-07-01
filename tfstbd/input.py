import tensorflow as tf
from tensorflow.python.data import Dataset
from tfmiss.text import lower_case, normalize_unicode, replace_string, split_chars, split_words, zero_digits
from tfmiss.preprocessing import spaces_after
from tfmiss.training import estimate_bucket_pipeline
from typing import Union
from .dataset import create_dataset
from .hparam import HParams


def raw_dataset(data_dir: str, phase: str, h_params: Union[HParams, None] = None) -> Dataset:
    builder = create_dataset(source_dirs=[], data_dir=data_dir, doc_size=1, num_repeats=1, test_re='')
    dataset = builder.as_dataset(phase)

    if h_params is not None and len(h_params.bucket_bounds) > 1:
        num_samples = h_params.mean_samples * h_params.samples_mult
        buck_bounds, batch_sizes, max_bound = estimate_bucket_pipeline(h_params.bucket_bounds, num_samples)
        dataset = dataset.filter(lambda example: example['length'] < max_bound)
        dataset = dataset.shuffle(max(batch_sizes) * 1000)
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            lambda example: example['length'],
            buck_bounds,
            batch_sizes,
        ))

    return dataset


def vocab_dataset(data_dir: str, h_params: HParams) -> Dataset:
    dataset = raw_dataset(data_dir, 'train', h_params)
    dataset = dataset.map(lambda ex: parse_documents(ex['document']), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def train_dataset(data_dir: str, phase: str, h_params: HParams) -> Dataset:
    def _separate_inputs(examples):
        features = {'document': examples['document']}

        tokens = split_chars(examples['token']) == 'B'
        sentences = split_chars(examples['sentence']) == 'B'
        labels = tf.where(sentences, 2, tf.where(tokens, 1, 0))

        weights = tf.ones_like(labels, dtype='float32').to_tensor(0.)[..., None]
        labels = tf.cast(labels, 'int32').to_tensor(0)[..., None]

        return features, labels, weights

    dataset = raw_dataset(data_dir, phase, h_params)
    dataset = dataset.map(_separate_inputs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def parse_documents(documents, raw_tokens=False):
    tokens = split_words(documents, extended=True)
    tokens, spaces = spaces_after(tokens)

    normals = normalize_unicode(tokens, 'NFKC')
    normals = lower_case(normals)
    normals = zero_digits(normals)

    if raw_tokens:
        return normals, spaces, tokens

    return normals, spaces
