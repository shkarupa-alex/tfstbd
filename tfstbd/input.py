from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfmiss.text import split_words
from tfmiss.training import estimate_bucket_pipeline
from .model import _normalize_tokens


def train_dataset(wild_card, h_params):
    dataset = _raw_dataset(wild_card, h_params)
    dataset = dataset.map(_separate_inputs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def vocab_dataset(wild_card, h_params):
    dataset = _raw_dataset(wild_card, h_params)
    dataset = dataset.map(_extract_tokens, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def _raw_dataset(wild_card, h_params):
    files = tf.data.Dataset.list_files(wild_card)
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if len(h_params.bucket_bounds) > 1:
        num_samples = h_params.mean_samples * h_params.samples_mult
        buck_bounds, batch_sizes, max_bound = estimate_bucket_pipeline(h_params.bucket_bounds, num_samples)
        dataset = dataset.filter(lambda example: example['length'] < max_bound)
        dataset = dataset.shuffle(max(batch_sizes) * 100)
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            lambda example: example['length'],
            buck_bounds,
            batch_sizes,
        ))
    else:
        tf.get_logger().warning('No bucket boundaries provided. Batch size will be reduced to 1')
        dataset = dataset.batch(1)

    return dataset


def _parse_example(protos):
    example = tf.io.parse_single_example(protos, features={
        'document': tf.io.FixedLenFeature((), tf.string),
        'length': tf.io.FixedLenFeature((), tf.int64),

        'space': tf.io.RaggedFeature(tf.int64),
        'token': tf.io.RaggedFeature(tf.int64),
        'sentence': tf.io.RaggedFeature(tf.int64),

        'token_weight': tf.io.RaggedFeature(tf.float32),
    })
    example['length'] = tf.cast(example['length'], tf.int32)
    example['space'] = tf.expand_dims(example['space'], axis=-1)
    example['token'] = tf.expand_dims(example['token'], axis=-1)
    example['sentence'] = tf.expand_dims(example['sentence'], axis=-1)

    return example


def _separate_inputs(examples):
    features = {'documents': examples['document']}
    labels = {
        'space': examples['space'],
        'token': examples['token'],
        'sentence': examples['sentence']
    }
    weights = {'token': examples['token_weight']}

    return features, labels, weights


def _extract_tokens(examples):
    documents = examples['document']
    tokens = split_words(documents, extended=True)
    normals = _normalize_tokens(tokens)

    return normals
