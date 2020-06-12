from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfmiss.training import estimate_bucket_pipeline
from .feature import document_features


def train_dataset(wild_card, params, cycle_length=5):
    files = tf.data.Dataset.list_files(wild_card)
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=cycle_length)
    dataset = dataset.map(_parse_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if len(params.bucket_bounds) > 1:
        num_samples = params.mean_samples * params.samples_mult
        buck_bounds, batch_sizes, max_bound = estimate_bucket_pipeline(params.bucket_bounds, num_samples)
        dataset = dataset.filter(lambda _, length: length < max_bound)
        dataset = dataset.shuffle(max(batch_sizes) * 100)
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            lambda _, length: length,
            buck_bounds,
            batch_sizes,
        ))
    else:
        tf.get_logger().warning('No bucket boundaries provided. Batch size will be reduced to 1')
        dataset = dataset.batch(1)

    dataset = dataset.map(
        lambda protos, _: _parse_examples(protos, params),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def _parse_length(proto):
    features = {'length': tf.io.FixedLenFeature((), tf.int64)}
    example = tf.io.parse_single_example(proto, features=features)

    return proto, tf.cast(example['length'], tf.int32)


def _parse_examples(protos, params):
    examples = tf.io.parse_example(protos, features={
        'document': tf.io.FixedLenFeature((), tf.string),
        'token_weights': tf.io.RaggedFeature(tf.float32),
        'space_ids': tf.io.RaggedFeature(tf.int64),
        'token_ids': tf.io.RaggedFeature(tf.int64),
        'sentence_ids': tf.io.RaggedFeature(tf.int64),
    })

    features = document_features(
        examples['document'],
        params.word_mean,
        params.word_std,
        params.ngram_minn,
        params.ngram_maxn
    )

    labels = {
        'space': tf.expand_dims(examples['space_ids'].to_tensor(0), axis=-1),
        'token': tf.expand_dims(examples['token_ids'].to_tensor(0), axis=-1),
        'sentence': tf.expand_dims(examples['sentence_ids'].to_tensor(0), axis=-1)
    }

    weights = {'token': tf.expand_dims(examples['token_weights'].to_tensor(0.), axis=-1)}

    return features, labels, weights
