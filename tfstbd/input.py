from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfmiss.training import estimate_bucket_pipeline
from .feature import document_features


def train_input_fn(wild_card, params, cycle_length=5):
    with tf.name_scope('input'):
        files = tf.data.Dataset.list_files(wild_card)

        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=cycle_length)
        dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if len(params.bucket_bounds) > 1:
            num_samples = params.mean_samples * params.samples_mult
            buck_bounds, batch_sizes, max_bound = estimate_bucket_pipeline(params.bucket_bounds, num_samples)
            dataset = dataset.filter(lambda example: example['length'] < max_bound)
            dataset = dataset.shuffle(max(batch_sizes) * 10)
            dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
                lambda example: tf.cast(example['length'], tf.int32),
                buck_bounds,
                batch_sizes,
            ))
        else:
            tf.get_logger().warning('No bucket boundaries provided. Batch size will be reduced to 1')
            dataset = dataset.batch(1)

        dataset = dataset.map(
            lambda examples: _prepare_train_examples(examples, params),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


def _parse_example(proto):
    parsed = tf.io.parse_single_example(proto, features={
        'document': tf.io.FixedLenFeature((), tf.string),
        'length': tf.io.FixedLenFeature((), tf.int64),
        # 'space_labels': tf.io.FixedLenFeature((), tf.string),
        # 'token_labels': tf.io.FixedLenFeature((), tf.string),
        'token_weights': tf.io.VarLenFeature(tf.float32),
        # 'sentence_labels': tf.io.FixedLenFeature((), tf.string),
        'space_ids': tf.io.VarLenFeature(tf.int64),
        'token_ids': tf.io.VarLenFeature(tf.int64),
        'sentence_ids': tf.io.VarLenFeature(tf.int64),
    })

    parsed['token_weights'] = tf.sparse.to_dense(parsed['token_weights'], 0.0)
    parsed['space_ids'] = tf.sparse.to_dense(parsed['space_ids'], 0)
    parsed['token_ids'] = tf.sparse.to_dense(parsed['token_ids'], 0)
    parsed['sentence_ids'] = tf.sparse.to_dense(parsed['sentence_ids'], 0)

    return parsed


def _prepare_train_examples(examples, params):
    # token_labels = tf.strings.split(examples['token_labels'], sep=',')
    # token_labels = token_labels.to_tensor(default_value='B')
    # token_labels = tf.expand_dims(token_labels, axis=-1)
    #
    # for k, v in features.items():
    #     if not isinstance(features[k], tf.RaggedTensor):
    #         continue
    #     features[k] = v.to_sparse()

    features = document_features(
        examples['document'],
        params.word_mean,
        params.word_std,
        params.ngram_minn,
        params.ngram_maxn
    )
    features['length'] = examples['length']

    # del features['document']
    # del features['words']

    labels = {
        'space': examples['space_ids'],
        'token': examples['token_ids'],
        'sentence': examples['sentence_ids']
    }
    weights = {'token': examples['token_weights']}

    return features, labels, weights

# def serve_input_fn(ngram_minn, ngram_maxn):
#     def serving_input_receiver_fn():
#         documents = tf.placeholder(dtype=tf.string, shape=[None], name='documents')
#         features = features_from_documens(documents, ngram_minn, ngram_maxn)
#
#         return tf.estimator.export.ServingInputReceiver(features, documents)
#
#     return serving_input_receiver_fn
