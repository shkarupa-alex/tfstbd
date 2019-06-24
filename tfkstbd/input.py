from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.lookup_ops import index_table_from_tensor
from .feature import document_features


def train_input(wild_card, buck_bounds, batch_sizes, word_mean, word_std, ngram_minn, ngram_maxn, cycle_length=5):
    token_table = index_table_from_tensor(vocabulary_list=['B', 'J'])
    sentence_table = index_table_from_tensor(vocabulary_list=['J', 'B'])

    def _parse_examples(examples):
        features = document_features(examples['document'], word_mean, word_std, ngram_minn, ngram_maxn)
        features['length'] = examples['length']

        tokens = tf.strings.split(examples['tokens'], sep=',')
        tokens = tf.ragged.map_flat_values(token_table.lookup, tokens)
        tokens = tokens.to_sparse()
        # tokens = tokens.to_tensor(default_value=0)
        # tokens = tf.expand_dims(tokens, axis=-1)

        sentences = tf.strings.split(examples['sentences'], sep=',')
        sentences = tf.ragged.map_flat_values(sentence_table.lookup, sentences)
        sentences = sentences.to_sparse()
        # sentences = sentences.to_tensor(default_value=0)
        # sentences = tf.expand_dims(sentences, axis=-1)

        del features['document']
        del features['length']
        for k, v in features.items():
            features[k] = v.to_sparse()
        #
        # {
        #     'document': documents,
        #     # 'words': tf.sparse.to_dense(words, default_value=''),  # Required to pass in prediction
        #     'word_ngrams': ngrams,
        #     'word_length': length,
        #     'word_nocase': no_case,
        #     'word_lower': lower_case,
        #     'word_upper': upper_case,
        #     'word_title': title_case,
        #     'word_mixed': mixed_case,
        # }

        # return features, sentences
        return features, {'tokens': tokens, 'sentences': sentences}

    with tf.name_scope('input'):
        files = tf.data.Dataset.list_files(wild_card)

        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=cycle_length)
        dataset = dataset.map(
            lambda proto: tf.io.parse_single_example(proto, features={
                'document': tf.io.FixedLenFeature((), tf.string),
                'length': tf.io.FixedLenFeature((), tf.int64),
                'tokens': tf.io.FixedLenFeature((), tf.string),
                'sentences': tf.io.FixedLenFeature((), tf.string),
            }),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.shuffle(max(batch_sizes) * 10)
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            lambda example: tf.cast(example['length'], tf.int32),
            buck_bounds,
            batch_sizes,
        ))
        dataset = dataset.map(_parse_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

# def serve_input_fn(ngram_minn, ngram_maxn):
#     def serving_input_receiver_fn():
#         documents = tf.placeholder(dtype=tf.string, shape=[None], name='documents')
#         features = features_from_documens(documents, ngram_minn, ngram_maxn)
#
#         return tf.estimator.export.ServingInputReceiver(features, documents)
#
#     return serving_input_receiver_fn
