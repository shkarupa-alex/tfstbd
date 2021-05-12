import tensorflow as tf
from tensorflow.python.data import Dataset
from tfmiss.text import lower_case, normalize_unicode, replace_string, split_chars, split_words, zero_digits
from tfmiss.training import estimate_bucket_pipeline
from typing import Union
from .dataset import create_dataset
from .hparam import HParams


def raw_dataset(data_dir: str, phase: str, h_params: Union[HParams, None] = None) -> Dataset:
    builder = create_dataset(source_dirs=[], data_dir=data_dir, doc_size=1, dash_weight=1, num_repeats=1, test_re='')
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
    def _extract_tokens(examples):
        documents = examples['document']
        tokens = split_words(documents, extended=True)
        normals = normalize_tokens(tokens)

        return normals

    dataset = raw_dataset(data_dir, 'train', h_params)
    dataset = dataset.map(_extract_tokens, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def train_dataset(data_dir: str, phase: str, h_params: HParams) -> Dataset:
    def _separate_inputs(examples):
        features = {'document': examples['document']}
        labels = {
            'space': tf.cast(split_chars(examples['space']) == 'S', 'int32').to_tensor(0)[..., None],
            'token': tf.cast(split_chars(examples['token']) == 'B', 'int32').to_tensor(0)[..., None],
            'sentence': tf.cast(split_chars(examples['sentence']) == 'B', 'int32').to_tensor(0)[..., None]
        }
        weight_ = examples['weight'][..., None]
        weights = {
            'space': tf.where(labels['space'] == 0, h_params.space_weight[0], h_params.space_weight[1]) * weight_,
            'token': tf.where(labels['token'] == 0, h_params.token_weight[0], h_params.token_weight[1]) * weight_,
            'sentence': tf.where(labels['sentence'] == 0, h_params.sent_weight[0], h_params.sent_weight[1]),
        }

        if h_params.rdw_loss:
            repdivwrap = split_chars(examples['repdivwrap']) != 'N'
            repdivwrap = tf.cast(repdivwrap, 'float32').to_tensor(0.)[..., None] * weight_
            features['repdivwrap'] = repdivwrap[:, :-1, :]

        return features, labels, weights

    dataset = raw_dataset(data_dir, phase, h_params)
    dataset = dataset.map(_separate_inputs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def normalize_tokens(tokens):
    tokens = normalize_unicode(tokens, 'NFKC')
    tokens = replace_string(  # accentuation
        tokens,
        ['\u0060', ' \u0301', '\u02CA', '\u02CB', '\u0300', '\u0301'],
        [''] * 6
    )
    tokens = lower_case(tokens)
    tokens = zero_digits(tokens)

    return tokens
