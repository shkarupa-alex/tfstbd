import os
import tensorflow as tf
from nlpvocab import Vocabulary
from ..config import build_config
from ..input import train_dataset
from ..model import build_model


class TestBuildModel(tf.test.TestCase):
    def tearDown(self):
        os.environ['TF_XLA_FLAGS'] = ''
        tf.config.optimizer.set_jit(False)
        super(TestBuildModel, self).tearDown()

    def test_lstm(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
        config = build_config({
            'bucket_bounds': [10, 20],
            'mean_samples': 16,
            'samples_mult': 10,
            'ngram_minn': 2,
            'ngram_maxn': 6,
            'unit_freq': 2,
            'seq_units': [1],
            'att_type': 'Multiplicative',
            'drop_reminder': False,
        })
        dataset = train_dataset(data_path, 'train', config)
        tokens = Vocabulary({
            '[UNK]': 11, '<кто>': 10, '<знает>': 2, '<зна': 4, 'ает>': 4, '<,>': 20, '<что>': 15, '<он>': 10,
            '<там>': 5, '<думал>': 1, '!..': 1
        })
        spaces = Vocabulary({'< >': 100, '<\n>': 10})
        model = build_model(config, tokens, spaces)

        model.compile(
            optimizer='Adam',
            loss=[None, None, 'sparse_categorical_crossentropy'],
            run_eagerly=False,
        )
        model.summary()
        history = model.fit(dataset, epochs=1, steps_per_epoch=2)
        self.assertGreater(len(history.history['loss']), 0)


if __name__ == "__main__":
    tf.test.main()
