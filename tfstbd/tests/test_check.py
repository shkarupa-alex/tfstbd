import os
import tensorflow as tf
from ..check import check_dataset


class TestCheckDataset(tf.test.TestCase):
    def test_normal(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
        check_dataset(data_path)


if __name__ == "__main__":
    tf.test.main()
