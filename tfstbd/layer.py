from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.layers.preprocessing.reduction import Reduction as _Reduction


class Reduction(_Reduction):
    def get_config(self):
        config = super(Reduction, self).get_config()
        config.update({
            'reduction': self.reduction,
            'axis': self.axis,
        })

        return config
