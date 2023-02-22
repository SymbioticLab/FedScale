from typing import List

import numpy as np
import tensorflow as tf

from fedscale.cloud.internal.model_adapter_base import ModelAdapterBase


class TFLiteModelAdapter(ModelAdapterBase):
    def __init__(self, model: tf.keras.Model):
        self.model = model

    def set_weights(self, weights: List[np.ndarray]):
        for var, weight in zip(self.model.weights, weights):
            var.assign(weight)

    def get_weights(self) -> List[np.ndarray]:
        return [np.asarray(var.read_value()) for var in self.model.weights]

    def get_model(self):
        return self.model
