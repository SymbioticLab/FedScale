from typing import List

import numpy as np
import tensorflow as tf

from fedscale.cloud.internal.model_wrapper_base import ModelWrapperBase


class TensorflowModelWrapper(ModelWrapperBase):
    def __init__(self, model: tf.keras.Model):
        self.model = model

    def set_weights(self, weights: List[np.ndarray]):
        for i, layer in enumerate(self.model.layers):
            if layer.trainable:
                layer.set_weights(weights[i])

    def get_weights(self) -> List[np.ndarray]:
        return [np.asarray(layer.get_weights()) for layer in self.model.layers if layer.trainable]

    def get_model(self):
        return self.model
