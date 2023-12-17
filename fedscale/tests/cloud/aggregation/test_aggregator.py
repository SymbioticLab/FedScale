import copy
import numpy as np
import tensorflow as tf
import torch

from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.internal.tensorflow_model_adapter import TensorflowModelAdapter
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter


class MockAggregator(Aggregator):
    def __init__(self, model_wrapper):
        self.model_weights = []
        self.model_in_update = 1
        self.tasks_round = 3
        self.model_wrapper = model_wrapper
        self.client_training_results = None


def multiply_weights(weights, factor):
    return {"update_weight": [weights_group * factor for weights_group in weights]}


class TestAggregator:
    def test_update_weight_aggregation_for_keras_model(self):
        x = tf.keras.Input(shape=(2,))
        y = tf.keras.layers.Dense(2, activation="softmax")(
            tf.keras.layers.Dense(4, activation="softmax")(x)
        )
        model = tf.keras.Model(x, y)
        model_adapter = TensorflowModelAdapter(model)
        aggregator = MockAggregator(model_adapter)
        weights = copy.deepcopy(model_adapter.get_weights())
        aggregator.update_weight_aggregation(multiply_weights(weights, 2))
        aggregator.model_in_update += 1
        aggregator.update_weight_aggregation(multiply_weights(weights, 2))
        aggregator.model_in_update += 1
        aggregator.update_weight_aggregation(multiply_weights(weights, 5))
        np.array_equal(
            aggregator.model_wrapper.get_weights(), multiply_weights(weights, 3)
        )

    def test_update_weight_aggregation_for_torch_model(self):
        model = torch.nn.Linear(3, 2)
        model_adapter = TorchModelAdapter(model)
        aggregator = MockAggregator(model_adapter)
        weights = copy.deepcopy(model_adapter.get_weights())
        aggregator.update_weight_aggregation(multiply_weights(weights, 2))
        aggregator.model_in_update += 1
        aggregator.update_weight_aggregation(multiply_weights(weights, 2))
        aggregator.model_in_update += 1
        aggregator.update_weight_aggregation(multiply_weights(weights, 5))
        np.array_equal(
            aggregator.model_wrapper.get_weights(), multiply_weights(weights, 3)
        )
