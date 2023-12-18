import abc
from typing import Any
import numpy as np


class ModelAdapterBase(abc.ABC):
    """
    Represents an adapter that operates on a framework-specific model.
    """

    @abc.abstractmethod
    def set_weights(
        self, weights: np.ndarray, is_aggregator=True, client_training_results=None
    ):
        """
        Set the model's weights to the numpy weights array.
        :param weights: numpy weights array
        :param is_aggregator: boolean indicating whether the caller is the aggregator
        :param client_training_results: list of gradients from every clients, for q-fedavg
        """
        pass

    @abc.abstractmethod
    def get_weights(self) -> np.ndarray:
        """
        Get the model's weights as a numpy weights array. Note that it doesn't contain layer names. Rather, index 0
        contains the model's first layer weights, and index N contains the N+1 layer's weights.
        :return: A numpy array
        """
        pass

    @abc.abstractmethod
    def get_model(self) -> Any:
        """
        Get the instantiated framework specific model including the architecture.
        """
        pass
