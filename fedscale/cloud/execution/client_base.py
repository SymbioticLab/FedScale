import abc

from fedscale.cloud.internal.model_adapter_base import ModelAdapterBase


class ClientBase(abc.ABC):
    """
    Represents a framework-agnostic FL client that can perform training and evaluation.
    """

    @abc.abstractmethod
    def train(self, client_data, model, conf):
        """
        Perform a training task.
        :param client_data: client training dataset
        :param model: the framework-specific model
        :param conf: job config
        :return: training results
        """
        pass

    @abc.abstractmethod
    def test(self, client_data, model, conf):
        """
        Perform a testing task.
        :param client_data: client evaluation dataset
        :param model: the framework-specific model
        :param conf: job config
        :return: testing results
        """
        pass

    @abc.abstractmethod
    def get_model_adapter(self, model) -> ModelAdapterBase:
        """
        Return framework-specific model adapter.
        :param model: the model
        :return: a model adapter containing the model
        """
        pass
