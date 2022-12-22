import abc


class ClientBase(abc.ABC):

    @abc.abstractmethod
    def train(self, client_data, model, conf):
        pass

    @abc.abstractmethod
    def test(self, client_data, model, conf):
        pass
