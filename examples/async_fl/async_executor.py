# -*- coding: utf-8 -*-
from fedscale.core.logger.execution import *
import pickle

from fedscale.core.execution.executor import Executor
from fedscale.core.execution.rlclient import RLClient


class AsyncExecutor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super().__init__(args)
        self.temp_model_path_version = lambda round: os.path.join(
            logDir, 'model_' + str(round) + '.pth.tar')

    def update_model_handler(self, model):
        """Update the model copy on this executor"""
        self.model = model
        self.round += 1

        # Dump latest model to disk
        with open(self.temp_model_path_version(self.round), 'wb') as model_out:
            logging.info(f"Received latest model saved at {self.temp_model_path_version(self.round)}")
            pickle.dump(self.model, model_out)

    def load_global_model(self, round):
        # load last global model
        if round == -1:
            with open(self.temp_model_path, 'rb') as model_in:
                model = pickle.load(model_in)
        else:
            round = min(round, self.round)
            with open(self.temp_model_path_version(round), 'rb') as model_in:
                model = pickle.load(model_in)
        return model

    def training_handler(self, clientId, conf, model=None):
        """Train model given client ids"""

        # load last global model
        client_model = self.load_global_model(-1) if model is None \
            else self.load_global_model(model)

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
        if args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets,
                                         batch_size=conf.batch_size, args=self.args,
                                         collate_fn=self.collate_fn
                                         )

            client = self.get_client_trainer(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)

        return train_res


if __name__ == "__main__":
    executor = AsyncExecutor(args)
    executor.run()
