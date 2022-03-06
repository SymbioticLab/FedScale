import sys, os
import pickle
from customized_client import Customized_Client
from customized_fllibs import init_model
from fl_client_libs import select_dataset, tokenizer, time, logging
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from executor import Executor
from argParser import args
from rlclient import RLClient
import torch

class Customized_Executor(Executor):

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)
        self.last_global_model = None

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

    def init_model(self):
        """Return the model architecture used in training"""
        return init_model()

    def training_handler(self, clientId, conf, model=None):
        """Train model given client ids"""

        # load last global model
        s_time = time.time()
        client_model = self.load_global_model() if model is None else model

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
        if args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets, batch_size=conf.batch_size, collate_fn=self.collate_fn)

            client = self.get_client_trainer(conf)
            # need to update model on executor
            train_res, client_model = client.train(client_data=client_data, model=client_model, conf=conf)

            # [Deprecated] we need to get runtime variance for BN, override by state_dict from the coordinator
            # self.model = client_model
        return train_res

if __name__ == "__main__":
    executor = Customized_Executor(args)
    executor.run()