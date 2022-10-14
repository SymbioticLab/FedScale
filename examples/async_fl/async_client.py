import copy
import logging
import math
import pickle

import torch
from torch.autograd import Variable

from fedscale.core.execution.client import Client
from fedscale.core.execution.optimizers import ClientOptimizer
from fedscale.dataloaders.nlp import mask_tokens


class Client(Client):
    """Basic client component in Federated Learning"""

    def train(self, client_data, model, conf):

        clientId = conf.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        tokenizer, device = conf.tokenizer, conf.device

        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(
            len(client_data.dataset), conf.local_steps * conf.batch_size)

        self.global_model = None
        if conf.gradient_policy == 'fed-prox':
            # could be move to optimizer
            self.global_model = [param.data.clone() for param in model.parameters()]

        prev_model_dict = copy.deepcopy(model.state_dict())
        optimizer = self.get_optimizer(model, conf)
        criterion = self.get_criterion(conf)
        error_type = None

        # NOTE: One may hope to run fixed number of epochs, instead of iterations
        # then replace the following with "while self.completed_steps < conf.local_steps * len(client_data)"
        while self.completed_steps < conf.local_steps:
            try:
                self.train_step(client_data, conf, model, optimizer, criterion)
            except Exception as ex:
                error_type = ex
                break

        state_dicts = model.state_dict()
        # In async, we need the delta_weight only
        model_param = {p: (state_dicts[p] - prev_model_dict[p]).data.cpu().numpy() 
                       for p in state_dicts}
        results = {'clientId': clientId, 'moving_loss': self.epoch_train_loss,
                   'trained_size': self.completed_steps*conf.batch_size, 
                   'success': self.completed_steps == conf.local_steps}

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['utility'] = math.sqrt(
            self.loss_squre)*float(trained_unique_samples)
        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results
