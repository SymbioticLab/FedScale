import logging
import math
import os
import sys

import numpy as np
import torch
from clip_norm import clip_grad_norm_
from torch.autograd import Variable

from fedscale.cloud.execution.torch_client import TorchClient


class Customized_Client(TorchClient):
    """
    Basic client component in Federated Learning
    Local differential privacy
    """

    def train(self, client_data, model, conf):

        client_id = conf.client_id
        logging.info(f"Start to train (CLIENT: {client_id}) ...")
        tokenizer, device = conf.tokenizer, conf.device
        last_model_params = [p.data.clone() for p in model.parameters()]

        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(
            len(client_data.dataset), conf.local_steps * conf.batch_size)
        self.global_model = None

        if conf.gradient_policy == 'fed-prox':
            # could be move to optimizer
            self.global_model = [param.data.clone() for param in model.parameters()]

        optimizer = self.get_optimizer(model, conf)
        criterion = self.get_criterion(conf)
        error_type = None

        # TODO: One may hope to run fixed number of epochs, instead of iterations
        while self.completed_steps < conf.local_steps:

            try:
                self.train_step(client_data, conf, model, optimizer, criterion)
            except Exception as ex:
                error_type = ex
                break

        delta_weight = []
        for param in model.parameters():
            delta_weight.append((param.data.cpu() - last_model_params[len(delta_weight)]))

        clip_grad_norm_(delta_weight, max_norm=conf.clip_threshold)

        # recover model weights
        idx = 0
        for param in model.parameters():
            param.data = last_model_params[idx] + delta_weight[idx]
            idx += 1
        sigma = conf.noise_factor * conf.clip_threshold
        state_dicts = model.state_dict()
        model_param = {p:  np.asarray(state_dicts[p].data.cpu().numpy() + \
            torch.normal(mean=0, std=sigma, size=state_dicts[p].data.shape).cpu().numpy()) for p in state_dicts}


        results = {'client_id': client_id, 'moving_loss': self.epoch_train_loss,
                   'trained_size': self.completed_steps*conf.batch_size, 'success': self.completed_steps > 0}
        results['utility'] = math.sqrt(
            self.loss_squared)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {client_id}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {client_id}) failed as {error_type}")

        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results
