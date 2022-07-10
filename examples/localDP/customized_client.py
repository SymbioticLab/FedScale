import logging
import math
import os
import sys

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.autograd import Variable

from fedscale.core.execution.client import Client
from fedscale.core.execution.optimizers import ClientOptimizer


class Customized_Client(Client):
    """
    Basic client component in Federated Learning
    Local differential privacy
    """
    def __init__(self, conf):
        self.optimizer = ClientOptimizer()
        self.privacy_engine = PrivacyEngine()

    def train(self, client_data, model, conf):
        clientId = conf.clientId

        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        device = conf.device

        last_model_params = [p.data.clone() for p in model.parameters()]

        trained_unique_samples = min(len(client_data.dataset), conf.local_steps* conf.batch_size)

        optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss().to(device=device)

        epoch_train_loss = 1e-4

        error_type = None
        completed_steps = 0

        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)

        model, optimizer, client_data = self.privacy_engine.make_private(module = model,
                                                                optimizer = optimizer,
                                                                data_loader = client_data,
                                                                noise_multiplier = conf.noise_factor,
                                                                max_grad_norm =conf.clip_threshold )

        model = model.to(device=device)
        model.train()

        # TODO: One may hope to run fixed number of epochs, instead of iterations
        while completed_steps < conf.local_steps:
            try:
                for data_pair in client_data:

                    (data, target) = data_pair

                    data, target = Variable(data).to(device=device), Variable(target).to(device=device)

                    output = model(data)
                    loss = criterion(output, target)

                    # only measure the loss of the first epoch
                    if completed_steps < len(client_data):
                        if epoch_train_loss == 1e-4:
                            epoch_train_loss = loss.item()
                        else:
                            epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * loss.item()

                    # ========= Define the backward loss ==============
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    completed_steps += 1

                    if completed_steps == conf.local_steps:
                        break

            except Exception as ex:
                error_type = ex
                break

        # Local differential privacy: https://arxiv.org/pdf/2009.03561.pdf
        # Add noise; clip norm of delta weight
        delta_weight = []
        for param in model.parameters():
            delta_weight.append((param.data - last_model_params[len(delta_weight)]))

        # recover model weights
        idx = 0
        for param in model.parameters():
            param.data += delta_weight[idx]
            idx += 1

        state_dicts = model.state_dict()
        model_param = {p:state_dicts[p].data.cpu().numpy() for p in state_dicts}

        eps = self.privacy_engine.get_epsilon(delta=conf.target_delta)
        results = {'clientId':clientId, 'moving_loss': epoch_train_loss, 'localDP_eps': eps,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = math.sqrt(epoch_train_loss)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results
