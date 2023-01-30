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
    """Basic client component in Federated Learning"""

    def train(self, client_data, model, conf):
        """We flip the label of the malicious client"""
        client_id = conf.client_id

        """1 out of malicious_factor client is malicious"""
        is_malicious = ((client_id+1) % conf.malicious_factor == 0)

        if is_malicious:
            label_mapping = list(range(conf.num_class))
            np.random.seed(client_id)
            np.random.shuffle(label_mapping)

        logging.info(f"Start to train (CLIENT: {client_id}) ...")
        device = conf.device

        last_model_params = [p.data.clone() for p in model.parameters()]
        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(len(client_data.dataset), conf.local_steps* conf.batch_size)

        optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss().to(device=device)

        epoch_train_loss = 1e-4

        error_type = None
        completed_steps = 0

        # TODO: One may hope to run fixed number of epochs, instead of iterations
        while completed_steps < conf.local_steps:
            try:
                for data_pair in client_data:

                    (data, target) = data_pair

                    if is_malicious:
                        for i in range(len(target)):
                            target[i] = label_mapping[target[i]]

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

        # Backdoor paper: https://arxiv.org/pdf/1911.07963.pdf
        # 1. clip norm of delta weight
        delta_weight = []
        for param in model.parameters():
            delta_weight.append(param.data - last_model_params[len(delta_weight)])
        clip_grad_norm_(delta_weight, max_norm=3.)

        # recover model weights
        idx = 0
        for param in model.parameters():
            param.data += delta_weight[idx]
            idx += 1

        state_dicts = model.state_dict()
        model_param = {p:state_dicts[p].data.cpu().numpy() for p in state_dicts}

        results = {'client_id':client_id, 'moving_loss': epoch_train_loss,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = math.sqrt(epoch_train_loss)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {client_id}) completes, {results}, is_malicious: {is_malicious}")
        else:
            logging.info(f"Training of (CLIENT: {client_id}) failed as {error_type}")

        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results

