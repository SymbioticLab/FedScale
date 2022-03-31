import torch
import logging
import math
from torch.autograd import Variable
import numpy as np

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../../'))

from client import Client

class Customized_Client(Client):
    """Basic client component in Federated Learning"""

    def train(self, client_data, model, conf):
        """We flip the label of the malicious client"""
        clientId = conf.clientId

        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        device = conf.device

        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(len(client_data.dataset), conf.local_steps* conf.batch_size)

        optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss().to(device=device)

        epoch_train_loss = 1e-4

        error_type = None
        completed_steps = 0

        targets = torch.zeros(32, dtype=torch.long)
        for i in range(len(targets)):
            targets[i] = 0

        # TODO: One may hope to run fixed number of epochs, instead of iterations
        while completed_steps < conf.local_steps:
            #(data, target) = data_pair
            data, target = torch.rand(32, 3, 256, 256, device=device), targets.to(device)
            # data, target = Variable(data).to(device=device), Variable(target).to(device=device)

            output = model(data)
            loss = criterion(output, target)

            # only measure the loss of the first epoch
            epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * loss.item()

            # ========= Define the backward loss ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            completed_steps += 1

            if completed_steps == conf.local_steps:
                break

        model_param = [param.data.cpu().numpy() for param in model.state_dict().values()]

        results = {'clientId':clientId, 'moving_loss': epoch_train_loss,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = math.sqrt(epoch_train_loss)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results

