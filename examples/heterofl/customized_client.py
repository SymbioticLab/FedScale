import os
import sys

import config
from customized_fllibs import split_model
from resnet_heterofl import resnet18

from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.fllibs import Variable, logging, math, np, os, torch


class Customized_Client(TorchClient):
    def __init__(self, conf):
        super().__init__(conf)
        self.model_rate = None
        self.param_idx = None
        self.local_parameters = None


    def make_model_rate(self):
        """get the model scaling rate"""
        if config.cfg['model_split_mode'] == 'dynamic':
            self.model_rate = np.random.choice(config.cfg['shrinkage'])
        elif config.cfg['model_split_mode'] == 'fix':
            for i in range(len(config.cfg['model_rate'])):
                if self.client_id % sum(config.cfg['proportion_of_model']) < \
                    sum(config.cfg['proportion_of_model'][:i+1]):
                    self.model_rate = config.cfg['model_rate'][i]
                    break
        return


    def train(self, client_data, model, conf):
        self.client_id = conf.client_id
        self.make_model_rate()
        logging.info(f"Start to split model (CLIENT: {self.client_id}, MODEL RATE: {self.model_rate}) ...")
        self.local_parameters = split_model(model, self.model_rate)
        self.local_model = resnet18(model_rate=self.model_rate)
        self.local_model.load_state_dict(self.local_parameters)
        logging.info(f"Start to train (CLIENT: {self.client_id}) ...")
        device = conf.device
        self.local_model = self.local_model.to(device=device)
        self.local_model.train(True)
        trained_unique_samples = min(len(client_data.dataset), conf.local_steps * conf.batch_size)
        optimizer =  torch.optim.SGD(self.local_model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)# make_optimizer(model, lr)
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)
        epoch_train_loss = 1e-4
        error_type = None
        completed_steps = 0
        loss_squared = 0
        completed_steps = 0
        while completed_steps < config.cfg['local_epochs']:
            try:
                if len(client_data) == 0:
                    logging.info(f"Error : data size = 0")
                    break
                for data_pair in client_data:
                    (data, target) = data_pair
                    data = Variable(data).to(device=device)
                    target = Variable(target).to(device=device)
                    output = self.local_model(data)
                    loss = criterion(output, target)
                    loss_list = loss.tolist()
                    loss = loss.mean()
                    temp_loss = sum(loss_list)/float(len(loss_list))
                    loss_squared = sum([l**2 for l in loss_list])/float(len(loss_list))
                    if completed_steps < len(client_data):
                        if epoch_train_loss == 1e-4:
                            epoch_train_loss = temp_loss
                        else:
                            epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * temp_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 1)
                    optimizer.step()
                logging.info(f"Client {self.client_id} completes local epoch: {completed_steps}, loss square: {loss_squared}")
                completed_steps += 1

            except Exception as ex:
                error_type = ex
                break
        results = {'client_id':self.client_id, 'moving_loss': epoch_train_loss,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = math.sqrt(loss_squared)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {self.client_id}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {self.client_id}) failed as {error_type}")

        results['wall_duration'] = 0
        results['model_rate'] = self.model_rate
        results['local_parameters'] = self.local_model.state_dict()
        return results