import logging
import torch
import math
import copy
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import sys, os
from customized_fllibs import Metric
import config
from resnet_fedhet import resnet18
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from client import Client

class Customized_Client(Client):
    """Basic client component in Federated Learning"""
    def __init__(self, conf):
        super().__init__(conf)
        # TODO: link system profile to here
        self.model_rate = 1
        self.param_idx = None
        self.local_parameters = None

    def make_model_rate(self):
        """get the model scaling rate"""
        if config.cfg['model_split_mode'] == 'dynamic':
            self.model_rate = np.random.choice(config.cfg['shrinkage'])
        elif config.cfg['model_split_mode'] == 'fix':
            pass
        return


    def split_model(self, global_model):
        """split global model into a sub local model"""
        self.make_model_rate()
        global_parameters = global_model.state_dict()
        local_parameters = OrderedDict()
        param_idx = OrderedDict()
        param_idxi = None
        # resnet
        for k, v in global_parameters.items():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                if parameter_type == 'weight':
                    if v.dim() > 1:
                        input_size = v.size(1)
                        output_size = v.size(0)
                        if 'conv1' in k or 'conv2' in k:
                            if param_idxi is None:
                                param_idxi = torch.arange(input_size, device=v.device)
                            input_idx_i = param_idxi
                            scaler_rate = self.model_rate
                            local_output_size = int(np.ceil(output_size * scaler_rate))
                            output_idx_i = torch.arange(output_size, device=v.device)[:local_output_size]
                            param_idxi = output_idx_i
                        elif 'shortcut' in k:
                            input_idx_i = param_idx[k.replace('shortcut', 'conv1')][1]
                            output_idx_i = param_idxi
                        elif 'linear' in k:
                            input_idx_i = param_idxi
                            output_idx_i = torch.arange(output_size, device=v.device)
                        else:
                            raise ValueError('Not valid k')
                        param_idx[k] = (output_idx_i, input_idx_i)
                    else:
                        input_idx_i = param_idxi
                        param_idx[k] = input_idx_i
                else:
                    input_size = v.size(0)
                    if 'linear' in k:
                        input_idx_i = torch.arange(input_size, device=v.device)
                        param_idx[k] = input_idx_i
                    else:
                        input_idx_i = param_idxi
                        param_idx[k] = input_idx_i
            else:
                pass
        for k, v in global_parameters.items():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                if 'weight' in parameter_type:
                    if v.dim() > 1:
                        local_parameters[k] = copy.deepcopy(v[torch.meshgrid(param_idx[k])])
                    else:
                        local_parameters[k] = copy.deepcopy(v[param_idx[k]])
                else:
                    local_parameters[k] = copy.deepcopy(v[param_idx[k]])
            else:
                local_parameters[k] = copy.deepcopy(v)
        local_model = resnet18(model_rate=self.model_rate)
        local_model.load_state_dict(local_parameters)
        self.param_idx = param_idx
        return local_model

    def train(self, client_data, model, conf):
        # consider only detection task
        clientId = conf.clientId
        logging.info(f"Start to split model (CLIENT: {clientId}) ...")
        model = self.split_model(model)
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        device = conf.device
        metric = Metric()
        model = model.to(device=device)
        model.train()
        trained_unique_samples = min(len(client_data.dataset), conf.local_steps * conf.batch_size)
        optimizer =  torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)# make_optimizer(model, lr)
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)
        epoch_train_loss = 1e-4
        error_type = None
        completed_steps = 0
        loss_squre = 0
        completed_steps = 0
        while completed_steps < conf.local_steps:
            try:
                if len(client_data) == 0:
                    logging.info(f"Error : data size = 0")
                    break
                for data_pair in client_data:
                    (data, target) = data_pair
                    data = Variable(data).to(device=device)
                    target = Variable(target).to(device=device)
                    output = model(data)
                    loss = criterion(output, target)
                    loss_list = loss.tolist()
                    loss = loss.mean()
                    temp_loss = sum(loss_list)/float(len(loss_list))
                    loss_squre = sum([l**2 for l in loss_list])/float(len(loss_list))
                    if completed_steps < len(client_data):
                        if epoch_train_loss == 1e-4:
                            epoch_train_loss = temp_loss
                        else:
                            epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * temp_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    completed_steps += 1
                    if completed_steps == conf.local_steps:
                        break

            except Exception as ex:
                error_type = ex
                break
        self.local_parameters = model.state_dict()
        model_param = [param.data.cpu().numpy() for param in model.state_dict().values()]
        results = {'clientId':clientId, 'moving_loss': epoch_train_loss,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = math.sqrt(loss_squre)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['wall_duration'] = 0
        results['model_rate'] = self.model_rate
        results['local_parameters'] = self.local_parameters
        return results