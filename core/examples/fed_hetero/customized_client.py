import logging
import torch
import math
import copy
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import sys, os
from client import Client
from customized_fllibs import Metric
sys.path.insert(1, os.path.join(sys.path[0], '../../'))

class Customized_Client(Client):
    """Basic client component in Federated Learning"""
    def __init__(self, conf):
        super().__init__(conf)
        # TODO: link system profile to here
        self.model_rate = 1
        self.param_idx = None
        self.local_parameters = None

    def make_model_rate(self, conf):
        """get the model scaling rate"""
        if conf.model_split_mode == 'dynamic':
            self.model_rate = np.random.choice(conf.shrinkage)
        elif conf.model_split_mode == 'fix':
            pass
        return


    def split_model(self, global_model, conf):
        """split global model into a sub local model"""
        self.make_model_rate(conf)
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
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(local_parameters)
        self.param_idx = param_idx
        return local_model

    def get_label_split(self, client_data):
        label = torch.tensor(client_data.target)
        label_split = {}
        label_split = torch.unique(label).tolist()
        return label_split

    def train(self, client_data, model, conf):
        # consider only detection task
        label_split = self.get_label_split(client_data)
        clientId = conf.clientId
        logging.ingo(f"Start to split model (CLIENT: {clientId}) ...")
        model = self.split_model(model, conf)
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        device = conf.device
        metric = Metric()
        model = model.to(device=device)
        model.train()
        optimizer =  torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)# make_optimizer(model, lr)
        completed_steps = 0
        while completed_steps < conf.local_steps:
            try:
                if len(client_data) == 0:
                    logging.info(f"Error : data size = 0")
                    break
                for i, input in enumerate(client_data):
                    # input = collate(input)
                    for k in input:
                        input[k] = torch.stack(input[k], 0)
                    input_size = input['img'].size(0)
                    input['label_split'] = torch.tensor(label_split)
                    input = input.to(device=device)
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate('Loss', input, output)
                    completed_steps = completed_steps + 1
            except Exception as ex:
                error_type = ex
                break
        self.local_parameters = model.state_dict()
        model_param = [param.data.cpu().numpy() for param in model.state_dict().values()]
        results = {'clientId':clientId, 'moving_loss': evaluation,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = 0 # disable square_loss

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {evaluation}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['update_weight'] = model_param
        results['wall_duration'] = 0
        results['param_idx'] = self.param_idx
        results['model_rate'] = self.model_rate
        results['label_split'] = label_split
        results['local_parameters'] = self.local_parameters
        return results