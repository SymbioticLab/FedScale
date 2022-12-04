import copy
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from resnet_heterofl import resnet18

import fedscale.cloud.config_parser as parser


def init_model():
    global tokenizer
    
    logging.info("Initializing the model ...")
    if parser.args.model == 'resnet_heterofl':
        model = resnet18()
    
    return model
    

def make_param_idx(model, model_rate):
    """
    "HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients"
    Enmao Diao, Jie Ding, Vahid Tarokh,
    ICLR 2021
    """
    param_idxi = None
    param_idx = OrderedDict()
    for k, v in model.state_dict().items():
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
                        scaler_rate = model_rate
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
    return param_idx


def split_model(global_model, model_rate):
    """
    "HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients"
    Enmao Diao, Jie Ding, Vahid Tarokh,
    ICLR 2021
    """
    param_idx = make_param_idx(global_model, model_rate)
    local_parameters = OrderedDict()
    # split global model into a sub local model
    for k, v in global_model.state_dict().items():
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
    return local_parameters

