import sys, os, logging
from resnet_fedhet import resnet18
import torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from argParser import args

def init_model():
    global tokenizer
    
    logging.info("Initializing the model ...")
    if args.model == 'resnet_fedhet':
        model = resnet18()
    
    return model

def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    else:
        raise ValueError('Not valid input type')
    return output

def Accuracy(output, target, topk=1):
    with torch.no_grad():
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc

def Perplexity(output, target):
    with torch.no_grad():
        ce = F.cross_entropy(output, target)
        perplexity = torch.exp(ce).item()
    return perplexity

class Metric(object):
    def __init__(self):
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Local-Loss': (lambda input, output: output['loss'].item()),
                       'Global-Loss': (lambda input, output: output['loss'].item()),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['score'], input['label'])),
                       'Local-Accuracy': (lambda input, output: recur(Accuracy, output['score'], input['label'])),
                       'Global-Accuracy': (lambda input, output: recur(Accuracy, output['score'], input['label'])),
                       'Perplexity': (lambda input, output: recur(Perplexity, output['score'], input['label'])),
                       'Local-Perplexity': (lambda input, output: recur(Perplexity, output['score'], input['label'])),
                       'Global-Perplexity': (lambda input, output: recur(Perplexity, output['score'], input['label']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation