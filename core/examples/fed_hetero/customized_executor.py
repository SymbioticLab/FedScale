from executor import Executor
from customized_client import Customized_Client
import torch
import pickle
import time 

class Customized_Executor(Executor):

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

