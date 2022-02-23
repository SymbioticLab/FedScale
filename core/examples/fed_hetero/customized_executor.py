import sys, os
from customized_client import Customized_Client
from customerized_fllibs import init_model

sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from executor import Executor

class Customized_Executor(Executor):

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

    def init_model(self):
        """Return the model architecture used in training"""
        return init_model()
