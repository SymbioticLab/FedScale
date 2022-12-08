import os
import sys

from customized_client import Customized_Client
from customized_fllibs import init_model

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.executor import Executor


class Customized_Executor(Executor):

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)


    def get_client_trainer(self, conf):
        return Customized_Client(conf)


    def init_model(self):
        """return PreActivated ResNet18"""
        return init_model()
        

if __name__ == "__main__":
    executor = Customized_Executor(parser.args)
    executor.run()