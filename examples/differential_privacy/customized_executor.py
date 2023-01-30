# -*- coding: utf-8 -*-

import os
import sys

from customized_client import Customized_Client

from fedscale.cloud.execution.executor import Executor
import fedscale.cloud.config_parser as parser

"""In this example, we only need to change the TorchClient Component we need to import"""

class Customized_Executor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

if __name__ == "__main__":
    executor = Customized_Executor(parser.args)
    executor.run()

