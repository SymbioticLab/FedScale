# -*- coding: utf-8 -*-

import os
import sys

from customized_client import Customized_Client
from opacus.validators import ModuleValidator

from fedscale.core.execution.executor import Executor
from fedscale.core.logger.execution import *

"""In this example, we only need to change the Client Component we need to import"""

class Customized_Executor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

    def init_model(self):
        """Return the model architecture used in training"""
        model = init_model()
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)

        return model


if __name__ == "__main__":
    executor = Customized_Executor(args)
    executor.run()

