# -*- coding: utf-8 -*-

import sys, os
from customized_client import Customized_Client

sys.path.insert(1, os.path.join(sys.path[0], '../../'))

from executor import Executor
from fl_client_libs import args

"""In this example, we only need to change the Client Component we need to import"""

class Customized_Executor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

if __name__ == "__main__":
    executor = Customized_Executor(args)
    executor.run()

