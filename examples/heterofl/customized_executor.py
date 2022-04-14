import sys, os
from customized_client import Customized_Client
from customized_fllibs import init_model
sys.path.insert(1, os.path.join(sys.path[0], '../../fedscale/core'))
from executor import Executor
from fl_client_libs import args

class Customized_Executor(Executor):

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)


    def get_client_trainer(self, conf):
        return Customized_Client(conf)


    def init_model(self):
        """return PreActivated ResNet18"""
        return init_model()
        

if __name__ == "__main__":
    executor = Customized_Executor(args)
    executor.run()