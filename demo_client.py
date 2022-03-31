import torch
import logging
import math
from torch.autograd import Variable
import numpy as np

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], './FedScale/core'))
from fedscale.core.client import Client
from fedscale.core.executor import Executor
from fedscale.core.fl_client_libs import args
### On CPU
args.use_cuda = "False"
Demo_Executor = Executor(args)
Demo_Executor.run()