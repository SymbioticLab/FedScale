import sys, os
from fedscale.core.client import Client
from fedscale.core.aggregator import Aggregator
from fedscale.core.fl_client_libs import args
Demo_Aggregator = Aggregator(args)
### On CPU
args.use_cuda = "False"
Demo_Aggregator.run()