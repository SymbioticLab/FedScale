import sys, os, logging
from resnet_fedhet import resnet18
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from argParser import args

def init_model():
    global tokenizer
    
    logging.info("Initializing the model ...")
    if args.model == 'resnet_fedhet':
        model = resnet18()
    
    return model
    