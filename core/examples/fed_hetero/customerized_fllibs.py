import sys, os, logging
sys.path.insert(1, os.path.join(sys.path[0], '../../'))

from argParser import args


def init_model():
    global tokenizer
    
    logging.info("Initializing the model ...")
    
    if args.model == 'resnet_fedhet':
        from resnet_fedhet import resnet18
        model = resnet18
    
    return model
    