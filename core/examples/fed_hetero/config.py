import yaml

global cfg
if 'cfg' not in globals():
    with open('FedScale/core/examples/fed_hetero/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)