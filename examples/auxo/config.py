import yaml, os
global auxo_config


# Get the value of the FEDSCALE_HOME environment variable
fedscale_home = os.environ.get('FEDSCALE_HOME')

# Check if the environment variable is set
if fedscale_home is not None:
    config_path = os.path.join(fedscale_home, 'examples', 'auxo', 'config.yml')

    # Now, open the file using the constructed path
    with open(config_path, 'r') as f:
        auxo_config = yaml.load(f, Loader=yaml.FullLoader)
else:
    print("FEDSCALE_HOME environment variable is not set.")