import yaml
import sys

# TODO: on the long run, replace this generator with Helm
def generate_aggr_template(dict, path):
    """ Generate YAML template for aggregator deployment and save it in the given file
    """
    config = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": dict["pod_name"]
        },
        "spec": {
            "restartPolicy": "OnFailure",
            "containers": [{
                "name": "fedscale-aggr",
                "image": "fedscale/fedscale-aggr",
                "imagePullPolicy": "Always",
                "ports": [
                    {
                        "containerPort": 30000
                    }
                ],
                "volumeMounts": [{
                    "mountPath": "FedScale/benchmark",
                    "name": "benchmark",
                    "readOnly": False
                    }
                ]
                }
            ],
            "volumes": [{
                "name": "benchmark",
                "hostPath": {
                    # directory location on host, assume it exists on all nodes
                    "path": dict["data_path"],
                    "type": "Directory"
                }
            }]
        }
    }
    with open(path, "w") as f:
        f.write(yaml.dump(config, default_flow_style=False))

    # validate the yaml file in case weird things happen
    with open(path, "r") as f:
        try:
            yaml.safe_load(f)
            return config
        except:
            sys.exit("Generated YAML is not valid, aborting...")


def generate_exec_template(dict, path):
    """ Generate YAML template for executor deployment and save it in the given file
    """
    config = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": dict["pod_name"]
        },
        "spec": {
            "restartPolicy": "OnFailure",
            "containers": [{
                "name": "fedscale-exec",
                "image": "fedscale/fedscale-exec",
                "imagePullPolicy": "Always",
                "ports": [
                    {
                        "containerPort": 32000
                    }
                ],
                "volumeMounts": [{
                    "mountPath": "FedScale/benchmark",
                    "name": "benchmark",
                    "readOnly": False
                    }
                ]
                }
            ],
            "volumes": [{
                "name": "benchmark",
                "hostPath": {
                    # directory location on host, assume it exists on all nodes
                    "path": dict["data_path"],
                    "type": "Directory"
                }
            }]
        }
    }
    if dict["use_cuda"]: 
        config["spec"]["containers"][0]["resources"] = {
            "limits": {
                "nvidia.com/gpu": 1 # request 1 GPU
            }
        }
        config["spec"]["tolerations"] = [{
            "key": "nvidia.com/gpu",
            "operator": "Exists",
            "effect": "NoSchedule"
        }]

    with open(path, "w") as f:
        f.write(yaml.dump(config, default_flow_style=False))
        
    # validate the yaml file in case weird things happen
    with open(path, "r") as f:
        try:
            yaml.safe_load(f)
            return config
        except:
            sys.exit("Generated YAML is not valid, aborting...")

# if __name__ == "__main__":
#     generate_aggr_template({"pod_name": "fedscale-aggr-pod", "data_path": "/users/yilegu/benchmark"}, "generated_aggr.yaml")
#     generate_exec_template({"pod_name": "fedscale-exec-pod", "data_path": "/users/yilegu/benchmark"}, "generated_exec.yaml")