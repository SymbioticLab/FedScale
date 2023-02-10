
  
# Running an FL Experiment
 
This tutorial will show you how to set up and start an FL experiment to train ShuffleNet on the FEMNIST dataset using FedScale.
 
## Preliminary

Check the [instructions](../README.md) to set up your environment
 and [instructions](../benchmark/dataset/README.md) to download the FEMNIST dataset.

Please make sure you are using the correct environment.
```bash
conda activate fedscale
```

## Create Your Experiment Profile
Go to `benchmark/configs/femnist/` directory and modify/create your **[configuration file](../benchmark/configs/femnist/conf.yml)** to submit your job.


Modify the configurations such as the number of participants per round, the aggregation algorithm, the client optimizer, the training model, etc. based on your need.
 
## Submit Your FL Job

Use `fedscale driver submit [conf_yml_path]` (Or `python docker/driver.py submit`) to submit your FL job. It will automatically launch the `aggregator.py` and `executor.py` to start the FL evaluation.
You can either choose to evaluate your large-scale FL experiment over a GPU cluster or test your code on your local machine.

To stop your job:
```
fedscale driver stop [job_name]
# Or python docker/driver.py stop [job_name] (specified in the yml config)
```

### Test on Your Local Machine by Submitting Config
 
It is more convenient to first test your code without a GPU cluster. 
First add an argument `- use_cuda:  False` under `job_conf` in your configuration file `benchmark/configs/femnist/conf.yml` if you are training without using any GPU.

Set `ps_ip` and `worker_ips` to be `localhost` and `localhost:[x]` by default, where x represent how many executors you want to run on your local machine.
Then run the following command to start your FL job:
```
python driver.py start benchmark/configs/femnist/conf.yml
```

### Test on Your Local Machine with Jupyter Notebook
We also provide jupyter notebook [examples](../examples/notebook/) to run your code locally.
You can first start running [server](../examples/notebook/fedscale_demo_server.ipynb), 
and then run the [client](../examples/notebook/fedscale_demo_client.ipynb).
 


### Evaluate on a GPU Cluster

Once you have followed the **[instructions](../fedscale/cloud/README.md)** to set up your experiment cluster, you will be able to submit your FL job to the cluster!
 
Change `ps_ip` and `worker_ips` to the host name of your nodes in the configuration file by `cat \etc\hosts`.
For example, set `10.0.0.2:[4,4]` as one of the `worker_ips`
means launching `4 executors` on each of the first two GPUs on `10.0.0.2` to train your model in a space/time sharing fashion.

Make sure the node you submit the job has access to the computation nodes.
Also make sure you have synchronized the code across all the nodes.
Then run the following command to submit your FL job:

```
fedscale driver submit [conf_yml_path] 
# Or python docker/driver.py submit benchmark/configs/femnist/conf.yml
```

## Monitor Your Training Progress
 
You can find the job logging `job_name` under the path `log_path` specified in the **[configuration file](../benchmark/configs/femnist/conf.yml)**. To check the training loss or test accuracy, you can do:
```
cat job_name_logging |grep 'Training loss'
cat job_name_logging |grep 'FL Testing'
```

You can also use **Tensorboard** or **WandB** to better visualize the progress, please refer to [Dashboard Tutorial](../fedscale/cloud/README.md#dashboard) for more details.


## K8S/Docker Deployment
We also support deploying FL experiments using k8s/docker, please follow instructions in  **[Containerized FedScale Tutorial](../docker/README.md)**.
