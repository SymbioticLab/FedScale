
  
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
Go to `../benchmark/configs/femnist/` directory and modify/create your **[configuration file](../benchmark/configs/femnist/conf.yml)** to submit your job.


Modify the configurations such as the number of participants per round, the aggregation algorithm, the client optimizer, the training model, etc. based on your need.
 
## Submit Your FL Job

Go to `../benchmark/` and use [manager.py](../benchmark/manager.py)
to submit your FL job.
`manager.py` will automatically launch the `aggregator.py` and `executor.py` to start the FL evaluation.
You can either choose to evaluate your large-scale FL experiment over a GPU cluster or test your code on your local machine.
 
 
### Test on Your Local Machine by Submitting Config
 
It is more convenient to first test your code without a GPU cluster. 
First add an argument `- use_cuda:  False` under `job_conf` in your configuration file `benchmark/configs/femnist/conf.yml` if you are training without using any GPU.

Set `ps_ip` and `worker_ips` to be `localhost` and `localhost:[x]` by default, where x represent how many executors you want to run on your local machine.
Go to `./benchmark/` and run the following command to start your FL job:
```
python manager.py start configs/femnist/conf.yml
```

### Test on Your Local Machine with Jupyter Notebook
We also provide jupyter notebook [examples](../benchmark/examples/notebook/) to run your code locally.
You can first start running [server](../benchmark/examples/notebook/fedscale_demo_server.ipynb), 
and then run the [client](../benchmark/examples/notebook/fedscale_demo_client.ipynb).
 


### Evaluate on a GPU Cluster

Once you have followed the **[instructions](../fedscale/core/README.md)** to set up your experiment cluster, you will be able to submit your FL job to the cluster!
 
Change `ps_ip` and `worker_ips` to the host name of your nodes in the configuration file by `cat \etc\hosts`.
For example, set `10.0.0.2:[4,4]` as one of the `worker_ips`
means launching `4 executors` on each of the first two GPUs on `10.0.0.2` to train your model in a space/time sharing fashion.

Make sure the node you submit the job has access to the computation nodes.
Also make sure you have synchronized the code across all the nodes.
Go to `../benchmark/` and run the following command to start your FL job:

```
python manager.py submit configs/femnist/conf.yml
```


## Monitor Your Training Progress
 
You can find the job logging `job_name` under the path `log_path` specified in the `conf.yml` file. To check the training loss or test accuracy, you can do:
```
cat job_name_logging |grep 'Training loss'
cat job_name_logging |grep 'FL Testing'
```
You can also use [Tensorboard](../fedscale/core/README.md#experiment-dashboard) to better visualize the progress.
