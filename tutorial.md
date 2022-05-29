
  
# Tutorial
 
This tutorial will show you how to set up and start a FL experiment over Femnist dataset using Fedscale.
 
## Preliminary

Check the [instructions](./README.md) to set up your environment
 and [instructions](./dataset/README.md) to download the Femnist dataset.

Please make sure you have entered the correct environment through 
```bash
conda activate fedscale
```

## Create your experiment profile
Go to `./evals/configs/femnist/` directory and modify/create your **[configuration file](./evals/configs/femnist/conf.yml)** to submit your job.


Modify the configurations such as the number of participants per round, the aggregation algorithm, the client optimizer, the training model, etc. based on your need .
 
## Submit your FL job

Go to `./evals/` and use [manager.py](./evals/manager.py)
to submit your FL job.
`manager.py` will automatically launch the `aggregator.py` and `executor.py` to start the FL simulation.
You can either choose to evaluate your large-scale FL experiment over a GPU cluster or test your code on your local machine.
 
 
 ### Test on your local machine by submitting config
 
It is much easy and convenient to first test your code without a GPU cluster. 
First add an argument `- use_cuda:  False` under `job_conf` in your configuration file `evals/configs/femnist/conf.yml` to indicate you are training without using GPU.

Set `ps_ip` and `worker_ips` to be `10.0.0.1` and `10.0.0.1:[x]` by default, where x represent how many executors you want to run on your local machine.

```
cd core/evals/
python manager.py start evals/configs/femnist/conf.yml
```

### Test on your local machine with jupyter notebook
We also provide jupyter notebook [examples](./examples/notebook/) to run your code locally.
You can first start running [server](./examples/notebook/fedscale_demo_server.ipynb), 
and then run the [client](./examples/notebook/fedscale_demo_client.ipynb).
 


### Evaluate on a GPU cluster

Once you have followed the **[instructions](./fedscale/core/README.md)** to set up your experiment cluster, you will be able to submit your FL job to the cluster!
 
Change `ps_ip` and `worker_ips` to the host name of your nodes in the configuration file by `cat \etc\hosts`.
For example, set `10.0.0.2:[4,4]` as one of the `worker_ips`
means launching `4 executors` on each of the first two GPUs on `10.0.0.2` to train your model in a space/time sharing fashion.

Make sure the node you submit the job has access to the computation nodes.
Also make sure you have synchronized the code across all the nodes.
 
```
cd core/evals/
python manager.py submit evals/configs/femnist/conf.yml
```


## Monitor your training progress
 
You can find the job logging `job_name` under the path `log_path` specified in the `conf.yml` file. To check the training loss or test accuracy, you can do:
```
cd core/evals
cat job_name_logging |grep 'Training loss'
cat job_name_logging |grep 'FL Testing'
```
You can also use [Tensorboard](./fedscale/core/README.md#experiment-dashboard) to better visualize the progress.
