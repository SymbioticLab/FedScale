
  
# Tutorial
 
This tutorial will show you how to set up and start a FL experiment over OpenImg dataset using Fedscale.
 
## Environment

Run the following commands to install FedScale.

```
git clone https://github.com/SymbioticLab/FedScale
cd FedScale
source install.sh
```
 
Please make sure you have enter the correct environment through 
```bash
# bash
conda activate fedscale
```

 
## Download OpenImg dataset
To download the OpenImg dataset, go to `./dataset`:
 
```
cd dataset/
bash download.sh -o
```

### Exploring the data
Under `data/openImg/client_data_mapping`, there is a list of `csv` files that map every image with its label to an owner. 
There are 13,771 unique clients who contain 1.3M samples in total in the OpenImg dataset.

```
cd data/openImg/client_data_mapping
head train.csv
```
| client_id | sample_path | label_name | label_id |
| ------ | ------ | ------ | ------ |
| 0 | 1ea021de60b3cd89___m_09j2d.jpg | _m_09j2d | 1 |
| 0 | cae40be4017c90fd___m_09j2d.jpg | _m_09j2d | 1 |
| 0 | fd30ab5d0338b876___m_09j2d.jpg | _m_09j2d | 1 |
 
### Exploring heterogeneity

Let's understand how client-level heterogeneity looks like in the OpenImg dataset.
We select and show few images in *people* label from different clients.
 
![](./figures/client1.png)
> Some images of *people* label from client 1.
 
![](./figures/client10.png)
> Some images of *people* label from client 2.


## Create your experiment profile
 

Go to `./core/evals/configs/openimage/` directory and modify/create your **[configuration file](https://github.com/SymbioticLab/FedScale/blob/master/core/evals/configs/openimage/conf.yml)** to submit your job.

Modify the configurations such as the number of participants per round, the aggregation algorithm, the client optimizer, the training model, etc. based on your need .
 
## Submit your FL job

Go to `./core/evals/` and use [manager.py](https://github.com/SymbioticLab/FedScale/blob/master/core/evals/manager.py)
to submit your FL job.
`manager.py` will automatically launch the `aggregator.py` and `executor.py` to start the FL simulation.
You can either choose to evaluate your large-scale FL experiment over a GPU cluster or test your code on your local machine.
 
### Evaluate on a GPU cluster

Once you have followed the **[instructions](https://github.com/SymbioticLab/FedScale/blob/master/core/README.md)** to set up your experiment cluster, you will be able to submit your FL job to the cluster!
 
Change `ps_ip` and `worker_ips` to the host name of your nodes in the configuration file by `cat \etc\hosts`.
For example, set `10.0.0.2:[4,4]` as one of the `worker_ips`
means launching `4 executors` on each of the first two GPUs on `10.0.0.2` to train your model in a space/time sharing fashion.

Make sure the node you submit the job has access to the computation nodes.
Also make sure you have synchronized the code across all the nodes.
 
```
cd core/evals/
python manager.py submit evals/configs/openimage/conf.yml
```
 
 ### Test on your local machine by submitting config
 
It is much easy and convenient to first test your code without a GPU cluster. 
First add an argument `- use_cuda:  False` under `job_conf` in your configuration file `evals/configs/openimage/conf.yml` to indicate you are training without using GPU.

Set `ps_ip` and `worker_ips` to be `10.0.0.1` and `10.0.0.1:[x]` by default, where x represent how many executors you want to run on your local machine.

```
cd core/evals/
python manager.py start evals/configs/openimage/conf.yml
```
### Test on your local machine with jupyter notebook
We also provide jupyter notebook [examples](./core/examples/notebook/) to run your code locally. You can first start running [server](./core/examples/notebook/fedscale_demo_server.ipynb), and then run the [client](./core/examples/notebook/fedscale_demo_client.ipynb)

 
## Monitor your training progress
 
You can find the job logging `job_name` under the path `log_path` specified in the `conf.yml` file. To check the training loss or test accuracy, you can do:
```
cd core/evals
cat job_name_logging |grep 'Training loss'
cat job_name_logging |grep 'FL Testing'
```
You can also use [Tensorboard](https://github.com/SymbioticLab/FedScale/blob/master/core/README.md#experiment-dashboard) to better visualize the progress.
