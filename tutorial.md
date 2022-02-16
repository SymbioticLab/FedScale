
# Tutorial
 
This tutorial will show you how to set up and start a FL experiment over OpenImg dataset using Fedscale.
 
## Environment
 
Our ```install.sh``` will install the following automatically:
 
* Anaconda Package Manager
* CUDA 10.2
 
Note: if you prefer different versions of conda and CUDA, please check  comments in `install.sh` for details.
 
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
## Create your experiment profile
 
Once you have followed the **[instructions](https://github.com/SymbioticLab/FedScale/blob/master/core/README.md)**
to set up your experiment cluster, you will be able to submit your FL job!
 
Go to `./core/evals/configs/openimage/` directory and modify/create your
**[configuration file](https://github.com/SymbioticLab/FedScale/blob/master/core/evals/configs/openimage/conf.yml)** to submit your job.
 
Change `ps_ip` and `worker_ips` to the host name of your nodes.
For example, set `10.0.0.2:[4,4]` as one of the `worker_ips`
means launching `4 executors` on the first two GPUs on the same machine each to train your model in a space/time sharing fashion.
 
Modify the rest of configurations
(the number of participants per round, the aggregation algorithm, the client optimizer, the training model, etc.)  as you need .
 
 
## Submit your FL job
 
Go to `./core/evals/` and use [manager.py](https://github.com/SymbioticLab/FedScale/blob/master/core/evals/manager.py)
to submit your FL job.
`manager.py` will automatically launch the `aggregator.py` and `executor.py` to start the FL simulation.
Make sure the node you submit the job has access to the computation nodes.
Also make sure you have synchronized the code across all the nodes.
 
```
cd core/evals/
python manager.py submit evals/configs/openimage/conf.yml
```
 
Modify the path or the name of `conf.yml` based on your setting.
 
 
## Monitor your training progress
 
You can find the job logging `job_name` under the path `log_path` specified in the `conf.yml` file. To check the training loss or test accuracy, you can do:
```
cd core/evals
cat job_name_logging |grep 'Training loss'
cat job_name_logging |grep 'FL Testing'
```
We are going to launch our Tensorboard to enable more efficient visualization for your job.
 
 
