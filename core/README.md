
## FedScale Automated Runtime: Evaluation Platform for Federated Learning

Existing FL evaluation platforms can hardly reproduce the scale of practical FL deployments and often fall short in providing user-friendly APIs, 
thus requiring great developer efforts to deploy new plugins. As such, we introduce FedScale Automated Runtime (FAR), 
an automated and easily-deployable evaluation platform, to simplify and standardize the FL experimental setup and model evaluation under a practical setting. 
FAR is based on our [Oort project](https://github.com/SymbioticLab/Oort), which has been shown to scale well and can emulate FL training of thousands of clients 
in each round.

## Preliminary

Our training evaluations rely on a distributed setting of GPUs/CPUs via the Parameter-Server (PS) architecture. 
In our paper, we used up to 68 GPUs to simulate the FL aggregation of 1300 participants in each round. 
Each training experiment is pretty time-consuming, as each GPU has to run multiple clients (1300/68 in our case) for each round. 

We outline some numbers on Tesla P100 GPUs for each line in our plots when using 100 participants/round for reference 
(we also provide estimated prices on [Google Cloud](https://cloud.google.com/products/calculator), but they may be inaccurate): 

| Setting      | Time to Target Accuracy  | Time to Converge |
| ----------- | ----------- | ----------- |
| YoGi             | 53  GPU hours (~$97)     |    121  GPU hours (~$230) |

Table 1: GPU hours on Openimage dataset with ShuffleNet

***Due to the high computation load on each GPU, we recommend the user make sure that each GPU is simulating no more than 20 clients. 
i.e., if the number of participants in each round is K, then we would better use at least K/20 GPUs.***

## Getting Started 


### Setting GPU Cluster

***Please assure that these paths are consistent across all nodes so that the simulator can find the right path.***

- ***Master Node***: Make sure that the master node (parameter server) has access to other worker nodes via ```ssh```. 

- ***All Nodes***: Follow [this](https://github.com/SymbioticLab/FedScale#getting-started) to install all necessary libs, and then download the datasets following [this](https://github.com/SymbioticLab/FedScale/blob/master/dataset/README.md).

### Setting Job Configuration

We provide an example of submitting a training job in ```FedScale/core/evals/manager.py```, whereby the user can submit jobs on the master node. 

- ```python manager.py submit [conf.yml]``` will submit a job with parameters specified in conf.yml on both the PS and worker nodes. 
We provide some example ```conf.yml``` in ```FedScale/core/evals/configs``` for each dataset. 
They are close to the settings used in our evaluations. Comments in our example will help you quickly understand how to specify these parameters. 

- ```python manager.py stop [job_name]``` will terminate the running ```job_name``` (specified in yml) on the used nodes. 


***We are working on building the leaderboards for better visualization. So far, all logs will be dumped to ```log_path``` (specified in the config file) on each node. 
```training_perf``` locates at the master node under this path, and the user can load it with ```pickle``` to check the time-to-accuracy performance. 
Meanwhile, the user can check ```/evals/[job_name]_logging``` to see whether the job is moving on.***



