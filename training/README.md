# Kuiper-Training

This folder contains scripts and instructions for reproducing the FL training experiments in our OSDI '21 paper. 
***Note that model training performance (both accuracy and time-to-accuracy performance) often shows certain variations. We report the mean value over five runs for each experiment in our paper.***

# Preliminary

Our training evaluations rely on a distributed setting of ***multiple GPUs*** via the Parameter-Server (PS) architecture. 
In our paper, we used up to 68 GPUs to simulate the FL aggregation of 1300 participants in each round. 
Each training experiment is pretty time-consuming, as each GPU has to run multiple clients (1300/68 in our case) for each round. 

We outline some numbers on Tesla P100 GPUs for each line in our plots when using 100 participants/round for reference (we also provide estimated prices on [Google Cloud](https://cloud.google.com/products/calculator), but they may be inaccurate): 

| Setting      | Time to Target Accuracy  | Time to Converge |
| ----------- | ----------- | ----------- |
| Kuiper+YoGi      | 27  GPU hours (~$53)    |    58 GPU hours (~$111)   |
| YoGi             | 53  GPU hours (~$97)     |    121  GPU hours (~$230) |

Table 1: GPU hours on Openimage dataset with ShuffleNet

***Due to the high computation load on each GPU, we recommend the reviewers make sure that each GPU is simulating no more than 20 clients. 
i.e., if the number of participants in each round is K, then we would better use at least K/20 GPUs.***

# Getting Started 


## Setting GPU Cluster

***Please assure that these paths are consistent across all nodes so that the simulator can find the right path.***

- ***Master Node***: Make sure that the master node (parameter server) has access to other worker nodes via ```ssh```. 

- ***All Nodes***: Follow [this](https://github.com/SymbioticLab/Kuiper#getting-started) to install all necessary libs, and then run the following command to download the datasets (We use the benchmarking dataset in the [FLPerf](https://github.com/SymbioticLab/FLPerf) repo.):

```
git clone https://github.com/SymbioticLab/FLPerf.git
cd FLPerf
# Download the open image dataset. Make sure you have at least 150 GB of storage capacity.
# Check ./download.sh -h for downloading different datasets. Refer to FLPerf if failed.
bash download.sh -o    
```


## Setting Job Configuration

We provide an example of submitting a training job in ```Kuiper/training/evals/manager.py```, whereby the user can submit jobs on the master node. 

- ```python manager.py submit [conf.yml]``` will submit a job with parameters specified in conf.yml on both the PS and worker nodes. 
We provide some example ```conf.yml``` in ```Kuiper/training/evals/configs``` for each dataset. 
They are close to the settings used in our evaluations. Comments in our example will help you quickly understand how to specify these parameters. 

- ```python manager.py stop [job_name]``` will terminate the running ```job_name``` (specified in yml) on the used nodes. 


***All logs will be dumped to ```log_path``` (specified in the config file) on each node. 
```training_perf``` locates at the master node under this path, and the user can load it with ```pickle``` to check the time-to-accuracy performance. 
Meanwhile, the user can check ```/evals/[job_name]_logging``` to see whether the job is moving on.***

# Run Experiment and Validate Results

***NOTE: To save reviewers' time, we recommend the reviewers only run Kuiper with YoGi on OpenImage dataset, as it validates our major claim about Kuiper's improvement over random selection and is the most efficient setting. Instead, FedProx takes ~2x more GPU hours than YoGi, while the NLP task takes more than 4x GPU hours even with YoGi. However, please feel free to run other experiments if time permits. Running all experiments requires > 4300 GPU hours.***

The output of the experiment will validate the following major claims in our paper:
1. Kuiper outperforms existing random participant selection by 1.2×-14.1× in time-to-accuracy performance, while achieving 1.3%-9.8% better final model accuracy (§7.2.1) -> Table 1 and Figure 9.
2. Kuiper achieves close-to-optimal model efficiency by adaptively striking the trade-off between statistical and system efficiency with different components (§7.2.2) -> Figure 11 and 12.
3. Kuiper outperforms its counterpart over a wide range of parameters and different scales of experiments, while being robust to outliers (§7.2.3) -> Figure 13, 14, and 15.

## Time to accuracy performance (Table 1 and Figure 9)

Please refer to ```Kuiper/training/evals/configs/{DATASET_NAME}/conf.yml``` to configurate the [node ips](https://github.com/SymbioticLab/Kuiper/blob/master/training/evals/configs/openimage/conf.yml#L5), and [NIC](https://github.com/SymbioticLab/Kuiper/blob/master/training/evals/configs/openimage/conf.yml#L25) for communication. 

For example, to run Kuiper with YoGi on OpenImage dataset and plot the figure, execute the following commands:

``` 
cd training/evals/
```

Change [sample_mode](https://github.com/SymbioticLab/Kuiper/blob/master/training/evals/configs/openimage/conf.yml#L38) to random and run the following command to run YoGi with random selection : 
```
python manager.py submit configs/openimage/conf.yml 
```

After the completion of training, then change [sample_mode](https://github.com/SymbioticLab/Kuiper/blob/master/training/evals/configs/openimage/conf.yml#L38) to kuiper and run the following command(again) to run YoGi with Kuiper: 
```
python manager.py submit configs/openimage/conf.yml 
```

After the experiments finishes, you can find `training_perf` of both experiment on master node's `log_path`. For example, if default config is used, `training_perf`s is available at `Kuiper/training/evals/logs/openimage_kuiper/{time_stamp}/aggregator/`. Run the following command to plot the figure: 
```
python plot_perf.py [path_to_training_perf_random/training_perf] [path_to_training_perf_kuiper/training_perf] 
```

This will produce a plot close to Figure 11(b) on page 10 of the paper. You might notice some variation compared to the original figure due to the randomness of the experiments.

## Performance breakdown (Figure 12)

Please specify the following parameters in ```Kuiper/training/evals/configs/{DATASET_NAME}/conf.yml``` to run the breakdown experiment:  

Run Kuiper w/o Sys by setting [round_penalty](https://github.com/SymbioticLab/Kuiper/blob/master/training/evals/configs/openimage/conf.yml#L41) to 0

Run Kuiper w/o Pacer by setting [pacer_delta](https://github.com/SymbioticLab/Kuiper/blob/master/training/evals/configs/openimage/conf.yml#L44) to 0

## Sensitivity Analysis (Figure 13, Figure 14, and Figure 15)

### Figure 14 

Change [round_penalty](https://github.com/SymbioticLab/Kuiper/blob/master/training/evals/configs/openimage/conf.yml#L41) (\alpha in Figure 13), while keeping other configurations the same. 

### Figure 13

Change [total_worker](https://github.com/SymbioticLab/Kuiper/blob/master/training/evals/configs/openimage/conf.yml#L33) (different number of participants K for Figure 14), while keeping other configurations the same. 

### Figure 15

***Experiments of outliers are extremely slow as we need to get the final accuracy of the training, so we recommend the reviewer to put this to the last.***
To run this, please first add ```- blacklist_rounds: 50``` to the config file in order to enable the blacklist. Then specify different degrees of outliers ```- malicious_clients: 0.1``` (i.e., 10% clients are corrupted). 

