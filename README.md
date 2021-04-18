# Kuiper

This repository contains scripts and instructions for reproducing the experiments in our OSDI '21 paper "Efficient Federated Learning via Guided Participant Selection". 

# Overview

* [Getting Started](#getting-started)
* [Run Experiments and Validate Results](#run-experiments-and-validate-results)
* [Repo Structure](#repo-structure)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

# Getting Started 

Our ```install.sh``` will install the following automatically:

* Anaconda Package Manager
* CUDA 10.2

Note: if you prefer different versions of conda and CUDA, please check  comments in `install.sh` for details.

Run the following commands to install kuiper. 

```
git clone https://github.com/SymbioticLab/Kuiper
cd Kuiper
source install.sh 
```

# Run Experiments and Validate Results

The output of the experiment will validate the following major claims in our evaluation (section 7 in paper):

####    **FL Training:**
1. Kuiper outperforms existing random participant selection by 1.2×-14.1× in time-to-accuracy performance, while achieving 1.3%-9.8% better final model accuracy (§7.2.1) -> Table 1 and Figure 9.
2. Kuiper achieves close-to-optimal model efficiency by adaptively striking the trade-off between statistical and system efficiency with different components (§7.2.2) -> Figure 11 and 12.
3. Kuiper outperforms its counterpart over a wide range of parameters and different scales of experiments, while being robust to outliers (§7.2.3) -> Figure 13, 14 and 15.

####    **FL Testing:**
1. Kuiper can serve developer testing criteria on data deviation while reducing costs by bounding the number of participants needed even without individual data characteristics(§7.3.1) —> Figure 16.
2. With the individual information, Kuiper improves the testing duration by 4.7× w.r.t. Mixed Integer Linear Programming (MILP) solver, and is able to efficiently enforce developer preferences across millions of clients (§7.3.2) -> Figure 17.

## Training

Please go to `./training` directory and follow the training [README](https://github.com/SymbioticLab/Kuiper/blob/master/training/README.md) to run training scripts.

## Testing

Please go to `./testing` directory and follow the testing [README](https://github.com/SymbioticLab/Kuiper/blob/master/testing/README.md) to run testing scripts.

# Repo Structure

```
Repo Root
|---- kuiper        # Kuiper code base.
|---- training
    |---- evals     # Submit/terminate training jobs
        |---- configs   # Configuration examples
|---- testing       # Testing scripts    
    
```

# Notes
please consider to cite our paper if you use the code or data in your research project.
```
@inproceedings{kuiper-osdi21,
  title={Kuiper: Efficient Federated Learning via Guided Participant Selection},
  author={Fan Lai, Xiangfeng Zhu, Harsha V. Madhyastha, Mosharaf Chowdhury},
  booktitle={USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year={2021}
}
```

# Acknowledgements

Thanks to Qihua Zhou for his [Falcon repo](https://github.com/kimihe/Falcon).

# Contact
Fan Lai (fanlai@umich.edu) and Xiangfeng Zhu (xzhu0027@gmail.com)


