# Kuiper-Testing

This folder contains scripts and instructions for reproducing the FL testing experiments in our OSDI '21 paper. 

### Preliminary

Testing evaluations in our paper run on a single machine (32 CPU cores, 384GB RAM). 

Before attempting to run testing script, you must download the datasets by running `bash download_result.sh`.

### Figure 16 - Preserving Data Representativeness 

```
python plot_figure16.py     # few seconds
open figure16.pdf
```

This will produce plots close to Figure 16 on page 12 of the paper. You might notice some variation compared to the original figure due to randomness of the experiments.

### Figure 17 - Enforcing Diverse Data Distribution 

Before running below script, you must install gurobi license by:

* Request an [academic license](https://www.gurobi.com/downloads/end-user-license-agreement-academic/) if possible. Otherwise, please contact us for a temporary license. 
* `grbgetkey [your license]` to install the license 

```
python plot_figure17.py   # > 70 hours
# or python plot_figure17.py -k # ~ 1.5 hour
open figure17a.pdf figure17b.pdf
``` 

This will produce plots close to Figure 17 on page 12 of the paper. You might notice some variation compared to the original figure as 1) we removed a few long-running queries; 2) gurobi performance shows certain variations. 

Note: To save reviewers time, `python plot_figure17.py -k` will only run and plot the lines for Kuiper. We hope the runtime will convince you that MILP is extremely slow :).
