
# Implementing FedScale Deployment with Custom Dataset and Model

FedScale ([fedscale.ai](http://fedscale.ai/)) offers high-level APIs for large-scale implementation of FL algorithms across diverse hardware and software backends, as well as for deploying and evaluating these algorithms. This guide provides a straightforward setup and deployment for the DLRM algorithm on the FedScale platform.

## Contents
[FedScale Operational Workflow Analysis](#fedscale-operational-workflow-analysis)

&nbsp;&nbsp;&nbsp;&nbsp;[Flowchart](#flowchart)

&nbsp;&nbsp;&nbsp;&nbsp;[Key File and Function Analysis](#key-file-and-function-analysis)

[Run and deploy](#run-and-deploy)

&nbsp;&nbsp;&nbsp;&nbsp;[Laptop configuration](#laptop-configuration)

&nbsp;&nbsp;&nbsp;&nbsp;[Configuration phase](#configuration-phase)

&nbsp;&nbsp;&nbsp;&nbsp;[Existing dataset and model training (Eg. Femnist dataset)](#existing-dataset-and-model-training-eg-femnist-dataset)

&nbsp;&nbsp;&nbsp;&nbsp;[Custom dataset & model training (Taobao click dataset & DLRM)](#custom-dataset--model-training-taobao-click-dataset--dlrm)

[FL optimization strategies in FedScale](#fl-optimization-strategies-in-fedscale)

&nbsp;&nbsp;&nbsp;&nbsp;[Oort sampler](#oort-sampler)

&nbsp;&nbsp;&nbsp;&nbsp;[Optimizer](#optimizer)

<br>

# FedScale Operational Workflow Analysis

## Flowchart

![flowchart.jpg](https://github.com/RohanYim/FedScale-DLRM-RunGuide/blob/main/images/flowchart.jpg)

## Key File and Function Analysis

### **aggregator.py: Simulates the Server Side**

- init_model(): Model initialization
- init_data(): Data initialization
- update_weight_aggregation(): Aggregate and update model parameters

### **executor.py：Simulates the Client Side**

- init_data(): Initialize data and distribute to simulation clients
- Train(): Train the model
- Test(): Test the model
- event_monitor(): Distribute and receive server events.

### **config_parser.py: dealing with yaml parameter inputs**

### **torch_cilent.py / terserflow_client.py: handles specific training steps (forward, backward)**

### **fllibs.py: model data initialization**

- init_model(): model initialization
- init_data(): data initialization

# Run and deploy

## Laptop configuration:

* MacOS

* MacBook Pro (M1 chip)

* 16G RAM

* No MPS acceleration

## Configuration phase

### Installation and import
```bash
# get code
git clone https://github.com/SymbioticLab/FedScale.git

# setup environment
cd FedScale

# Replace ~/.bashrc with ~/.bash_profile for MacOS
echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc 
echo alias fedscale=\'bash $FEDSCALE_HOME/fedscale.sh\' >> ~/.bashrc 
conda init bash
. ~/.bashrc

# Configure fedscale virtual environment
# MAC M1 chips need to run arm architecture configuration environment-arm.yml
conda env create -f environment-arm.yml
conda activate fedscale    # conda deactivate
pip install -e .
```

### Environment configuration issues:

**Conflict between virtual environment and local environment:**

The conda virtual environment was not created according to the specified version of python (in this case it was created based on python 3.8, but in fact it was based on python 3.6).

```bash
(fedscale) haoransong@HaorandeMacBook-Pro docker % which python
alias python="/usr/local/bin/python3.6"
/Users/haoransong/opt/anaconda3/envs/fedscale/bin/python
```

If the alias for the python path was previously set in `~/.bash_profile`, it may cause the python version to point to the wrong version in the virtual environment, try:

```bash
# Change bash file
vim ~/.bash_profile

# Remove python alias line in bash_profile
alias python="/usr/local/bin/python3.6"

# Applying a new bash file
source .bash_profile
```

Update the python version of the virtual environment

```bash
# activate the virtual environment
conda activate fedscale

# under virtual environment do
conda install python=3.8

# reactivate virtual environment
conda deactivate
conda activate fedscale
```

**MacOS M1 Chip Tensorflow Issues**

M1 chip may have **["zsh: illegal hardware instruction python" when installing Tensorflow on macbook pro M1 [duplicate]](https://stackoverflow.com/questions/65383338/zsh-illegal-hardware-instruction-python-when-installing-tensorflow-on-macbook), try:**

[Download TF for M1 whl](https://drive.google.com/drive/folders/1oSipZLnoeQB0Awz8U68KYeCPsULy_dQ7)

Run

```bash
pip install ~/Downloads/tensorflow-2.4.1-py3-none-any.whl
```

## Existing dataset and model training (Eg. Femnist dataset)

### Dataset download

```bash
# fedscale provides bash files for dataset downloads
cd ./benchmart/dataset
bash download.sh download [dataset_name]
```

Problems that may arise:

```bash
download.sh: line 358: wget: command not found
```

`wget` is a common command line tool used to download files from the Internet. The `wget` command is not found on your system, try:

```bash
brew install wget
# and run the command again
bash download.sh download [dataset_name]
```

### Configure the `config` file:

```bash
cd /FedScale/benchmark/configs/femnist

# modify conf.yml
```

If running locally, you will need to change the ip to the local loopback address: `127.0.0.1`.

```yaml
# Configuration file of FAR training experiment

# ========== Cluster configuration ========== 
# ip address of the parameter server (need 1 GPU process)
ps_ip: 127.0.0.1
# ip address of each worker:# of available gpus process on each gpu in this node
# Note that if we collocate ps and worker on same GPU, then we need to decrease this number of available processes on that GPU by 1
# E.g., master node has 4 available processes, then 1 for the ps, and worker should be set to: worker:3
worker_ips:
    - 127.0.0.1:[4]
```

### Running a Demo with Jupyter Notebook

**Running a project in a virtual environment in Jupyter Notebook**

```bash
# Install a Conda Plug-In
conda install nb_conda

# Activate Virtual Environment
conda activate fedscale

# Install a Conda Plug-In in Virtual Environment
conda install ipykernel
# if you are in python3, try
pip3 install ipykernel

# Add environment to Jupyter with custom name
python -m ipykernel install --name fedscale
# if Permission denied, try
sudo python -m ipykernel install --name fedscale
```

**Aggregator(server side)：**

```python
import sys, os

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.aggregation.aggregator import Aggregator
Demo_Aggregator = Aggregator(parser.args)
### On CPU
parser.args.use_cuda = "False"
Demo_Aggregator.run()
```

**Executor(Client side)：**

```python
import torch
import logging
import math
from torch.autograd import Variable
import numpy as np

import sys, os

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.execution.executor import Executor
### On CPU
parser.args.use_cuda = "False"
Demo_Executor = Executor(parser.args)
Demo_Executor.run()
```

### View results on tersorboard:

```bash
tensorboard --logdir=<path_to_log_folder> --port=6007 --bind_all
```

![femnist_train.png](https://github.com/RohanYim/FedScale-DLRM-RunGuide/blob/main/images/femnist_train.png)

## Custom dataset & model training (Taobao click dataset & DLRM)

### Dataset Download

Download [Taobao Click Dataset](https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom)

Put all files under `$FEDSCALE_HOME/benchmark/dataset/taobao` folder
### Configure config file:

```bash
mkdir /FedScale/benchmark/configs/taobao

# create conf.yml
```

```bash
# Configuration file of FAR training experiment

# ========== Cluster configuration ==========
# ip address of the parameter server (need 1 GPU process)
ps_ip: 127.0.0.1

# ip address of each worker:# of available gpus process on each gpu in this node
# Note that if we collocate ps and worker on same GPU, then we need to decrease this number of available processes on that GPU by 1
# E.g., master node has 4 available processes, then 1 for the ps, and worker should be set to: worker:3
worker_ips:
    - 127.0.0.1:[2]

exp_path: $FEDSCALE_HOME/fedscale/cloud

# Entry function of executor and aggregator under $exp_path
executor_entry: execution/executor.py

aggregator_entry: aggregation/aggregator.py

auth:
    ssh_user: ""
    ssh_private_key: ~/.ssh/id_rsa

# cmd to run before we can indeed run FAR (in order)
setup_commands:
    - source $HOME/anaconda3/bin/activate fedscale

# ========== Additional job configuration ==========
# Default parameters are specified in config_parser.py, wherein more description of the parameter can be found

job_conf:
    - job_name: taobao # change job name
    - log_path: $FEDSCALE_HOME/benchmark
    - wandb_token: 4221994eb764b3c6244c61a8c6ba5410xxxxxxxxxx # add wandb api
    - task: recommendation # change task name
    - num_participants: 50
    - data_set: taobao # modify data_set
    - data_dir: $FEDSCALE_HOME/benchmark/dataset/taobao # modify data_set path
	# Delete data_map_file
    - device_conf_file: $FEDSCALE_HOME/benchmark/dataset/data/device_info/client_device_capacity
    - device_avail_file: $FEDSCALE_HOME/benchmark/dataset/data/device_info/client_behave_trace
    - model: dlrm # Modify the used model
    - eval_interval: 5
    - rounds: 1000
    - filter_less: 21
    - num_loaders: 2
    - local_steps: 5
    - learning_rate: 0.01
    - batch_size: 256
    - test_bsz: 256
    - use_cuda: False
    - save_checkpoint: False
	# Add new training parameters
    - sparse_feature_number: 200000 200000 200000 200000 200000 200000 200000 200000 200000 200000 200000 200000
    - sparse_feature_dim: 16
    - dense_feature_dim: 10
    - bot_layer_sizes: 122 64 16
    - top_layer_sizes: 512 256 1
    - num_field: 7
```

Modify `config_parser.py` to take on the new variables

```python
# for dlrm
parser.add_argument("--dense_feature_dim", type=int, default=16)
parser.add_argument("--bot_layer_sizes", type=int, nargs='+', default=[64, 128, 64])
parser.add_argument("--sparse_feature_number", type=int, nargs='+',default=[10000, 10000, 10000])
parser.add_argument("--sparse_feature_dim", type=int, default=16)
parser.add_argument("--top_layer_sizes", type=int, nargs='+', default=[512, 256, 1])
parser.add_argument("--num_field", type=int, default=26)
parser.add_argument("--sync_mode", type=str, default=None)
```

### Processing data sets

Preprocessing:

```python
# fllibs.py
def init_dataset():
	...
	from fedscale.dataloaders.dlrm_taobao import Taobao
	import pandas as pd
	def manual_train_test_split(df, test_size=0.2):
	    indices = df.index.tolist()
	    test_indices = random.sample(indices, int(len(indices) * test_size))
	
	    test_df = df.loc[test_indices]
	    train_df = df.drop(test_indices)
	    
	    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
	
	logging.info("Getting taobao dataset...")
	n_rows = 200000
	df_user_profile = pd.read_csv(parser.args.data_dir + '/user_profile.csv', nrows=n_rows)
    df_raw_sample = pd.read_csv(parser.args.data_dir + '/raw_sample.csv', nrows=n_rows)
    df_ad_feature = pd.read_csv(parser.args.data_dir + '/ad_feature.csv', nrows=n_rows)
	df_raw_sample.rename(columns={'user': 'userid'}, inplace=True)
	df_merged = pd.merge(df_raw_sample, df_user_profile, how='left', on='userid')
	df_merged = pd.merge(df_merged, df_ad_feature, how='left', on='adgroup_id')
	df_merged.columns = df_merged.columns.str.strip()
	
	missing_values = df_merged.isna().any()
	columns_with_nan = missing_values[missing_values].index.tolist()
	
	for column in columns_with_nan:
	    mode_value = df_merged[column].mode()[0]
	    df_merged[column].fillna(mode_value, inplace=True)
	
	train_df, test_df = manual_train_test_split(df_merged, test_size=0.2)
	logging.info('Before Taobao')
	train_dataset = Taobao(train_df)
	test_dataset = Taobao(test_df)
	logging.info('Got dataset!')
```

Handle dense and sparse data:

```python
# dataloaders/dlrm_taobao.py
class Taobao(Dataset):
    def __init__(self, df):
        self.data_frame = df

        logging.info("init taobao...")

        self.sparse_features = [
            'userid', 'adgroup_id', 'cms_segid', 
            'cate_id', 'campaign_id', 'customer', 'brand'
        ]
        self.index_mappings = {
            feature: {v: k for k, v in enumerate(self.data_frame[feature].unique())} 
            for feature in self.sparse_features
        }

        dense_features = ['final_gender_code', 'pid', 'age_level','cms_group_id', 'shopping_level', 'occupation', 'new_user_class_level', 'time_stamp', 'price', 'pvalue_level']
        self.dense_features = dense_features
        for feature in self.dense_features:
            self.data_frame[feature] = self.data_frame[feature].astype(np.float32)
        
        self.targets = self.data_frame['clk'].values.astype(np.float32)
        logging.info("init taobao done...")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sparse_feature_values = [
            self.index_mappings[feature][self.data_frame.loc[idx, feature]]
            for feature in self.index_mappings
        ]
        sparse_x = np.array(sparse_feature_values, dtype=np.int64)
        dense_x_values = [
            self.data_frame.loc[idx, feature]
            for feature in self.dense_features
        ]
        dense_x = np.array(dense_x_values, dtype=np.float32)
        label = self.targets[idx].astype(np.float32)
    
        return dense_x, sparse_x, label
```

### DLRM model

```python
# utils/models/recommendation/dlrm.py
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

MIN_FLOAT = torch.finfo(torch.float32).min / 100.0

class DLRM(nn.Module):
    def __init__(self,
                 args,
                 sync_mode=None):
        super(DLRM, self).__init__()
        logging.info('Init DLRM')
        self.dense_feature_dim = args.dense_feature_dim
        self.bot_layer_sizes = args.bot_layer_sizes
        self.sparse_feature_number = args.sparse_feature_number
        self.sparse_feature_dim = args.sparse_feature_dim
        self.top_layer_sizes = args.top_layer_sizes
        self.num_field = args.num_field

        self.bot_mlp = MLPLayer(input_shape=self.dense_feature_dim,
                                units_list=self.bot_layer_sizes,
                                last_action="relu")

        self.top_mlp = MLPLayer(input_shape=int(self.num_field * (self.num_field + 1) / 2) + self.sparse_feature_dim,
                                units_list=self.top_layer_sizes,last_action='sigmoid')
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=self.sparse_feature_dim)
            for size in self.sparse_feature_number
        ])

    def forward(self, sparse_inputs, dense_inputs):
        x = self.bot_mlp(dense_inputs)

        batch_size, d = x.shape
        sparse_embs = [self.embeddings[i](sparse_inputs[:, i]).view(-1, self.sparse_feature_dim) for i in range(sparse_inputs.shape[1])]

        T = torch.cat(sparse_embs + [x], axis=1).view(batch_size, -1, d)

        Z = torch.bmm(T, T.transpose(1, 2))
        Zflat = torch.triu(Z, diagonal=1) + torch.tril(torch.ones_like(Z) * MIN_FLOAT, diagonal=0)
        Zflat = Zflat.masked_select(Zflat > MIN_FLOAT).view(batch_size, -1)

        R = torch.cat([x] + [Zflat], axis=1)

        y = self.top_mlp(R)
        return y

class MLPLayer(nn.Module):
    def __init__(self, input_shape, units_list=None, l2=0.01, last_action=None):
        super(MLPLayer, self).__init__()

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.l2 = l2
        self.last_action = last_action
        self.mlp = nn.Sequential()

        for i in range(len(units_list)-1):
            self.mlp.add_module('dense_%d' % i, nn.Linear(units_list[i], units_list[i + 1]))
            if i != len(units_list) - 2 or last_action is not None:
                self.mlp.add_module('relu_%d' % i, nn.ReLU())
            self.mlp.add_module('norm_%d' % i, nn.BatchNorm1d(units_list[i + 1]))
        if last_action == 'sigmoid':
            self.mlp.add_module('sigmoid', nn.Sigmoid())

    def forward(self, inputs):
        return self.mlp(inputs)
```

Modify the corresponding file on the client side to take on the new model

```python
# fllibs.py
def import_libs():
	...
	elif parser.args.task == 'recommendation':
        global DLRM
        from fedscale.utils.models.recommendation.dlrm import DLRM

def init_model():
	...
	elif parser.args.task == 'recommendation':
        if parser.args.model == 'dlrm':
            model = DLRM(parser.args)
            logging.info('Got DLRM！')
        else:
            logging.info('Recommendation model does not exist!')
```

```python
# torch_client.py
def get_criterion(self, conf):
    criterion = None
		# new
    elif conf.task == 'recommendation':
        criterion = torch.nn.BCEWithLogitsLoss()
    return criterion

def train_step(self, client_data, conf, model, optimizer, criterion):
    logging.info("start training step....")
    for data_pair in client_data:
        if conf.task == 'nlp':
            ...
		# new
        elif conf.task == 'recommendation':
            dense_x, sparse_x, target = data_pair
       
				...

        if conf.task == "detection":
            ...
        elif conf.task == 'recommendation' and conf.model == 'dlrm':
            logging.info("start dlrm training step....")
            dense_features = dense_x.float()
            sparse_features = sparse_x.long()
            target = target.float().view(-1, 1)  
        ...

        target = Variable(target).to(device=self.device)

        if conf.task == 'nlp':
            ...
        elif conf.task == 'recommendation':
            outputs = model(sparse_features, dense_features)
            loss = criterion(outputs, target)
        ...

        # ======== collect training feedback for other decision components [e.g., oort selector] ======

        if conf.task == 'nlp' or (conf.task == 'text_clf' and conf.model == 'albert-base-v2') or conf.task == 'recommendation':
            loss_list = [loss.item()]  # [loss.mean().data.item()]

        ...
```

```python
# model_test_module.py
with torch.no_grad():
    if parser.args.task == 'detection':
        ...
elif parser.args.task == 'recommendation':
    logging.info("Testing for dlrm...")
    total_loss = 0.0
    total_examples = 0
    correct_predictions = 0
    for batch in test_data:
        dense_x, sparse_x, labels = batch
        dense_x = dense_x.float()
        sparse_x = sparse_x.long()
        labels = labels.float().view(-1, 1)

        outputs = model(sparse_x, dense_x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)  
        total_examples += labels.size(0)

        predicted_probs = torch.sigmoid(outputs)  
        predicted_labels = (predicted_probs > 0.5).float() 
        correct_predictions += (predicted_labels == labels).sum().item()

    logging.info(f'Test set: Loss: {total_loss:.4f}')
    return correct_predictions,correct_predictions,total_loss,{'top_1': correct_predictions, 'top_5': correct_predictions, 'test_loss': total_loss, 'test_len': total_examples}
```

### Run

```bash
python docker/driver.py start benchmark/configs/taobao/conf.yml
```

### View results on tersorboard:

![dlrm_train.png](https://github.com/RohanYim/FedScale-DLRM-RunGuide/blob/main/images/dlrm_train.png)

### Shortcomings and future work:

1. Model Accuracy Issues

    This post just tries to build FedScale-based custom model training locall, and it does not consider the results of the model. I will consider cloud training to improve the training efficiency and complete the optimization of model accuracy.

2. DLRM optimization:
    - In [AdaEmbed: Adaptive Embedding for Large-Scale Recommendation Models](https://www.usenix.org/conference/osdi23/presentation/lai), it is mentioned that the efficiency of DLRM can be optimized by dynamically pruning the embedding table during the training process, which can be followed up by the implementation of AdaEmbed to address the problem of oversized table.
    - The embedding table within the model represents a significant portion of the model parameters. When these are transmitted back to the server, it results in substantial bandwidth consumption and also risks exposing users' private data. Moreover, since the embedding table contains user-specific personalized information, it is not suited for server-side parameter aggregation and subsequent distribution to all clients. Addressing these concerns is crucial for future optimizations.

# FL optimization strategies in FedScale

## **Oort sampler**:

Oort prioritizes clients that can both complete training quickly and provide data with the greatest model accuracy gains.

The `init_client_manager` method determines which manager to use based on the parameters provided (via `args.sample_mode`) and returns the initialized client manager instance.

## Optimizer:

### q-fedavg: [Fair Resource Allocation in Federated Learning](https://arxiv.org/pdf/1905.10497.pdf)

Takes into account the unfairness between clients by introducing a hyperparameter **`q`** to control how updates are aggregated, where each client's updates are weighted by the power of its loss, with a view to achieving a fairer resource allocation.

### **FedYogi: [Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295.pdf)**

Adaptive optimization method where the server-side optimizer uses model updates computed by the client to adjust the global model.