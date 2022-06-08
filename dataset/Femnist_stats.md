 
# Understand the heterogeneous FL data [[Jupyter Notebook](/dataset/Femnist_stats.ipynb) ]
 

## Download the Femnist dataset and FedScale
Follow the download instruction in `/content/FedScale/dataset/download.sh` to download the FEMNIST dataset.

```{code-cell} 
wget -O  /content/femnist.tar.gz https://fedscale.eecs.umich.edu/dataset/femnist.tar.gz
tar -xf  /content/femnist.tar.gz -C /content/
rm -f  /content/femnist.tar.gz
echo -e "${GREEN}FEMNIST dataset downloaded!${NC}"        
```


```sh
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Make sure you have downloaded and installed FedScale 
from fedscale.core.utils.femnist import FEMNIST
from fedscale.core.utils.utils_data import get_data_transform
from fedscale.core.utils.divide_data import DataPartitioner
from fedscale.core.argParser import args 
```


## Data Loader

```{code-cell}
train_transform, test_transform = get_data_transform('mnist')
train_dataset = FEMNIST('/content/femnist', dataset='train', transform=train_transform)
test_dataset = FEMNIST('/content/femnist', dataset='test', transform=test_transform)
```

Partition the dataset by the `client_data_mapping` file, which gives the real-world client-level heterogeneoity.

```{code-cell}
args.task = 'cv'
training_sets = DataPartitioner(data=train_dataset, args=args, numOfClass=62)
training_sets.partition_data_helper(num_clients=None, data_map_file='/content/femnist/client_data_mapping/train.csv')
#testing_sets = DataPartitioner(data=test_dataset, args=args, numOfClass=62, isTest=True)
#testing_sets.partition_data_helper(num_clients=None, data_map_file='/content/femnist/client_data_mapping/train.csv')
```


## Print and plot statistics of the dataset.

```{code-cell} 
print(f'Total number of data smaples: {training_sets.getDataLen()}')
print(f'Total number of clients: {training_sets.getClientLen()}') 
```

> Total number of data smaples: 637877 
> 
> Total number of clients: 2800

```{code-cell} 
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
size_dist = training_sets.getSize()['size']
 
n_bins = 20
axs[0].hist(size_dist, bins=n_bins) 
axs[0].set_title('Client data size distribution')

label_dist = training_sets.getClientLabel()
axs[1].hist(label_dist, bins=n_bins) 
axs[1].set_title('Client label distribution')
```

![image](/dataset/femnist/client_label_heter.png)
 
## Visiualize the clients' data.

```{code-cell} 
rank=1
isTest = False
dropLast =  True
partition = training_sets.use(rank - 1, isTest)
num_loaders = min(int(len(partition)/ args.batch_size/2),  args.num_loaders)
dataloader = DataLoader(partition, batch_size=16, shuffle=True, pin_memory=True, timeout=60, num_workers=num_loaders, drop_last=dropLast)
```

```{code-cell}
for data in iter(dataloader):
    plt.imshow(np.transpose(data[0][0].numpy(), (1, 2, 0)))
    break
```


![image](/dataset/femnist/sample_data.png)
