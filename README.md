
# FedScale

FedScale is a scalable and extensible open-source federated learning (FL) engine. 
It provides high-level APIs to implement FL algorithms, deploy them at scale across diverse hardware and software backends, and evaluate them at scale.
FedScale also includes the largest FL benchmark that contains FL tasks ranging from image classification and object detection to language modeling and speech recognition. 
Moreover, it provides datasets to faithfully emulate FL training environments where FL will realistically be deployed.

[http://fedscale.ai](http://fedscale.ai)

## Getting Started

### Quick Installation (Linux)

You can simply run `install.sh`.

```
source install.sh # Add `--cuda` if you want CUDA 
pip install -e .
```

Update `install.sh` if you prefer different versions of conda/CUDA.

### Installation from Source (Linux/MacOS)

If you have [Anaconda](https://www.anaconda.com/products/distribution#download-section) installed and cloned FedScale, here are the instructions.
```
cd FedScale

# Please replace ~/.bashrc with ~/.bash_profile for MacOS
echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc 
conda init bash
. ~/.bashrc

conda env create -f environment.yml
conda activate fedscale
pip install -e .
```

Finally, install NVIDIA [CUDA 10.2](https://developer.nvidia.com/cuda-downloads) or above if you want to use FedScale with GPU support.


### Tutorials

Now that you have FedScale installed, you can start exploring FedScale following one of these introductory tutorials.

1. [Explore FedScale datasets](./dataset/Femnist_stats.md)
2. [Deploy your FL experiment](./tutorial.md)
3. [Implement an FL algorithm](./examples/README.md)


## FedScale Datasets

***We are adding more datasets! Please contribute!***

FedScale consists of 20+ large-scale, heterogeneous FL datasets covering computer vision (CV), natural language processing (NLP), and miscellaneous tasks. 
Each one is associated with its training, validation, and testing datasets. 
Please go to the `./dataset` directory and follow the dataset [README](./dataset/README.md) for more details.

## FedScale Runtime
FedScale Runtime is an scalable and extensible deployment as well as evaluation platform to simplify and standardize FL experimental setup and model evaluation. 
It evolved from our prior system, [Oort](https://github.com/SymbioticLab/Oort), which has been shown to scale well and can emulate FL training of thousands of clients in each round.

Please go to `./core` directory and follow the [README](./fedscale/core/README.md) to set up FL training scripts.


## Repo Structure

```
Repo Root
|---- dataset     # FedScale benchmarking datasets
|---- fedscale    # FedScale source code
  |---- core      # Experiment platform of FedScale
|---- examples    # Examples of new plugins
|---- evals       # Backend for FL job submission
    
```

## References
Please read and/or cite as appropriate to use FedScale code or data or learn more about FedScale.

```bibtex
@inproceedings{fedscale-icml,
  title={{FedScale}: Benchmarking Model and System Performance of Federated Learning at Scale},
  author={Fan Lai and Yinwei Dai and Sanjay S. Singapuram and Jiachen Liu and Xiangfeng Zhu and Harsha V. Madhyastha and Mosharaf Chowdhury},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```

and  

```bibtex
@inproceedings{oort-osdi21,
  title={Oort: Efficient Federated Learning via Guided Participant Selection},
  author={Fan Lai and Xiangfeng Zhu and Harsha V. Madhyastha and Mosharaf Chowdhury},
  booktitle={USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year={2021}
}
```

## Contributions and Communication
Please submit [issues](https://github.com/SymbioticLab/FedScale/issues) or [pull requests](https://github.com/SymbioticLab/FedScale/pulls) as you find bugs or improve FedScale.

If you have any questions or comments, please join our [Slack](https://join.slack.com/t/fedscale/shared_invite/zt-uzouv5wh-ON8ONCGIzwjXwMYDC2fiKw) channel, or email us ([fedscale@googlegroups.com](mailto:fedscale@googlegroups.com)). 

