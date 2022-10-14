# Computer vision models

This folder contains 70+ computer vision models. Some are from [Imgclsmob](https://github.com/osmr/imgclsmob/blob/master/pytorch/README.md). We reimplement some of them, add new APIs (e.g., num_classes), and support them in FedScale benchmarking. 
**Please acknowledge to [Imgclsmob](https://github.com/osmr/imgclsmob) if you use any of the model herein**. 

The full list of supported models are available [here](https://github.com/SymbioticLab/FedScale/blob/master/fedscale/utils/models/model_provider.py#L179). Note that for small images (e.g., FMNIST), we suggest using models with ```-cifar``` suffix, as they have smaller kernels and strides. Meanwhile, please ignore the suffix ``-cifar10`` or ``-cifar100`` as their model num_classes will be automatically overrided by the ``--num_classes`` setting of the dataset.

**We are adding new models, and appreciate if you can consider contributing yours! Please feel free to report bugs.**