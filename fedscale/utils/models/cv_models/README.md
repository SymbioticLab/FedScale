# Computer vision models

This folder contains 70+ computer vision models from [Imgclsmob](https://github.com/osmr/imgclsmob/blob/master/pytorch/README.md). We borrow their implementations, and change model APIs (e.g., num_classes), and integrate them into FedScale benchmarking. 
**Please acknowledge to [Imgclsmob](https://github.com/osmr/imgclsmob) if you use any of the model herein**. 

The full list of supported models are available [here](https://github.com/SymbioticLab/FedScale/blob/master/fedscale/utils/models/model_provider.py#L179). Note that for small images (e.g., FMNIST), we suggest using models with ```-cifar``` suffix, as they have smaller kernels and strides.