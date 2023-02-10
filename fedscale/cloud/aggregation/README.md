# Aggregation for Mobiles

This document contains explanation and instruction of aggregation for mobiles.

An example android aggregator accompanied by 
- [MNN](https://github.com/SymbioticLab/FedScale/fedscale/edge/mnn/). The android app has [MNN](https://github.com/alibaba/MNN) backend support for training and testing.
- [TFLite](https://github.com/SymbioticLab/FedScale/fedscale/edge/tflite/). The android app has [TFLite](https://www.tensorflow.org/lite) backend support for training and testing.

## MNN

`fedscale/cloud/aggregation/aggregator_mnn.py` contains an inherited version of aggregator. While keeping all functionalities of the original [aggregator](https://github.com/SymbioticLab/FedScale/blob/master/fedscale/cloud/aggregation/aggregator.py), it adds support to do bijective conversion between PyTorch model and MNN model. It uses JSON to communicate with android client.

**Note**:
MNN does not support direct conversion from MNN to PyTorch model, so we did a manual conversion from MNN to JSON, then from JSON to PyTorch model. We currently only support Convolution (including Linear) and BatchNorm conversion. We welcome contribution to support more conversion for operators with trainable parameters.

`scripts/convert.sh` contains model conversion code. It will clone MNN and build converter. You do not need to manually run this script. This script is run internally inside android aggregator.

`fedscale/utils/models/simple/linear_model.py` contains a simple linear model with Flatten->Linear->Softmax, used for simple test of our sample android app.

`fedscale/utils/models/mnn_convert.py` contains all the code necessary for MNN<->PyTorch model conversion.

In order to run this aggregator with default setting in order to test sample app, please run
```
git clone https://github.com/SymbioticLab/FedScale.git
cd FedScale
source install.sh
pip install -e .
cd fedscale/cloud/aggregation
python3 aggregator_mnn.py --experiment_mode=mobile --num_participants=1 --model=linear
```
and configure your android app according to the [tutorial](https://github.com/SymbioticLab/FedScale/fedscale/edge/mnn/README.md).

## TFLite

`fedscale/cloud/aggregation/aggregator_tflite.py` contains an inherited version of aggregator. While keeping all functionalities of the original [aggregator](https://github.com/SymbioticLab/FedScale/blob/master/fedscale/cloud/aggregation/aggregator.py), it adds support to do bijective conversion between tensorflow model and TFLite model.

`fedscale/utils/models/tflite_model_provider.py` contains a simple linear model with Flatten->Dense->Dense, used for simple test of our sample android app. Please feel free to contribute to it and add more models.

`fedscale/cloud/internal/tflite_model_adapter.py` defer from `fedscale/cloud/internal/tensorflow_model_adapter.py` in the way that TFLite adapter skip layers without weights, such as Flatten.

In order to run this aggregator with default setting in order to test sample app, please run
```
git clone https://github.com/SymbioticLab/FedScale.git
cd FedScale
source install.sh
pip install -e .
cd fedscale/cloud/aggregation
python3 aggregator_tflite.py --experiment_mode=mobile --num_participants=1 --engine=tensorflow
```
and configure your android app according to the [tutorial](https://github.com/SymbioticLab/FedScale/fedscale/edge/tflite/README.md).

---
If you need any other help, feel free to contact FedScale team or the developer [website](https://continue-revolution.github.io) [email](mailto:continuerevolution@gmail.com) of this android aggregator.
