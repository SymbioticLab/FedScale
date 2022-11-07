# Android Aggregator

An example android aggregator accompanied by an [sample android app](https://github.com/SymbioticLab/FedScale/fedscale/core/execution/android). The android app has [MNN](https://github.com/alibaba/MNN) backend support.

`android_aggregator.py` contains an inherited version of aggregator. While keeping all functionalities of the original [aggregator](https://github.com/SymbioticLab/FedScale/blob/master/fedscale/core/aggregation/aggregator.py), it adds support to do bijective conversion between PyTorch model and MNN model. It uses JSON to communicate with android client.

**Note**:
MNN does not support direct conversion from MNN to PyTorch model, so we did a manual conversion from MNN to JSON, then from JSON to PyTorch model. We currently only support Convolution (including Linear) and BatchNorm conversion. We welcome contribution to support more conversion for operators with trainable parameters.

`convert.sh` contains model conversion code. It will clone MNN and build converter. You do not need to manually run this script. This script is run internally inside android aggregator.

`linear_model.py` contains a simple linear model with Flatten->Linear->Softmax, used for simple test of our sample android app.

In order to run this aggregator, please run
```
git clone https://github.com/SymbioticLab/FedScale.git
cd FedScale
source install.sh
pip install -e .
cd fedscale/core/aggregation/android
python3 android_aggregator.py --experiment_mode=mobile --num_participants=1 --model=linear
```
and configure your android app according to the [tutorial](https://github.com/SymbioticLab/FedScale/fedscale/core/execution/android/README.md).

---
If you need any other help, feel free to contact FedScale team or the developer [website](https://continue-revolution.github.io) [email](mailto:continuerevolution@gmail.com) of this android aggregator.
