import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

cache_path = "../../cloud/aggregation/cache/"
script_path = "../../../scripts/"


def build_simple_linear(args):
    return torch.nn.Sequential([
        torch.nn.Flatten(),
        torch.nn.Linear(np.prod(args.input_shape), args.num_classes),
        torch.nn.Softmax(dim=1)
    ])


_models = {
    'linear': build_simple_linear,
}


def get_mnn_model(name: str, args):
    if name not in _models:
        raise ValueError(f"Unsupported model: {name}")
    return _models[name](args)


def init_keymap(model_weights: dict) -> dict:
    """
    Match keys from mnn to torch.

    MNN do not support mnn->torch conversion 
    and do not keep keys inside state_dict when converted from torch model.

    MNN can be converted to JSON, which has a list of operations.
    Some operations have trainable parameters.
    All operations have a type.

    We currently support getting the key map of 
    two types of operations which have trainable operations:
        1. Convolution: weight, bias.
        2. BatchNorm: slopeData, meanData, varData, biasData.
    This method initialize keymap as 
        idx: an unsigned integer representing the index of
             the operation inside oplist which has trainable parameters.
            ->
        (
            key     : a string representing the key of state_dict.
            type    : a string representing the type of the operation in MNN.
                      "Convolution"|"BatchNorm"
            shape   : a tuple representing the original shape in torch.
            has_bias: a bool representing whether the operation has bias.
        ).
    Example: 4->("linear", "Convolution", (10, 2352)), True)

    Args:
        model_weights (dict): PyTorch model weights in state_dict.

    Returns:
        dict: MNN oplist index -> PyTorch state_dict key map.
    """
    # load converted JSON file to mnn_json
    subprocess.call([
        f"{script_path}MNNDump2Json",
        f"{cache_path}model.mnn",
        f"{cache_path}model.json"])
    with open('../../cloud/aggregation/cache/model.json') as f:
        mnn_json = json.load(f)
    keymap = {}
    torch_keys = set()
    for key in model_weights.keys():
        torch_keys.add('.'.join(key.split('.')[:-1]))
    for idx, op in enumerate(mnn_json["oplists"]):
        if "Convolution" in op["type"]:
            for key in torch_keys:
                if f'{key}.weight' in model_weights.keys():
                    mnn_weight = torch.tensor(op['main']['weight'])
                    torch_weight = model_weights[f'{key}.weight']
                    torch_weight_flat = torch_weight.reshape(-1)
                    if mnn_weight.shape == torch_weight_flat.shape and (
                            mnn_weight - torch_weight_flat).max() < 1e-4:
                        keymap[idx] = (
                            key, "Convolution", tuple(torch_weight.shape),
                            f'{key}.bias' in model_weights.keys())
                        torch_keys.remove(key)
                        break
        elif op["type"] == "BatchNorm":
            for key in torch_keys:
                if f'{key}.weight' in model_weights.keys():
                    mnn_weight = torch.tensor(op['main']['slopeData'])
                    torch_weight = model_weights[f'{key}.weight']
                    if mnn_weight.shape == torch_weight.shape and (
                            mnn_weight - torch_weight).max() < 1e-4:
                        keymap[idx] = (
                            key, "BatchNorm", tuple(torch_weight.shape), True)
                        torch_keys.remove(key)
                        break
    return keymap


def torch_to_mnn(model, input_shape: Tensor):
    """Convert torch model to mnn binary.

    Args:
        model (Module): Pytorch model to be converted.
        input_shape (Tensor): Shape of input to the model.

    Returns:
        JSON object: MNN model in JSON format.
    """
    # PyTorch -> ONNX
    input_data = torch.randn(input_shape)
    input_names = ["input"]
    output_names = ["output"]
    Path(cache_path).mkdir(exist_ok=True)
    torch.onnx.export(
        model, input_data, f"{cache_path}model.onnx", verbose=True,
        training=torch.onnx.TrainingMode.TRAINING, do_constant_folding=False,
        input_names=input_names, output_names=output_names)

    # ONNX -> MNN
    subprocess.call([
        f"{script_path}MNNConvert", "-f", "ONNX", "--modelFile",
        f"{cache_path}model.onnx", "--MNNModel", f"{cache_path}model.mnn", "--forTraining"])

    return Path(f'{cache_path}model.mnn').read_bytes()


def mnn_to_torch(keymap: dict, mnn_model_binary: bytes, client_id: str):
    """
    Extract trainable parameters from mnn json.
    Then convert it to state_dict, matching pytorch model.
    Args:
        keymap (dict): Key map from MNN oplist index to PyTorch state_dict key.
        data (JSON object): MNN model in JSON.
    Returns:
        dict: Returned the converted state_dict.
    """
    Path(f'{cache_path}{client_id}.mnn').write_bytes(mnn_model_binary)
    subprocess.call([
        f"{script_path}MNNDump2Json",
        f"{cache_path}{client_id}.mnn",
        f"{cache_path}{client_id}.json"
    ])
    data = json.load(f"{cache_path}{client_id}.json")
    Path(f'{cache_path}{client_id}.mnn').unlink()
    Path(f'{cache_path}{client_id}.json').unlink()
    state_dict = {}
    for idx, val in keymap.items():
        key, mnn_type, shape, has_bias = val
        if mnn_type == 'Convolution':
            state_dict[f'{key}.weight'] = np.asarray(
                data['oplists'][idx]['main']['weight'],
                dtype=np.float32).reshape(shape)
            if has_bias:
                state_dict[f'{key}.bias'] = np.asarray(
                    data['oplists'][idx]['main']['bias'],
                    dtype=np.float32)
        elif mnn_type == 'BatchNorm':
            state_dict[f'{key}.weight'] = np.asarray(
                data['oplists'][idx]['main']['slopeData'],
                dtype=np.float32)
            state_dict[f'{key}.bias'] = np.asarray(
                data['oplists'][idx]['main']['biasData'],
                dtype=np.float32)
            state_dict[f'{key}.running_mean'] = np.asarray(
                data['oplists'][idx]['main']['meanData'],
                dtype=np.float32)
            state_dict[f'{key}.running_var'] = np.asarray(
                data['oplists'][idx]['main']['varData'],
                dtype=np.float32)
        else:
            logging.ERROR(f"Unsupported MNN Type: {mnn_type}")
    return state_dict
