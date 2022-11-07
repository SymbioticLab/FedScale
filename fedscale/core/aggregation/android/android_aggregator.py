import json
import subprocess
import numpy as np
import logging
from pathlib import Path

import torch

from fedscale.core.aggregation.aggregator import Aggregator
from fedscale.core.aggregation.android.linear_model import LinearModel
import fedscale.core.config_parser as parser


class Android_Aggregator(Aggregator):
    """This aggregator collects training/testing feedbacks from Android MNN APPs.

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. 
                           Defaults to the setup in arg_parser.py.
    """
    def __init__(self, args):
        super().__init__(args)
        
        # == mnn model and keymap ==
        self.mnn_json = None
        self.keymap_mnn2torch = {}
        self.input_shape = args.input_shape
    
    def init_model(self):
        """
        Load the model architecture and convert to mnn.
        NOTE: MNN does not support dropout.
        """
        if self.args.model == 'linear':
            self.model = LinearModel()
            self.model_weights = self.model.state_dict()
        else:
            super().init_model()
        self.torch_to_mnn(True)
        self.init_keymap()

    def init_keymap(self):
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
        """
        torch_keys = set()
        for key in self.model_weights.keys():
            torch_keys.add('.'.join(key.split('.')[:-1]))
        for idx, op in enumerate(self.mnn_json["oplists"]):
            if "Convolution" in op["type"]:
                for key in torch_keys:
                    if f'{key}.weight' in self.model_weights.keys():
                        mnn_weight = torch.tensor(op['main']['weight'])
                        torch_weight = self.model_weights[f'{key}.weight']
                        torch_weight_flat = torch_weight.reshape(-1)
                        if mnn_weight.shape == torch_weight_flat.shape and (
                                mnn_weight - torch_weight_flat).max() < 1e-4:
                            self.keymap_mnn2torch[idx] = (
                                key, "Convolution", tuple(torch_weight.shape),
                                f'{key}.bias' in self.model_weights.keys())
                            torch_keys.remove(key)
                            break
            elif op["type"] == "BatchNorm":
                for key in torch_keys:
                    if f'{key}.weight' in self.model_weights.keys():
                        mnn_weight = torch.tensor(op['main']['slopeData'])
                        torch_weight = self.model_weights[f'{key}.weight']
                        if mnn_weight.shape == torch_weight.shape and (
                                mnn_weight - torch_weight).max() < 1e-4:
                            self.keymap_mnn2torch[idx] = (
                                key, "BatchNorm", tuple(torch_weight.shape), True)
                            torch_keys.remove(key)
                            break

    def mnn_to_torch(self, data):
        """
        Extract trainable parameters from mnn json.
        Then convert it to state_dict, matching pytorch model.

        Args:
            data (JSON object): mnn model in JSON.

        Returns:
            dictionary: Returned the converted state_dict.
        """
        state_dict = {}
        for idx, val in self.keymap_mnn2torch.items():
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

    def torch_to_mnn(self, is_install=False):
        """Convert torch model to mnn json.

        Args:
            is_install (bool, optional): 
                Whether need to install and make MNN to build converter.
                Defaults to False.
        """
        # PyTorch -> ONNX
        input_data = torch.randn(self.input_shape)
        input_names = ["input"]
        output_names = ["output"]
        Path("cache").mkdir(exist_ok=True)
        torch.onnx.export(
            self.model, input_data, "cache/model.onnx", verbose=True,
            training=torch.onnx.TrainingMode.TRAINING, do_constant_folding=False,
            input_names=input_names, output_names=output_names)
        
        # ONNX -> MNN -> JSON
        if is_install:
            subprocess.run(["sh", "convert.sh", "--install"])
        else:
            subprocess.run(["sh", "convert.sh"])
            
        # load converted JSON file to mnn_json
        with open('cache/model.json') as f:
            self.mnn_json = json.load(f)

    def round_weight_handler(self, last_model):
        """
        Update model when the round completes.
        Then convert new model to mnn json.
        
        Args:
            last_model (list): A list of global model weight in last round.
        """
        super().round_weight_handler(last_model)
        if self.round > 1:
            self.torch_to_mnn()

    def deserialize_response(self, responses):
        """
        Deserialize the response from executor.
        If the response contains mnn json model, convert to pytorch state_dict.
        
        Args:
            responses (byte stream): Serialized response from executor.

        Returns:
            string, bool, or bytes: The deserialized response object from executor.
        """
        data = json.loads(responses.decode('utf-8'))
        if "update_weight" in data:
            data["update_weight"] = self.mnn_to_torch(
                json.loads(data["update_weight"]))
        return data

    def serialize_response(self, responses):
        """
        Serialize the response to send to server upon assigned job completion.
        If the responses is the pytorch model, change it to mnn_json.

        Args:
            responses (ServerResponse): Serialized response from server.

        Returns:
            bytes: The serialized response object to server.
        """
        if responses == self.model:
            responses = self.mnn_json
        data = json.dumps(responses)
        return data.encode('utf-8')


if __name__ == "__main__":
    aggregator = Android_Aggregator(parser.args)
    aggregator.run()
