import json

import fedscale.cloud.config_parser as parser
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.utils.models.simple.linear_model import LinearModel
from fedscale.utils.models.mnn_convert import *


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
        self.mnn_json = torch_to_mnn(self.model, self.input_shape, True)
        self.keymap_mnn2torch = init_keymap(self.model_weights, self.mnn_json)

    def round_weight_handler(self, last_model):
        """
        Update model when the round completes.
        Then convert new model to mnn json.
        
        Args:
            last_model (list): A list of global model weight in last round.
        """
        super().round_weight_handler(last_model)
        if self.round > 1:
            self.mnn_json = torch_to_mnn(self.model, self.input_shape)

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
            data["update_weight"] = mnn_to_torch(
                self.keymap_mnn2torch,
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
