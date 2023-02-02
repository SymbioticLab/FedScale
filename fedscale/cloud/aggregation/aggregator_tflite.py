import logging
import numpy as np

import fedscale.cloud.config_parser as parser
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.internal.tflite_model_adapter import TFLiteModelAdapter
from fedscale.utils.models.tflite_model_provider import *


class TFLiteAggregator(Aggregator):
    """This aggregator collects training/testing feedbacks from Android TFLite APPs.

    Args:
        args (dictionary): Variable arguments for FedScale runtime config. 
                           Defaults to the setup in arg_parser.py.
    """

    def __init__(self, args):
        super().__init__(args)
        self.tflite_model = None

    def init_model(self):
        """
        Load the model architecture and convert to TFLite.
        """
        self.model_wrapper = TFLiteModelAdapter(
            build_simple_linear(self.args))
        self.model_weights = self.model_wrapper.get_weights()
        self.tflite_model = convert_and_save(
            TFLiteModel(self.model_wrapper.get_model()))

    def update_weight_aggregation(self, update_weights):
        """
        Update model when the round completes.
        Then convert new model to TFLite.

        Args:
            update_weights (list): A list of global model weight in last round.
        """
        super().update_weight_aggregation(update_weights)
        if self.model_in_update == self.tasks_round:
            self.tflite_model = convert_and_save(
                TFLiteModel(self.model_wrapper.get_model()))

    def deserialize_response(self, responses):
        """
        Deserialize the response from executor.
        If the response contains mnn json model, convert to pytorch state_dict.

        Args:
            responses (byte stream): Serialized response from executor.

        Returns:
            string, bool, or bytes: The deserialized response object from executor.
        """
        data = super().deserialize_response(responses)
        if "update_weight" in data:
            path = f'cache/{data["client_id"]}.ckpt'
            with open(path, 'wb') as model_file:
                model_file.write(data["update_weight"])
            restored_tensors = [
                np.asarray(tf.raw_ops.Restore(file_pattern=path, tensor_name=var.name,
                                              dt=var.dtype, name='restore')) for var in self.model_wrapper.get_model().weights if var.trainable]
            os.remove(path)
            data["update_weight"] = restored_tensors
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
        if type(responses) is list:
            tmp = [np.array_equal(a, b) for a, b in zip(
                responses, self.model_wrapper.get_weights())]
            logging.info(f"Serializing {responses}, original weights {self.model_wrapper.get_weights()}, check type - {type(responses) is list}, length - {len(responses) > 0}, equal - {tmp}")
        if type(responses) is list:
            responses = self.tflite_model
        return super().serialize_response(responses)


if __name__ == "__main__":
    aggregator = TFLiteAggregator(parser.args)
    aggregator.run()
