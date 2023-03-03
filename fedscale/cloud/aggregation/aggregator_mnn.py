import fedscale.cloud.config_parser as parser
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.utils.models.mnn_model_provider import *
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter


class MNNAggregator(Aggregator):
    """This aggregator collects training/testing feedbacks from Android MNN APPs.

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. 
                           Defaults to the setup in arg_parser.py.
    """

    def __init__(self, args):
        super().__init__(args)

        # == mnn model and keymap ==
        self.mnn_model = None
        self.keymap_mnn2torch = {}
        self.input_shape = args.input_shape

    def init_model(self):
        """
        Load the model architecture and convert to mnn.
        NOTE: MNN does not support dropout.
        """
        self.model_wrapper = TorchModelAdapter(
            get_mnn_model(self.args.model, self.args))
        self.model_weights = self.model_wrapper.get_weights()
        self.mnn_model = torch_to_mnn(
            self.model_wrapper.get_model(), self.input_shape)
        self.keymap_mnn2torch = init_keymap(
            self.model_wrapper.get_model().state_dict())

    def update_weight_aggregation(self, update_weights):
        """
        Update model when the round completes.
        Then convert new model to mnn json.

        Args:
            last_model (list): A list of global model weight in last round.
        """
        super().update_weight_aggregation(update_weights)
        if self.model_in_update == self.tasks_round:
            self.mnn_model = torch_to_mnn(
                self.model_wrapper.get_model(), self.input_shape)

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
            data["update_weight"] = mnn_to_torch(
                self.keymap_mnn2torch,
                data["update_weight"],
                data["client_id"])
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
            responses = self.mnn_model
        return super().serialize_response(responses)


if __name__ == "__main__":
    aggregator = MNNAggregator(parser.args)
    aggregator.run()
