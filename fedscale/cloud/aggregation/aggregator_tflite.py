import os
import numpy as np

import fedscale.cloud.config_parser as parser
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.internal.tflite_model_adapter import TFLiteModelAdapter
from fedscale.utils.models.tflite_model_provider import *
from fedscale.cloud.channels import job_api_pb2
from fedscale.cloud.fllibs import *


class TFLiteAggregator(Aggregator):
    """This aggregator collects training/testing feedbacks from Android TFLite APPs.

    Args:
        args (dictionary): Variable arguments for FedScale runtime config. 
                           Defaults to the setup in arg_parser.py.
    """

    def __init__(self, args):
        super().__init__(args)
        self.tflite_model = None
        self.base = None

    def init_model(self):
        """
        Load the model architecture and convert to TFLite.
        """
        model, self.base = get_tflite_model(self.args.model, self.args)
        self.model_wrapper = TFLiteModelAdapter(model)
        self.tflite_model = convert_and_save(model, self.base, self.args)
        self.model_weights = self.model_wrapper.get_weights()

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
                self.model_wrapper.get_model(), self.base, self.args)

    def deserialize_response(self, responses):
        """
        Deserialize the response from executor.
        If the response contains mnn model, convert to pytorch state_dict.

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
                np.asarray(tf.raw_ops.Restore(
                    file_pattern=path, tensor_name=var.name,
                    dt=var.dtype, name='restore')
                ) for var in self.model_wrapper.get_model().weights]
            os.remove(path)
            data["update_weight"] = restored_tensors
        return data
    
    def serialize_response(self, responses):
        """ Serialize the response to send to server upon assigned job completion

        Args:
            responses (ServerResponse): Serialized response from server.

        Returns:
            bytes: The serialized response object to server.

        """
        if type(responses) is list:
            responses = self.tflite_model
        return super().serialize_response(responses)

    # def create_client_task(self, executor_id):
    #     """Issue a new client training task to specific executor

    #     Args:
    #         executorId (int): Executor Id.

    #     Returns:
    #         tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

    #     """
    #     next_client_id = self.resource_manager.get_next_task(executor_id)
    #     train_config = None
    #     # NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
    #     if next_client_id is not None:
    #         config = self.get_client_conf(next_client_id)
    #         train_config = {'client_id': next_client_id, 'task_config': config}
    #     return train_config, self.tflite_model_bytes
    
    def CLIENT_PING(self, request, context):
        """Handle client ping requests

        Args:
            request (PingRequest): Ping request info from executor.

        Returns:
            ServerResponse: Server response to ping request

        """
        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id may use the same executor_id (VMs) in simulations
        executor_id, client_id = request.executor_id, request.client_id
        response_data = response_msg = commons.DUMMY_RESPONSE

        if len(self.individual_client_events[executor_id]) == 0:
            # send dummy response
            current_event = commons.DUMMY_EVENT
            response_data = response_msg = commons.DUMMY_RESPONSE
        else:
            # NOTE: This is a temp solution to bypass the following errors:
            # 1. problem:   server->client update_model package dropped, server->client model_test in error
            #    solution:  ignore update_model, send model in model_test package
            # 2. problem:   server->client client_train package dropped, server->client dummy_event forever
            #    solution:  keep event inside queue until client confirm event completed
            #    pitfall:   simulation executor L388 multi-thread may ping the same event more than once
            #               update_model no confirmation, no way to tell if update_model finished
            current_event = self.individual_client_events[executor_id][0]
            while current_event == commons.UPDATE_MODEL:
                self.individual_client_events[executor_id].popleft()
                current_event = self.individual_client_events[executor_id][0]
            if current_event == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(
                    executor_id)
                if response_msg is None:
                    current_event = commons.DUMMY_EVENT
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(
                            commons.CLIENT_TRAIN)
            elif current_event == commons.MODEL_TEST:
                response_msg = self.get_test_config(client_id)
                response_data = self.tflite_model
            elif current_event == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)
                self.individual_client_events[executor_id].popleft()

        response_msg, response_data = self.serialize_response(
            response_msg), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        response = job_api_pb2.ServerResponse(event=current_event,
                                              meta=response_msg, data=response_data)
        if current_event != commons.DUMMY_EVENT:
            logging.info(
                f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")

        return response
    
    def CLIENT_EXECUTE_COMPLETION(self, request, context):
        """FL clients complete the execution task.

        Args:
            request (CompleteRequest): Complete request info from executor.

        Returns:
            ServerResponse: Server response to job completion request

        """

        executor_id, client_id, event = request.executor_id, request.client_id, request.event
        execution_status, execution_msg = request.status, request.msg
        meta_result, data_result = request.meta_result, request.data_result

        if event in (commons.MODEL_TEST, commons.UPLOAD_MODEL):
            if execution_status is False:
                logging.error(
                    f"Executor {executor_id} fails to run client {client_id}, due to {execution_msg}")
            else:
                self.add_event_handler(
                    executor_id, event, meta_result, data_result)
            event_pop = self.individual_client_events[executor_id].popleft()
            logging.info(f"Event {event_pop} popped from queue.")
        else:
            logging.error(
                f"Received undefined event {event} from client {client_id}")
        return self.CLIENT_PING(request, context)


if __name__ == "__main__":
    aggregator = TFLiteAggregator(parser.args)
    aggregator.run()
