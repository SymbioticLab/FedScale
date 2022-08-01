# -*- coding: utf-8 -*-
import pickle

import fedscale.core.channels.job_api_pb2 as job_api_pb2
from fedscale.core.execution.executor import Executor
from fedscale.core.execution.rlclient import RLClient
from fedscale.core.logger.execution import *
from fedscale.core import commons

class AsyncExecutor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super().__init__(args)
        self.temp_model_path_version = lambda round: os.path.join(
            logDir, 'model_' + str(round) + '.pth.tar')

    def update_model_handler(self, model):
        """Update the model copy on this executor"""
        self.model = model
        self.round += 1

        # Dump latest model to disk
        with open(self.temp_model_path_version(self.round), 'wb') as model_out:
            logging.info(f"Received latest model saved at {self.temp_model_path_version(self.round)}")
            pickle.dump(self.model, model_out)

    def load_global_model(self, round=None):
        # load last global model
        if round == -1:
            with open(self.temp_model_path, 'rb') as model_in:
                model = pickle.load(model_in)
        else:
            round = min(round, self.round) if round is not None else self.round
            with open(self.temp_model_path_version(round), 'rb') as model_in:
                model = pickle.load(model_in)
        return model

    def training_handler(self, clientId, conf, model=None):
        """Train model given client ids"""

        # Here model is model_id
        client_model = self.load_global_model(-1) if model is None \
            else self.load_global_model(model)

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
        if args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets,
                                         batch_size=conf.batch_size, args=self.args,
                                         collate_fn=self.collate_fn
                                         )

            client = self.get_client_trainer(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)

        return train_res

    def check_model_version(self, model_id):
        return os.path.exists(self.temp_model_path_version(round))

    def event_monitor(self):
        """Activate event handler once receiving new message
        """
        logging.info("Start monitoring events ...")
        self.client_register()

        while self.received_stop_request == False:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event

                if current_event == commons.CLIENT_TRAIN:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    if train_model is not None and not self.check_model_version(train_model):
                        # The executor may have not received the model due to async grpc
                        self.event_queue.append(request)
                        time.sleep(1)
                        continue

                    train_config['model'] = train_model
                    train_config['client_id'] = int(train_config['client_id'])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    future_call = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                        job_api_pb2.CompleteRequest(client_id=str(client_id), executor_id=self.executor_id,
                                                    event=commons.UPLOAD_MODEL, status=True, msg=None,
                                                    meta_result=None, data_result=self.serialize_response(train_res)
                                                    ))
                    future_call.add_done_callback(lambda _response: self.dispatch_worker_events(_response.result()))

                elif current_event == commons.MODEL_TEST:
                    self.Test(self.deserialize_response(request.meta))

                elif current_event == commons.UPDATE_MODEL:
                    broadcast_config = self.deserialize_response(request.data)
                    self.UpdateModel(broadcast_config)
                    time.sleep(5)

                elif current_event == commons.SHUT_DOWN:
                    self.Stop()

                elif current_event == commons.DUMMY_EVENT:
                    pass
            else:
                time.sleep(10)
                self.client_ping()

if __name__ == "__main__":
    executor = AsyncExecutor(args)
    executor.run()
