import logging

from fedscale.cloud.execution.executor import *
from utils.helper import *
import copy

class AuxoExecutor(Executor):
    def __init__(self, args):
        super().__init__(args)

        self.round = [0]
        self.model_adapter = [self.get_client_trainer(args).get_model_adapter(init_model())]


    def UpdateModel(self, model_weights, cohort_id):
        """Receive the broadcasted global model for current round

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model config
            cohort_id (int): The cohort id
        """
        self.round[cohort_id] += 1
        self.model_adapter[cohort_id].set_weights(model_weights)

    def training_handler(self, client_id, conf, model, cohort_id):
        """Train model given client id

        Args:
            client_id (int): The client id.
            conf (dictionary): The client runtime config.
            cohort_id (int): The cohort id.

        Returns:
            dictionary: The train result

        """
        self.model_adapter[cohort_id].set_weights(model)
        conf.client_id = client_id
        conf.tokenizer = tokenizer
        client_data = self.training_sets if self.args.task == "rl" else \
            select_dataset(client_id, self.training_sets,
                           batch_size=conf.batch_size, args=self.args,
                           collate_fn=self.collate_fn
                           )
        client = self.get_client_trainer(self.args)
        train_res = client.train(
            client_data=client_data, model=self.model_adapter[cohort_id].get_model(), conf=conf)

        return train_res

    def Train(self, config):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:
            tuple (int, dictionary): The client id and train result

        """
        client_id, train_config, cohort_id = config['client_id'], config['task_config'], config['cohort_id']

        if 'model' not in config or not config['model']:
            raise "The 'model' object must be a non-null value in the training config."
        client_conf = self.override_conf(train_config)
        train_res = self.training_handler(
            client_id=client_id, conf=client_conf, model=config['model'], cohort_id=cohort_id)

        # Report execution completion meta information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=str(client_id), executor_id=self.executor_id,
                event=generate_msg(commons.CLIENT_TRAIN, cohort_id), status=True, msg=None,
                meta_result=None, data_result=None
            )
        )
        self.dispatch_worker_events(response)
        logging.info(f"Client {client_id} finished training. ")

        return client_id, train_res

    def testing_handler(self, cohort_id):
        """Test model

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
            config (dictionary): Variable arguments from coordinator.
        Returns:
            dictionary: The test result

        """
        test_config = self.override_conf({
            'rank': self.this_rank,
            'memory_capacity': self.args.memory_capacity,
            'tokenizer': tokenizer
        })
        client = self.get_client_trainer(test_config)
        data_loader = select_dataset(self.this_rank, self.testing_sets,
                                     batch_size=self.args.test_bsz, args=self.args,
                                     isTest=True, collate_fn=self.collate_fn)

        test_results = client.test(data_loader, self.model_adapter[cohort_id].get_model(), test_config)
        gc.collect()

        return test_results

    def Test(self, config, cohort_id):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group

        Args:
            config (dictionary): The client testing config.

        """
        test_res = self.testing_handler(cohort_id)
        test_res = {'executorId': self.this_rank, 'results': test_res}

        # Report execution completion information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=self.executor_id, executor_id=self.executor_id,
                event=generate_msg(commons.MODEL_TEST, cohort_id), status=True, msg=None,
                meta_result=None, data_result=self.serialize_response(test_res)
            )
        )
        self.dispatch_worker_events(response)

    def _init_split(self, cohort_id, new_cohort_id):
        if len(self.model_adapter) <= new_cohort_id:
            self.model_adapter.append(copy.deepcopy(self.model_adapter[cohort_id]))
            self.round.append(copy.deepcopy(self.round[cohort_id]))

    def event_monitor(self):
        """Activate event handler once receiving new message
        """
        logging.info("Start monitoring events ...")
        self.client_register()

        while not self.received_stop_request:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event
                event_type, cohort_id = decode_msg(current_event)
                if event_type != commons.DUMMY_EVENT:
                    logging.info("Received message: {}".format(current_event))

                if event_type == commons.CLIENT_TRAIN:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    train_config['model'] = train_model
                    train_config['client_id'] = int(train_config['client_id'])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    future_call = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                        job_api_pb2.CompleteRequest(client_id=str(client_id), executor_id=self.executor_id,
                                                    event=generate_msg(commons.UPLOAD_MODEL, cohort_id), status=True, msg=None,
                                                    meta_result=None, data_result=self.serialize_response(train_res)
                                                    ))
                    future_call.add_done_callback(lambda _response: self.dispatch_worker_events(_response.result()))

                elif event_type == commons.MODEL_TEST:
                    self.Test(self.deserialize_response(request.meta), cohort_id)

                elif event_type == commons.UPDATE_MODEL:
                    model_weights = self.deserialize_response(request.data)
                    self.UpdateModel(model_weights, cohort_id)

                elif event_type == 'split':
                    new_cohort_id = len(self.model_adapter)
                    self._init_split(cohort_id, new_cohort_id)

                elif event_type == commons.SHUT_DOWN:
                    self.Stop()

                elif event_type == commons.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                try:
                    self.client_ping()
                except Exception as e:
                    logging.info(f"Caught exception {e} from aggregator, terminating executor {self.this_rank} ...")
                    self.Stop()

if __name__ == "__main__":
    executor = AuxoExecutor(parser.args)
    executor.run()
