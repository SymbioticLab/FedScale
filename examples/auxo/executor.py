import logging

from fedscale.cloud.execution.executor import *
from utils.helper import *
import copy
from config import auxo_config

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
        if len(client_data) == 0:
            state_dicts = self.model_adapter[cohort_id].get_model().state_dict()
            logging.info(f"Client {client_id} has no data, return empty result")
            return {'client_id': client_id, 'moving_loss': 0,
                   'trained_size': 0, 'utility': 0, 'wall_duration': 0,
                   'update_weight': {p: state_dicts[p].data.cpu().numpy()
                       for p in state_dicts},
                   'success': 1}
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
        logging.info(f"[Cohort {cohort_id}] Client {client_id} finished training. ")

        return client_id, train_res

    def _init_train_test_data(self):

        if self.args.data_set == 'femnist':
            from utils.openimg import OpenImage
            train_transform, test_transform = get_data_transform('mnist')
            train_dataset = OpenImage(self.args.data_dir, dataset='femnist', transform=train_transform, client_mapping_file = auxo_config['train_data_map_file']  )
            test_dataset = OpenImage(self.args.data_dir, dataset='femnist', transform=test_transform, client_mapping_file = auxo_config['test_data_map_file'] )
        else:
            raise NotImplementedError
        return train_dataset, test_dataset


    def init_data(self):
        """Return the training and testing dataset

        Returns:
            Tuple of DataPartitioner class: The partioned dataset class for training and testing

        """
        train_dataset, test_dataset = self._init_train_test_data()
        if self.args.task == "rl":
            return train_dataset, test_dataset
        if self.args.task == 'nlp':
            self.collate_fn = collate
        elif self.args.task == 'voice':
            self.collate_fn = voice_collate_fn
        # load data partitionxr (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(
            data=train_dataset, args=self.args, numOfClass=self.args.num_class)
        training_sets.partition_data_helper(
            num_clients=self.args.num_participants, data_map_file=auxo_config['train_data_map_file'])

        testing_sets = DataPartitioner(
            data=test_dataset, args=self.args, numOfClass=self.args.num_class)
        testing_sets.partition_data_helper(
            num_clients=self.args.num_participants, data_map_file=auxo_config['test_data_map_file'])

        logging.info("Data partitioner completes ...")

        return training_sets, testing_sets


    def testing_handler(self, client_list, cohort_id=0):
        """Test model

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
            config (dictionary): Variable arguments from coordinator.
        Returns:
            dictionary: The test result

        """

        test_num_clt = max(len(client_list) // self.num_executors, 1)
        test_client_id_list = client_list[(self.this_rank - 1) * test_num_clt: self.this_rank * test_num_clt]
        logging.info(f"[Cohort {cohort_id}] Test client ID: {test_client_id_list}")
        testResults_accum = {'top_1': 0, 'top_5': 0, 'test_loss': 0, 'test_len': 0}

        test_config = self.override_conf({
            'rank': self.this_rank,
            'memory_capacity': self.args.memory_capacity,
            'tokenizer': tokenizer
        })
        for clt in test_client_id_list:
            client = self.get_client_trainer(test_config)
            data_loader = select_dataset(clt, self.testing_sets,
                                         batch_size=self.args.test_bsz, args=self.args,
                                         isTest=False, collate_fn=self.collate_fn)
            if len(data_loader) > 0:
                test_results = client.test(data_loader, self.model_adapter[cohort_id].get_model(), test_config)
                testResults_accum['top_1'] += test_results['top_1']
                testResults_accum['top_5'] += test_results['top_5']
                testResults_accum['test_loss'] += test_results['test_loss']
                testResults_accum['test_len'] += test_results['test_len']

        # testRes = {'top_1': correct, 'top_5': top_5,
        #            'test_loss': sum_loss, 'test_len': test_len}

        gc.collect()

        return testResults_accum

    def Test(self, config, cohort_id):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group

        Args:
            config (dictionary): The client testing config.

        """
        test_res = self.testing_handler(config['client_id'], cohort_id)
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
