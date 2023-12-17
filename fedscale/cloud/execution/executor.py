# -*- coding: utf-8 -*-
import collections
import gc
import pickle
import random
import time
from argparse import Namespace

import numpy as np
import torch
import wandb

import fedscale.cloud.channels.job_api_pb2 as job_api_pb2
import fedscale.cloud.logger.executor_logging as logger
from fedscale.cloud.channels.channel_context import ClientConnections
from fedscale.cloud.execution.tensorflow_client import TensorflowClient
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.execution.data_processor import collate, voice_collate_fn
from fedscale.cloud.execution.rl_client import RLClient
from fedscale.cloud.fllibs import *
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset


class Executor(object):
    """Abstract class for FedScale executor.

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """

    def __init__(self, args):
        # initiate the executor log path, and executor ips
        logger.initiate_client_setting()

        self.model_adapter = self.get_client_trainer(args).get_model_adapter(
            init_model()
        )

        self.args = args
        self.num_executors = args.num_executors
        # ======== env information ========
        self.this_rank = args.this_rank
        self.executor_id = str(self.this_rank)

        # ======== model and data ========
        self.training_sets = self.test_dataset = None

        # ======== channels ========
        self.aggregator_communicator = ClientConnections(args.ps_ip, args.ps_port)

        # ======== runtime information ========
        self.collate_fn = None
        self.round = 0
        self.start_run_time = time.time()
        self.received_stop_request = False
        self.event_queue = collections.deque()

        if args.wandb_token != "":
            os.environ["WANDB_API_KEY"] = args.wandb_token
            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(
                    project=f"fedscale-{args.job_name}",
                    name=f"executor{args.this_rank}-{args.time_stamp}",
                    group=f"{args.time_stamp}",
                )
            else:
                logging.error("Warning: wandb has already been initialized")

        else:
            self.wandb = None
        super(Executor, self).__init__()

    def setup_env(self):
        """Set up experiments environment"""
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")
        self.setup_seed(seed=1)

    def setup_communication(self):
        """Set up grpc connection"""
        self.init_control_communication()
        self.init_data_communication()

    def setup_seed(self, seed=1):
        """Set random seed for reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        self.aggregator_communicator.connect_to_server()

    def init_data_communication(self):
        """In charge of jumbo data traffics (e.g., fetch training result)"""
        pass

    def init_data(self):
        """Return the training and testing dataset

        Returns:
            Tuple of DataPartitioner class: The partioned dataset class for training and testing

        """
        train_dataset, test_dataset = init_dataset()
        if self.args.task == "rl":
            return train_dataset, test_dataset
        if self.args.task == "nlp":
            self.collate_fn = collate
        elif self.args.task == "voice":
            self.collate_fn = voice_collate_fn
        # load data partitionxr (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(
            data=train_dataset, args=self.args, numOfClass=self.args.num_class
        )
        training_sets.partition_data_helper(
            num_clients=self.args.num_participants,
            data_map_file=self.args.data_map_file,
        )

        testing_sets = DataPartitioner(
            data=test_dataset,
            args=self.args,
            numOfClass=self.args.num_class,
            isTest=True,
        )
        testing_sets.partition_data_helper(num_clients=self.num_executors)

        logging.info("Data partitioner completes ...")

        return training_sets, testing_sets

    def run(self):
        """Start running the executor by setting up execution and communication environment, and monitoring the grpc message."""
        self.setup_env()
        self.training_sets, self.testing_sets = self.init_data()
        self.setup_communication()
        self.event_monitor()

    def dispatch_worker_events(self, request):
        """Add new events to worker queues

        Args:
            request (string): Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        """
        self.event_queue.append(request)

    def deserialize_response(self, responses):
        """Deserialize the response from server

        Args:
            responses (byte stream): Serialized response from server.

        Returns:
            ServerResponse defined at job_api.proto: The deserialized response object from server.

        """
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """Serialize the response to send to server upon assigned job completion

        Args:
            responses (string, bool, or bytes): TorchClient responses after job completion.

        Returns:
            bytes stream: The serialized response object to server.

        """
        return pickle.dumps(responses)

    def UpdateModel(self, model_weights):
        """Receive the broadcasted global model for current round

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model config

        """
        self.round += 1
        self.model_adapter.set_weights(model_weights, is_aggregator=False)

    def Train(self, config):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:
            tuple (int, dictionary): The client id and train result

        """
        client_id, train_config = config["client_id"], config["task_config"]

        if "model" not in config or not config["model"]:
            raise "The 'model' object must be a non-null value in the training config."
        client_conf = self.override_conf(train_config)
        train_res = self.training_handler(
            client_id=client_id, conf=client_conf, model=config["model"]
        )

        # Report execution completion meta information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=str(client_id),
                executor_id=self.executor_id,
                event=commons.CLIENT_TRAIN,
                status=True,
                msg=None,
                meta_result=None,
                data_result=None,
            )
        )
        self.dispatch_worker_events(response)

        return client_id, train_res

    def Test(self, config):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group

        Args:
            config (dictionary): The client testing config.

        """
        test_res = self.testing_handler(model=config["model"])
        test_res = {"executorId": self.this_rank, "results": test_res}

        # Report execution completion information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=self.executor_id,
                executor_id=self.executor_id,
                event=commons.MODEL_TEST,
                status=True,
                msg=None,
                meta_result=None,
                data_result=self.serialize_response(test_res),
            )
        )
        self.dispatch_worker_events(response)

    def Stop(self):
        """Stop the current executor"""
        logging.info(f"Terminating the executor ...")
        self.aggregator_communicator.close_sever_connection()
        self.received_stop_request = True
        if self.wandb != None:
            self.wandb.finish()

    def report_executor_info_handler(self):
        """Return the statistics of training dataset

        Returns:
            int: Return the statistics of training dataset, in simulation return the number of clients

        """
        return self.training_sets.getSize()

    def override_conf(self, config):
        """Override the variable arguments for different client

        Args:
            config (dictionary): The client runtime config.

        Returns:
            dictionary: Variable arguments for client runtime config.

        """
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)

    def get_client_trainer(self, conf):
        """
        Returns a framework-specific client that handles training and evaluation.
        :param conf: job config
        :return: framework-specific client instance
        """
        if conf.engine == commons.TENSORFLOW:
            return TensorflowClient(conf)
        elif conf.engine == commons.PYTORCH:
            if conf.task == "rl":
                return RLClient(conf)
            else:
                return TorchClient(conf)
        raise "Currently, FedScale supports tensorflow and pytorch."

    def training_handler(self, client_id, conf, model):
        """Train model given client id

        Args:
            client_id (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result

        """
        self.model_adapter.set_weights(model, is_aggregator=False)
        conf.client_id = client_id
        conf.tokenizer = tokenizer
        client_data = (
            self.training_sets
            if self.args.task == "rl"
            else select_dataset(
                client_id,
                self.training_sets,
                batch_size=conf.batch_size,
                args=self.args,
                collate_fn=self.collate_fn,
            )
        )
        client = self.get_client_trainer(self.args)
        train_res = client.train(
            client_data=client_data, model=self.model_adapter.get_model(), conf=conf
        )

        return train_res

    def testing_handler(self, model):
        """Test model

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
            config (dictionary): Variable arguments from coordinator.
        Returns:
            dictionary: The test result

        """
        self.model_adapter.set_weights(model, is_aggregator=False)
        test_config = self.override_conf(
            {
                "rank": self.this_rank,
                "memory_capacity": self.args.memory_capacity,
                "tokenizer": tokenizer,
            }
        )
        client = self.get_client_trainer(test_config)
        data_loader = select_dataset(
            self.this_rank,
            self.testing_sets,
            batch_size=self.args.test_bsz,
            args=self.args,
            isTest=True,
            collate_fn=self.collate_fn,
        )

        test_results = client.test(
            data_loader, model=self.model_adapter.get_model(), conf=test_config
        )
        self.log_test_result(test_results)
        gc.collect()

        return test_results

    def client_register(self):
        """Register the executor information to the aggregator"""
        start_time = time.time()
        while time.time() - start_time < 180:
            try:
                response = self.aggregator_communicator.stub.CLIENT_REGISTER(
                    job_api_pb2.RegisterRequest(
                        client_id=self.executor_id,
                        executor_id=self.executor_id,
                        executor_info=self.serialize_response(
                            self.report_executor_info_handler()
                        ),
                    )
                )
                self.dispatch_worker_events(response)
                break
            except Exception as e:
                logging.warning(
                    f"Failed to connect to aggregator {e}. Will retry in 5 sec."
                )
                time.sleep(5)

    def client_ping(self):
        """Ping the aggregator for new task"""
        response = self.aggregator_communicator.stub.CLIENT_PING(
            job_api_pb2.PingRequest(
                client_id=self.executor_id, executor_id=self.executor_id
            )
        )
        self.dispatch_worker_events(response)

    def event_monitor(self):
        """Activate event handler once receiving new message"""
        logging.info("Start monitoring events ...")
        self.client_register()

        while not self.received_stop_request:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event

                if current_event == commons.CLIENT_TRAIN:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    train_config["model"] = train_model
                    train_config["client_id"] = int(train_config["client_id"])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    future_call = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                        job_api_pb2.CompleteRequest(
                            client_id=str(client_id),
                            executor_id=self.executor_id,
                            event=commons.UPLOAD_MODEL,
                            status=True,
                            msg=None,
                            meta_result=None,
                            data_result=self.serialize_response(train_res),
                        )
                    )
                    future_call.add_done_callback(
                        lambda _response: self.dispatch_worker_events(
                            _response.result()
                        )
                    )

                elif current_event == commons.MODEL_TEST:
                    test_config = self.deserialize_response(request.meta)
                    test_model = self.deserialize_response(request.data)
                    test_config["model"] = test_model
                    test_config["client_id"] = int(test_config["client_id"])
                    self.Test(test_config)

                elif current_event == commons.UPDATE_MODEL:
                    model_weights = self.deserialize_response(request.data)
                    self.UpdateModel(model_weights)

                elif current_event == commons.SHUT_DOWN:
                    self.Stop()

                elif current_event == commons.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                try:
                    self.client_ping()
                except Exception as e:
                    logging.info(
                        f"Caught exception {e} from aggregator, terminating executor {self.this_rank} ..."
                    )
                    self.Stop()

    def log_test_result(self, test_res):
        """Log test results to wandb server if enabled"""
        acc = round(test_res["top_1"] / test_res["test_len"], 4)
        acc_5 = round(test_res["top_5"] / test_res["test_len"], 4)
        test_loss = test_res["test_loss"] / test_res["test_len"]
        if self.wandb != None:
            self.wandb.log(
                {
                    "Test/round_to_top1_accuracy": acc,
                    "Test/round_to_top5_accuracy": acc_5,
                    "Test/round_to_loss": test_loss,
                },
                step=self.round,
            )


if __name__ == "__main__":
    executor = Executor(parser.args)
    executor.run()
