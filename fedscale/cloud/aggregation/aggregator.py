# -*- coding: utf-8 -*-
import collections
import copy
import math
import os
import pickle
import random
import threading
import time
from concurrent import futures

import grpc
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

import fedscale.cloud.channels.job_api_pb2_grpc as job_api_pb2_grpc
import fedscale.cloud.logger.aggregator_logging as logger
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.channels import job_api_pb2
from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud.internal.tensorflow_model_adapter import TensorflowModelAdapter
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter
from fedscale.cloud.resource_manager import ResourceManager
from fedscale.cloud.fllibs import *
from torch.utils.tensorboard import SummaryWriter

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB


class Aggregator(job_api_pb2_grpc.JobServiceServicer):
    """This centralized aggregator collects training/testing feedbacks from executors

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """

    def __init__(self, args):
        # init aggregator loger
        logger.initiate_aggregator_setting()

        logging.info(f"Job args {args}")
        self.args = args
        self.experiment_mode = args.experiment_mode
        self.device = args.cuda_device if args.use_cuda else torch.device("cpu")

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.0
        self.round_duration = 0.0
        self.resource_manager = ResourceManager(self.experiment_mode)
        self.client_manager = self.init_client_manager(args=args)

        # ======== model and data ========
        self.model_wrapper = None
        self.model_in_update = 0
        self.update_lock = threading.Lock()
        # all weights including bias/#_batch_tracked (e.g., state_dict)
        self.model_weights = None
        self.temp_model_path = os.path.join(
            logger.logDir, "model_" + str(args.this_rank) + ".npy"
        )
        self.last_saved_round = 0

        # ======== channels ========
        self.connection_timeout = self.args.connection_timeout
        self.executors = None
        self.grpc_server = None

        # ======== Event Queue =======
        self.individual_client_events = {}  # Unicast
        self.server_events_queue = collections.deque()
        self.broadcast_events_queue = collections.deque()  # Broadcast

        # ======== runtime information ========
        self.tasks_round = 0
        self.num_of_clients = 0

        # NOTE: sampled_participants = sampled_executors in deployment,
        # because every participant is an executor. However, in simulation mode,
        # executors is the physical machines (VMs), thus:
        # |sampled_executors| << |sampled_participants| as an VM may run multiple participants
        self.sampled_participants = []
        self.sampled_executors = []

        self.round_stragglers = []
        self.model_update_size = 0.0

        self.collate_fn = None
        self.round = 0

        self.start_run_time = time.time()
        self.client_conf = {}

        self.stats_util_accumulator = []
        self.loss_accumulator = []
        self.client_training_results = []

        # number of registered executors
        self.registered_executor_info = set()
        self.test_result_accumulator = []
        self.testing_history = {
            "data_set": args.data_set,
            "model": args.model,
            "sample_mode": args.sample_mode,
            "gradient_policy": args.gradient_policy,
            "task": args.task,
            "perf": collections.OrderedDict(),
        }
        self.log_writer = SummaryWriter(log_dir=logger.logDir)
        if args.wandb_token != "":
            os.environ["WANDB_API_KEY"] = args.wandb_token
            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(
                    project=f"fedscale-{args.job_name}",
                    name=f"aggregator{args.this_rank}-{args.time_stamp}",
                    group=f"{args.time_stamp}",
                )
                self.wandb.config.update(
                    {
                        "num_participants": args.num_participants,
                        "data_set": args.data_set,
                        "model": args.model,
                        "gradient_policy": args.gradient_policy,
                        "eval_interval": args.eval_interval,
                        "rounds": args.rounds,
                        "batch_size": args.batch_size,
                        "use_cuda": args.use_cuda,
                    }
                )
            else:
                logging.error("Warning: wandb has already been initialized")
            # self.wandb.run.name = f'{args.job_name}-{args.time_stamp}'
        else:
            self.wandb = None

        # ======== Task specific ============
        self.init_task_context()

    def setup_env(self):
        """Set up experiments environment and server optimizer"""
        self.setup_seed(seed=1)

    def setup_seed(self, seed=1):
        """Set global random seed for better reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        logging.info(f"Initiating control plane communication ...")
        if self.experiment_mode == commons.SIMULATION_MODE:
            num_of_executors = 0
            for ip_numgpu in self.args.executor_configs.split("="):
                ip, numgpu = ip_numgpu.split(":")
                for numexe in numgpu.strip()[1:-1].split(","):
                    for _ in range(int(numexe.strip())):
                        num_of_executors += 1
            self.executors = list(range(num_of_executors))
        else:
            self.executors = list(range(self.args.num_participants))

        # initiate a server process
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
        )
        job_api_pb2_grpc.add_JobServiceServicer_to_server(self, self.grpc_server)
        port = "[::]:{}".format(self.args.ps_port)

        logging.info(
            f"%%%%%%%%%% Opening aggregator server using port {port} %%%%%%%%%%"
        )

        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()

    def init_data_communication(self):
        """For jumbo traffics (e.g., training results)."""
        pass

    def init_model(self):
        """Initialize the model"""
        if self.args.engine == commons.TENSORFLOW:
            self.model_wrapper = TensorflowModelAdapter(init_model())
        elif self.args.engine == commons.PYTORCH:
            self.model_wrapper = TorchModelAdapter(
                init_model(),
                optimizer=TorchServerOptimizer(
                    self.args.gradient_policy, self.args, self.device
                ),
            )
        else:
            raise ValueError(f"{self.args.engine} is not a supported engine.")
        self.model_weights = self.model_wrapper.get_weights()

    def init_task_context(self):
        """Initiate execution context for specific tasks"""
        if self.args.task == "detection":
            cfg_from_file(self.args.cfg_file)
            np.random.seed(self.cfg.RNG_SEED)
            self.imdb, _, _, _ = combined_roidb(
                "voc_2007_test", ["DATA_DIR", self.args.data_dir], server=True
            )

    def init_client_manager(self, args):
        """Initialize client sampler

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

        Returns:
            ClientManager: The client manager class

        Currently we implement two client managers:

        1. Random client sampler - it selects participants randomly in each round
        [Ref]: https://arxiv.org/abs/1902.01046

        2. Oort sampler
        Oort prioritizes the use of those clients who have both data that offers the greatest utility
        in improving model accuracy and the capability to run training quickly.
        [Ref]: https://www.usenix.org/conference/osdi21/presentation/lai

        """

        # sample_mode: random or oort
        client_manager = ClientManager(args.sample_mode, args=args)

        return client_manager

    def load_client_profile(self, file_path):
        """For Simulation Mode: load client profiles/traces

        Args:
            file_path (string): File path for the client profiles/traces

        Returns:
            dictionary: Return the client profiles/traces

        """
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, "rb") as fin:
                # {client_id: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile

    def client_register_handler(self, executorId, info):
        """Triggered once receive new executor registration.

        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        logging.info(f"Loading {len(info['size'])} client traces ...")
        for _size in info["size"]:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = (
                (self.num_of_clients + 1) % len(self.client_profiles)
                if len(self.client_profiles) > 0
                else 1
            )
            systemProfile = self.client_profiles.get(
                mapped_id, {"computation": 1.0, "communication": 1.0}
            )

            client_id = (
                (self.num_of_clients + 1)
                if self.experiment_mode == commons.SIMULATION_MODE
                else executorId
            )
            self.client_manager.register_client(
                executorId, client_id, size=_size, speed=systemProfile
            )
            self.client_manager.registerDuration(
                client_id,
                batch_size=self.args.batch_size,
                local_steps=self.args.local_steps,
                upload_size=self.model_update_size,
                download_size=self.model_update_size,
            )
            self.num_of_clients += 1

        logging.info(
            "Info of all feasible clients {}".format(self.client_manager.getDataInfo())
        )

    def executor_info_handler(self, executorId, info):
        """Handler for register executor info and it will start the round after number of
        executor reaches requirement.

        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        self.registered_executor_info.add(executorId)
        logging.info(
            f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}"
        )

        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout
        if self.experiment_mode == commons.SIMULATION_MODE:
            if len(self.registered_executor_info) == len(self.executors):
                self.client_register_handler(executorId, info)
                # start to sample clients
                self.round_completion_handler()
        else:
            # In real deployments, we need to register for each client
            self.client_register_handler(executorId, info)
            if len(self.registered_executor_info) == len(self.executors):
                self.round_completion_handler()

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """Record sampled client execution information in last round. In the SIMULATION_MODE,
        further filter the sampled_client and pick the top num_clients_to_collect clients.

        Args:
            sampled_clients (list of int): Sampled clients from client manager
            num_clients_to_collect (int): The number of clients actually needed for next round.

        Returns:
            Tuple: (the List of clients to run, the List of stragglers in the round, a Dict of the virtual clock of each
            client, the duration of the aggregation round, and the durations of each client's task).

        """
        if self.experiment_mode == commons.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            completionTimes = []
            completed_client_clock = {}
            # 1. remove dummy clients that are not available to the end of training
            for client_to_run in sampled_clients:
                client_cfg = self.client_conf.get(client_to_run, self.args)

                exe_cost = self.client_manager.get_completion_time(
                    client_to_run,
                    batch_size=client_cfg.batch_size,
                    local_steps=client_cfg.local_steps,
                    upload_size=self.model_update_size,
                    download_size=self.model_update_size,
                )

                roundDuration = exe_cost["computation"] + exe_cost["communication"]
                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager.isClientActive(
                    client_to_run, roundDuration + self.global_virtual_clock
                ):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                    completed_client_clock[client_to_run] = exe_cost

            num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))
            # 2. get the top-k completions to remove stragglers
            workers_sorted_by_completion_time = sorted(
                range(len(completionTimes)), key=lambda k: completionTimes[k]
            )
            top_k_index = workers_sorted_by_completion_time[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            stragglers = [
                sampledClientsReal[k]
                for k in workers_sorted_by_completion_time[num_clients_to_collect:]
            ]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            return (
                clients_to_run,
                stragglers,
                completed_client_clock,
                round_duration,
                completionTimes[:num_clients_to_collect],
            )
        else:
            completed_client_clock = {
                client: {"computation": 1, "communication": 1}
                for client in sampled_clients
            }
            completionTimes = [1 for c in sampled_clients]
            return (
                sampled_clients,
                sampled_clients,
                completed_client_clock,
                1,
                completionTimes,
            )

    def run(self):
        """Start running the aggregator server by setting up execution
        and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.client_profiles = self.load_client_profile(
            file_path=self.args.device_conf_file
        )

        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        self.model_update_size = (
            sys.getsizeof(pickle.dumps(self.model_wrapper)) / 1024.0 * 8.0
        )  # kbits

        self.event_monitor()
        self.stop()

    def _is_first_result_in_round(self):
        return self.model_in_update == 1

    def _is_last_result_in_round(self):
        return self.model_in_update == self.tasks_round

    def select_participants(self, select_num_participants, overcommitment=1.3):
        """Select clients for next round.

        Args:
            select_num_participants (int): Number of clients to select.
            overcommitment (float): Overcommit ration for next round.

        Returns:
            list of int: The list of sampled clients id.

        """
        return sorted(
            self.client_manager.select_participants(
                int(select_num_participants * overcommitment),
                cur_time=self.global_virtual_clock,
            ),
        )

    def client_completion_handler(self, results):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache

        Args:
            results (dictionary): client's training result

        """
        # Format:
        #       -results = {'client_id':client_id, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        if self.args.gradient_policy in ["q-fedavg"]:
            self.client_training_results.append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results["utility"])
        self.loss_accumulator.append(results["moving_loss"])

        self.client_manager.register_feedback(
            results["client_id"],
            results["utility"],
            auxi=math.sqrt(results["moving_loss"]),
            time_stamp=self.round,
            duration=self.virtual_client_clock[results["client_id"]]["computation"]
            + self.virtual_client_clock[results["client_id"]]["communication"],
        )

        # ================== Aggregate weights ======================
        self.update_lock.acquire()

        self.model_in_update += 1
        self.update_weight_aggregation(results)

        self.update_lock.release()

    def update_weight_aggregation(self, results):
        """Updates the aggregation with the new results.

        :param results: the results collected from the client.
        """
        update_weights = results["update_weight"]
        if type(update_weights) is dict:
            update_weights = [x for x in update_weights.values()]
        if self._is_first_result_in_round():
            self.model_weights = update_weights
        else:
            self.model_weights = [
                weight + update_weights[i]
                for i, weight in enumerate(self.model_weights)
            ]
        if self._is_last_result_in_round():
            self.model_weights = [
                np.divide(weight, self.tasks_round) for weight in self.model_weights
            ]
            self.model_wrapper.set_weights(
                copy.deepcopy(self.model_weights),
                client_training_results=self.client_training_results,
            )

    def aggregate_test_result(self):
        accumulator = self.test_result_accumulator[0]
        for i in range(1, len(self.test_result_accumulator)):
            if self.args.task == "detection":
                for key in accumulator:
                    if key == "boxes":
                        for j in range(596):
                            accumulator[key][j] = (
                                accumulator[key][j]
                                + self.test_result_accumulator[i][key][j]
                            )
                    else:
                        accumulator[key] += self.test_result_accumulator[i][key]
            else:
                for key in accumulator:
                    accumulator[key] += self.test_result_accumulator[i][key]
        self.testing_history["perf"][self.round] = {
            "round": self.round,
            "clock": self.global_virtual_clock,
        }
        for metric_name in accumulator.keys():
            if metric_name == "test_loss":
                self.testing_history["perf"][self.round]["loss"] = (
                    accumulator["test_loss"]
                    if self.args.task == "detection"
                    else accumulator["test_loss"] / accumulator["test_len"]
                )
            elif metric_name not in ["test_len"]:
                self.testing_history["perf"][self.round][metric_name] = (
                    accumulator[metric_name] / accumulator["test_len"]
                )

        round_perf = self.testing_history["perf"][self.round]
        logging.info(
            "FL Testing in round: {}, virtual_clock: {}, results: {}".format(
                self.round, self.global_virtual_clock, round_perf
            )
        )

    def update_default_task_config(self):
        """Update the default task configuration after each round"""
        if self.round % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate * self.args.decay_factor,
                self.args.min_learning_rate,
            )

    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        self.global_virtual_clock += self.round_duration
        self.round += 1
        last_round_avg_util = sum(self.stats_util_accumulator) / max(
            1, len(self.stats_util_accumulator)
        )
        # assign avg reward to explored, but not ran workers
        for client_id in self.round_stragglers:
            self.client_manager.register_feedback(
                client_id,
                last_round_avg_util,
                time_stamp=self.round,
                duration=self.virtual_client_clock[client_id]["computation"]
                + self.virtual_client_clock[client_id]["communication"],
                success=False,
            )

        avg_loss = sum(self.loss_accumulator) / max(1, len(self.loss_accumulator))
        logging.info(
            f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, Planned participants: "
            + f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}"
        )

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        # update select participants
        self.sampled_participants = self.select_participants(
            select_num_participants=self.args.num_participants,
            overcommitment=self.args.overcommitment,
        )
        (
            clients_to_run,
            round_stragglers,
            virtual_client_clock,
            round_duration,
            flatten_client_duration,
        ) = self.tictak_client_tasks(
            self.sampled_participants, self.args.num_participants
        )

        logging.info(f"Selected participants to run: {clients_to_run}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clients_to_run)
        self.tasks_round = len(clients_to_run)

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id) for c_id in self.sampled_participants]
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.flatten_client_duration = np.array(flatten_client_duration)
        self.round_duration = round_duration
        self.model_in_update = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []
        self.loss_accumulator = []
        self.update_default_task_config()

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0 or self.round == 1:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.START_ROUND)

    def log_train_result(self, avg_loss):
        """Log training result on TensorBoard and optionally WanDB"""
        self.log_writer.add_scalar("Train/round_to_loss", avg_loss, self.round)
        self.log_writer.add_scalar(
            "Train/time_to_train_loss (min)", avg_loss, self.global_virtual_clock / 60.0
        )
        self.log_writer.add_scalar(
            "Train/round_duration (min)", self.round_duration / 60.0, self.round
        )
        self.log_writer.add_histogram(
            "Train/client_duration (min)", self.flatten_client_duration, self.round
        )

        if self.wandb != None:
            self.wandb.log(
                {
                    "Train/round_to_loss": avg_loss,
                    "Train/round_duration (min)": self.round_duration / 60.0,
                    "Train/client_duration (min)": self.flatten_client_duration,
                    "Train/time_to_round (min)": self.global_virtual_clock / 60.0,
                },
                step=self.round,
            )

    def log_test_result(self):
        """Log testing result on TensorBoard and optionally WanDB"""
        self.log_writer.add_scalar(
            "Test/round_to_loss",
            self.testing_history["perf"][self.round]["loss"],
            self.round,
        )
        self.log_writer.add_scalar(
            "Test/round_to_accuracy",
            self.testing_history["perf"][self.round]["top_1"],
            self.round,
        )
        self.log_writer.add_scalar(
            "Test/time_to_test_loss (min)",
            self.testing_history["perf"][self.round]["loss"],
            self.global_virtual_clock / 60.0,
        )
        self.log_writer.add_scalar(
            "Test/time_to_test_accuracy (min)",
            self.testing_history["perf"][self.round]["top_1"],
            self.global_virtual_clock / 60.0,
        )

    def save_model(self):
        """Save model to the wandb server if enabled"""
        if parser.args.save_checkpoint and self.last_saved_round < self.round:
            self.last_saved_round = self.round
            np.save(self.temp_model_path, self.model_weights)
            if self.wandb != None:
                artifact = self.wandb.Artifact(
                    name="model_" + str(self.this_rank), type="model"
                )
                artifact.add_file(local_path=self.temp_model_path)
                self.wandb.log_artifact(artifact)

    def deserialize_response(self, responses):
        """Deserialize the response from executor

        Args:
            responses (byte stream): Serialized response from executor.

        Returns:
            string, bool, or bytes: The deserialized response object from executor.
        """
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """Serialize the response to send to server upon assigned job completion

        Args:
            responses (ServerResponse): Serialized response from server.

        Returns:
            bytes: The serialized response object to server.

        """
        return pickle.dumps(responses)

    def testing_completion_handler(self, client_id, results):
        """Each executor will handle a subset of testing dataset

        Args:
            client_id (int): The client id.
            results (dictionary): The client test results.

        """

        results = results["results"]

        # List append is thread-safe
        self.test_result_accumulator.append(results)

        # Have collected all testing results

        if len(self.test_result_accumulator) == len(self.executors):
            self.aggregate_test_result()
            # Dump the testing result
            with open(os.path.join(logger.logDir, "testing_perf"), "wb") as fout:
                pickle.dump(self.testing_history, fout)

            self.save_model()

            if len(self.loss_accumulator):
                logging.info("logging test result")
                self.log_test_result()

            self.broadcast_events_queue.append(commons.START_ROUND)

    def broadcast_aggregator_events(self, event):
        """Issue tasks (events) to aggregator worker processes by adding grpc request event
        (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        """
        self.broadcast_events_queue.append(event)

    def dispatch_client_events(self, event, clients=None):
        """Issue tasks (events) to clients

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.
            clients (list of int): target client ids for event.

        """
        if clients is None:
            clients = self.sampled_executors

        for client_id in clients:
            self.individual_client_events[client_id].append(event)

    def get_client_conf(self, client_id):
        """Training configurations that will be applied on clients,
        developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: TorchClient training config.

        """
        conf = {
            "learning_rate": self.args.learning_rate,
        }
        return conf

    def create_client_task(self, executor_id):
        """Issue a new client training task to specific executor

        Args:
            executorId (int): Executor Id.

        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        next_client_id = self.resource_manager.get_next_task(executor_id)
        train_config = None
        # NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
        if next_client_id is not None:
            config = self.get_client_conf(next_client_id)
            train_config = {"client_id": next_client_id, "task_config": config}
        return train_config, self.model_wrapper.get_weights()

    def get_test_config(self, client_id):
        """FL model testing on clients, developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: The testing config for new task.

        """
        return {"client_id": client_id}, self.model_wrapper.get_weights()

    def get_shutdown_config(self, client_id):
        """Shutdown config for client, developers can further define personalized client config here.

        Args:
            client_id (int): TorchClient id.

        Returns:
            dictionary: Shutdown config for new task.

        """
        return {"client_id": client_id}

    def add_event_handler(self, client_id, event, meta, data):
        """Due to the large volume of requests, we will put all events into a queue first.

        Args:
            client_id (int): The client id.
            event (string): grpc event MODEL_TEST or UPLOAD_MODEL.
            meta (dictionary or string): Meta message for grpc communication, could be event.
            data (dictionary): Data transferred in grpc communication, could be model parameters, test result.

        """
        self.server_events_queue.append((client_id, event, meta, data))

    def CLIENT_REGISTER(self, request, context):
        """FL TorchClient register to the aggregator

        Args:
            request (RegisterRequest): Registeration request info from executor.

        Returns:
            ServerResponse: Server response to registeration request

        """

        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id uses the same executor_id (VMs) in simulations
        executor_id = request.executor_id
        executor_info = self.deserialize_response(request.executor_info)
        if executor_id not in self.individual_client_events:
            # logging.info(f"Detect new client: {executor_id}, executor info: {executor_info}")
            self.individual_client_events[executor_id] = collections.deque()
        else:
            logging.info(f"Previous client: {executor_id} resumes connecting")

        # We can customize whether to admit the clients here
        self.executor_info_handler(executor_id, executor_info)
        dummy_data = self.serialize_response(commons.DUMMY_RESPONSE)

        return job_api_pb2.ServerResponse(
            event=commons.DUMMY_EVENT, meta=dummy_data, data=dummy_data
        )

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
            current_event = self.individual_client_events[executor_id].popleft()
            if current_event == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(executor_id)
                if response_msg is None:
                    current_event = commons.DUMMY_EVENT
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(
                            commons.CLIENT_TRAIN
                        )
            elif current_event == commons.MODEL_TEST:
                response_msg, response_data = self.get_test_config(client_id)
            elif current_event == commons.UPDATE_MODEL:
                response_data = self.model_wrapper.get_weights()
            elif current_event == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg
        ), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        response = job_api_pb2.ServerResponse(
            event=current_event, meta=response_msg, data=response_data
        )
        if current_event != commons.DUMMY_EVENT:
            logging.info(f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")

        return response

    def CLIENT_EXECUTE_COMPLETION(self, request, context):
        """FL clients complete the execution task.

        Args:
            request (CompleteRequest): Complete request info from executor.

        Returns:
            ServerResponse: Server response to job completion request

        """

        executor_id, client_id, event = (
            request.executor_id,
            request.client_id,
            request.event,
        )
        execution_status, execution_msg = request.status, request.msg
        meta_result, data_result = request.meta_result, request.data_result

        if event == commons.CLIENT_TRAIN:
            # Training results may be uploaded in CLIENT_EXECUTE_RESULT request later,
            # so we need to specify whether to ask client to do so (in case of straggler/timeout in real FL).
            if execution_status is False:
                logging.error(
                    f"Executor {executor_id} fails to run client {client_id}, due to {execution_msg}"
                )

            # TODO: whether we should schedule tasks when client_ping or client_complete
            if self.resource_manager.has_next_task(executor_id):
                # NOTE: we do not pop the train immediately in simulation mode,
                # since the executor may run multiple clients
                if (
                    commons.CLIENT_TRAIN
                    not in self.individual_client_events[executor_id]
                ):
                    self.individual_client_events[executor_id].append(
                        commons.CLIENT_TRAIN
                    )

        elif event in (commons.MODEL_TEST, commons.UPLOAD_MODEL):
            self.add_event_handler(executor_id, event, meta_result, data_result)
        else:
            logging.error(f"Received undefined event {event} from client {client_id}")

        return self.CLIENT_PING(request, context)

    def event_monitor(self):
        """Activate event handler according to the received new message"""
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events to clients
            if len(self.broadcast_events_queue) > 0:
                current_event = self.broadcast_events_queue.popleft()

                if current_event in (commons.UPDATE_MODEL, commons.MODEL_TEST):
                    self.dispatch_client_events(current_event)

                elif current_event == commons.START_ROUND:
                    self.dispatch_client_events(commons.CLIENT_TRAIN)

                elif current_event == commons.SHUT_DOWN:
                    self.dispatch_client_events(commons.SHUT_DOWN)
                    break

            # Handle events queued on the aggregator
            elif len(self.server_events_queue) > 0:
                (
                    client_id,
                    current_event,
                    meta,
                    data,
                ) = self.server_events_queue.popleft()

                if current_event == commons.UPLOAD_MODEL:
                    self.client_completion_handler(self.deserialize_response(data))
                    if len(self.stats_util_accumulator) == self.tasks_round:
                        self.round_completion_handler()

                elif current_event == commons.MODEL_TEST:
                    self.testing_completion_handler(
                        client_id, self.deserialize_response(data)
                    )

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)

    def stop(self):
        """Stop the aggregator"""
        logging.info(f"Terminating the aggregator ...")
        if self.wandb != None:
            self.wandb.finish()
        time.sleep(5)


if __name__ == "__main__":
    aggregator = Aggregator(parser.args)
    aggregator.run()
