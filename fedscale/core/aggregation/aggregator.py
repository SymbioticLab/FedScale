# -*- coding: utf-8 -*-

import pickle
import threading
from concurrent import futures

import grpc
import torch
from torch.utils.tensorboard import SummaryWriter

import fedscale.core.channels.job_api_pb2_grpc as job_api_pb2_grpc
from fedscale.core import commons
from fedscale.core.channels import job_api_pb2
from fedscale.core.logger.aggragation import *
from fedscale.core.resource_manager import ResourceManager

MAX_MESSAGE_LENGTH = 1*1024*1024*1024  # 1GB


class Aggregator(job_api_pb2_grpc.JobServiceServicer):
    """This centralized aggregator collects training/testing feedbacks from executors
    
    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """
    def __init__(self, args):
        logging.info(f"Job args {args}")

        self.args = args
        self.experiment_mode = args.experiment_mode
        self.device = args.cuda_device if args.use_cuda else torch.device(
            'cpu')

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.
        self.round_duration = 0.
        self.resource_manager = ResourceManager(self.experiment_mode)
        self.client_manager = self.init_client_manager(args=args)

        # ======== model and data ========
        self.model = None
        self.model_in_update = 0
        self.update_lock = threading.Lock()
        # all weights including bias/#_batch_tracked (e.g., state_dict)
        self.model_weights = collections.OrderedDict()
        self.last_gradient_weights = []  # only gradient variables
        self.model_state_dict = None
        # NOTE: if <param_name, param_tensor> (e.g., model.parameters() in PyTorch), then False
        # True, if <param_name, list_param_tensors> (e.g., layer.get_weights() in Tensorflow)
        self.using_group_params = self.args.engine == commons.TENSORFLOW

        # ======== channels ========
        self.connection_timeout = self.args.connection_timeout
        self.executors = None
        self.grpc_server = None

        # ======== Event Queue =======
        self.individual_client_events = {}    # Unicast
        self.sever_events_queue = collections.deque()
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
        self.model_update_size = 0.

        self.collate_fn = None
        self.task = args.task
        self.round = 0

        self.start_run_time = time.time()
        self.client_conf = {}

        self.stats_util_accumulator = []
        self.loss_accumulator = []
        self.client_training_results = []

        # number of registered executors
        self.registered_executor_info = set()
        self.test_result_accumulator = []
        self.testing_history = {'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                                'gradient_policy': args.gradient_policy, 'task': args.task, 'perf': collections.OrderedDict()}

        self.log_writer = SummaryWriter(log_dir=logDir)

        # ======== Task specific ============
        self.init_task_context()

    def setup_env(self):
        """Set up experiments environment and server optimizer
        """
        self.setup_seed(seed=1)
        self.optimizer = ServerOptimizer(
            self.args.gradient_policy, self.args, self.device)

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
                ip, numgpu = ip_numgpu.split(':')
                for numexe in numgpu.strip()[1:-1].split(','):
                    for _ in range(int(numexe.strip())):
                        num_of_executors += 1
            self.executors = list(range(num_of_executors))
        else:
            self.executors = list(range(self.args.num_participants))

        # initiate a server process
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ],
        )
        job_api_pb2_grpc.add_JobServiceServicer_to_server(
            self, self.grpc_server)
        port = '[::]:{}'.format(self.args.ps_port)

        logging.info(f'%%%%%%%%%% Opening aggregator sever using port {port} %%%%%%%%%%')

        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()

    def init_data_communication(self):
        """For jumbo traffics (e.g., training results).
        """
        pass

    def init_model(self):
        """Load the model architecture
        """
        assert self.args.engine == commons.PYTORCH, "Please define model for non-PyTorch models"

        self.model = init_model()

        # Initiate model parameters dictionary <param_name, param>
        self.model_weights = self.model.state_dict()

    def init_task_context(self):
        """Initiate execution context for specific tasks
        """
        if self.args.task == "detection":
            cfg_from_file(self.args.cfg_file)
            np.random.seed(self.cfg.RNG_SEED)
            self.imdb, _, _, _ = combined_roidb(
                "voc_2007_test", ['DATA_DIR', self.args.data_dir], server=True)

    def init_client_manager(self, args):
        """ Initialize client sampler

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
        
        Returns:
            clientManager: The client manager class

        Currently we implement two client managers:

        1. Random client sampler - it selects participants randomly in each round 
        [Ref]: https://arxiv.org/abs/1902.01046

        2. Oort sampler
        Oort prioritizes the use of those clients who have both data that offers the greatest utility
        in improving model accuracy and the capability to run training quickly.
        [Ref]: https://www.usenix.org/conference/osdi21/presentation/lai

        """

        # sample_mode: random or oort
        client_manager = clientManager(args.sample_mode, args=args)

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
            with open(file_path, 'rb') as fin:
                # {clientId: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile

    def client_register_handler(self, executorId, info):
        """Triggered once receive new executor registration.
        
        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        logging.info(f"Loading {len(info['size'])} client traces ...")
        for _size in info['size']:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = (self.num_of_clients+1) % len(
                self.client_profiles) if len(self.client_profiles) > 0 else 1
            systemProfile = self.client_profiles.get(
                mapped_id, {'computation': 1.0, 'communication': 1.0})

            clientId = (
                self.num_of_clients+1) if self.experiment_mode == commons.SIMULATION_MODE else executorId
            self.client_manager.register_client(
                executorId, clientId, size=_size, speed=systemProfile)
            self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
                                                 upload_step=self.args.local_steps, upload_size=self.model_update_size, download_size=self.model_update_size)
            self.num_of_clients += 1

        logging.info("Info of all feasible clients {}".format(
            self.client_manager.getDataInfo()))

    def executor_info_handler(self, executorId, info):
        """Handler for register executor info and it will start the round after number of
        executor reaches requirement.
        
        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        self.registered_executor_info.add(executorId)
        logging.info(f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}")

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
            tuple: Return the sampled clients and client execution information in the last round.

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

                exe_cost = self.client_manager.getCompletionTime(client_to_run,
                                                                 batch_size=client_cfg.batch_size, upload_step=client_cfg.local_steps,
                                                                 upload_size=self.model_update_size, download_size=self.model_update_size)

                roundDuration = exe_cost['computation'] + \
                    exe_cost['communication']
                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                    completed_client_clock[client_to_run] = exe_cost

            num_clients_to_collect = min(
                num_clients_to_collect, len(completionTimes))
            # 2. get the top-k completions to remove stragglers
            sortedWorkersByCompletion = sorted(
                range(len(completionTimes)), key=lambda k: completionTimes[k])
            top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            dummy_clients = [sampledClientsReal[k]
                             for k in sortedWorkersByCompletion[num_clients_to_collect:]]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            return (clients_to_run, dummy_clients,
                    completed_client_clock, round_duration,
                    completionTimes[:num_clients_to_collect])
        else:
            completed_client_clock = {
                client: {'computation': 1, 'communication': 1} for client in sampled_clients}
            completionTimes = [1 for c in sampled_clients]
            return (sampled_clients, sampled_clients, completed_client_clock,
                    1, completionTimes)

    def run(self):
        """Start running the aggregator server by setting up execution 
        and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        self.save_last_param()
        self.model_update_size = sys.getsizeof(
            pickle.dumps(self.model))/1024.0*8.  # kbits
        self.client_profiles = self.load_client_profile(
            file_path=self.args.device_conf_file)

        self.event_monitor()

    def select_participants(self, select_num_participants, overcommitment=1.3):
        """Select clients for next round.

        Args: 
            select_num_participants (int): Number of clients to select.
            overcommitment (float): Overcommit ration for next round.

        Returns:
            list of int: The list of sampled clients id.

        """
        return sorted(self.client_manager.select_participants(
            int(select_num_participants*overcommitment),
            cur_time=self.global_virtual_clock),
        )

    def client_completion_handler(self, results):
        """We may need to keep all updates from clients, 
        if so, we need to append results to the cache
        
        Args:
            results (dictionary): client's training result
        
        """
        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        if self.args.gradient_policy in ['q-fedavg']:
            self.client_training_results.append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])

        self.client_manager.register_feedback(results['clientId'], results['utility'],
                                          auxi=math.sqrt(
                                              results['moving_loss']),
                                          time_stamp=self.round,
                                          duration=self.virtual_client_clock[results['clientId']]['computation'] +
                                          self.virtual_client_clock[results['clientId']
                                                                    ]['communication']
                                          )

        # ================== Aggregate weights ======================
        self.update_lock.acquire()

        self.model_in_update += 1
        if self.using_group_params == True:
            self.aggregate_client_group_weights(results)
        else:
            self.aggregate_client_weights(results)

        self.update_lock.release()

    def aggregate_client_weights(self, results):
        """May aggregate client updates on the fly
        
        Args:
            results (dictionary): client's training result
        
        [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
        H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
        """
        # Start to take the average of updates, and we do not keep updates to save memory
        # Importance of each update is 1/#_of_participants
        # importance = 1./self.tasks_round

        for p in results['update_weight']:
            param_weight = results['update_weight'][p]
            if isinstance(param_weight, list):
                param_weight = np.asarray(param_weight, dtype=np.float32)
            param_weight = torch.from_numpy(
                param_weight).to(device=self.device)

            if self.model_in_update == 1:
                self.model_weights[p].data = param_weight
            else:
                self.model_weights[p].data += param_weight

        if self.model_in_update == self.tasks_round:
            for p in self.model_weights:
                d_type = self.model_weights[p].data.dtype

                self.model_weights[p].data = (
                    self.model_weights[p]/float(self.tasks_round)).to(dtype=d_type)

    def aggregate_client_group_weights(self, results):
        """Streaming weight aggregation. Similar to aggregate_client_weights, 
        but each key corresponds to a group of weights (e.g., for Tensorflow)
        
        Args:
            results (dictionary): Client's training result
        
        """
        for p_g in results['update_weight']:
            param_weights = results['update_weight'][p_g]
            for idx, param_weight in enumerate(param_weights):
                if isinstance(param_weight, list):
                    param_weight = np.asarray(param_weight, dtype=np.float32)
                param_weight = torch.from_numpy(
                    param_weight).to(device=self.device)

                if self.model_in_update == 1:
                    self.model_weights[p_g][idx].data = param_weight
                else:
                    self.model_weights[p_g][idx].data += param_weight

        if self.model_in_update == self.tasks_round:
            for p in self.model_weights:
                for idx in range(len(self.model_weights[p])):
                    d_type = self.model_weights[p][idx].data.dtype

                    self.model_weights[p][idx].data = (
                        self.model_weights[p][idx].data/float(self.tasks_round)
                    ).to(dtype=d_type)

    def save_last_param(self):
        """ Save the last model parameters
        """
        if self.args.engine == commons.TENSORFLOW:
            self.last_gradient_weights = [
                layer.get_weights() for layer in self.model.layers]
        else:
            self.last_gradient_weights = [
                p.data.clone() for p in self.model.parameters()]

    def round_weight_handler(self, last_model):
        """Update model when the round completes
        
        Args:
            last_model (list): A list of global model weight in last round.
        
        """
        if self.round > 1:
            if self.args.engine == commons.TENSORFLOW:
                for layer in self.model.layers:
                    layer.set_weights([p.cpu().detach().numpy()
                                      for p in self.model_weights[layer.name]])
            else:
                self.model.load_state_dict(self.model_weights)
                current_grad_weights = [param.data.clone()
                                        for param in self.model.parameters()]
                self.optimizer.update_round_gradient(
                    last_model, current_grad_weights, self.model)

    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        self.global_virtual_clock += self.round_duration
        self.round += 1

        if self.round % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        # handle the global update w/ current and last
        self.round_weight_handler(self.last_gradient_weights)

        avgUtilLastround = sum(self.stats_util_accumulator) / \
            max(1, len(self.stats_util_accumulator))
        # assign avg reward to explored, but not ran workers
        for clientId in self.round_stragglers:
            self.client_manager.register_feedback(clientId, avgUtilLastround,
                                              time_stamp=self.round,
                                              duration=self.virtual_client_clock[clientId]['computation'] +
                                              self.virtual_client_clock[clientId]['communication'],
                                              success=False)

        avg_loss = sum(self.loss_accumulator) / \
            max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, Planned participants: " +
                     f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        # update select participants
        self.sampled_participants = self.select_participants(
            select_num_participants=self.args.num_participants, overcommitment=self.args.overcommitment)
        (clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration) = self.tictak_client_tasks(
            self.sampled_participants, self.args.num_participants)

        logging.info(f"Selected participants to run: {clientsToRun}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)
        self.tasks_round = len(clientsToRun)

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(
                self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id)
                                      for c_id in self.sampled_participants]

        self.save_last_param()
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.flatten_client_duration = numpy.array(flatten_client_duration)
        self.round_duration = round_duration
        self.model_in_update = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.START_ROUND)

    def log_train_result(self, avg_loss):
        """Log training result on TensorBoard
        """
        self.log_writer.add_scalar('Train/round_to_loss', avg_loss, self.round)
        self.log_writer.add_scalar(
            'FAR/time_to_train_loss (min)', avg_loss, self.global_virtual_clock/60.)
        self.log_writer.add_scalar(
            'FAR/round_duration (min)', self.round_duration/60., self.round)
        self.log_writer.add_histogram(
            'FAR/client_duration (min)', self.flatten_client_duration, self.round)

    def log_test_result(self):
        """Log testing result on TensorBoard
        """
        self.log_writer.add_scalar(
            'Test/round_to_loss', self.testing_history['perf'][self.round]['loss'], self.round)
        self.log_writer.add_scalar(
            'Test/round_to_accuracy', self.testing_history['perf'][self.round]['top_1'], self.round)
        self.log_writer.add_scalar('FAR/time_to_test_loss (min)', self.testing_history['perf'][self.round]['loss'],
                                   self.global_virtual_clock/60.)
        self.log_writer.add_scalar('FAR/time_to_test_accuracy (min)', self.testing_history['perf'][self.round]['top_1'],
                                   self.global_virtual_clock/60.)

    def deserialize_response(self, responses):
        """Deserialize the response from executor
        
        Args:
            responses (byte stream): Serialized response from executor.

        Returns:
            string, bool, or bytes: The deserialized response object from executor.
        """
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """ Serialize the response to send to server upon assigned job completion

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

        results = results['results']

        # List append is thread-safe
        self.test_result_accumulator.append(results)

        # Have collected all testing results

        if len(self.test_result_accumulator) == len(self.executors):
            
            aggregate_test_result(
                self.test_result_accumulator, self.args.task, \
                self.round, self.global_virtual_clock, self.testing_history)
            # Dump the testing result
            with open(os.path.join(logDir, 'testing_perf'), 'wb') as fout:
                pickle.dump(self.testing_history, fout)

            if len(self.loss_accumulator):
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

    def get_client_conf(self, clientId):
        """Training configurations that will be applied on clients,
        developers can further define personalized client config here.

        Args:
            clientId (int): The client id.

        Returns:
            dictionary: Client training config.

        """
        conf = {
            'learning_rate': self.args.learning_rate,
            'model': None  # none indicates we are using the global model
        }
        return conf

    def create_client_task(self, executorId):
        """Issue a new client training task to specific executor
        
        Args:
            executorId (int): Executor Id.
        
        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        next_clientId = self.resource_manager.get_next_task(executorId)
        train_config = None
        # NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
        model = None
        if next_clientId != None:
            config = self.get_client_conf(next_clientId)
            train_config = {'client_id': next_clientId, 'task_config': config}
        return train_config, model

    def get_test_config(self, client_id):
        """FL model testing on clients, developers can further define personalized client config here.
        
        Args:
            client_id (int): The client id.
        
        Returns:
            dictionary: The testing config for new task.
        
        """
        return {'client_id': client_id}

    def get_global_model(self):
        """Get global model that would be used by all FL clients (in default FL)

        Returns:
            PyTorch or TensorFlow module: Based on the executor's machine learning framework, initialize and return the model for training.

        """
        return self.model

    def get_shutdown_config(self, client_id):
        """Shutdown config for client, developers can further define personalized client config here.

        Args:
            client_id (int): Client id.
        
        Returns:
            dictionary: Shutdown config for new task.

        """
        return {'client_id': client_id}

    def add_event_handler(self, client_id, event, meta, data):
        """ Due to the large volume of requests, we will put all events into a queue first.

        Args:
            client_id (int): The client id.
            event (string): grpc event MODEL_TEST or UPLOAD_MODEL.
            meta (dictionary or string): Meta message for grpc communication, could be event.
            data (dictionary): Data transferred in grpc communication, could be model parameters, test result.

        """
        self.sever_events_queue.append((client_id, event, meta, data))

    def CLIENT_REGISTER(self, request, context):
        """FL Client register to the aggregator
        
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

        return job_api_pb2.ServerResponse(event=commons.DUMMY_EVENT,
                                          meta=dummy_data, data=dummy_data)

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
            current_event = self.individual_client_events[executor_id].popleft(
            )
            if current_event == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(
                    client_id)
                if response_msg is None:
                    current_event = commons.DUMMY_EVENT
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(
                                commons.CLIENT_TRAIN)
            elif current_event == commons.MODEL_TEST:
                response_msg = self.get_test_config(client_id)
            elif current_event == commons.UPDATE_MODEL:
                response_data = self.get_global_model()
            elif current_event == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        response = job_api_pb2.ServerResponse(event=current_event,
                                          meta=response_msg, data=response_data)
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

        executor_id, client_id, event = request.executor_id, request.client_id, request.event
        execution_status, execution_msg = request.status, request.msg
        meta_result, data_result = request.meta_result, request.data_result

        if event == commons.CLIENT_TRAIN:
            # Training results may be uploaded in CLIENT_EXECUTE_RESULT request later,
            # so we need to specify whether to ask client to do so (in case of straggler/timeout in real FL).
            if execution_status is False:
                logging.error(f"Executor {executor_id} fails to run client {client_id}, due to {execution_msg}")
            if self.resource_manager.has_next_task(executor_id):
                # NOTE: we do not pop the train immediately in simulation mode,
                # since the executor may run multiple clients
                self.individual_client_events[executor_id].append(
                        commons.CLIENT_TRAIN)

        elif event in (commons.MODEL_TEST, commons.UPLOAD_MODEL):
            self.add_event_handler(
                executor_id, event, meta_result, data_result)
        else:
            logging.error(f"Received undefined event {event} from client {client_id}")
        return self.CLIENT_PING(request, context)

    def event_monitor(self):
        """Activate event handler according to the received new message
        """
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
            elif len(self.sever_events_queue) > 0:
                client_id, current_event, meta, data = self.sever_events_queue.popleft()

                if current_event == commons.UPLOAD_MODEL:
                    self.client_completion_handler(
                        self.deserialize_response(data))
                    if len(self.stats_util_accumulator) == self.tasks_round:
                        self.round_completion_handler()

                elif current_event == commons.MODEL_TEST:
                    self.testing_completion_handler(
                        client_id, self.deserialize_response(data))

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)

    def stop(self):
        """Stop the aggregator
        """
        logging.info(f"Terminating the aggregator ...")
        time.sleep(5)


if __name__ == "__main__":
    aggregator = Aggregator(args)
    aggregator.run()
