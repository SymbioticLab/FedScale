# -*- coding: utf-8 -*-

from fl_aggregator_libs import *
from random import Random
from resource_manager import ResourceManager
import grpc
import job_api_pb2_grpc
import job_api_pb2
import io
import torch
import pickle


MAX_MESSAGE_LENGTH = 50000000

class ExecutorConnections(object):
    """"Helps aggregator manage its grpc connection with executors."""

    class _ExecutorContext(object):
        def __init__(self, executorId):
            self.id = executorId
            self.address = None
            self.channel = None
            self.stub = None
            
    def __init__(self, config, base_port=50000):
        self.executors = {}
        self.base_port = base_port

        executorId = 0
        for ip_numgpu in config.split("="):
            ip, numgpu = ip_numgpu.split(':')
            for numexe in numgpu.strip()[1:-1].split(','):
                for _ in range(int(numexe.strip())):
                    executorId += 1
                    self.executors[executorId] = ExecutorConnections._ExecutorContext(executorId)
                    self.executors[executorId].address = '{}:{}'.format(ip, self.base_port + executorId)

    def __len__(self):
        return len(self.executors)

    def __iter__(self):
        return iter(self.executors)

    def open_grpc_connection(self):
        for executorId in self.executors:
            logging.info('%%%%%%%%%% Opening grpc connection to ' + self.executors[executorId].address + ' %%%%%%%%%%')
            channel = grpc.insecure_channel(
                self.executors[executorId].address,
                options=[
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ]
            )
            self.executors[executorId].channel = channel
            self.executors[executorId].stub = job_api_pb2_grpc.JobServiceStub(channel)

    def close_grpc_connection(self):
        for executorId in self.executors:
            logging.info(f'%%%%%%%%%% Closing grpc connection with {executorId} %%%%%%%%%%')
            self.executors[executorId].channel.close()

    def get_stub(self, executorId):
        return self.executors[executorId].stub


class Aggregator(object):
    """This centralized aggregator collects training/testing feedbacks from executors"""
    def __init__(self, args):
        logging.info(f"Job args {args}")

        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device('cpu')
        self.executors = ExecutorConnections(args.executor_configs, args.base_port)

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.
        self.round_duration = 0.
        self.resource_manager = ResourceManager()
        self.client_manager = self.init_client_manager(args=args)

        # ======== model and data ========
        self.model = None

        # list of parameters in model.parameters()
        self.model_in_update = []
        self.last_global_model = []

        # ======== channels ========
        self.server_event_queue = {}
        self.client_event_queue = Queue()
        self.control_manager = None
        # event queue of its own functions
        self.event_queue = collections.deque()

        # ======== runtime information ========
        self.tasks_round = 0
        self.sampled_participants = []

        self.round_stragglers = []
        self.model_update_size = 0.

        self.collate_fn = None
        self.task = args.task
        self.epoch = 0

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
        
        # ======== Task specific ============
        self.imdb = None           # object detection


    def setup_env(self):
        self.setup_seed(seed=self.this_rank)

        # set up device
        if self.args.use_cuda and self.device == None:
            for i in range(torch.cuda.device_count()):
                try:
                    self.device = torch.device('cuda:'+str(i))
                    torch.cuda.set_device(i)
                    _ = torch.rand(1).to(device=self.device)
                    logging.info(f'End up with cuda device ({self.device})')
                    break
                except Exception as e:
                    assert i != torch.cuda.device_count()-1, 'Can not find available GPUs'

        self.init_control_communication(self.args.ps_ip, self.args.manager_port, self.executors)
        self.init_data_communication()
        self.optimizer = ServerOptimizer(self.args.gradient_policy, self.args, self.device)

    def setup_seed(self, seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def init_control_communication(self, ps_ip, ps_port, executors):
        # Create communication channel between aggregator and worker
        # This channel serves control messages
        logging.info(f"Start to initiate {ps_ip}:{ps_port} for control plane communication ...")

        dummy_que = {executorId:Queue() for executorId in executors}
        # create multiple queue for each aggregator_executor pair
        for executorId in executors:
            BaseManager.register('get_server_event_que'+str(executorId), callable=lambda: dummy_que[executorId])

        dummy_client_que = Queue()
        BaseManager.register('get_client_event', callable=lambda: dummy_client_que)

        self.control_manager = BaseManager(address=(ps_ip, ps_port), authkey=b'FLPerf')
        self.control_manager.start()

        #self.server_event_queue = torch.multiprocessing.Manager().dict()
        for executorId in self.executors:
            self.server_event_queue[executorId] = eval('self.control_manager.get_server_event_que'+str(executorId)+'()')

        self.client_event_queue = self.control_manager.get_client_event()


    def init_data_communication(self):
        #dist.init_process_group(self.args.backend, rank=self.this_rank, world_size=len(self.executors) + 1)
        pass

    def init_model(self):
        """Load model"""
        if self.args.task == "detection":
            cfg_from_file(self.args.cfg_file)
            np.random.seed(self.cfg.RNG_SEED)
            self.imdb, _, _, _ = combined_roidb("voc_2007_test", ['DATA_DIR', self.args.data_dir], server=True)

        return init_model()

    def init_client_manager(self, args):
        """
            Currently we implement two client managers:
            1. Random client sampler
                - it selects participants randomly in each round
                - [Ref]: https://arxiv.org/abs/1902.01046
            2. Oort sampler
                - Oort prioritizes the use of those clients who have both data that offers the greatest utility
                  in improving model accuracy and the capability to run training quickly.
                - [Ref]: https://www.usenix.org/conference/osdi21/presentation/lai
        """

        # sample_mode: random or kuiper
        client_manager = clientManager(args.sample_mode, args=args)

        return client_manager

    def load_client_profile(self, file_path):
        # load client profiles
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fin:
                # {clientId: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile

    def executor_info_handler(self, executorId, info):

        self.registered_executor_info.add(executorId)
        logging.info(f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}")
        # have collected all executors
        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout

        if len(self.registered_executor_info) == len(self.executors):

            clientId = 1
            logging.info(f"Loading {len(info['size'])} client traces ...")

            for _size in info['size']:
                # since the worker rankId starts from 1, we also configure the initial dataId as 1
                mapped_id = clientId%len(self.client_profiles) if len(self.client_profiles) > 0 else 1
                systemProfile = self.client_profiles.get(mapped_id, {'computation': 1.0, 'communication':1.0})
                self.client_manager.registerClient(executorId, clientId, size=_size, speed=systemProfile)
                self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
                    upload_epoch=self.args.local_steps, upload_size=self.model_update_size, download_size=self.model_update_size)
                clientId += 1

            logging.info("Info of all feasible clients {}".format(self.client_manager.getDataInfo()))

            # start to sample clients
            self.round_completion_handler()


    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """We try to remove dummy events as much as possible, by removing the stragglers/offline clients in overcommitment"""

        sampledClientsReal = []
        completionTimes = []
        completed_client_clock = {}
        # 1. remove dummy clients that are not available to the end of training
        for client_to_run in sampled_clients:
            client_cfg = self.client_conf.get(client_to_run, self.args)

            exe_cost = self.client_manager.getCompletionTime(client_to_run,
                                    batch_size=client_cfg.batch_size, upload_epoch=client_cfg.local_steps,
                                    upload_size=self.model_update_size, download_size=self.model_update_size)

            roundDuration = exe_cost['computation'] + exe_cost['communication']
            # if the client is not active by the time of collection, we consider it is lost in this round
            if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
                sampledClientsReal.append(client_to_run)
                completionTimes.append(roundDuration)
                completed_client_clock[client_to_run] = exe_cost

        num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))
        # 2. get the top-k completions to remove stragglers
        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
        top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
        clients_to_run = [sampledClientsReal[k] for k in top_k_index]

        dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[num_clients_to_collect:]]
        round_duration = completionTimes[top_k_index[-1]]

        return clients_to_run, dummy_clients, completed_client_clock, round_duration


    def run(self):
        self.setup_env()
        self.model = self.init_model()
        self.save_last_param()

        self.model_update_size = sys.getsizeof(pickle.dumps(self.model))/1024.0*8. # kbits
        self.client_profiles = self.load_client_profile(file_path=self.args.device_conf_file)
        self.event_monitor()


    def select_participants(self, select_num_participants, overcommitment=1.3):
        return sorted(self.client_manager.resampleClients(int(select_num_participants*overcommitment), cur_time=self.global_virtual_clock))


    def client_completion_handler(self, results):
        """We may need to keep all updates from clients, if so, we need to append results to the cache"""
        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': epoch_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}
        if self.args.gradient_policy in ['q-fedavg']:
            self.client_training_results.append(results)

        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])

        self.client_manager.registerScore(results['clientId'], results['utility'], auxi=math.sqrt(results['moving_loss']),
                    time_stamp=self.epoch,
                    duration=self.virtual_client_clock[results['clientId']]['computation']+self.virtual_client_clock[results['clientId']]['communication']
                )

        device = self.device
        """
            [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data". 
            H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
        """
        # Start to take the average of updates, and we do not keep updates to save memory
        # Importance of each update is 1/#_of_participants
        importance = 1./self.tasks_round
        if len(self.model_in_update) == 0:
            self.model_in_update = [True]

            for idx, param in enumerate(self.model.state_dict().values()):
                param.data = (torch.from_numpy(results['update_weight'][idx]).to(device=device)*importance).to(dtype=param.data.dtype)
        else: 
            for idx, param in enumerate(self.model.state_dict().values()):
                param.data +=(torch.from_numpy(results['update_weight'][idx]).to(device=device)*importance).to(dtype=param.data.dtype)


    def save_last_param(self):
        self.last_global_model = [param.data.clone() for param in self.model.parameters()]

    def round_weight_handler(self, last_model, current_model):
        if self.epoch > 1:
            self.optimizer.update_round_gradient(last_model, current_model, self.model)
            
    def round_completion_handler(self):
        self.global_virtual_clock += self.round_duration
        self.epoch += 1

        if self.epoch % self.args.decay_epoch == 0:
            self.args.learning_rate = max(self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        # handle the global update w/ current and last
        self.round_weight_handler(self.last_global_model, [param.data.clone() for param in self.model.parameters()])

        avgUtilLastEpoch = sum(self.stats_util_accumulator)/max(1, len(self.stats_util_accumulator))
        # assign avg reward to explored, but not ran workers
        for clientId in self.round_stragglers:
            self.client_manager.registerScore(clientId, avgUtilLastEpoch,
                                    time_stamp=self.epoch,
                                    duration=self.virtual_client_clock[clientId]['computation']+self.virtual_client_clock[clientId]['communication'],
                                    success=False)

        avg_loss = sum(self.loss_accumulator)/max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, Epoch: {self.epoch}, Planned participants: " + \
            f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # update select participants
        self.sampled_participants = self.select_participants(select_num_participants=self.args.total_worker, overcommitment=self.args.overcommitment)
        clientsToRun, round_stragglers, virtual_client_clock, round_duration = self.tictak_client_tasks(self.sampled_participants, self.args.total_worker)

        logging.info(f"Selected participants to run: {clientsToRun}:\n{virtual_client_clock}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)
        self.tasks_round = len(clientsToRun)

        self.save_last_param()
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.round_duration = round_duration
        self.model_in_update = []
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []

        if self.epoch >= self.args.epochs:
            self.event_queue.append('stop')
        elif self.epoch % self.args.eval_interval == 0:
            self.event_queue.append('update_model')
            self.event_queue.append('test')
        else:
            self.event_queue.append('update_model')
            self.event_queue.append('start_round')


    def testing_completion_handler(self, results):
        self.test_result_accumulator.append(results)

        # Have collected all testing results
        if len(self.test_result_accumulator) == len(self.executors):
            accumulator = self.test_result_accumulator[0]
            for i in range(1, len(self.test_result_accumulator)):
                if self.args.task == "detection":
                    for key in accumulator:
                        if key == "boxes":
                            for j in range(self.imdb.num_classes):
                                accumulator[key][j] = accumulator[key][j] + self.test_result_accumulator[i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[i][key]
                else:
                    for key in accumulator:
                        accumulator[key] += self.test_result_accumulator[i][key]
            if self.args.task == "detection":
                self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator), 4),
                    'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator), 4),
                    'loss': accumulator['test_loss'],
                    'test_len': accumulator['test_len']
                    }
            else:
                self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                    'loss': accumulator['test_loss']/accumulator['test_len'],
                    'test_len': accumulator['test_len']
                    }


            logging.info("FL Testing in epoch: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
                    .format(self.epoch, self.global_virtual_clock, self.testing_history['perf'][self.epoch]['top_1'],
                    self.testing_history['perf'][self.epoch]['top_5'], self.testing_history['perf'][self.epoch]['loss'],
                    self.testing_history['perf'][self.epoch]['test_len']))

            # Dump the testing result
            with open(os.path.join(logDir, 'testing_perf'), 'wb') as fout:
                pickle.dump(self.testing_history, fout)

            self.event_queue.append('start_round')

    def get_client_conf(self, clientId):
        # learning rate scheduler
        conf = {}
        conf['learning_rate'] = self.args.learning_rate
        return conf

    def event_monitor(self):
        logging.info("Start monitoring events ...")
        start_time = time.time()
        time.sleep(20)

        while time.time() - start_time < 2000:
            try:
                self.executors.open_grpc_connection()
                for executorId in self.executors:
                    response = self.executors.get_stub(executorId).ReportExecutorInfo(
                        job_api_pb2.ReportExecutorInfoRequest())
                    self.executor_info_handler(executorId, {"size": response.training_set_size})
                break
                
            except Exception as e:
                self.executors.close_grpc_connection()
                logging.warning(f"{e}: Have not received executor information. This may due to slow data loading (e.g., Reddit)")
                time.sleep(30)

        logging.info("Have received all executor information")

        while True:
            if len(self.event_queue) != 0:
                event_msg = self.event_queue.popleft()
                send_msg = {'event': event_msg}

                if event_msg == 'update_model':
                    serialized_tensors = []
                    # TODO: do serialization in parallel
                    for param in self.model.state_dict().values():
                        buffer = io.BytesIO()
                        torch.save(param.data.to(device='cpu'), buffer)
                        buffer.seek(0)
                        serialized_tensors.append(buffer.read())

                    update_model_request = job_api_pb2.UpdateModelRequest()
                    for executorId in self.executors:
                        _ = self.executors.get_stub(executorId).UpdateModel(
                            job_api_pb2.UpdateModelRequest(serialized_tensor=param)
                                for param in serialized_tensors)

                elif event_msg == 'start_round':
                    for executorId in self.executors:
                        next_clientId = self.resource_manager.get_next_task()

                        if next_clientId is not None:
                            # TODO: Remove sending train request via server_event_queue
                            config = self.get_client_conf(next_clientId)
                            self.server_event_queue[executorId].put({'event': 'train', 'clientId':next_clientId, 'conf': config})

                            _ = self.executors.get_stub(executorId).Train(
                                job_api_pb2.TrainRequest(
                                    client_id=next_clientId,
                                    serialized_train_config=pickle.dumps(config)))

                elif event_msg == 'stop':
                    for executorId in self.executors:
                        _ = self.executors.get_stub(executorId).Stop(job_api_pb2.StopRequest())

                    self.stop()
                    break

                elif event_msg == 'test':
                    for executorId in self.executors:
                        response = self.executors.get_stub(executorId).Test(job_api_pb2.TestRequest())
                        self.testing_completion_handler(pickle.loads(response.serialized_test_response))

            elif not self.client_event_queue.empty():

                event_dict = self.client_event_queue.get()
                event_msg, executorId, results = event_dict['event'], event_dict['executorId'], event_dict['return']

                if event_msg != 'train_nowait':
                    logging.info(f"Round {self.epoch}: Receive (Event:{event_msg.upper()}) from (Executor:{executorId})")

                # collect training returns from the executor
                if event_msg == 'train_nowait':
                    # pop a new client to run
                    next_clientId = self.resource_manager.get_next_task()

                    if next_clientId is not None:
                        config = self.get_client_conf(next_clientId)
                        runtime_profile = {'event': 'train', 'clientId':next_clientId, 'conf': config}
                        self.server_event_queue[executorId].put(runtime_profile)


                elif event_msg == 'train':
                    # push training results
                    self.client_completion_handler(results)

                    if len(self.stats_util_accumulator) == self.tasks_round:
                        self.round_completion_handler()

                else:
                    logging.error("Unknown message types!")

            # execute every 100 ms
            time.sleep(0.1)

        self.executors.close_grpc_connection()


    def stop(self):
        logging.info(f"Terminating the aggregator ...")
        time.sleep(5)
        self.control_manager.shutdown()

if __name__ == "__main__":
    aggregator = Aggregator(args)
    aggregator.run()
