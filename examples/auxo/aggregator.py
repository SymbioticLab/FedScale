# -*- coding: utf-8 -*-
import logging

from fedscale.cloud.aggregation.aggregator import *
from client_manager import HeterClientManager
from resource_manager import AuxoResourceManager
from utils.helper import *

class AuxoAggregator(Aggregator):
    def __init__(self, args):
        super().__init__(args)

        self.sampled_participants = [[]]
        self.sampled_executors = [[]]
        self.round_stragglers = [[]]
        self.stats_util_accumulator = [[]]
        self.loss_accumulator = [[]]
        self.client_training_results = [[]]
        self.test_result_accumulator = [[]]
        self.virtual_client_clock = [[]]
        self.testing_history = [{'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                                'gradient_policy': args.gradient_policy, 'task': args.task,
                                'perf': collections.OrderedDict()}]

        self.model_in_update = [0]
        # self.last_saved_round = [0]
        self.tasks_round = [0]
        self.global_virtual_clock = [0.]
        self.round_duration = [0.]
        self.model_update_size = [0.]
        self.round = [0]

        self.stop_cluster = 0
        self.split_cluster = 1
        self.num_split = 2
        self.resource_manager = AuxoResourceManager(self.experiment_mode)

    def init_model(self):
        """Initialize the model"""
        if self.args.engine == commons.TENSORFLOW:
            self.model_wrapper = [TensorflowModelAdapter(init_model())]
        elif self.args.engine == commons.PYTORCH:
            self.model_wrapper = [TorchModelAdapter(
                init_model(),
                optimizer=TorchServerOptimizer(
                    self.args.gradient_policy, self.args, self.device))]
        else:
            raise ValueError(f"{self.args.engine} is not a supported engine.")
        self.model_weights = [self.model_wrapper[0].get_weights()]

    def init_client_manager(self, args):
        """ Initialize client sampler

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

        3. Auxo: Client Heterogeneity Manager
        [Ref]: https://arxiv.org/abs/2210.16656
        """

        # sample_mode: random or oort
        client_manager = HeterClientManager(args.sample_mode, args=args)

        return client_manager

    def event_monitor(self):
        """Activate event handler according to the received new message
        """
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events to clients
            if len(self.broadcast_events_queue) > 0:
                current_event = self.broadcast_events_queue.popleft()
                logging.info(f"Event {current_event} is broadcasted to clients")
                event_type, cohort_id = decode_msg(current_event)

                if event_type in (commons.UPDATE_MODEL, commons.MODEL_TEST):
                    self.dispatch_client_events(current_event)

                elif event_type == commons.START_ROUND:
                    self.dispatch_client_events(generate_msg(commons.CLIENT_TRAIN, cohort_id))

                elif event_type == 'split':
                    self.dispatch_client_events(current_event)

                elif event_type == commons.SHUT_DOWN:
                    self.dispatch_client_events(current_event)
                    break

            # Handle events queued on the aggregator
            elif len(self.server_events_queue) > 0:

                client_id, current_event, meta, data = self.server_events_queue.popleft()
                logging.info(f"Event {current_event} is received from client {client_id}")
                event_type, cohort_id = decode_msg(current_event)

                if event_type == commons.UPLOAD_MODEL:
                    self.client_completion_handler(
                        self.deserialize_response(data), cohort_id)
                    logging.info(f"[Cohort {cohort_id}] Client {client_id} has completed the task. {len(self.stats_util_accumulator[cohort_id])} v.s. {self.tasks_round[cohort_id]}")
                    if len(self.stats_util_accumulator[cohort_id]) == self.tasks_round[cohort_id]:
                        self.round_completion_handler(cohort_id)

                elif event_type == commons.MODEL_TEST:
                    self.testing_completion_handler(
                        client_id, self.deserialize_response(data), cohort_id)
                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)

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
        dummy_data = self.serialize_response(generate_msg(commons.DUMMY_RESPONSE, 0))

        return job_api_pb2.ServerResponse(event=generate_msg(commons.DUMMY_EVENT, 0),
                                          meta=dummy_data, data=dummy_data)

    def get_test_config(self, client_id, cohort_id=0):
        """FL model testing on clients, developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: The testing config for new task.

        """
        num_client = self.client_manager.schedule_plan()
        client_list = self.select_participants(num_client, overcommitment = 1, cohort_id = cohort_id, test=True)
        return {'client_id': client_list}


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
        response_data = response_msg = generate_msg(commons.DUMMY_RESPONSE, 0)

        if len(self.individual_client_events[executor_id]) == 0:
            # send dummy response
            current_event = generate_msg(commons.DUMMY_EVENT, 0)
            response_data = response_msg = current_event
        else:
            current_event = self.individual_client_events[executor_id].popleft()
            event_type, cohort_id = decode_msg(current_event)
            if event_type == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(
                    executor_id, cohort_id)
                if response_msg is None:
                    current_event = generate_msg(commons.DUMMY_EVENT, 0)
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(
                            commons.CLIENT_TRAIN)
            elif event_type == commons.MODEL_TEST:
                # TODO: remove fedscale test and add individual client testing
                response_msg = self.get_test_config(client_id, cohort_id)
            elif event_type == commons.UPDATE_MODEL:
                response_data = self.model_wrapper[cohort_id].get_weights()
            elif event_type == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        response = job_api_pb2.ServerResponse(event=current_event,
                                              meta=response_msg, data=response_data)
        if decode_msg(current_event)[0] != commons.DUMMY_EVENT:
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
        event_type, cohort_id = decode_msg(event)
        if event_type == commons.CLIENT_TRAIN:
            # Training results may be uploaded in CLIENT_EXECUTE_RESULT request later,
            # so we need to specify whether to ask client to do so (in case of straggler/timeout in real FL).
            if execution_status is False:
                logging.error(f"Executor {executor_id} fails to run client {client_id}, due to {execution_msg}")

            if self.resource_manager.has_next_task(executor_id, cohort_id):
                # NOTE: we do not pop the train immediately in simulation mode,
                # since the executor may run multiple clients
                if commons.CLIENT_TRAIN not in self.individual_client_events[executor_id]:
                    self.individual_client_events[executor_id].append(generate_msg(commons.CLIENT_TRAIN, cohort_id))

        elif event_type in (commons.MODEL_TEST, commons.UPLOAD_MODEL):
            self.add_event_handler(
                executor_id, event, meta_result, data_result)
        else:
            logging.error(f"Received undefined event {event} from client {client_id}")

        return self.CLIENT_PING(request, context)


    def create_client_task(self, executor_id, cohort_id):
        """Issue a new client training task to specific executor

        Args:
            executorId (int): Executor Id.
            cohort_id (int): Cohort Id.

        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        next_client_id = self.resource_manager.get_next_task(executor_id, cohort_id)
        train_config = None
        # NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
        if next_client_id is not None:
            config = self.get_client_conf(next_client_id)
            train_config = {'client_id': next_client_id, 'task_config': config, 'cohort_id': cohort_id}

        return train_config, self.model_wrapper[cohort_id].get_weights()


    def client_completion_handler(self, results, cohort_id):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache

        Args:
            results (dictionary): client's training result

        """
        # Format:
        #       -results = {'client_id':client_id, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        if self.args.gradient_policy in ['q-fedavg']:
            self.client_training_results[cohort_id].append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator[cohort_id].append(results['utility'])
        self.loss_accumulator[cohort_id].append(results['moving_loss'])

        self.client_manager.register_feedback(results['client_id'], results['utility'],
                                              auxi=math.sqrt(
                                                  results['moving_loss']),
                                              time_stamp=self.round[cohort_id],
                                              duration=self.virtual_client_clock[cohort_id][results['client_id']]['computation'] +
                                                       self.virtual_client_clock[cohort_id][results['client_id']]['communication'],
                                              w_new = results['update_weight'],
                                              w_old = self.model_wrapper[cohort_id].get_weights(),
                                              cohort_id=cohort_id
                                              )

        # ================== Aggregate weights ======================
        self.update_lock.acquire()

        self.model_in_update[cohort_id] += 1
        self.update_weight_aggregation(results, cohort_id)

        self.update_lock.release()

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect, cohort_id):
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

                exe_cost = self.client_manager.get_completion_time(client_to_run,
                                                                   batch_size=client_cfg.batch_size,
                                                                   local_steps=client_cfg.local_steps,
                                                                   upload_size=self.model_update_size,
                                                                   download_size=self.model_update_size)

                roundDuration = exe_cost['computation'] + \
                                exe_cost['communication']
                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock[cohort_id]):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                    completed_client_clock[client_to_run] = exe_cost

            num_clients_to_collect = min(
                num_clients_to_collect, len(completionTimes))
            # 2. get the top-k completions to remove stragglers
            workers_sorted_by_completion_time = sorted(
                range(len(completionTimes)), key=lambda k: completionTimes[k])
            top_k_index = workers_sorted_by_completion_time[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            stragglers = [sampledClientsReal[k]
                          for k in workers_sorted_by_completion_time[num_clients_to_collect:]]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            return (clients_to_run, stragglers,
                    completed_client_clock, round_duration,
                    completionTimes[:num_clients_to_collect])
        else:
            completed_client_clock = {
                client: {'computation': 1, 'communication': 1} for client in sampled_clients}
            completionTimes = [1 for c in sampled_clients]
            return (sampled_clients, sampled_clients, completed_client_clock,
                    1, completionTimes)

    def update_default_task_config(self, cohort_id):
        """Update the default task configuration after each round
        """
        # TODO: fix the lr update
        if self.round[cohort_id] % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate * self.args.decay_factor, self.args.min_learning_rate)

    def select_participants(self, select_num_participants, overcommitment=1.3, cohort_id=0, test=False):
        """Select clients for next round.

        Args:
            select_num_participants (int): Number of clients to select.
            overcommitment (float): Overcommit ration for next round.

        Returns:
            list of int: The list of sampled clients id.

        """
        return sorted(self.client_manager.select_participants(
            int(select_num_participants * overcommitment),
            cur_time=self.global_virtual_clock[cohort_id],
            cohort_id=cohort_id,
            test=test)
        )



    def _init_split_config(self, cohort_id):
        def increment_config( in_list ):
            in_list.append([])
            in_list[cohort_id ] = []

        self.model_wrapper.append(copy.deepcopy(self.model_wrapper[cohort_id]))
        self.global_virtual_clock.append(copy.deepcopy(self.global_virtual_clock[cohort_id]))
        self.model_in_update.append(0)
        increment_config(self.round_stragglers)
        increment_config(self.virtual_client_clock )
        increment_config(self.round_duration)
        increment_config(self.test_result_accumulator )
        increment_config(self.stats_util_accumulator)
        increment_config(self.client_training_results)
        increment_config(self.tasks_round)
        increment_config(self.loss_accumulator)
        self.model_weights.append(copy.deepcopy(self.model_weights[cohort_id]))
        self.testing_history.append(copy.deepcopy(self.testing_history[cohort_id]))
        self.round.append(copy.deepcopy(self.round[cohort_id]))
        increment_config(self.sampled_participants)

    def _split_participant_list(self, cohort_id, num_split = 2):

        for s in range(num_split-1):
            self._init_split_config(cohort_id)
        cohort_id_list = [cohort_id, self.split_cluster - 1] if num_split == 2 else [*range(num_split)]

        for cid in range(num_split):

            # num_client_per_round  = self.args.num_participants * self.client_manager.get_cohort_size(cohort_id_list[cid]) // self.total_clients
            num_client_per_round = self.client_manager.schedule_plan(self.round[cohort_id_list[cid]] , cid )
            num_client_per_round = max(num_client_per_round,1 )

            self.sampled_participants[cohort_id_list[cid]] = self.select_participants(select_num_participants = num_client_per_round, \
                                     overcommitment=self.args.overcommitment, cohort_id = cohort_id_list[cid])
            clients_to_run, round_stragglers, virtual_client_clock, round_duration, _ = \
                self.tictak_client_tasks(self.sampled_participants[cohort_id_list[cid]], num_client_per_round, cohort_id_list[cid])
            self.round_stragglers[cohort_id_list[cid]] = round_stragglers
            self.resource_manager.register_tasks(clients_to_run, cohort_id_list[cid])
            self.tasks_round[cohort_id_list[cid]] = len(clients_to_run)
            self.virtual_client_clock[cohort_id_list[cid]] = virtual_client_clock
            self.round_duration[cohort_id_list[cid]] = round_duration
            self.model_in_update[cohort_id_list[cid]] = 0
            self.test_result_accumulator[cohort_id_list[cid]] = []
            self.stats_util_accumulator[cohort_id_list[cid]] = []
            self.client_training_results[cohort_id_list[cid]] = []
            self.loss_accumulator[cohort_id_list[cid]] = []

        return cohort_id_list


    def _is_first_result_in_round(self, cohort_id):
        return self.model_in_update[cohort_id] == 1

    def _is_last_result_in_round(self, cohort_id):
        return self.model_in_update[cohort_id] == self.tasks_round[cohort_id]


    def update_weight_aggregation(self, results, cohort_id = 0):
        """Updates the aggregation with the new results.

        :param results: the results collected from the client.
        """
        update_weights = results['update_weight']
        if type(update_weights) is dict:
            update_weights = [x for x in update_weights.values()]
        if self._is_first_result_in_round(cohort_id):
            self.model_weights[cohort_id] = update_weights
        else:
            self.model_weights[cohort_id] = [weight + update_weights[i] for i, weight in enumerate(self.model_weights[cohort_id])]
        if self._is_last_result_in_round(cohort_id):
            self.model_weights[cohort_id] = [np.divide(weight, self.tasks_round[cohort_id]) for weight in self.model_weights[cohort_id]]
            self.model_wrapper[cohort_id].set_weights(copy.deepcopy(self.model_weights[cohort_id]))


    def round_completion_handler(self, cohort_id = 0):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        self.global_virtual_clock[cohort_id] += self.round_duration[cohort_id]
        self.round[cohort_id] += 1
        last_round_avg_util = sum(self.stats_util_accumulator[cohort_id]) / max(1, len(self.stats_util_accumulator[cohort_id]))
        # assign avg reward to explored, but not ran workers
        for client_id in self.round_stragglers[cohort_id]:
            self.client_manager.register_feedback(client_id, last_round_avg_util,
                                                  time_stamp=self.round[cohort_id],
                                                  duration=self.virtual_client_clock[cohort_id][client_id]['computation'] +
                                                           self.virtual_client_clock[cohort_id][client_id]['communication'],
                                                  success=False)

        avg_loss = sum(self.loss_accumulator[cohort_id]) / max(1, len(self.loss_accumulator[cohort_id]))
        logging.info(f"[Cohort {cohort_id}] Wall clock: {round(self.global_virtual_clock[cohort_id])} s, round: {self.round[cohort_id]}, Planned participants: " +
                     f"{len(self.sampled_participants[cohort_id])}, Succeed participants: {len(self.stats_util_accumulator[cohort_id])}, Training loss: {avg_loss}")

        at_split = False
        if self.round[cohort_id] > 1: # TODO: replace with clustering start round
            at_split = self.client_manager.cohort_clustering(self.round[cohort_id], cohort_id )

        # TODO: add split and non-split logic: update the stats and select participants
        if at_split:
            self.split_cluster = len(self.client_manager.feasibleClients)
            self.resource_manager.split(cohort_id)
            new_cohort_id_list = self._split_participant_list(cohort_id, self.num_split)

        else:
            num_client_per_round = self.client_manager.schedule_plan(self.round[cohort_id], cohort_id)
            # update select participants
            self.sampled_participants[cohort_id] = self.select_participants(
                select_num_participants=num_client_per_round, overcommitment=self.args.overcommitment, cohort_id=cohort_id)
            (clients_to_run, round_stragglers, virtual_client_clock, round_duration,
             flatten_client_duration) = self.tictak_client_tasks(
                self.sampled_participants[cohort_id], num_client_per_round, cohort_id)

            logging.info(f"Selected participants to run: {clients_to_run}")

            # Issue requests to the resource manager; Tasks ordered by the completion time
            self.resource_manager.register_tasks(clients_to_run, cohort_id)
            self.tasks_round[cohort_id] = len(clients_to_run)

            # Update executors and participants
            if self.experiment_mode == commons.SIMULATION_MODE:
                self.sampled_executors = list(
                    self.individual_client_events.keys())
            else:
                self.sampled_executors = [str(c_id)
                                          for c_id in self.sampled_participants]
            self.round_stragglers[cohort_id] = round_stragglers
            self.virtual_client_clock[cohort_id] = virtual_client_clock
            self.flatten_client_duration = np.array(flatten_client_duration)
            self.round_duration[cohort_id] = round_duration
            self.model_in_update[cohort_id] = 0
            self.test_result_accumulator[cohort_id] = []
            self.stats_util_accumulator[cohort_id] = []
            self.client_training_results[cohort_id] = []
            self.loss_accumulator[cohort_id] = []
            self.update_default_task_config(cohort_id)

        if self.round[cohort_id] >= self.args.rounds:
            self.broadcast_aggregator_events(generate_msg(commons.SHUT_DOWN))
        elif at_split:
            self.broadcast_aggregator_events(generate_msg('split', cohort_id=cohort_id))
            for cid in new_cohort_id_list:
                self.broadcast_aggregator_events(generate_msg(commons.UPDATE_MODEL, cohort_id=cid))
                self.broadcast_aggregator_events(generate_msg(commons.START_ROUND, cohort_id=cid))
        elif self.round[cohort_id] % self.args.eval_interval == 0 or self.round[cohort_id] == 1:
            self.broadcast_aggregator_events(generate_msg(commons.UPDATE_MODEL, cohort_id=cohort_id))
            self.broadcast_aggregator_events(generate_msg(commons.MODEL_TEST, cohort_id=cohort_id))
        else:
            self.broadcast_aggregator_events(generate_msg(commons.UPDATE_MODEL, cohort_id=cohort_id))
            self.broadcast_aggregator_events(generate_msg(commons.START_ROUND, cohort_id=cohort_id))

    def aggregate_test_result(self, cohort_id):
        accumulator = self.test_result_accumulator[cohort_id][0]
        for i in range(1, len(self.test_result_accumulator[cohort_id])):
            if self.args.task == "detection":
                for key in accumulator:
                    if key == "boxes":
                        for j in range(596):
                            accumulator[key][j] = accumulator[key][j] + \
                                                  self.test_result_accumulator[cohort_id][i][key][j]
                    else:
                        accumulator[key] += self.test_result_accumulator[cohort_id][i][key]
            else:
                for key in accumulator:
                    accumulator[key] += self.test_result_accumulator[cohort_id][i][key]
        self.testing_history[cohort_id]['perf'][self.round[cohort_id]] = {'round': self.round[cohort_id], 'clock': self.global_virtual_clock[cohort_id]}
        for metric_name in accumulator.keys():
            if metric_name == 'test_loss':
                self.testing_history[cohort_id]['perf'][self.round[cohort_id]]['loss'] = accumulator['test_loss'] \
                    if self.args.task == "detection" else accumulator['test_loss'] / accumulator['test_len']
            elif metric_name not in ['test_len']:
                self.testing_history[cohort_id]['perf'][self.round[cohort_id]][metric_name] \
                    = accumulator[metric_name] / accumulator['test_len']

        round_perf = self.testing_history[cohort_id]['perf'][self.round[cohort_id]]
        logging.info(
            "FL Testing in round: {}, virtual_clock: {}, results: {}"
            .format(self.round[cohort_id], self.global_virtual_clock[cohort_id], round_perf))

    def testing_completion_handler(self, client_id, results, cohort_id):
        """Each executor will handle a subset of testing dataset

        Args:
            client_id (int): The client id.
            results (dictionary): The client test results.

        """

        results = results['results']
        # List append is thread-safe
        self.test_result_accumulator[cohort_id].append(results)

        # Have collected all testing results
        if len(self.test_result_accumulator[cohort_id]) == len(self.executors):
            self.aggregate_test_result(cohort_id)
            self.broadcast_aggregator_events(generate_msg(commons.START_ROUND, cohort_id=cohort_id))


if __name__ == "__main__":
    aggregator = AuxoAggregator(parser.args)
    aggregator.run()
