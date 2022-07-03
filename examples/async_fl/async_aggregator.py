# -*- coding: utf-8 -*-

from fedscale.core.logger.aggragation import *
from fedscale.core.aggregation.aggregator import Aggregator
from fedscale.core import commons
from fedscale.core.channels import job_api_pb2

import torch
import sys, os

logging.info(f"===={os.path.dirname(os.path.abspath(__file__))}")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from resource_manager import ResourceManager

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB


class AsyncAggregator(Aggregator):
    """This centralized aggregator collects training/testing feedbacks from executors"""

    def __init__(self, args):
        Aggregator.__init__(self, args)
        self.resource_manager = ResourceManager(self.experiment_mode)
        self.async_buffer_size = args.async_buffer
        self.client_round_duration = {}
        self.client_start_time = {}
        self.round_stamp = [0]
        self.client_model_version = {}
        self.virtual_client_clock = {}

    def executor_info_handler(self, executorId, info):

        self.registered_executor_info.add(executorId)
        logging.info(
            f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}")

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

        if self.experiment_mode == commons.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            startTimes = []
            endTimes = []
            completed_client_clock = {}

            start_time = self.global_virtual_clock
            constant_checin_period = self.args.arrival_interval
            # 1. remove dummy clients that are not available to the end of training
            for client_to_run in sampled_clients:
                client_cfg = self.client_conf.get(client_to_run, self.args)
                exe_cost = self.client_manager.getCompletionTime(client_to_run,
                                                                 batch_size=client_cfg.batch_size,
                                                                 upload_step=client_cfg.local_steps,
                                                                 upload_size=self.model_update_size,
                                                                 download_size=self.model_update_size)

                roundDuration = exe_cost['computation'] + \
                    exe_cost['communication']
                # if the client is not active by the time of collection, we consider it is lost in this round
                start_time += constant_checin_period
                if self.client_manager.isClientActive(client_to_run, roundDuration + start_time):
                    sampledClientsReal.append(client_to_run)
                    completed_client_clock[client_to_run] = exe_cost
                    startTimes.append(start_time)
                    self.client_start_time[client_to_run] = start_time
                    self.client_round_duration[client_to_run] = roundDuration
                    endTimes.append(roundDuration + start_time)

            num_clients_to_collect = min(
                num_clients_to_collect, len(sampledClientsReal))
            # 2. sort & execute clients based on completion time
            sortedWorkersByCompletion = sorted(
                range(len(endTimes)), key=lambda k: endTimes[k])
            top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]
            return (clients_to_run,
                    start_time,
                    completed_client_clock)  # dict : string the speed for each client

        else:
            completed_client_clock = {
                client: {'computation': 1, 'communication': 1} for client in sampled_clients}
            completionTimes = [1 for c in sampled_clients]
            return (sampled_clients, sampled_clients, completed_client_clock,
                    1, completionTimes)

    def client_completion_handler(self, results):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache"""
        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        if self.args.gradient_policy in ['q-fedavg']:
            self.client_training_results.append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])

        self.client_manager.registerScore(results['clientId'], results['utility'],
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
        """May aggregate client updates on the fly"""
        """
            [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
            H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
        """
        # Start to take the average of updates, and we do not keep updates to save memory
        # Importance of each update is 1/#_of_participants
        # importance = 1./self.tasks_round
        client_staleness = self.round - \
            self.client_model_version[results['clientId']]
        importance = 1. / math.sqrt(1 + client_staleness)

        for p in results['update_weight']:
            param_weight = results['update_weight'][p]
            if isinstance(param_weight, list):
                param_weight = np.asarray(param_weight, dtype=np.float32)
            param_weight = torch.from_numpy(
                param_weight).to(device=self.device)

            if self.model_in_update == 1:
                self.model_weights[p].data = param_weight * importance
            else:
                self.model_weights[p].data += param_weight * importance

        if self.model_in_update == self.async_buffer_size:
            for p in self.model_weights:
                d_type = self.model_weights[p].data.dtype

                self.model_weights[p].data = (
                    self.model_weights[p] / float(self.async_buffer_size)).to(dtype=d_type)

    def aggregate_client_group_weights(self, results):
        """Streaming weight aggregation. Similar to aggregate_client_weights,
        but each key corresponds to a group of weights (e.g., for Tensorflow)"""

        client_staleness = self.round - \
            self.client_model_version[results['clientId']]
        importance = 1. / math.sqrt(1 + client_staleness)

        for p_g in results['update_weight']:
            param_weights = results['update_weight'][p_g]
            for idx, param_weight in enumerate(param_weights):
                if isinstance(param_weight, list):
                    param_weight = np.asarray(param_weight, dtype=np.float32)
                param_weight = torch.from_numpy(
                    param_weight).to(device=self.device)

                if self.model_in_update == 1:
                    self.model_weights[p_g][idx].data = param_weight * importance
                else:
                    self.model_weights[p_g][idx].data += param_weight * importance

        if self.model_in_update == self.async_buffer_size:
            for p in self.model_weights:
                for idx in range(len(self.model_weights[p])):
                    d_type = self.model_weights[p][idx].data.dtype

                    self.model_weights[p][idx].data = (
                        self.model_weights[p][idx].data /
                        float(self.async_buffer_size)
                    ).to(dtype=d_type)

    def round_completion_handler(self):
        # += self.round_duration
        self.global_virtual_clock = self.round_stamp[-1]
        # self.round_stamp.append(self.global_virtual_clock)
        self.round += 1

        if self.round % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate * self.args.decay_factor, self.args.min_learning_rate)

        # handle the global update w/ current and last
        self.round_weight_handler(self.last_gradient_weights)

        avg_loss = sum(self.loss_accumulator) / \
            max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, Remaining participants: " +
                     f"{self.resource_manager.get_remaining()}, Succeed participants: " +
                     f"{len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        # update select participants
        if self.round % self.args.checkin_period == 1:  # num_participantsants > buffer_size * sample_interval
            self.sampled_participants = self.select_participants(
                select_num_participants=self.args.num_participantsants, overcommitment=self.args.overcommitment)
            (clientsToRun, clientsStartTime, virtual_client_clock) = self.tictak_client_tasks(
                self.sampled_participants, len(self.sampled_participants))

            logging.info(f"{len(clientsToRun)} clients with constant arrival following the order: {clientsToRun}")

            # Issue requests to the resource manager; Tasks ordered by the completion time
            self.resource_manager.register_tasks(clientsToRun)
            self.virtual_client_clock.update(virtual_client_clock)

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id)
                                      for c_id in self.sampled_participants]

        self.save_last_param()
        #self.round_stragglers = round_stragglers

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
        """Result will be post on TensorBoard"""
        self.log_writer.add_scalar('Train/round_to_loss', avg_loss, self.round)
        self.log_writer.add_scalar(
            'FAR/time_to_train_loss (min)', avg_loss, self.global_virtual_clock / 60.)
        self.log_writer.add_scalar(
            'FAR/round_duration (min)', self.round_duration / 60., self.round)

    def find_latest_model(self, start_time):
        for i, time_stamp in enumerate(reversed(self.round_stamp)):
            if start_time >= time_stamp:
                return len(self.round_stamp) - i
        return None

    def get_client_conf(self, clientId):
        """Training configurations that will be applied on clients"""
        start_time = self.client_start_time[clientId]
        model_id = self.find_latest_model(start_time)
        self.client_model_version[clientId] = model_id
        end_time = self.client_round_duration[clientId] + start_time
        logging.info(f"Client {clientId} train on model {model_id} during {start_time}-{end_time}")

        conf = {
            'learning_rate': self.args.learning_rate,
            'model': model_id  # none indicates we are using the global model
        }
        return conf

    def create_client_task(self, executorId):
        """Issue a new client training task to the executor"""

        next_clientId = self.resource_manager.get_next_task(executorId)
        train_config = None
        # NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
        model = None
        # TODO: useless model?
        if next_clientId != None:
            config = self.get_client_conf(next_clientId)
            train_config = {'client_id': next_clientId, 'task_config': config}
        return train_config, model

    def CLIENT_REGISTER(self, request, context):
        """FL Client register to the aggregator"""

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

    def event_monitor(self):
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
                    if len(self.stats_util_accumulator) == self.async_buffer_size:
                        clientID = self.deserialize_response(data)['clientId']
                        self.round_stamp.append(
                            self.client_round_duration[clientID] + self.client_start_time[clientID])
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
        logging.info(f"Terminating the aggregator ...")
        time.sleep(5)


if __name__ == "__main__":
    aggregator = AsyncAggregator(args)
    aggregator.run()
