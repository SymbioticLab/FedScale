# -*- coding: utf-8 -*-

import os
import sys

import torch

from fedscale.core import commons
from fedscale.core.aggregation.aggregator import Aggregator
from fedscale.core.channels import job_api_pb2
from fedscale.core.logger.aggragation import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from resource_manager import ResourceManager

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB

# NOTE: We are supporting and improving the following implementation (Async FL) in FedScale:
    # - "PAPAYA: Practical, Private, and Scalable Federated Learning", MLSys, 2022
    # - "Federated Learning with Buffered Asynchronous Aggregation", AISTATS, 2022

# We appreciate you to contribute and/or report bugs. Thank you!

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
        self.weight_tensor_type = {}

        # We need to keep the test model for specific round to avoid async mismatch
        self.test_model = None
        self.aggregate_update = {}
        self.importance_sum = 0

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):

        if self.experiment_mode == commons.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            startTimes = []
            endTimes = []
            completed_client_clock = {}

            start_time = self.global_virtual_clock
            constant_checkin_period = self.args.arrival_interval
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
                start_time += constant_checkin_period
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

    def save_last_param(self):
        """ Save the last model parameters
        """
        self.last_gradient_weights = [
            p.data.clone() for p in self.model.parameters()]
        self.model_weights = copy.deepcopy(self.model.state_dict())
        self.weight_tensor_type = {p: self.model_weights[p].data.dtype \
                                        for p in self.model_weights}

    def aggregate_client_weights(self, results):
        """May aggregate client updates on the fly"""
        """
            "PAPAYA: PRACTICAL, PRIVATE, AND SCALABLE FEDERATED LEARNING". MLSys, 2022
        """
        # Start to take the average of updates, and we do not keep updates to save memory
        # Importance of each update is 1/staleness
        client_staleness = self.round - self.client_model_version[results['clientId']]
        importance = 1. #/ (math.sqrt(1 + client_staleness))

        new_round_aggregation = (self.model_in_update == 1)
        if new_round_aggregation:
            self.importance_sum = 0
        self.importance_sum += importance

        for p in results['update_weight']:
            # Different to core/executor, update_weight here is (train_model_weight - untrained)
            param_weight = results['update_weight'][p]

            if isinstance(param_weight, list):
                param_weight = np.asarray(param_weight, dtype=np.float32)
            param_weight = torch.from_numpy(
                param_weight).to(device=self.device)

            # if self.model_weights[p].data.dtype in (
            #     torch.float, torch.double, torch.half, 	
            #     torch.bfloat16, torch.chalf, torch.cfloat, torch.cdouble
            # ):  
                # Only assign importance to floats (trainable variables) 
            if new_round_aggregation:
                self.aggregate_update[p] = param_weight * importance
            else:
                self.aggregate_update[p] += param_weight * importance
                
                # self.model_weights[p].data += param_weight * importance
            # else:
            #     # Non-floats (e.g., num_batches), no need to aggregate but need to track
            #     self.aggregate_update[p] = param_weight

        if self.model_in_update == self.async_buffer_size:
            for p in self.model_weights:
                d_type = self.weight_tensor_type[p]
                self.model_weights[p].data = (
                    self.model_weights[p].data + self.aggregate_update[p]/self.importance_sum
                ).to(dtype=d_type)

    def round_completion_handler(self):
        self.global_virtual_clock = self.round_stamp[-1]
        self.round += 1

        if self.round % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate * self.args.decay_factor, self.args.min_learning_rate)

        # handle the global update w/ current and last
        self.round_weight_handler(self.last_gradient_weights)

        avg_loss = sum(self.loss_accumulator) / \
            max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, asyn running participants: " +
                     f"{self.resource_manager.get_task_length()}, aggregating {len(self.stats_util_accumulator)} participants, " +
                     f"training loss: {avg_loss}")

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        # update select participants
        # NOTE: we simulate async, while have to sync every 20 rounds to avoid large division to trace
        if self.resource_manager.get_task_length() < self.async_buffer_size:
            self.sampled_participants = self.select_participants(
                select_num_participants=self.async_buffer_size*2, overcommitment=self.args.overcommitment)
            (clientsToRun, clientsStartTime, virtual_client_clock) = self.tictak_client_tasks(
                self.sampled_participants, len(self.sampled_participants))

            logging.info(f"{len(clientsToRun)} clients with constant arrival following the order: {clientsToRun}")
            logging.info(f"====Register {len(clientsToRun)} to queue")
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
        self.loss_accumulator = []

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0:
            self.test_model = copy.deepcopy(self.model)
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.START_ROUND)

    def find_latest_model(self, start_time):
        for i, time_stamp in enumerate(reversed(self.round_stamp)):
            if start_time >= time_stamp:
                return len(self.round_stamp) - i
        return 1

    def get_test_config(self, client_id):
        """FL model testing on clients, developers can further define personalized client config here.
        
        Args:
            client_id (int): The client id.
        
        Returns:
            dictionary: The testing config for new task.
        
        """
        # Get the straggler round-id
        client_tasks = self.resource_manager.client_run_queue
        current_pending_length = min(
            self.resource_manager.client_run_queue_idx, len(client_tasks)-1)

        current_pending_clients = client_tasks[current_pending_length:]
        straggler_round = 1e10
        for client in current_pending_clients:
            straggler_round = min(
                self.find_latest_model(self.client_start_time[client]), straggler_round)

        return {'client_id': client_id, 
                'straggler_round': straggler_round, 
                'test_model': self.test_model}

    def get_client_conf(self, clientId):
        """Training configurations that will be applied on clients"""
        conf = {
            'learning_rate': self.args.learning_rate,
        }
        return conf

    def create_client_task(self, executorId):
        """Issue a new client training task to the executor"""

        next_clientId = self.resource_manager.get_next_task(executorId)
        train_config = None
        model = None

        if next_clientId != None:
            config = self.get_client_conf(next_clientId)
            start_time = self.client_start_time[next_clientId]
            model_id = self.find_latest_model(start_time)
            self.client_model_version[next_clientId] = model_id
            end_time = self.client_round_duration[next_clientId] + start_time

            # The executor has already received the model, thus transfering id is enough
            model = model_id
            train_config = {'client_id': next_clientId, 'task_config': config}

            logging.info(f"Client {next_clientId} train on model {model_id} during {int(start_time)}-{int(end_time)}")

        return train_config, model

    def log_train_result(self, avg_loss):
        """Result will be post on TensorBoard"""
        self.log_writer.add_scalar('Train/round_to_loss', avg_loss, self.round)
        self.log_writer.add_scalar(
            'FAR/time_to_train_loss (min)', avg_loss, self.global_virtual_clock / 60.)
        self.log_writer.add_scalar(
            'FAR/round_duration (min)', self.round_duration / 60., self.round)

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
                    if self.model_in_update == self.async_buffer_size:
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

if __name__ == "__main__":
    aggregator = AsyncAggregator(args)
    aggregator.run()
