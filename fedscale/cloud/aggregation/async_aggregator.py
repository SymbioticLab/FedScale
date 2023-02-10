# -*- coding: utf-8 -*-
import copy
from collections import deque
from heapq import heappush, heappop
import numpy as np
from overrides import overrides

from fedscale.cloud.fllibs import *
from fedscale.cloud.aggregation.aggregator import Aggregator


class AsyncAggregator(Aggregator):
    """Represents an async aggregator implementing the FedBuff algorithm.
    Currently, this aggregator only supports simulation mode."""

    def _new_task(self, event_time):
        """Generates a new task that starts at event_time, and inserts it into the min heap.

        :param event_time: the time to start the new task.
        """
        client = self.client_manager.select_participants(1, cur_time=event_time)[0]
        client_cfg = self.client_conf.get(client, self.args)

        exe_cost = self.client_manager.get_completion_time(
            client,
            batch_size=client_cfg.batch_size,
            local_steps=client_cfg.local_steps,
            upload_size=self.model_update_size,
            download_size=self.model_update_size)
        self.virtual_client_clock[client] = exe_cost
        duration = exe_cost['computation'] + \
                   exe_cost['communication']
        end_time = event_time + duration
        heappush(self.min_pq, (event_time, 'start', client))
        heappush(self.min_pq, (end_time, 'end', client))
        self.client_task_start_times[client] = event_time
        self.client_task_model_version[client] = self.round

    @overrides
    def create_client_task(self, executor_id):
        """Issue a new client training task to the specific executor.

        Args:
            executor_id (int): Executor Id.

        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        next_client_id = self.resource_manager.get_next_task(executor_id)
        config = self.get_client_conf(next_client_id)
        train_config = {'client_id': next_client_id, 'task_config': config}
        model_version = self.client_task_model_version[next_client_id]
        return train_config, self.model_cache[self.round - model_version]

    @overrides
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
            self.model_cache.appendleft(self.model_wrapper.get_weights())
            if len(self.model_cache) > self.args.max_staleness + 1:
                self.model_cache.pop()
            clients_to_run = []
            durations = []
            final_time = self.global_virtual_clock
            if not self.min_pq:
                self._new_task(self.global_virtual_clock)
            while len(clients_to_run) < num_clients_to_collect:
                event_time, event_type, client = heappop(self.min_pq)
                if event_type == 'start':
                    self.current_concurrency += 1
                    if self.current_concurrency < self.args.max_concurrency:
                        self._new_task(event_time)
                else:
                    self.current_concurrency -= 1
                    if self.current_concurrency == self.args.max_concurrency - 1:
                        self._new_task(event_time)
                    if self.round - self.client_task_model_version[client] <= self.args.max_staleness:
                        clients_to_run.append(client)
                    durations.append(event_time - self.client_task_start_times[client])
                    final_time = event_time
            self.global_virtual_clock = final_time
            return clients_to_run, [], self.virtual_client_clock, 0, durations
        else:
            # Dummy placeholder for non-simulations.
            completed_client_clock = {
                client: {'computation': 1, 'communication': 1} for client in sampled_clients}
            times = [1 for _ in sampled_clients]
            return sampled_clients, sampled_clients, completed_client_clock, 1, times

    @overrides
    def setup_env(self):
        """Set up environment and variables."""
        self.setup_seed(seed=1)
        self.virtual_client_clock = {}
        self.min_pq = []
        self.model_cache = deque()
        self.client_task_start_times = {}
        self.client_task_model_version = {}
        self.current_concurrency = 0
        self.aggregation_denominator = 0

    @overrides
    def update_weight_aggregation(self, results):
        """Updates the aggregation with the new results.

        Implements the aggregation mechanism implemented in FedBuff
        https://arxiv.org/pdf/2106.06639.pdf (Nguyen et al., 2022)

        :param results: the results collected from the client.
        """
        update_weights = results['update_weight']
        # Aggregation weight is derived from equation from "staleness scaling" section in the referenced FedBuff paper.
        inverted_staleness = 1 / (1 + self.round - self.client_task_model_version[results['client_id']]) ** 0.5
        self.aggregation_denominator += inverted_staleness
        if type(update_weights) is dict:
            update_weights = [x for x in update_weights.values()]
        if self._is_first_result_in_round():
            self.model_weights = [weight * inverted_staleness for weight in update_weights]
        else:
            self.model_weights = [weight + inverted_staleness * update_weights[i] for i, weight in
                                  enumerate(self.model_weights)]
        if self._is_last_result_in_round():
            self.model_weights = [np.divide(weight, self.aggregation_denominator) for weight in self.model_weights]
            self.model_wrapper.set_weights(copy.deepcopy(self.model_weights))
            self.aggregation_denominator = 0


if __name__ == "__main__":
    aggregator = AsyncAggregator(parser.args)
    aggregator.run()
