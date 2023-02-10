import math
import random
from collections import OrderedDict

import torch

import config
import customized_fllibs
from customized_fllibs import make_param_idx
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.logger.aggregation_logging import *


class Customized_Aggregator(Aggregator):

    def __init__(self, args):
        super().__init__(args)
        self.param_idx = {}


    def init_model(self):
        return customized_fllibs.init_model()

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        sampledClientsReal = []
        completionTimes = []
        completed_client_clock = {}
        # 1. remove dummy clients that are not available to the end of training
        for client_to_run in sampled_clients:
            client_cfg = self.client_conf.get(client_to_run, self.args)

            exe_cost = self.client_manager.get_completion_time(client_to_run,
                                    batch_size=client_cfg.batch_size, local_steps=client_cfg.local_steps,
                                    upload_size=self.model_update_size, download_size=self.model_update_size)

            roundDuration = exe_cost['computation'] + exe_cost['communication']
            # if the client is not active by the time of collection, we consider it is lost in this round
            if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
                sampledClientsReal.append(client_to_run)
                completionTimes.append(roundDuration)
                completed_client_clock[client_to_run] = exe_cost

        num_clients_to_collect = min(int(num_clients_to_collect * config.cfg['participation_rate']), len(completionTimes))
        # 2. get the top-k completions to remove stragglers
        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
        top_k_index = random.sample(sortedWorkersByCompletion, k=num_clients_to_collect)
        clients_to_run = [sampledClientsReal[k] for k in top_k_index]

        stragglers = [sampledClientsReal[k] for k in sortedWorkersByCompletion[num_clients_to_collect:]]
        round_duration = completionTimes[top_k_index[-1]]
        completionTimes.sort()

        return clients_to_run, [], completed_client_clock, round_duration, completionTimes[:num_clients_to_collect]

    def client_completion_handler(self, results):
        self.client_training_results.append(results)
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])
        self.client_manager.registerScore(results['client_id'], results['utility'], auxi=math.sqrt(results['moving_loss']),
            time_stamp=self.epoch,
            duration=self.virtual_client_clock[results['client_id']]['computation']+self.virtual_client_clock[results['client_id']]['communication']
        )

        self.update_lock.acquire()
        self.model_in_update += 1

        if self.model_in_update == self.tasks_round:
            self.combine_models()

        self.update_lock.release()


    def get_param_idx(self, model_rate):
        if model_rate not in self.param_idx.keys():
            self.param_idx[model_rate] = make_param_idx(self.model, model_rate)
        return self.param_idx[model_rate]

    def combine_models(self):
        """
        "HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients"
        Enmao Diao, Jie Ding, Vahid Tarokh,
        ICLR 2021
        """
        count = OrderedDict()
        # combine sub-models into global model
        for k, v in self.model.state_dict().items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(self.client_training_results)):
                param_idx = self.get_param_idx(self.client_training_results[m]['model_rate'])
                local_parameters = self.client_training_results[m]['local_parameters']
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            if 'linear' in k:
                                param_idx[k] = list(param_idx[k])
                                tmp_v[torch.meshgrid(param_idx[k])] += local_parameters[k]
                                count[k][torch.meshgrid(param_idx[k])] += 1
                            else:
                                tmp_v[torch.meshgrid(param_idx[k])] += local_parameters[k]
                                count[k][torch.meshgrid(param_idx[k])] += 1
                        else:
                            tmp_v[param_idx[k]] += local_parameters[k]
                            count[k][param_idx[k]] += 1
                    else:
                        if 'linear' in k:
                            param_idx[k] = param_idx[k]
                            tmp_v[param_idx[k]] += local_parameters[k]
                            count[k][param_idx[k]] += 1
                        else:
                            tmp_v[param_idx[k]] += local_parameters[k]
                            count[k][param_idx[k]] += 1
                else:
                    tmp_v += local_parameters[k]
                    count[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        return


if __name__ == "__main__":
    aggregator = Customized_Aggregator(parser.args)
    aggregator.run()