import customized_fllibs
from customized_fllibs import make_param_idx
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from fedscale.core.aggregator import Aggregator
from fedscale.core.fl_aggregator_libs import *


class Customized_Aggregator(Aggregator):

    def __init__(self, args):
        super().__init__(args)
        self.param_idx = {}


    def init_model(self):
        return customized_fllibs.init_model()
        

    def client_completion_handler(self, results):
        self.client_training_results.append(results)
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])
        self.client_manager.registerScore(results['clientId'], results['utility'], auxi=math.sqrt(results['moving_loss']),
            time_stamp=self.epoch,
            duration=self.virtual_client_clock[results['clientId']]['computation']+self.virtual_client_clock[results['clientId']]['communication']
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
    aggregator = Customized_Aggregator(args)
    aggregator.run()