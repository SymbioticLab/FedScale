from sklearn import model_selection
import customized_fllibs

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from aggregator import Aggregator
from fl_aggregator_libs import *


class Customized_Aggregator(Aggregator):

    def __init__(self, args):
        super().__init__(args)
        self.param_idx = {}

    def init_model(self):
        # return super().init_model()
        return customized_fllibs.init_model()


    def client_completion_handler(self, results):
        self.client_training_results.append(results)
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])
        self.client_manager.registerScore(results['clientId'], results['utility'], auxi=math.sqrt(results['moving_loss']),
            time_stamp=self.epoch,
            duration=self.virtual_client_clock[results['clientId']]['computation']+self.virtual_client_clock[results['clientId']]['communication']
        )
        device = self.device

        self.update_lock.acquire()

        # ================== Aggregate weights ======================
        self.model_in_update += 1
        
        if self.model_in_update == self.tasks_round:
            self.combine_models()

        self.update_lock.release()

    def get_param_idx(self, model_rate):
        if model_rate in self.param_idx.keys():
            return self.param_idx[model_rate]
        else:
            param_idxi = None
            param_idx = OrderedDict()
            for k, v in self.model.state_dict().items():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv1' in k or 'conv2' in k:
                                if param_idxi is None:
                                    param_idxi = torch.arange(input_size, device=v.device)
                                input_idx_i = param_idxi
                                scaler_rate = model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                output_idx_i = torch.arange(output_size, device=v.device)[:local_output_size]
                                param_idxi = output_idx_i
                            elif 'shortcut' in k:
                                input_idx_i = param_idx[k.replace('shortcut', 'conv1')][1]
                                output_idx_i = param_idxi
                            elif 'linear' in k:
                                input_idx_i = param_idxi
                                output_idx_i = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            param_idx[k] = (output_idx_i, input_idx_i)
                        else:
                            input_idx_i = param_idxi
                            param_idx[k] = input_idx_i
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i = torch.arange(input_size, device=v.device)
                            param_idx[k] = input_idx_i
                        else:
                            input_idx_i = param_idxi
                            param_idx[k] = input_idx_i
                else:
                    pass
            self.param_idx[model_rate] = param_idx
            return param_idx

    def combine_models(self):
        logging.info("COMBINING MODEL")
        count = OrderedDict()
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