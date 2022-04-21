
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from fedscale.core.aggregator import Aggregator
from fedscale.core.fl_aggregator_libs import *


class Customized_Aggregator(Aggregator):

    def __init__(self, args):
        super().__init__(args)

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
                                          duration=self.virtual_client_clock[results['clientId']]['computation' ]
                                                   +self.virtual_client_clock[results['clientId']]['communication']
                                          )

        device = self.device
        """
            [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
            H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
        """

        self.update_lock.acquire()

        # ================== Aggregate weights ======================

        self.model_in_update += 1

        if self.model_in_update == 1:
            self.model_state_dict = self.model.state_dict()
            for idx, param in enumerate(self.model_state_dict.values()):
                param.data = (torch.from_numpy(np.asarray(results['update_weight'][idx])).to(device=device))
        else:
            for idx, param in enumerate(self.model_state_dict.values()):
                param.data += (torch.from_numpy(np.asarray(results['update_weight'][idx])).to(device=device))

        if self.model_in_update == self.tasks_round:
            for idx, param in enumerate(self.model_state_dict.values()):
                param.data = (param.dat a /float(self.tasks_round)).to(dtype=param.data.dtype)

            self.model.load_state_dict(self.model_state_dict)

        self.update_lock.release()

if __name__ == "__main__":
    aggregator = Customized_Aggregator(args)
    aggregator.run()