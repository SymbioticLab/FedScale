from fl_aggregator_libs import *
from aggregator import Aggregator
from customized_fllibs import init_model


class Customized_Aggregator(Aggregator):

    def __init__(self, args):
        super().__init__(args)

    def init_model(self):
        # return super().init_model()
        return init_model()

    
    def client_completion_handler(self, results):
        self.client_training_results.append(results)
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])
        self.client_manager.registerScore(results['clientId'], results['utility'], auxi=math.sqrt(results['moving_loss']),
            time_stamp=self.epoch,
            duration=self.virtual_client_clock[results['clientId']]['computation']+self.virtual_client_clock[results['clientId']]['communication']
        )
        device = self.device
    
    def combine_models(self):
        count = OrderedDict()
        for k, v in self.model.state_dict().items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(self.client_training_results)):
                param_idx = self.client_training_results[m]['param_idx']
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
    
    def round_completion_handler(self):
        self.global_virtual_clock += self.round_duration
        self.epoch += 1

        if self.epoch % self.args.decay_epoch == 0:
            self.args.learning_rate = max(self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        # [Customized] handle the global update w/ current and last
        self.combine_models()

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

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_writer.add_scalar('Train/round_to_loss', avg_loss, self.epoch)

            self.log_writer.add_scalar('FAR/time_to_train_loss (min)', avg_loss, self.global_virtual_clock/60.)
            self.log_writer.add_scalar('FAR/round_duration (min)', self.round_duration/60., self.epoch)
            self.log_writer.add_histogram('FAR/client_duration (min)', self.flatten_client_duration, self.epoch)

        # update select participants
        self.sampled_participants = self.select_participants(
                        select_num_participants=self.args.total_worker, overcommitment=self.args.overcommitment)
        clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration = self.tictak_client_tasks(
                        self.sampled_participants, self.args.total_worker)

        logging.info(f"Selected participants to run: {clientsToRun}:\n{virtual_client_clock}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)
        self.tasks_round = len(clientsToRun)

        self.save_last_param()
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.flatten_client_duration = numpy.array(flatten_client_duration)
        self.round_duration = round_duration
        self.model_in_update = 0
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