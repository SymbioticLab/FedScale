
class ServerOptimizer(object):
 
    def __init__(self, mode, args, device, sample_seed=233):
        
        self.mode = mode
        self.args = args
        self.device = device

        if mode == 'fed-yogi':
            from utils.yogi import YoGi
            self.gradient_controller = YoGi(eta=args.yogi_eta, tau=args.yogi_tau, beta=args.yogi_beta, beta2=args.yogi_beta2)
        
        
    def update_round_gradient(self, last_model, current_model, target_model):
        
        if self.mode == 'fed-yogi':
            """
            "Adaptive Federated Optimizations", 
            Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konecný, Sanjiv Kumar, H. Brendan McMahan,
            ICLR 2021.
            """
            last_model = [x.to(device=self.device) for x in last_model]
            current_model = [x.to(device=self.device) for x in current_model]

            diff_weight = self.gradient_controller.update([pb-pa for pa, pb in zip(last_model, current_model)])

            for idx, param in enumerate(target_model.parameters()):
                param.data = last_model[idx] + diff_weight[idx]

            
        elif self.mode =='q-fedavg':
            """
            "Fair Resource Allocation in Federated Learning", Tian Li, Maziar Sanjabi, Ahmad Beirami, Virginia Smith, ICLR 2020.
            """
            learning_rate, qfedq = self.args.learning_rate, self.args.qfed_q
            Deltas, hs = None, 0.
            last_model = [x.to(device=self.device) for x in last_model]

            for result in self.client_training_results:
                # plug in the weight updates into the gradient
                grads = [(u - torch.from_numpy(v).to(device=self.device)) * 1.0 / learning_rate for u, v in zip(last_model, result['update_weight'])]
                loss = result['moving_loss']

                if Deltas is None:
                    Deltas = [np.float_power(loss+1e-10, qfedq) * grad for grad in grads]
                else:
                    for idx in range(len(Deltas)):
                        Deltas[idx] += np.float_power(loss+1e-10, qfedq) * grads[idx]

                # estimation of the local Lipchitz constant
                hs += (qfedq * np.float_power(loss+1e-10, (qfedq-1)) * torch.sum(torch.stack([torch.square(grad).sum() for grad in grads])) + (1.0/learning_rate) * np.float_power(loss+1e-10, qfedq))

            # update global model
            for idx, param in enumerate(target_model.parameters()):
                param.data = last_model[idx] - Deltas[idx]/(hs+1e-10)

        else:
            # The default optimizer, FedAvg, has been applied in aggregator.py on the fly
            pass


class ClientOptimizer(object):
 
    def __init__(self, sample_seed=233): 
        pass
    
    def update_client_weight(self, conf, model, global_model = None):
        if conf.gradient_policy == 'fed-prox':
            for idx, param in enumerate(model.parameters()):
                param.data += conf.learning_rate * conf.proxy_mu * (param.data - global_model[idx])
        
        
