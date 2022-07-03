
class ClientOptimizer(object):
 
    def __init__(self, sample_seed=233): 
        pass
    
    def update_client_weight(self, conf, model, global_model = None):
        if conf.gradient_policy == 'fed-prox':
            for idx, param in enumerate(model.parameters()):
                param.data += conf.learning_rate * conf.proxy_mu * (param.data - global_model[idx])