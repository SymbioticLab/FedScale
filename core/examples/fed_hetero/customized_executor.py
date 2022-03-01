import sys, os
import pickle
from customized_client import Customized_Client
from customized_fllibs import init_model

sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from executor import Executor
from argParser import args

class Customized_Executor(Executor):

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

    def init_model(self):
        """Return the model architecture used in training"""
        return init_model()
    
    def update_model_handler(self, request):
        """Update the model copy on this executor"""
        temp_model = pickle.loads(request.serialized_tensor)
        for p, tp in zip(self.model.state_dict().values(), temp_model.state_dict().values()):
            p.data = tp.to(device=self.device)
        del temp_model

        self.epoch += 1
        if self.epoch % self.args.dump_epoch == 0 and self.this_rank == 1:
            with open(self.temp_model_path+'_'+str(self.epoch), 'wb') as model_out:
                pickle.dump(self.model, model_out)

        # Dump latest model to disk
        with open(self.temp_model_path, 'wb') as model_out:
            pickle.dump(self.model, model_out)

if __name__ == "__main__":
    executor = Customized_Executor(args)
    executor.run()