import pytest
import numpy as np
import torch

from fedscale.cloud.aggregation.aggregator import ServerOptimizer
from test_aggregator import Struct

@pytest.fixture()
def optimizer_example():
    args = {'learning_rate':0.1,
            'qfed_q':1,
            'yogi_eta':3e-3 ,
            'yogi_tau': 1e-8,
            'yogi_beta': 0.9,
            'yogi_beta2': 0.99
            }
    optimizer = ServerOptimizer(mode='fed-yogi', args = Struct(**args), device = 'cpu')

    return optimizer

@pytest.fixture()
def last_model_weight_example():
    model_weights = [torch.tensor([1.1]) ]
    return model_weights


@pytest.fixture()
def cur_model_weight_example():
    model_weights = [torch.tensor([1.2]) ]
    return model_weights

@pytest.mark.usefixtures("env_setup")
class target_model_weight_example():
    def __init__(self):
        self.para = [Struct(**{'data': 0})]

    def parameters(self):
        return self.para


class TestOptimizer():
    def test_yogi(self, optimizer_example,
                  last_model_weight_example,
                  cur_model_weight_example
                   ):
        target_model_example = target_model_weight_example()
        optimizer_example.update_round_gradient(last_model_weight_example,
                                                cur_model_weight_example,
                                                target_model_example)
        assert target_model_example.parameters()[0].data == torch.tensor([1.2])

    def test_qfedavg(self):
        # TODO: qfedavg bug
        pass

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))