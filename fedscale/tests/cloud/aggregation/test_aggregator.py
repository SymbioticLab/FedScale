import pytest
import numpy as np
from fedscale.cloud.aggregation.aggregator import Aggregator

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

@pytest.fixture()
def aggregator_example():
    args = {'num_of_clients':10,
            'batch_size':8,
            'local_steps': 5,
            'experiment_mode': 'simulation',
            'use_cuda': False,
            'sample_mode': 'random',
            'task': 'cv',
            'data_set': 'femnist',
            'model': 'resnet18',
            'gradient_policy': None,
            'filter_less': 10,
            'filter_more': 100,
            'device_avail_file': None,
            'engine': 'pytorch',
            'connection_timeout': 5
            }
    args_struct = Struct(**args)
    aggregator_example = Aggregator(args_struct)
    aggregator_example.client_profiles = {}
    aggregator_example.num_of_clients = 0
    aggregator_example.model_update_size = 10
    aggregator_example.executors = list(range(2))
    aggregator_example.virtual_client_clock = {0: {'computation':1, 'communication':1},
                                               1: {'computation':1, 'communication':1}}
    aggregator_example.using_group_params = False
    aggregator_example.model_weights = {'weight':Struct(**{'data':0}),
                                        'bias':Struct(**{'data':0})}
    aggregator_example.tasks_round = 3
    return aggregator_example


@pytest.fixture()
def executor_example():
    exec_info = {'executorId': 0,
                'info': {'size': [0,1]}}
    return exec_info


@pytest.fixture()
def client_result_example():
    result = {
        'utility':1,
        'moving_loss':0.1,
        'clientId':0,
        'update_weight': {'weight':[1], 'bias':[1]}
    }
    return result


class TestAggregator:
    def test_executor_register(self, executor_example, aggregator_example):
        aggregator_example.executor_info_handler(**executor_example)

    def test_weight_aggegation(self, client_result_example, aggregator_example ):
        aggregator_example.client_completion_handler(client_result_example)
        for p in aggregator_example.model_weights:
            assert aggregator_example.model_weights[p].data == 1
        aggregator_example.client_completion_handler(client_result_example)
        for p in aggregator_example.model_weights:
            assert aggregator_example.model_weights[p].data == 2
        assert aggregator_example.model_in_update == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))