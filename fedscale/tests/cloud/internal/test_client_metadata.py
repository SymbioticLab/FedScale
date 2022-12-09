import pytest
import numpy as np

from fedscale.cloud.internal.client_metadata import ClientMetadata


@pytest.fixture()
def client_metadata():
    traces = {
        'active': [1, 3, 6, 9],
        'inactive': [1, 4, 7, 10],
        'finish_time': 10
    }
    return ClientMetadata("host", "client_1", {'computation': 1, 'communication': 1}, traces)


class TestClientMetadata:
    def test_get_completion_time(self, client_metadata):
        np.random.seed(seed=1)
        client_metadata.compute_speed = 1000
        client_metadata.bandwidth = 1
        assert client_metadata.get_completion_time(batch_size=1, local_steps=1, upload_size=1, download_size=1) \
               == {'communication': 2.0, 'computation': 3.0}

    def test_get_completion_time_lognormal(self, client_metadata):
        np.random.seed(seed=1)
        assert client_metadata.get_completion_time_lognormal(batch_size=1, local_steps=1, upload_size=1,
                                                             download_size=1) \
               == {'communication': 2.0, 'computation': 0.03601894790301564}
        assert client_metadata.get_completion_time_lognormal(batch_size=2, local_steps=2, upload_size=2,
                                                             download_size=2) \
               == {'communication': 4.0, 'computation': 0.037663009234622354}

    def test_is_active(self, client_metadata):
        assert not client_metadata.is_active(2)
        assert client_metadata.is_active(6)
        assert not client_metadata.is_active(8)
        assert client_metadata.is_active(9)
        assert not client_metadata.is_active(10)
        assert not client_metadata.is_active(15)
