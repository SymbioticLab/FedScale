import numpy as np


class ClientMetadata:
    """
    Contains the server-side metadata for a single client.
    """

    def __init__(self, host_id, client_id, speed, traces=None):
        """
        Initializes the ClientMetadata.
        :param host_id: id of the executor which is handling this client.
        :param client_id: id of the client.
        :param speed: computation and communication speed of the client.
        :param traces: list of client availability traces.
        """
        self.host_id = host_id
        self.client_id = client_id
        self.compute_speed = speed['computation']
        self.bandwidth = speed['communication']
        self.score = 0
        self.traces = traces
        self.behavior_index = 0

    def get_score(self):
        return self.score

    def register_reward(self, reward):
        """
        Registers the rewards of the client
        :param reward: int
        """
        self.score = reward

    def is_active(self, cur_time):
        """
        Decides whether the client is active at given cur_time
        :param cur_time: time in seconds
        :return: boolean
        """
        if self.traces is None:
            return True

        norm_time = cur_time % self.traces['finish_time']

        if norm_time > self.traces['inactive'][self.behavior_index]:
            self.behavior_index += 1

        self.behavior_index %= len(self.traces['active'])

        if self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]:
            return True

        return False

    def get_completion_time(self, batch_size, local_steps, upload_size, download_size, augmentation_factor=3.0):
        """
           Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers,
                                backward-pass takes around 2x the latency, so we multiple it by 3x;
           Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        return {'computation': augmentation_factor * batch_size * local_steps*float(self.compute_speed)/1000.,
                'communication': (upload_size+download_size)/float(self.bandwidth)}

    def get_completion_time_lognormal(self, batch_size, local_steps, upload_size, download_size,
                                      mean_seconds_per_sample=0.005, tail_skew=0.6):
        """
        Computation latency: compute_speed is the inference latency of models (ms/sample). The calculation assumes
        that client computation speed is a lognormal distribution (see PAPAPYA / GFL papers), and uses the parameters
        to sample a client task completion time.
        Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        :param batch_size: size of each training batch
        :param local_steps: number of local client training steps
        :param upload_size: size of model download (MB)
        :param download_size: size of model upload (MB)
        :param mean_seconds_per_sample: mean seconds to process a single training example. This can be adjusted based on
        on-device benchmarks.
        :param tail_skew: the skew of the lognormal distribution used to model device speed.
        :return: dict of computation and communication times for the client's training task.
        """
        device_speed = max(0.0001, np.random.lognormal(1, tail_skew, 1)[0])
        return {'computation': device_speed * mean_seconds_per_sample * batch_size * local_steps,
                'communication': (upload_size + download_size) / float(self.bandwidth)}
