import logging
import math
import pickle
from random import Random
from typing import Dict, List

from fedscale.cloud.internal.client_metadata import ClientMetadata


class ClientManager:

    def __init__(self, mode, args, sample_seed=233):
        self.client_metadata = {}
        self.client_on_hosts = {}
        self.mode = mode
        self.filter_less = args.filter_less
        self.filter_more = args.filter_more

        self.ucb_sampler = None

        if self.mode == 'oort':
            from thirdparty.oort.oort import create_training_selector
            self.ucb_sampler = create_training_selector(args=args)

        self.feasibleClients = []
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.count = 0
        self.feasible_samples = 0
        self.user_trace = None
        self.args = args

        if args.device_avail_file is not None:
            with open(args.device_avail_file, 'rb') as fin:
                self.user_trace = pickle.load(fin)
            self.user_trace_keys = list(self.user_trace.keys())

    def register_client(self, host_id: int, client_id: int, size: int, speed: Dict[str, float],
                        duration: float = 1) -> None:
        """Register client information to the client manager.

        Args:
            host_id (int): executor Id.
            client_id (int): client Id.
            size (int): number of samples on this client.
            speed (Dict[str, float]): device speed (e.g., compuutation and communication).
            duration (float): execution latency.

        """
        uniqueId = self.getUniqueId(host_id, client_id)
        user_trace = None if self.user_trace is None else self.user_trace[self.user_trace_keys[int(
            client_id) % len(self.user_trace)]]

        self.client_metadata[uniqueId] = ClientMetadata(host_id, client_id, speed, user_trace)

        # remove clients
        if size >= self.filter_less and size <= self.filter_more:
            self.feasibleClients.append(client_id)
            self.feasible_samples += size

            if self.mode == "oort":
                feedbacks = {'reward': min(size, self.args.local_steps * self.args.batch_size),
                             'duration': duration,
                             }
                self.ucb_sampler.register_client(client_id, feedbacks=feedbacks)
        else:
            del self.client_metadata[uniqueId]

    def getAllClients(self):
        return self.feasibleClients

    def getAllClientsLength(self):
        return len(self.feasibleClients)

    def getClient(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)]

    def registerDuration(self, client_id, batch_size, local_steps, upload_size, download_size):
        if self.mode == "oort" and self.getUniqueId(0, client_id) in self.client_metadata:
            exe_cost = self.client_metadata[self.getUniqueId(0, client_id)].get_completion_time(
                batch_size=batch_size, local_steps=local_steps,
                upload_size=upload_size, download_size=download_size
            )
            self.ucb_sampler.update_duration(
                client_id, exe_cost['computation'] + exe_cost['communication'])

    def get_completion_time(self, client_id, batch_size, local_steps, upload_size, download_size):
        return self.client_metadata[self.getUniqueId(0, client_id)].get_completion_time(
            batch_size=batch_size, local_steps=local_steps,
            upload_size=upload_size, download_size=download_size
        )

    def registerSpeed(self, host_id, client_id, speed):
        uniqueId = self.getUniqueId(host_id, client_id)
        self.client_metadata[uniqueId].speed = speed

    def registerScore(self, client_id, reward, auxi=1.0, time_stamp=0, duration=1., success=True):
        self.register_feedback(client_id, reward, auxi=auxi, time_stamp=time_stamp, duration=duration, success=success)

    def register_feedback(self, client_id: int, reward: float, auxi: float = 1.0, time_stamp: float = 0,
                          duration: float = 1., success: bool = True) -> None:
        """Collect client execution feedbacks of last round.

        Args:
            client_id (int): client Id.
            reward (float): execution utilities (processed feedbacks).
            auxi (float): unprocessed feedbacks.
            time_stamp (float): current wall clock time.
            duration (float): system execution duration.
            success (bool): whether this client runs successfully.

        """
        # currently, we only use distance as reward
        if self.mode == "oort":
            feedbacks = {
                'reward': reward,
                'duration': duration,
                'status': True,
                'time_stamp': time_stamp
            }

            self.ucb_sampler.update_client_util(client_id, feedbacks=feedbacks)

    def registerClientScore(self, client_id, reward):
        self.client_metadata[self.getUniqueId(0, client_id)].register_reward(reward)

    def get_score(self, host_id, client_id):
        uniqueId = self.getUniqueId(host_id, client_id)
        return self.client_metadata[uniqueId].get_score()

    def getClientsInfo(self):
        clientInfo = {}
        for i, client_id in enumerate(self.client_metadata.keys()):
            client = self.client_metadata[client_id]
            clientInfo[client.client_id] = client.distance
        return clientInfo

    def next_client_id_to_run(self, host_id):
        init_id = host_id - 1
        lenPossible = len(self.feasibleClients)

        while True:
            client_id = str(self.feasibleClients[init_id])
            csize = self.client_metadata[client_id].size
            if csize >= self.filter_less and csize <= self.filter_more:
                return int(client_id)

            init_id = max(
                0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))

    def getUniqueId(self, host_id, client_id):
        return str(client_id)
        # return (str(host_id) + '_' + str(client_id))

    def clientSampler(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)].size

    def clientOnHost(self, client_ids, host_id):
        self.client_on_hosts[host_id] = client_ids

    def getCurrentclient_ids(self, host_id):
        return self.client_on_hosts[host_id]

    def getClientLenOnHost(self, host_id):
        return len(self.client_on_hosts[host_id])

    def getClientSize(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)].size

    def getSampleRatio(self, client_id, host_id, even=False):
        totalSampleInTraining = 0.

        if not even:
            for key in self.client_on_hosts.keys():
                for client in self.client_on_hosts[key]:
                    uniqueId = self.getUniqueId(key, client)
                    totalSampleInTraining += self.client_metadata[uniqueId].size

            # 1./len(self.client_on_hosts.keys())
            return float(self.client_metadata[self.getUniqueId(host_id, client_id)].size) / float(totalSampleInTraining)
        else:
            for key in self.client_on_hosts.keys():
                totalSampleInTraining += len(self.client_on_hosts[key])

            return 1. / totalSampleInTraining

    def getFeasibleClients(self, cur_time):
        if self.user_trace is None:
            clients_online = self.feasibleClients
        else:
            clients_online = [client_id for client_id in self.feasibleClients if self.client_metadata[self.getUniqueId(
                0, client_id)].is_active(cur_time)]

        logging.info(f"Wall clock time: {round(cur_time)}, {len(clients_online)} clients online, " +
                     f"{len(self.feasibleClients) - len(clients_online)} clients offline")

        return clients_online

    def isClientActive(self, client_id, cur_time):
        return self.client_metadata[self.getUniqueId(0, client_id)].is_active(cur_time)

    def select_participants(self, num_of_clients: int, cur_time: float = 0) -> List[int]:
        """Select participating clients for current execution task.

        Args:
            num_of_clients (int): number of participants to select.
            cur_time (float): current wall clock time.

        Returns:
            List[int]: indices of selected clients.

        """
        self.count += 1

        clients_online = self.getFeasibleClients(cur_time)

        if len(clients_online) <= num_of_clients:
            return clients_online

        pickled_clients = None
        clients_online_set = set(clients_online)

        if self.mode == "oort" and self.count > 1:
            pickled_clients = self.ucb_sampler.select_participant(
                num_of_clients, feasible_clients=clients_online_set)
        else:
            self.rng.shuffle(clients_online)
            client_len = min(num_of_clients, len(clients_online) - 1)
            pickled_clients = clients_online[:client_len]

        return pickled_clients

    def resampleClients(self, num_of_clients, cur_time=0):
        return self.select_participants(num_of_clients, cur_time)

    def getAllMetrics(self):
        if self.mode == "oort":
            return self.ucb_sampler.getAllMetrics()
        return {}

    def getDataInfo(self):
        return {'total_feasible_clients': len(self.feasibleClients), 'total_num_samples': self.feasible_samples}

    def getClientReward(self, client_id):
        return self.ucb_sampler.get_client_reward(client_id)

    def get_median_reward(self):
        if self.mode == 'oort':
            return self.ucb_sampler.get_median_reward()
        return 0.
