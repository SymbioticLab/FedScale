
from fedscale.cloud.client_manager import *
from collections import  defaultdict
from client_metadata import AuxoClientMetadata
import logging
import numpy as np
from sklearn import preprocessing
from collections import defaultdict
import copy
from clustering import QTable
from sklearn.manifold import TSNE
from config import auxo_config


class HeterClientManager(ClientManager):
    def __init__(self, mode, args, sample_seed=233):
        '''
        Manage cohort membership;
        Manane client selection;
        Manage cohort training resources usage
        '''
        super().__init__(mode, args, sample_seed)

        self.round_clt = defaultdict(list)
        self.grad_div_dict = defaultdict(list)
        # self.split_round = defaultdict(bool)
        self.stop_cluster = False
        self.gradient_list = defaultdict(list)
        self.feasibleClients = [[]]

        self.total_res = args.num_participants
        self.round_acc =  defaultdict(dict)
        self.num_cluster = 1
        self.latest_acc_list = {0:1}

        logging.info(f'Client manager initialized with auxo config: {auxo_config}')


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

        self.client_metadata[uniqueId] = AuxoClientMetadata(host_id, client_id, speed, user_trace)
        # remove clients
        # if size >= self.filter_less and size <= self.filter_more:
        self.feasibleClients[0].append(client_id)
        self.feasible_samples += size

        if self.mode == "oort":
            feedbacks = {'reward': min(size, self.args.local_steps * self.args.batch_size),
                         'duration': duration,
                         }
            self.ucb_sampler.register_client(client_id, feedbacks=feedbacks)
        # else:
        #     del self.client_metadata[uniqueId]


    def getUniqueId(self, host_id, client_id):
        return int(client_id)
        # return (str(host_id) + '_' + str(client_id))

    def getFeasibleClients(self, cur_time: float, cohort_id: int = 0):
        if self.user_trace is None:
            clients_online = self.feasibleClients[cohort_id]
        else:
            clients_online = [client_id for client_id in self.feasibleClients[cohort_id] if self.client_metadata[self.getUniqueId(
                0, client_id)].is_active(cur_time)]

        logging.info(f"Wall clock time: {cur_time}, {len(clients_online)} clients online, " +
                     f"{len(self.feasibleClients[cohort_id]) - len(clients_online)} clients offline")

        return clients_online


    def select_participants(self, num_of_clients: int, cur_time: float = 0, cohort_id: int = 0, test: bool = False) -> List[int]:
        """Select participating clients for current execution task.

        Args:
            num_of_clients (int): number of participants to select.
            cur_time (float): current wall clock time.

        Returns:
            List[int]: indices of selected clients.

        """

        clients_online = self.getFeasibleClients(cur_time, cohort_id)

        if len(clients_online) <= num_of_clients:
            return clients_online

        self.gradient_list[cohort_id] = []
        pickled_clients = None
        clients_online_set = set(clients_online)

        if test:
            pivot_client = self.reward_qtable.return_pivot_client(cohort_id)
            pivot_client = list(set(pivot_client) & set(self.feasibleClients[cohort_id]))
            self.rng.shuffle(pivot_client)
            extra_clt = list(set(clients_online) - set(pivot_client))
            self.rng.shuffle(extra_clt)

            pickled_clients = pivot_client + extra_clt
            client_len = min(num_of_clients, len(pickled_clients) - 1)
            pickled_clients = pickled_clients[:client_len]

        elif self.mode == "oort":
            pickled_clients = self.ucb_sampler.select_participant(
                num_of_clients, feasible_clients=clients_online_set)
        else:
            self.rng.shuffle(clients_online)
            client_len = min(num_of_clients, len(clients_online) - 1)
            pickled_clients = clients_online[:client_len]
        return pickled_clients

    def getDataInfo(self):
        train_ratio = self.args.num_participants / len(self.feasibleClients[0])
        avg_train_times = self.args.rounds * self.args.num_participants / len(
            self.feasibleClients[0])
        known_clt = len(self.feasibleClients[0]) // 20

        split_round = auxo_config['split_round']
        exploredecay = auxo_config['exploredecay']
        explorerate = auxo_config['explorerate']
        self.reduction = auxo_config['reduction']
        metric = auxo_config['metric']

        self.reward_qtable = QTable(1 + len(self.feasibleClients[0]), known_clt = known_clt,
                                    split_round = split_round,\
                                    elbow_constant=0.97, train_ratio=train_ratio, avg_train_times=avg_train_times, \
                                    epsilon=explorerate, epsilon_decay = exploredecay,\
                                    metric=metric)

        return {'total_feasible_clients': len(self.feasibleClients[0]), 'total_num_samples': self.feasible_samples}

    def get_cohort_size(self, cohort_id):
        return len(self.feasibleClients[cohort_id])

    def register_feedback(self, client_id: int, reward: float, auxi: float = 1.0, time_stamp: float = 0,
                          duration: float = 1., success: bool = True, w_new = None, w_old = None, cohort_id = 0) -> None:
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

        if w_new is not None:
            grad_norm = self._register_client_grad(client_id, w_new, w_old, cohort_id)
        return

    def _register_client_grad(self, client_id, w_new, w_old, cohort_id):
        if not self.stop_cluster:
            grad_norm = self.client_metadata[client_id].register_gradient(w_new, w_old)
            self.round_clt[cohort_id].append(client_id)
            self.gradient_list[cohort_id].append(grad_norm)
            return grad_norm
        return 0

    def cohort_clustering(self, round, cohort_id=0):
        '''Clustering cohort results for current rounds; Update reward table
        Args:
            round: current round
            cohort_id: current cohort
        Returns:
            whether to split the cohort
        '''
        if round < auxo_config['start_round']:
            return False
        global_index = [int(gid) for gid in self.round_clt[cohort_id]]  # [1,2,3,5,7]
        logging.info(f'Clustering clients global index {global_index}')
        if len(global_index) < 5:
            return False
        gradient_list = [self.client_metadata[clt].gradient for clt in self.round_clt[cohort_id]]

        # if self.reduction:
        #     norm_grad = TSNE(n_components=3, init='random').fit_transform(np.array(gradient_list))
       #  elif self.distance == 'kl':
       #      norm_grad = centered_grad = np.array(gradient_list)
       # else:
        avg_grad = np.mean(gradient_list, axis=0)
        centered_grad = gradient_list - avg_grad
        norm_grad = preprocessing.normalize(centered_grad)

        # update sub reward
        logging.info(f'Update intra-cluster relation ')
        self.reward_qtable.knn_update_subR(cohort_id, norm_grad, global_index, False if round > 500 else True)
        logging.info(f'Update inter-cluster relation ')
        split = self.reward_qtable.update_mainR(cohort_id, centered_grad, global_index, False if round > 500 else True)

        if split:
            self._split(len(self.feasibleClients))
            logging.info(f'SPLIT at round {round} ')
            self.feasibleClients.append([])
            for i in range(len(self.feasibleClients)):
                self.feasibleClients[i] = list(np.argwhere(self.reward_qtable.y_kmeans == i).reshape(-1))

        # update feasible clients
        if len(self.feasibleClients) > 1 and split == False:
            for clt in self.round_clt[cohort_id]:  # str
                new_label = int(self.reward_qtable.y_kmeans[int(clt)])
                if int(clt) in self.feasibleClients[cohort_id] and new_label != cohort_id:
                    self.feasibleClients[cohort_id].remove(int(clt))
                    self.feasibleClients[new_label].append(int(clt))
                elif int(clt) not in self.feasibleClients[cohort_id] and new_label == cohort_id:
                    for c in range(len(self.feasibleClients)):
                        if int(clt) in self.feasibleClients[c]:
                            self.feasibleClients[c].remove(int(clt))
                    self.feasibleClients[cohort_id].append(int(clt))

        self.round_clt[cohort_id] = []

        self._print_cohort(round)
        return split

    def _print_cohort(self, round):
        size_ratio = [len(cluster) for cluster in self.feasibleClients]
        logging.info(f'Round {round} FeasibleClients client number : {size_ratio}')

    def schedule_plan(self, round=0, cohort_id=0) -> int:
        """ Schedule the training resources for each cohort

        Args:
            round:
            cohort_id:

        Returns:
            int: number of training resources for each cohort
        """
        # TODO: schedule based on accuracy
        return self.total_res // self.num_cluster

    def _split(self, cohort_id):
        self.num_cluster += 1
        if cohort_id not in self.latest_acc_list:
            self.latest_acc_list[cohort_id] = 1

    def update_eval(self, r, acc, clusterID):
        self.round_acc[r][clusterID] = acc
        if len(self.round_acc[r]) == self.num_cluster:
            self.latest_acc_list = self.round_acc[r]