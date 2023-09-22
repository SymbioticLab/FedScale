import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from random import Random
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
import logging, time
from utils.klkmeans import KLKmeans


class QEntry():
    def __init__(self, init_R=1, r0=1, train_times_R=0):
        self.R = init_R
        self.sub_R = [init_R / 2, init_R / 2]
        self.lr = 0.9
        self.r0 = r0  # lambda : np.random.normal(r0, r0/20)
        self.train_times_R = train_times_R

    def update_R(self, new_reward):
        old_R = self.R
        # self.R = self.lr * new_reward + self.R * (1 - self.lr)
        self.R = new_reward + self.R
        delta_R = self.R - old_R
        self.sub_R[0] += delta_R / 2
        self.sub_R[1] += delta_R / 2

    def update_sub(self, sub_id):
        self.sub_R[sub_id] += self.r0  # ()
        self.sub_R[1 - sub_id] -= self.r0  # ()

    def reset_sub(self):
        self.sub_R = [self.R / 2, self.R / 2]
        self.train_times_R = 0


class QTable():
    def __init__(self, num_client, train_ratio=0.1, base_reward=5, sample_seed=233, epsilon=0.01, epsilon_decay=0.99, \
                 known_clt=50, elbow_constant=0.45, avg_train_times=4, split_round=None, merge=False, metric='cosine'):
        # list dict
        self.Qtable = [{0: QEntry()} for row in range(num_client)]  # init with one model
        self.num_client = num_client
        self.y_kmeans = np.zeros(num_client)
        self.base_reward = base_reward
        self.known_clt = known_clt
        self.num_model = 1
        self.init_round = {0: False}
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.sub_num = {}
        self.split_counter = defaultdict(int)
        self.epoch = defaultdict(int)
        self.min_round = 1
        self.elbow_constant = elbow_constant
        self.train_ratio = train_ratio
        self.min_clt = 100 / train_ratio  # minimal # clt per round --> 50
        self.avg_train_times = avg_train_times
        self.pivot_clients = defaultdict(list)
        self.pivot_clients[0] = [*range(self.num_client)]
        self.min_cluster_size = num_client // 10
        self.split_round = split_round
        self.merge_action = merge
        self.metric = metric
        if metric == 'kl':
            self._initialize_kl()

    def _initialize_kl(self):
        """Initialize KL divergence metric if applicable."""
        def KL(a, b):
            epsilon = 0.00001
            a += epsilon
            b += epsilon
            return np.sum(np.where(a != 0, a * np.log(a / b), 0))
        self.kl = KL

    def update_R(self, cid, mid, new_reward):
        self.Qtable[cid][mid].update_R(new_reward)

        if new_reward < 0:
            self.Qtable[cid][mid].reset_sub()

    def update_R_batch(self, cid_list, mid, reward_list, remain_round):
        for cid, reward in zip(cid_list, reward_list):
            self.update_R(cid, mid, reward)
            self.Qtable[cid][mid].train_times_R += 1

    def update_subR(self, cid, mid, sub_id):
        self.Qtable[cid][mid].update_sub(sub_id)
        self.Qtable[cid][mid].train_times_R += 1

    def update_subR_batch(self, cid_list, mid, sub_id):
        for cid in cid_list:
            self.update_subR(cid, mid, sub_id)

    def get_subid(self, cid, mid):
        # ( cid ) prefer which subcluster in mid
        if self.Qtable[cid][mid].sub_R[0] == self.Qtable[cid][mid].sub_R[1] or self.Qtable[cid][mid].R < 0:
            return 2
        else:
            return int(self.Qtable[cid][mid].sub_R[0] < self.Qtable[cid][mid].sub_R[1])

    def subcluster_policy(self, mid, clt_list):
        sub_num = [0, 0, 0]
        sub_label_list = []
        for cid in clt_list:
            sub_id = self.get_subid(cid, mid)
            sub_num[sub_id] += 1
            sub_label_list.append(sub_id)
        return sub_num, sub_label_list

    def grow_table(self, mid):
        self.split_counter[mid] = 0
        new_mid = self.num_model
        self.init_round[new_mid] = False
        self.init_round[mid] = False
        self.epoch[new_mid] = self.epoch[mid]
        for cid in range(self.num_client):
            sub_a = self.Qtable[cid][mid].sub_R[0]
            sub_b = self.Qtable[cid][mid].sub_R[1]
            a_R = 1 if sub_a > sub_b else 0
            b_R = 1 - a_R if sub_a != sub_b else 0

            reward = self.Qtable[cid][mid].R
            times = self.Qtable[cid][mid].train_times_R
            self.Qtable[cid][mid] = QEntry(init_R=reward + a_R, train_times_R=times * a_R)
            self.Qtable[cid][new_mid] = QEntry(init_R=reward + b_R, train_times_R=times * b_R)
        self.num_model += 1

    def shrink_table(self, mid):
        self.num_model -= 1
        for cid in range(self.num_client):
            self.Qtable[cid].pop(mid, None)
            tmp_dict = self.Qtable[cid]
            for m in range(self.num_model):
                self.Qtable[cid][m] = list(tmp_dict.values())[m]

    def model_policy(self, cid):

        train_times = [self.Qtable[cid][m].train_times_R for m in self.Qtable[cid]]
        new_client = False if np.sum(train_times) > 0 else True

        self.epsilon *= self.epsilon_decay
        if new_client or self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.num_model - 1)
        else:
            reward_list = self.get_model_reward(cid)
            return np.argmax(reward_list)

    def model_policy_batch(self, client_list):
        for cid in client_list:
            self.y_kmeans[cid] = self.model_policy(cid)

        self.y_kmeans[0] = -1

    def get_model_reward(self, cid):
        return [self.Qtable[cid][m].R for m in self.Qtable[cid]]

    def dist_to_reward(self, dist_list, known_ratio):

        avg_dist = np.mean(dist_list)
        std_dist = np.std(dist_list)

        R0 = self.base_reward
        slope = self.base_reward / (avg_dist + std_dist)
        return [(R0 - slope * d) * known_ratio for d in dist_list]

    def count_known_main(self, global_index, mid):
        known_index_list = [i for i, cid in enumerate(global_index) \
                            if self.Qtable[cid][mid].train_times_R > 0 and self.Qtable[cid][mid].R > 1]
        # local clt index
        return known_index_list

    def split(self, mid):
        # whether to split: enough client subreward info + minimal size satisfy + elbow test
        if self.split_round is not None and self.epoch[mid] in self.split_round:
            return True
        elif self.split_round is None:
            # whether to split: enough client subreward info + minimal size satisfy + elbow test
            mid_client = np.argwhere(self.y_kmeans == mid)  # global idx
            mid_client = [i[0] for i in mid_client]  # [1,3,5,7,9]
            if len(mid_client) < self.min_clt:  # ensure around 50 participate every round
                return False

            sub_num = [0, 0, 0]
            for cid in mid_client:
                sub_num[self.get_subid(cid, mid)] += 1
            logging.info(f'sub_num : {sub_num} ')
            min_size = self.known_clt * 0.9 ** (self.num_model - 1)
            ratio = sub_num[0] / max(sub_num[1], 1)

            within_ratio = True if (ratio > 0.5 and ratio < 2) or self.num_model == 1 else False
            return sub_num[0] > min_size and sub_num[1] > min_size and within_ratio
        else:
            return False

    def knn_update_subR(self, mid, X_sub, global_index, keep_split=True):
        # update cluster membership for each clusters
        self.epoch[mid] += 1
        if self.split_round is not None:
            if self.epoch[mid] > max(self.split_round) + 1:
                keep_split = False
            else:
                keep_split = True

        if self.init_round[mid] == False:
            # First round of clustering
            if self.num_model > 1:
                known_clt_list = self.count_known_main(global_index, mid)
                if len(known_clt_list) <= 1:
                    return
                X_sub = X_sub[known_clt_list]
                global_index = known_clt_list

            if self.metric == 'kl':
                clustering = KLKmeans(n_clusters=2, init_center = X_sub[:2] )
                clustering.fit(X_sub)
                labels = clustering.labels_

            else:
                clustering = MiniBatchKMeans(n_clusters=2,
                                             random_state=0,
                                             batch_size=10).fit(X_sub)
                labels = clustering.labels_  # return labels

            for clt in range(len(global_index)):
                self.update_subR(global_index[clt], mid, int(labels[clt]))
            self.init_round[mid] = True

        elif keep_split:
            # Continuous clustering of subsequent rounds
            sub_num, sub_label_list = self.subcluster_policy(mid, global_index)

            if sub_num[0] == sub_num[1] == 0:
                return

            elif sub_num[0] == 0 or sub_num[1] == 0:
                sub_size = len(global_index) // 4
                if self.metric == 'kl':
                    neigh = NearestNeighbors(n_neighbors=sub_size, metric=lambda a, b: self.kl(a, b)).fit(X_sub)
                else:
                    neigh = NearestNeighbors(n_neighbors=sub_size).fit(X_sub)
                center_solo = np.argwhere(np.array(sub_label_list) != 2)[0][0]
                near_clt_id = neigh.kneighbors([X_sub[center_solo]])[1][0]
                for clt in near_clt_id:
                    self.update_subR(global_index[clt], mid, sub_label_list[center_solo])
                return

            else:
                labeled_id = np.argwhere(np.array(sub_label_list) != 2).reshape(-1)
                knn_label = list(filter(lambda score: score != 2, sub_label_list))
                knn_data = X_sub[labeled_id]
                if self.metric == 'kl':
                    neigh = KNeighborsClassifier(n_neighbors=1, metric=lambda a, b: self.kl(a, b)).fit(knn_data,
                                                                                                       knn_label)
                else:
                    neigh = KNeighborsClassifier(n_neighbors=1).fit(knn_data, knn_label)
                # labels =  neigh.predict(X_sub)
                unseen_id = np.argwhere(np.array(sub_label_list) == 2).reshape(-1)
                if len(unseen_id) < 1:
                    return
                labels = neigh.predict(X_sub[unseen_id])
                pred_prob = neigh.predict_proba(X_sub[unseen_id])
                for i, unseen_clt in enumerate(unseen_id):
                    conf = pred_prob[i][0] / max(pred_prob[i][1], 0.1)
                    if conf > 1.5 or conf < 0.67:
                        self.update_subR(global_index[unseen_clt], mid, int(labels[i]))
                return

    def update_mainR(self, mid, X_sub, global_index, remain_round=True):
        # update cluster membership for each clusters
        if self.split_round is not None:
            if self.epoch[mid] > max(self.split_round) + 1:
                remain_round = False
            else:
                remain_round = True

        if self.num_model > 1:
            known_clt_list = self.count_known_main(global_index, mid)
            known_ratio = len(known_clt_list) / len(global_index)

            if len(known_clt_list) < 2 and self.train_ratio < 1:
                # X_subcen = np.mean(X_sub, axis=0)
                # print("know too less")
                return False
            else:
                X_known = X_sub[known_clt_list]
                X_subcen = np.mean(X_known, axis=0)

            X_ = X_sub - X_subcen
            square_dist = np.sum(X_ ** 2, axis=1)
            dist_list = np.sqrt(square_dist)
            reward_list = self.dist_to_reward(dist_list, known_ratio)
            self.update_R_batch(global_index, mid, reward_list, remain_round)

        split = False
        self.split_counter[mid] += 1
        if self.split(mid):
            # if self.split_counter[mid] > self.min_round :
            split = True
            self.grow_table(mid)

        if self.num_model > 2 and self.split_counter[mid] > 5 and self.merge(mid):
            self.shrink_table(mid)
            print(f"<<<  Merge cluster {mid}")
            logging.info(f"<<<  Merge cluster {mid}")
        self.model_policy_batch([*range(self.num_client)])  # update kmeans

        return split

    def update_pivot_client(self, mid):
        trained_clt = set()
        c = set()
        for cid in range(1, self.num_client):
            if self.Qtable[cid][mid].train_times_R > 0 and self.Qtable[cid][mid].R > 1:
                c.add(cid)
            #    tmp_dict = self.Qtable[cid]
            #    if np.argsort(tmp_dict.values())[-1] == mid:
            if self.Qtable[cid][mid].R > 1:
                trained_clt.add(cid)
        print(len(c), len(trained_clt))
        self.pivot_clients[mid] = list(trained_clt)
        # TODO: can have many overlap clients, instead choose clients with highest score

    def return_pivot_client(self, mid):
        # Return the clients that belong to the cohort
        self.update_pivot_client(mid)
        return self.pivot_clients[mid]

    def merge(self, mid):
        if self.merge_action == False:
            return False

        mid_client = np.argwhere(self.y_kmeans == mid)
        if len(mid_client) < self.min_cluster_size:
            return True
        return False

    def count_trained_clt(self):
        trained_clt = []
        for mid in range(self.num_model):
            self.update_pivot_client(mid)
            trained_clt += self.pivot_clients[mid]
        print("Trained clients :", len(set(trained_clt)))
        return list(set(trained_clt))

    def plot(self, X, epoch):
        # Visualize the clustering result
        trained_clt = self.count_trained_clt()
        plt.scatter(X[trained_clt, 0], X[trained_clt, 1], c=self.y_kmeans[trained_clt], s=30, cmap='viridis')
        plt.title(f"Epoch {epoch}: {len(np.unique(self.y_kmeans[trained_clt], axis=0))} clusters")
        plt.savefig(f"epoch_{epoch}.png")
        plt.show()

        if len(np.unique(self.y_kmeans[trained_clt], axis=0)) > 1:
            silhouette_avg = silhouette_score(X[trained_clt], self.y_kmeans[trained_clt])
            print("Silhouette score is ", silhouette_avg)
