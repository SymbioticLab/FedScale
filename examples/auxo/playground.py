import time
import numpy as np
import random
import sys
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from clustering import QTable

DEFAULT_SAMPLE_SIZE = 50
DEFAULT_TOTAL_EPOCH = 800
DEFAULT_TOTAL_SAMPLE = 1000
DEFAULT_KNOWN_CLT = 150
NUM_CENTERS = 4
random.seed(100)

def bipartition_cluster(total_epoch, total_sample, sample_size, known_clt):
    start_time = time.time()

    learning_rate = 0.1
    avg_train_times = 4 if len(sys.argv) <= 3 else total_epoch * sample_size / total_sample
    print(f"{sample_size}/{total_sample} of samples cluster for {total_epoch} epochs")

    X, y_true = make_blobs(n_samples=total_sample, centers=NUM_CENTERS, cluster_std=2, random_state=100)

    Q_table = QTable(
        total_sample,
        train_ratio=0.9,
        elbow_constant=0.8,
        merge=False,
        known_clt=known_clt,
        avg_train_times=avg_train_times,
        # split_round=[200]
    )

    for epoch in range(total_epoch):
        num_list = random.sample(range(total_sample), sample_size)

        for mid in range(Q_table.num_model):
            client_id = np.argwhere(Q_table.y_kmeans[num_list] == mid)
            global_index = [num_list[i[0]] for i in client_id]

            if len(client_id) < 5:
                continue

            X_sub = X[global_index]
            Q_table.knn_update_subR(mid, X_sub, global_index)
            split = Q_table.update_mainR(mid, X_sub, global_index, epoch <= total_epoch * 0.8)

            if split or epoch % 200 == 100 and mid == 0:
                if split:
                    print(f"SPLIT at round {epoch}")
                print(f'Epoch: {epoch}')
                Q_table.plot(X, epoch)

    print(f"Split to {Q_table.num_model} models")
    print(f"Time usage: {time.time() - start_time}")

if __name__ == "__main__":
    total_epoch = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_TOTAL_EPOCH
    total_sample = int(sys.argv[1]) if len(sys.argv) > 3 else DEFAULT_TOTAL_SAMPLE
    sample_size = int(sys.argv[2]) if len(sys.argv) > 3 else DEFAULT_SAMPLE_SIZE
    known_clt = DEFAULT_KNOWN_CLT if len(sys.argv) <= 3 else 1000

    bipartition_cluster(total_epoch, total_sample, sample_size, known_clt)
