import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from common import init_model
from configure import ACC, LOCAL_ACC, GLOBAL_ACC, CLIENT_EPOCHS, BATCH_SIZE


class Particle:
    """PSO 기반 연합 학습에 사용하는 클라이언트 파티클."""
    def __init__(self, particle_num, initial_model_weights, x_train, y_train):
        self.particle_id = particle_num
        self.current_weights = initial_model_weights
        self.pbest_weights = initial_model_weights
        self.pbest_score = float('inf')
        self.x_train = x_train
        self.y_train = y_train
        self.velocities = [np.random.rand(*layer.shape) / 5 - 0.10 for layer in initial_model_weights]
        self.pso_params = {'inertia': ACC, 'local_acc': LOCAL_ACC, 'global_acc': GLOBAL_ACC}

    def train_locally(self, shared_model, return_loss=False):
        """로컬 데이터를 사용하여 모델을 학습하고 가중치를 반환합니다."""
        shared_model.set_weights(self.current_weights)
        history = shared_model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=CLIENT_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            verbose=0,
        )
        self.current_weights = shared_model.get_weights()
        if return_loss:
            last_val_loss = history.history.get('val_loss', [1.0])[-1]
            return self.current_weights, last_val_loss
        return self.current_weights

    def update_velocity_and_position(self, gbest_weights):
        """PSO 동역학에 따라 파티클의 속도와 위치(가중치)를 업데이트합니다."""
        new_weights = [None] * len(self.current_weights)
        local_rand, global_rand = random.random(), random.random()

        for i, layer_weights in enumerate(self.current_weights):
            new_velocity = (
                self.pso_params['inertia'] * self.velocities[i] +
                self.pso_params['local_acc'] * local_rand * (self.pbest_weights[i] - layer_weights) +
                self.pso_params['global_acc'] * global_rand * (gbest_weights[i] - layer_weights)
            )
            self.velocities[i] = new_velocity
            new_weights[i] = layer_weights + new_velocity
        self.current_weights = new_weights


def build_validation_dataset(x_val, y_val, batch_size):
    """빠른 평가를 위해 최적화된 tf.data.Dataset을 생성합니다."""
    return tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
        .batch(batch_size) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE)


def prepare_worker_models(server_model, num_workers):
    """병렬 실행을 위해 단일 GPU에 모델 복사본들을 생성합니다."""
    base_weights = server_model.get_weights()
    logical_gpus = tf.config.list_logical_devices('GPU')
    worker_models = []

    if logical_gpus:
        print(f"Creating {num_workers} model replicas on /GPU:0...")
        with tf.device('/GPU:0'):
            for _ in range(num_workers):
                model_replica = init_model()
                model_replica.set_weights(base_weights)
                worker_models.append(model_replica)
    else:
        print("No GPU found. Creating a single model replica on CPU.")
        model_replica = init_model()
        model_replica.set_weights(base_weights)
        worker_models.append(model_replica)
    return worker_models


def _evaluate_weights_chunk(model, weights_chunk, val_ds, progress_bar=None):
    """단일 모델 복사본에서 가중치 목록을 순차적으로 평가합니다."""
    losses = []
    for weights in weights_chunk:
        model.set_weights(weights)
        result = model.evaluate(val_ds, verbose=0)
        loss = result[0] if isinstance(result, (list, tuple)) else float(result)
        losses.append(loss)
        if progress_bar:
            progress_bar.update(1)
    return losses


def parallel_evaluate_weights(weights_list, worker_models, val_ds, tqdm_desc=None):
    """단일 GPU의 여러 모델 복사본을 사용하여 가중치 목록을 병렬로 평가합니다."""
    num_workers = len(worker_models)
    if num_workers <= 1:
        progress_bar = tqdm(total=len(weights_list), desc=tqdm_desc, leave=False) if tqdm_desc else None
        return _evaluate_weights_chunk(worker_models[0], weights_list, val_ds, progress_bar)

    chunks = [[] for _ in range(num_workers)]
    index_map = [[] for _ in range(num_workers)]
    for i, w in enumerate(weights_list):
        worker_index = i % num_workers
        chunks[worker_index].append(w)
        index_map[worker_index].append(i)

    progress_bar = tqdm(total=len(weights_list), desc=tqdm_desc, leave=False) if tqdm_desc else None

    final_losses = [None] * len(weights_list)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            if chunks[i]:
                future = executor.submit(_evaluate_weights_chunk, worker_models[i], chunks[i], val_ds, progress_bar)
                futures.append((future, i))

        for future, worker_idx in futures:
            worker_losses = future.result()
            original_indices = index_map[worker_idx]
            for i, loss in enumerate(worker_losses):
                final_losses[original_indices[i]] = loss

    if progress_bar:
        progress_bar.close()

    return final_losses
