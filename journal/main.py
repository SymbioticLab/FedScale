"""
지능형 연합 학습: 수동 특징 DQN 클라이언트 선택 + PSO 모델 통합
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import time

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common import *
from configure import *
# PCA를 사용하지 않는 수동 특징 기반 DQN 에이전트를 임포트합니다.
from smart_dqn_selector import Smart_DQN_Selector
from pso_utils import (
    Particle,
    build_validation_dataset,
    parallel_evaluate_weights,
    prepare_worker_models,
)

# --- 전역 랜덤 시드 고정 ---
set_global_determinism(SEED)

def main():
    """수동 특징 DQN과 PSO를 사용한 연합 학습 실험을 실행하는 메인 함수입니다."""
    # --- GPU 설정 ---
    configure_gpu_memory_growth()

    # --- 1. 데이터 로딩 및 준비 ---
    print("Loading and preparing dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = load_dataset(name=DATASET_NAME)
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=SEED)
    val_ds = build_validation_dataset(x_val, y_val, batch_size=BATCH_SIZE)
    client_data = client_data_config(x_train, y_train, iid=IID, alpha=ALPHA)
    client_diversities = calculate_diversities(client_data, NUMOFCLASSES)
    print("Dataset prepared successfully.")

    # --- 2. 모델, 파티클, DQN 에이전트 초기화 ---
    print("Initializing server model, particles, and Smart DQN agent...")
    server_model = init_model()
    initial_weights = server_model.get_weights()
    
    # 수동 특징 DQN 에이전트 설정
    num_select = int(NUMOFCLIENTS * SELECT_CLIENTS)
    NUM_MANUAL_FEATURES = 3 # 다양성, 손실, 기여도
    dqn_selector = Smart_DQN_Selector(
        num_clients=NUMOFCLIENTS,
        num_features=NUM_MANUAL_FEATURES
    )
    dqn_selector.set_diversities(client_diversities)

    # 파티클(클라이언트) 생성
    particles = [
        Particle(i, initial_weights, client_data[i][0], client_data[i][1])
        for i in range(NUMOFCLIENTS)
    ]
    global_weights = initial_weights
    previous_accuracy = 0.0
    server_evaluate_history = []
    print(f"{len(particles)} particles and Smart DQN agent initialized.")

    # --- 3. 병렬 평가를 위한 워커 모델 준비 ---
    NUM_EVAL_WORKERS = 5
    worker_models = prepare_worker_models(server_model, NUM_EVAL_WORKERS)
    
    # --- 4. 메인 연합 학습 루프 ---
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        epoch_start_time = time.time()

        # a. 클라이언트 프로파일링 (로컬 손실만 필요)
        #    참고: 기여도 업데이트를 위해 모든 클라이언트의 손실을 알아야 하므로,
        #    선택되지 않은 클라이언트도 프로파일링이 필요합니다.
        local_losses = []
        desc = " [1/5] Client Profiling "
        for p in tqdm(particles, desc=desc):
            p.current_weights = global_weights
            # train_locally는 가중치도 반환하지만, 여기서는 손실(loss)만 필요합니다.
            _, loss = p.train_locally(server_model, return_loss=True)
            local_losses.append(loss)
        
        # b. 클라이언트 선택 (DQN 에이전트)
        tqdm.write(f" [2/5] Selecting {num_select} clients with Smart DQN agent...")
        # 선택 전에 모든 클라이언트의 최신 손실 정보를 에이전트에 반영
        dqn_selector.last_local_losses = np.array(local_losses)
        selected_client_indices, current_state = dqn_selector.select_clients(num_select)
        
        selected_particles = [particles[i] for i in selected_client_indices]
        # 프로파일링 단계에서 이미 학습된 가중치를 가져옴
        initial_particle_weights = [p.current_weights for p in selected_particles]

        # c. PSO 기반 모델 통합
        desc = " [3/5] Finding gbest  "
        scores = parallel_evaluate_weights(initial_particle_weights, worker_models, val_ds, tqdm_desc=desc)

        best_particle_idx = int(np.argmin(scores))
        gbest_score = float(scores[best_particle_idx])
        gbest_weights = initial_particle_weights[best_particle_idx]

        for i, p in enumerate(selected_particles):
            p.pbest_score = scores[i]
            p.pbest_weights = initial_particle_weights[i]

        tqdm.write(f"Initial gbest score (validation loss): {gbest_score:.4f}")

        # d. PSO 반복 탐색
        desc = " [4/5] PSO Iterations "
        pso_iterator = tqdm(range(PSO_ITERATIONS), desc=desc)
        for _ in pso_iterator:
            for p in selected_particles:
                p.update_velocity_and_position(gbest_weights)
            
            current_weights_list = [p.current_weights for p in selected_particles]
            loss_list = parallel_evaluate_weights(current_weights_list, worker_models, val_ds)

            for p, loss in zip(selected_particles, loss_list):
                if loss < p.pbest_score:
                    p.pbest_score = loss
                    p.pbest_weights = p.current_weights
                if loss < gbest_score:
                    gbest_score = loss
                    gbest_weights = p.current_weights
            pso_iterator.set_postfix(loss=f"{gbest_score:.4f}")

        # e. 글로벌 모델 업데이트 및 평가
        server_model.set_weights(gbest_weights)
        global_weights = gbest_weights
        print(f"\n [5/5] Evaluating global model...")
        final_loss, final_acc = server_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)
        server_evaluate_history.append([final_loss, final_acc])

        # f. 보상 계산 및 에이전트 학습
        reward = (final_acc - previous_accuracy) * 100
        
        # 에이전트의 수동 지표(손실, 기여도) 업데이트
        selected_losses = [local_losses[i] for i in selected_client_indices]
        dqn_selector.update_metrics(selected_client_indices, selected_losses, reward)
        
        # 다음 상태는 다음 라운드 시작 시 내부적으로 다시 계산됨
        next_state = dqn_selector.get_state()
        
        dqn_selector.remember(current_state, selected_client_indices, reward, next_state)
        DQN_BATCH_SIZE = 64
        if len(dqn_selector.memory) >= 400:
            dqn_selector.replay(batch_size=DQN_BATCH_SIZE)

        previous_accuracy = final_acc
        print(f"Time taken: {time.time() - epoch_start_time:.2f} seconds")
    
    # --- 5. 결과 저장 ---
    write_csv("Ours", server_evaluate_history)
    print("\nAll training rounds are complete. Results saved to CSV.")

if __name__ == "__main__":
    main()
