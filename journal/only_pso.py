# ==============================================================================
# 연합 학습(Federated Learning)과 입자 군집 최적화(PSO) 결합 스크립트
# ==============================================================================
#
# ## 설명
# 이 스크립트는 연합 학습(FL) 과정에서 FedAvg와 같은 전통적인 가중치 평균 방식 대신,
# 입자 군집 최적화(PSO)를 사용하여 글로벌 모델을 통합하는 알고리즘을 구현합니다.
#
# ## 핵심 로직
# 1.  초기화: 하나의 글로벌 모델과 다수의 클라이언트(파티클)를 초기화합니다.
#     각 파티클은 모델 가중치의 한 버전을 소유합니다.
# 2.  로컬 학습: 매 라운드마다 일부 클라이언트를 무작위로 선택하여, 각자의
#     로컬 데이터로 모델 가중치를 학습 및 업데이트합니다.
# 3.  PSO 기반 통합:
#     - 로컬 학습이 끝나면, 각 클라이언트가 업데이트한 가중치들은 PSO 파티클의
#       '초기 위치'가 됩니다.
#     - PSO 알고리즘은 검증 데이터셋으로 각 파티클의 성능(손실 값)을 평가하며
#       최적의 가중치 집합(`gbest`)을 반복적으로 탐색합니다.
#     - 이 평가 과정이 전체 알고리즘에서 가장 연산 비용이 높은 부분입니다.
# 4.  글로벌 모델 업데이트: PSO 탐색을 통해 찾은 최적의 가중치(`gbest_weights`)를
#     다음 라운드의 새로운 글로벌 모델로 설정합니다.
#
# ## 핵심 특징: 단일 GPU 병렬 평가 최적화
# PSO의 평가 단계를 획기적으로 가속하기 위해, 이 스크립트는 단일 GPU 환경에서
# 다중 스레드를 활용한 병렬 평가 전략을 사용합니다.
# - 하나의 GPU에 여러 개의 모델 복사본(`worker_models`)을 생성합니다.
# - `ThreadPoolExecutor`를 사용하여 모든 파티클의 가중치 평가를 이 복사본들에
#   분산시켜 병렬로 처리합니다.
# - 이 접근법은 GPU 활용률을 극대화하여, 순차적으로 평가하는 방식에 비해
#   PSO 탐색 시간을 극적으로 단축시킵니다.
#
# ==============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import time

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common import *
from configure import *
from pso_utils import (
    Particle,
    build_validation_dataset,
    parallel_evaluate_weights,
    prepare_worker_models,
)

# --- 전역 랜덤 시드 고정 ---
# 실험 결과의 재현성을 보장합니다.
set_global_determinism(SEED)


# =================================================
# ============== 메인 실행 로직 ===================
# =================================================

def main():
    """연합 학습 및 PSO 실험을 실행하는 메인 함수입니다."""
    
    # --- GPU 설정 ---
    configure_gpu_memory_growth()

    # --- 1. 데이터 로딩 및 준비 ---
    print("Loading and preparing dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = load_dataset(name=DATASET_NAME)
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=SEED)
    
    val_ds = build_validation_dataset(x_val, y_val, batch_size=BATCH_SIZE)
    client_data = client_data_config(x_train, y_train, iid=IID, alpha=ALPHA)
    print("Dataset prepared successfully.")

    # --- 2. 모델 및 파티클 초기화 ---
    print("Initializing server model and client particles...")
    server_model = init_model()
    initial_weights = server_model.get_weights()
    
    particles = [
        Particle(
            particle_num=i,
            initial_model_weights=initial_weights,
            x_train=client_data[i][0],
            y_train=client_data[i][1]
        ) for i in range(NUMOFCLIENTS)
    ]
    global_weights = initial_weights
    server_evaluate_history = []
    print(f"{len(particles)} particles initialized.")

    # --- 3. 병렬 평가를 위한 워커 모델 준비 ---
    NUM_EVAL_WORKERS = 5  # 추천: 2-4
    worker_models = prepare_worker_models(server_model, NUM_EVAL_WORKERS)

    # --- 4. 메인 연합 학습 루프 ---
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        epoch_start_time = time.time()

        # a. 클라이언트 선택
        num_to_select = int(NUMOFCLIENTS * SELECT_CLIENTS)
        selected_particles = random.sample(particles, num_to_select)
        print(f"Selected {len(selected_particles)} clients for this round.")

        # b. 로컬 학습
        initial_particle_weights = []
        desc = " [1/3] Local Training  "
        for particle in tqdm(selected_particles, desc=desc):
            particle.current_weights = global_weights
            weights = particle.train_locally(server_model)
            initial_particle_weights.append(weights)
        
        # c. 초기 PSO 평가 (초기 gbest 탐색)
        desc = " [2/3] Finding gbest "
        scores = parallel_evaluate_weights(initial_particle_weights, worker_models, val_ds, tqdm_desc=desc)
        
        best_particle_idx = int(np.argmin(scores))
        gbest_score = float(scores[best_particle_idx])
        gbest_weights = initial_particle_weights[best_particle_idx]

        for i, particle in enumerate(selected_particles):
            particle.pbest_score = float(scores[i])
            particle.pbest_weights = initial_particle_weights[i]

        print(f"Initial gbest score (validation loss): {gbest_score:.4f}")
        
        # d. PSO 반복 탐색
        desc = " [3/3] PSO Iterations"
        pso_iterator = tqdm(range(PSO_ITERATIONS), desc=desc)
        for _ in pso_iterator:
            # 모든 파티클의 위치(가중치)를 먼저 업데이트
            for particle in selected_particles:
                particle.update_velocity_and_position(gbest_weights)
            
            # 모든 새로운 위치를 병렬로 평가
            current_weights_list = [p.current_weights for p in selected_particles]
            loss_list = parallel_evaluate_weights(current_weights_list, worker_models, val_ds)
            
            # pbest와 gbest 갱신
            for particle, loss in zip(selected_particles, loss_list):
                if loss < particle.pbest_score:
                    particle.pbest_score = loss
                    particle.pbest_weights = particle.current_weights
                if loss < gbest_score:
                    gbest_score = loss
                    gbest_weights = particle.current_weights
            
            pso_iterator.set_postfix(loss=f"{gbest_score:.4f}")

        # e. 글로벌 모델 업데이트 및 최종 평가
        server_model.set_weights(gbest_weights)
        global_weights = gbest_weights

        print(f"Epoch {epoch + 1} finished. Global model updated with gbest.")
        final_loss, final_acc = server_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)
        server_evaluate_history.append([final_loss, final_acc])
        
        epoch_end_time = time.time()
        print(f"Time taken: {epoch_end_time - epoch_start_time:.2f} seconds")

    # --- 5. 결과 저장 ---
    write_csv("only_pso", server_evaluate_history)
    print("\nAll training rounds are complete. Results saved to CSV.")


if __name__ == "__main__":
    main()
