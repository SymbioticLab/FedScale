# ====================================================================================
# 지능형 연합 학습: 수동 특징 DQN 클라이언트 선택 + FedAvg 모델 통합
# ====================================================================================
#
# ## 설명
# 이 스크립트는 FEMNIST와 같이 PCA가 잘 작동하지 않는 데이터셋을 위해 설계되었습니다.
# `Smart_DQN_Selector`를 사용하여 (다양성, 손실, 기여도) 같은 수동 특징을 기반으로
# 클라이언트를 선택하고, FedAvg 방식으로 모델을 통합합니다.
#
# ## 핵심 로직
# 1.  **초기 설정**: 모든 클라이언트의 데이터 다양성을 미리 계산하여 에이전트에 설정합니다.
# 2.  **클라이언트 프로파일링**: 매 라운드, 모든 클라이언트의 '로컬 손실'을 수집합니다.
# 3.  **상태 생성 및 클라이언트 선택**: DQN 에이전트가 내부적으로 수동 특징들을
#     결합하여 상태를 만들고, 이를 기반으로 최적의 클라이언트를 선택합니다.
# 4.  **FedAvg 기반 모델 통합**: 선택된 클라이언트들의 가중치를 FedAvg로 통합합니다.
# 5.  **글로벌 모델 업데이트 및 에이전트 학습**:
#      - 글로벌 모델을 업데이트하고 보상을 계산합니다.
#      - 보상과 로컬 손실을 에이전트에 전달하여 내부 지표(기여도)를 업데이트합니다.
#      - 경험 리플레이를 통해 에이전트를 학습시킵니다.
#
# ====================================================================================

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

# --- 전역 랜덤 시드 고정 ---
set_global_determinism(SEED)


class Particle:
    """각 클라이언트의 데이터와 모델 가중치를 관리하는 클래스입니다."""
    def __init__(self, particle_num, initial_model_weights, x_train, y_train):
        self.particle_id = particle_num
        self.current_weights = initial_model_weights
        self.x_train = x_train
        self.y_train = y_train

    def train_locally(self, shared_model):
        """로컬 데이터를 사용하여 모델을 학습하고, 가중치와 마지막 검증 손실을 반환합니다."""
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
        last_val_loss = history.history.get('val_loss', [1.0])[-1]
        return self.current_weights, last_val_loss

def main():
    """수동 특징 DQN과 FedAvg를 사용한 연합 학습 실험을 실행하는 메인 함수입니다."""
    # --- GPU 설정 ---
    configure_gpu_memory_growth()

    # --- 1. 데이터 로딩 및 준비 ---
    print("Loading and preparing dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = load_dataset(name=DATASET_NAME)
    x_train, _, y_train, _ = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=SEED)
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
    
    # --- 3. 메인 연합 학습 루프 ---
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        epoch_start_time = time.time()

        # a. 클라이언트 프로파일링 (로컬 손실 수집)
        local_losses = []
        desc = " [1/4] Client Profiling "
        for p in tqdm(particles, desc=desc):
            p.current_weights = global_weights
            # 이 단계에서 업데이트된 가중치는 FedAvg에 사용됨
            _, loss = p.train_locally(server_model)
            local_losses.append(loss)
        
        # b. 클라이언트 선택 (DQN 에이전트)
        tqdm.write(f" [2/4] Selecting {num_select} clients with Smart DQN agent...")
        # 에이전트에 최신 손실 정보를 반영
        dqn_selector.last_local_losses = np.array(local_losses)
        selected_client_indices, current_state = dqn_selector.select_clients(num_select)
        
        # c. 글로벌 모델 통합 (FedAvg 사용)
        tqdm.write(" [3/4] Aggregating models using FedAvg...")
        # 프로파일링 단계에서 이미 학습된 가중치를 가져옴
        weights_to_aggregate = [particles[i].current_weights for i in selected_client_indices]
        global_weights = fedAVG(weights_to_aggregate)
        
        if global_weights is not None:
            server_model.set_weights(global_weights)
        
        # d. 글로벌 모델 평가
        print("\n [4/4] Evaluating global model...")
        final_loss, final_acc = server_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)
        server_evaluate_history.append([final_loss, final_acc])

        # e. 보상 계산 및 에이전트 학습
        reward = (final_acc - previous_accuracy) * 100
        
        # 에이전트의 수동 지표(기여도) 업데이트
        selected_losses = [local_losses[i] for i in selected_client_indices]
        dqn_selector.update_metrics(selected_client_indices, selected_losses, reward)
        
        # 다음 상태는 에이전트가 내부적으로 다시 계산함
        next_state = dqn_selector.get_state()
        
        dqn_selector.remember(current_state, selected_client_indices, reward, next_state)
        DQN_BATCH_SIZE = 32
        if len(dqn_selector.memory) >= MIN_REPLAY_SIZE:
            dqn_selector.replay(batch_size=DQN_BATCH_SIZE)

        previous_accuracy = final_acc
        print(f"Time taken: {time.time() - epoch_start_time:.2f} seconds")
    
    # --- 4. 결과 저장 ---
    write_csv("only_dqn", server_evaluate_history)
    print("\nAll training rounds are complete. Results saved to CSV.")

if __name__ == "__main__":
    main()
