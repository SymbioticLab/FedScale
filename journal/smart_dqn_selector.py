# ==============================================================================
# 수동 특징 기반 스마트 DQN 클라이언트 선택 에이전트
# ==============================================================================
#
# ## 설명
# 이 스크립트는 PCA 기반의 특징 대신, 사람이 직접 정의한 해석 가능한 특징들을
# 상태(State)로 사용하여 클라이언트를 선택하는 DQN 에이전트를 정의합니다.
# FEMNIST와 같이 클라이언트 간 공통 패턴을 찾기 어려운 데이터셋에서 더 강건하게
# 작동할 수 있도록 설계되었습니다.
#
# ## 핵심 로직
# 1.  **상태(State) 정의**: 각 클라이언트의 다음 세 가지 수동 특징을 결합하여
#     하나의 긴 벡터를 상태로 정의합니다.
#     -   **데이터 다양성 (고정값)**: 각 클라이언트가 가진 데이터의 분포(엔트로피).
#     -   **최근 로컬 손실 (변동값)**: 직전 라운드의 로컬 학습에서 기록한 검증 손실.
#     -   **누적 기여도 점수 (변동값)**: 해당 클라이언트가 선택되었을 때, 글로벌
#         모델 성능 향상에 얼마나 기여했는지를 나타내는 이동 평균 점수.
# 2.  **클라이언트 선택**: 현재 상태를 신경망에 입력하여 각 클라이언트를 선택했을 때의
#     미래 보상(Q-값)을 예측하고, Epsilon-Greedy 정책에 따라 클라이언트를 선택합니다.
# 3.  **학습 (Replay)**: 경험 리플레이 메모리에서 미니배치를 샘플링하여,
#     (상태, 행동, 보상) 관계를 학습하고 Q-값 예측의 정확도를 높입니다.
#     이 버전에서는 학습 효율을 위해 Replay 메서드를 벡터화했습니다.
#
# ==============================================================================

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class Smart_DQN_Selector:
    """수동 특징(다양성, 손실, 기여도)을 기반으로 클라이언트를 선택하는 DQN 에이전트입니다."""
    def __init__(self, num_clients, num_features, learning_rate=0.001, gamma=0.95,
                 initial_epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
        self.num_clients = num_clients
        self.num_features = num_features  # 클라이언트당 특징 수 (예: 3개)
        self.state_size = num_clients * num_features
        self.memory = deque(maxlen=2000)

        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate

        self.model = self._build_model()
        
        # --- 클라이언트별 지표(수동 특징) 관리 ---
        self.client_diversities = np.zeros(num_clients)
        self.last_local_losses = np.ones(num_clients) # 0으로 나누는 것을 방지하기 위해 1로 초기화
        self.contribution_scores = np.zeros(num_clients)
        self.selection_counts = np.zeros(num_clients)

    def _build_model(self):
        """DQN을 위한 신경망 모델을 구성합니다."""
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.num_clients, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
        
    def set_diversities(self, diversities):
        """초기 데이터 다양성 점수를 설정합니다."""
        self.client_diversities = np.array(diversities)

    def update_metrics(self, indices, local_losses, reward):
        """선택된 클라이언트들의 손실 및 기여도 지표를 업데이트합니다."""
        for i, client_idx in enumerate(indices):
            self.last_local_losses[client_idx] = local_losses[i]
            self.selection_counts[client_idx] += 1
            
            # 이동 평균 기여도 업데이트
            old_score = self.contribution_scores[client_idx]
            count = self.selection_counts[client_idx]
            new_score = old_score + (1 / count) * (reward - old_score)
            self.contribution_scores[client_idx] = new_score

    def get_state(self):
        """현재 클라이언트 지표를 바탕으로 상태 벡터를 생성하고 반환합니다."""
        # 지표들을 0과 1 사이로 정규화
        # 분모가 0이 되는 경우를 방지하기 위해 작은 값(epsilon)을 더함
        epsilon = 1e-9
        norm_diversities = self.client_diversities / (np.max(self.client_diversities) + epsilon)
        # 손실은 값이 클수록 나쁜 지표이므로, (1 - 정규화된 손실)을 사용해 값이 클수록 좋게 변환
        # norm_losses = 1.0 - (self.last_local_losses / (np.max(self.last_local_losses) + epsilon))
        # 손실은 값이 클수록 "학습이 필요한/정보가 많은" 클라이언트라고 가정 -> 값이 클수록 좋게 사용
        norm_losses = self.last_local_losses / (np.max(self.last_local_losses) + epsilon)
        norm_contributions = self.contribution_scores / (np.max(np.abs(self.contribution_scores)) + epsilon)

        # 상태 벡터 구성: 모든 클라이언트의 [다양성, 손실, 기여도]를 1차원으로 펼침
        state = np.array([norm_diversities, norm_losses, norm_contributions]).T.flatten().reshape(1, self.state_size)
        return state

    def remember(self, state, action_indices, reward, next_state):
        """경험을 메모리에 저장합니다. (행동은 인덱스 리스트)"""
        self.memory.append((state, action_indices, reward, next_state))

    def select_clients(self, num_to_select):
        """현재 상태를 바탕으로 Epsilon-Greedy 정책에 따라 클라이언트를 선택합니다."""
        state = self.get_state()
        if np.random.rand() <= self.epsilon:
            # 탐험: 무작위 선택
            selected_indices = np.random.choice(self.num_clients, num_to_select, replace=False)
        else:
            # 활용: Q-값이 가장 높은 클라이언트 선택
            q_values = self.model.predict(state, verbose=0)[0]
            # argsort는 오름차순이므로, 뒤에서부터 num_to_select개를 가져옴
            selected_indices = np.argsort(q_values)[-num_to_select:]
        
        return selected_indices.tolist(), state

    def replay(self, batch_size):
        """메모리에서 미니배치를 샘플링하여 모델을 학습합니다. (벡터화된 버전)"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        # 배치 데이터를 NumPy 배열로 변환
        states = np.array([transition[0] for transition in minibatch]).reshape(batch_size, -1)
        next_states = np.array([transition[3] for transition in minibatch]).reshape(batch_size, -1)

        # Q-값을 한 번에 예측
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        # 각 샘플에 대한 타겟 Q-값 계산
        for i, (state, action_indices, reward, next_state) in enumerate(minibatch):
            # 표준 DQN 타겟: 보상 + 할인된 미래 최대 Q-값
            target = reward + self.gamma * np.amax(next_q_values[i])
            
            # 선택되었던 클라이언트들의 Q-값을 새로운 타겟 값으로 업데이트
            for action_index in action_indices:
                current_q_values[i][action_index] = target

        # 업데이트된 Q-값(target)을 사용하여 모델을 한 번에 학습
        self.model.fit(states, current_q_values, epochs=1, verbose=0)

        # Epsilon 값을 감소시켜 점차 탐험을 줄임
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
