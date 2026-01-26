# configure.py

from keras.optimizers.legacy import SGD
import numpy as np

SEED = 20260101

# =================================================================
# === 1. 데이터셋 선택 (이 부분만 수정하여 실험) ===
# 사용 가능한 이름: "CIFAR10", "CIFAR100", "FEMNIST", "MNIST"
DATASET_NAME = "FEMNIST"
# =================================================================

# client config
NUMOFCLIENTS = 50   # 클라이언트(파티클) 수
EPOCHS = 50         # 전체 반복 횟수
CLIENT_EPOCHS = 5   # 각 클라이언트의 반복 횟수
BATCH_SIZE = 32     # 배치 크기

IID = False         # IID(True) / Non-IID(False)
ALPHA = 0.5
DROP_RATE = 0       # 0 ~ 1.0 float value
SELECT_CLIENTS = 0.5  # c

# pso setting
ACC = 0.5           # 가속도 계수
LOCAL_ACC = 0.7     # 로컬 가속도 계수
GLOBAL_ACC = 1.4    # 글로벌 가속도 계수

TOP_N = int(NUMOFCLIENTS * SELECT_CLIENTS)
PSO_ITERATIONS = 10

# =================================================================
# === 2. 데이터셋에 따른 자동 설정 (수정 불필요) ===
if DATASET_NAME == "CIFAR10":
    NUMOFCLASSES = 10
    INPUT_SHAPE = (32, 32, 3)
elif DATASET_NAME == "CIFAR100":
    NUMOFCLASSES = 100
    INPUT_SHAPE = (32, 32, 3)
elif DATASET_NAME == "FEMNIST":
    NUMOFCLASSES = 62 # FEMNIST는 숫자+영대소문자=62개 클래스
    INPUT_SHAPE = (28, 28, 1) # 흑백 이미지
elif DATASET_NAME == "MNIST":
    NUMOFCLASSES = 10
    INPUT_SHAPE = (28, 28, 1) # 흑백 이미지
else:
    raise ValueError("지원하지 않는 데이터셋 이름입니다. DATASET_NAME을 확인해주세요.")
# =================================================================

# model config
LOSS = "categorical_crossentropy"  # Loss function
lr = 0.001

def build_optimizer():
    """Return a fresh optimizer instance per model to avoid shared state."""
    return SGD(learning_rate=lr, momentum=0.9, nesterov=False)
