# common.py

import os
import cv2 # OpenCV 라이브러리 추가
from keras.datasets import cifar10, cifar100, mnist
from keras.utils import to_categorical
from collections import Counter, defaultdict
import numpy as np
import random
import csv

from build_model import Model
from configure import * # configure 파일의 모든 설정 가져오기

def write_csv(method_name, rows):
    # 파일명에 데이터셋 이름이 포함되도록 수정
    file_name = '[{dset}]_{name}_LR_{lr}_CLI_{cli}_{cli_epoch}_EPOCHS_{epochs}_SELECT_{select}.csv'
    file_name = file_name.format(
        dset=DATASET_NAME, name=method_name, lr=lr, cli=NUMOFCLIENTS, 
        cli_epoch=CLIENT_EPOCHS, epochs=EPOCHS, select=SELECT_CLIENTS
    )
    split_folder = "iid" if IID else "non-iid"
    output_dir = os.path.join("results", DATASET_NAME, split_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        for row in rows:
            wr.writerow(row)

def configure_gpu_memory_growth():
    """Enable GPU memory growth when GPUs are available."""
    try:
        import tensorflow as tf
    except ImportError:
        print("[Warning] TensorFlow is not available; skipping GPU configuration.")
        return

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"[Warning] Could not set memory growth: {e}")

def set_global_determinism(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(seed)

# --- FEMNIST 로드를 위한 헬퍼 함수 ---
def load_client_images(path):
    """지정된 경로에서 클래스별 이미지를 로드하는 함수"""
    images = []
    labels = []
    class_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for class_label in class_folders:
        class_path = os.path.join(path, class_label)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            # 이미지를 흑백(Grayscale)으로 읽고 28x28로 리사이즈
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (28, 28))
                images.append(img.reshape(28, 28, 1)) # 채널 차원 추가
                labels.append(int(class_label))
    return images, labels

# --- 데이터셋별 로딩 함수들 ---
def load_femnist_dataset(data_path="/home/jsy/backup/jsy/journal/data/femnist/3500_client_img"):
    """FEMNIST 데이터셋 로드 (사용자 제공 코드 기반)"""
    print("Loading FEMNIST dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"FEMNIST 데이터 경로를 찾을 수 없습니다: {data_path}")
    
    client_folders = sorted([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])
    selected_clients = client_folders[:NUMOFCLIENTS]
    print(f"Selected {len(selected_clients)} clients from {len(client_folders)} available clients")
    
    all_train_x, all_train_y = [], []
    all_test_x, all_test_y = [], []
    
    for client_id in selected_clients:
        client_path = os.path.join(data_path, client_id)
        train_path = os.path.join(client_path, "train")
        if os.path.exists(train_path):
            train_x, train_y = load_client_images(train_path)
            all_train_x.extend(train_x)
            all_train_y.extend(train_y)
        
        test_path = os.path.join(client_path, "test")
        if os.path.exists(test_path):
            test_x, test_y = load_client_images(test_path)
            all_test_x.extend(test_x)
            all_test_y.extend(test_y)
    
    X_train = np.array(all_train_x, dtype=np.float32) / 255.0
    Y_train = to_categorical(np.array(all_train_y), num_classes=NUMOFCLASSES)
    X_test = np.array(all_test_x, dtype=np.float32) / 255.0
    Y_test = to_categorical(np.array(all_test_y), num_classes=NUMOFCLASSES)
    
    return (X_train, Y_train), (X_test, Y_test)

def load_standard_dataset(dataset_loader):
    """CIFAR10, CIFAR100, MNIST 등 Keras 기본 데이터셋 로드"""
    (X_train, Y_train), (X_test, Y_test) = dataset_loader()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # MNIST, FEMNIST 같이 채널이 없는 데이터에 채널 차원 추가
    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    Y_train = to_categorical(Y_train, num_classes=NUMOFCLASSES)
    Y_test = to_categorical(Y_test, num_classes=NUMOFCLASSES)
    return (X_train, Y_train), (X_test, Y_test)

# --- 메인 데이터 로딩 함수 (수정됨) ---
def load_dataset(name):
    print(f"===== Loading Dataset: {name} =====")
    if name == "CIFAR10":
        return load_standard_dataset(cifar10.load_data)
    elif name == "CIFAR100":
        return load_standard_dataset(cifar100.load_data)
    elif name == "FEMNIST":
        return load_femnist_dataset()
    elif name == "MNIST":
        return load_standard_dataset(mnist.load_data)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

def init_model():
    # INPUT_SHAPE을 configure에서 직접 가져와 사용
    model = Model(loss=LOSS, optimizer=build_optimizer(), classes=NUMOFCLASSES)
    fl_model = model.fl_paper_model(train_shape=INPUT_SHAPE)
    return fl_model

def fedAVG(server_weight):
    """입력된 가중치 리스트의 평균을 계산합니다."""
    if not server_weight:
        print("[Warning] No client weights received for FedAvg. Returning None.")
        return None
    avg_weight = [np.zeros_like(w) for w in server_weight[0]]
    for client_weights in server_weight:
        for i, w in enumerate(client_weights):
            avg_weight[i] += w
    for i in range(len(avg_weight)):
        avg_weight[i] /= len(server_weight)
    return avg_weight

def calculate_diversities(client_data, num_classes):
    """모든 클라이언트의 데이터 다양성(엔트로피)을 계산합니다."""
    from scipy.stats import entropy

    diversities = []
    for _, y_data in client_data:
        labels = np.argmax(y_data, axis=1)
        if len(labels) == 0:
            diversities.append(0)
            continue
        counts = np.bincount(labels, minlength=num_classes)
        probs = counts / len(labels)
        diversities.append(entropy(probs, base=2))
    return diversities

def client_data_config(x_train, y_train, iid=False, alpha=0.5):
    client_data = [() for _ in range(NUMOFCLIENTS)]

    if iid:
        # IID: 무작위 균등 분할 (효율적으로 수정)
        num_of_each_dataset = x_train.shape[0] // NUMOFCLIENTS
        all_indices = np.arange(x_train.shape[0])
        np.random.shuffle(all_indices)
        
        for i in range(NUMOFCLIENTS):
            start_index = i * num_of_each_dataset
            end_index = (i + 1) * num_of_each_dataset
            split_data_index = all_indices[start_index:end_index]
            
            new_x_train = x_train[split_data_index]
            new_y_train = y_train[split_data_index]
            client_data[i] = (new_x_train, new_y_train)

            # 클래스 분포 출력
            label_distribution = np.argmax(new_y_train, axis=1)
            class_counts = Counter(label_distribution)
            print(f"Client {i} label distribution: {dict(sorted(class_counts.items()))}")
    
    else:
        print("Non-IID")
        # Non-IID: 클래스별로 클라이언트에 편향되게 분배
        labels = np.argmax(y_train, axis=1)
        num_classes = np.max(labels) + 1
        client_data = [([], []) for _ in range(NUMOFCLIENTS)]

        # 클래스별 인덱스 모음
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)

        # Dirichlet 분포 기반 분할
        for cls in range(num_classes):
            cls_idx = class_indices[cls]
            np.random.shuffle(cls_idx)

            proportions = np.random.dirichlet(alpha=[alpha] * NUMOFCLIENTS)
            proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            split_indices = np.split(cls_idx, proportions)

            for client_id, client_idx in enumerate(split_indices):
                client_data[client_id][0].extend(x_train[client_idx])
                client_data[client_id][1].extend(y_train[client_idx])

        # 배열로 변환 + 분포 출력
        for i in range(NUMOFCLIENTS):
            x_np = np.asarray(client_data[i][0])
            y_np = np.asarray(client_data[i][1])
            client_data[i] = (x_np, y_np)

            label_distribution = np.argmax(y_np, axis=1)
            class_counts = Counter(label_distribution)
            # print(f"Non-IID (Dirichlet α={alpha}) - Client {i} label distribution: {dict(sorted(class_counts.items()))}")

    return client_data
