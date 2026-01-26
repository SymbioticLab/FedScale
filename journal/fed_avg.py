import random
import time # tqdm 사용 시 시간 측정을 위해 추가하면 좋습니다.

# tqdm 임포트
from tqdm import tqdm

from common import *
from configure import *

# 모든 랜덤 시드 고정
set_global_determinism(SEED)

def client_update(client_idx, client_model, global_weights, epoch_num, client_data):
    """단일 클라이언트의 로컬 업데이트를 수행합니다."""
    # 현재 글로벌 모델 가중치로 클라이언트 모델 설정 (첫 에포크 제외)
    if epoch_num != 0:
        client_model.set_weights(global_weights)

    # 로컬 학습 수행
    client_model.fit(client_data[client_idx][0], client_data[client_idx][1],
                     epochs=CLIENT_EPOCHS,
                     batch_size=BATCH_SIZE,
                     verbose=0,
                     validation_split=0.2,
                    )
    return client_model

if __name__ == "__main__":
    # --- GPU 설정 ---
    configure_gpu_memory_growth()

    # --- 1. 데이터 로딩 및 준비 ---
    print("Loading and preparing dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = load_dataset(name=DATASET_NAME)
    # FedAvg는 검증셋을 직접 사용하지 않으므로, 여기서는 분리하지 않아도 됩니다.
    # x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=SEED)
    client_data = client_data_config(x_train_full, y_train_full, iid=IID, alpha=ALPHA) # 전체 학습 데이터 사용
    print("Dataset prepared successfully.")

    # --- 2. 모델 초기화 ---
    print("Initializing server and client models...")
    server_model = init_model()
    # 각 클라이언트가 독립적인 모델 인스턴스를 갖도록 수정
    client_models = [init_model() for _ in range(NUMOFCLIENTS)]
    
    # 초기 글로벌 가중치 설정 (첫 에포크 client_update에서 사용)
    # FedAvg는 첫 에포크에선 avg_weight를 사용하지 않으므로 초기화 방식 변경
    global_weights = server_model.get_weights()
    server_evaluate_history = []
    print(f"{NUMOFCLIENTS} client models initialized.")

    # --- 3. 메인 연합 학습 루프 ---
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        epoch_start_time = time.time()
        
        server_weight_updates = [] # 이번 라운드에 집계할 가중치들을 저장할 리스트
        
        # a. 클라이언트 랜덤 선택
        num_to_select = int(max(NUMOFCLIENTS * SELECT_CLIENTS, 1))
        selected_client_indices = random.sample(range(NUMOFCLIENTS), num_to_select)
        print(f"Selected {len(selected_client_indices)} clients for this round.")

        # b. 로컬 학습 (tqdm 적용)
        # tqdm으로 클라이언트 학습 루프를 감쌉니다.
        desc = " Client Local Training "
        for client_idx in tqdm(selected_client_indices, desc=desc):
            client_model = client_models[client_idx] # 해당 인덱스의 클라이언트 모델 가져오기
            
            # 클라이언트 업데이트 함수 호출 (global_weights 전달)
            updated_client_model = client_update(client_idx, client_model, global_weights, epoch, client_data)
            
            # 통신 드랍 시뮬레이션 (선택 사항)
            rand_comm = random.randint(0, 99)
            if rand_comm >= DROP_RATE: # DROP_RATE 확률로 통신 실패
                # 통신 성공 시 가중치 추가 (copy.deepcopy는 메모리 효율상 제거 가능)
                server_weight_updates.append(updated_client_model.get_weights())
            # else:
            #     print(f"Communication dropped for client {client_idx + 1}")

        # c. 글로벌 모델 통합 (FedAvg)
        if server_weight_updates: # 통신 성공한 클라이언트가 있을 경우에만 집계
            print(f"Aggregating models from {len(server_weight_updates)} clients using FedAvg...")
            global_weights = fedAVG(server_weight_updates)
            server_model.set_weights(global_weights)
        else:
            print("[Warning] No client updates received in this round. Skipping FedAvg.")
        
        # d. 서버 모델 평가
        print(f"\nEpoch {epoch + 1} finished. Evaluating global model...")
        eval_result = server_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)
        server_evaluate_history.append(eval_result) # evaluate 결과 전체(loss, acc) 저장
        
        epoch_end_time = time.time()
        print(f"Time taken: {epoch_end_time - epoch_start_time:.2f} seconds")

    # --- 4. 결과 저장 ---
    # CSV 저장 형식을 [loss, accuracy]로 명확히 함
    write_csv("FedAvg", [[loss, acc] for loss, acc in server_evaluate_history])
    print("\nAll training rounds are complete. Results saved to CSV.")
