# Journal FL Experiments

이 프로젝트는 여러 연합 학습 실험 스크립트를 포함합니다. 각 스크립트는
`configure.py`의 설정을 읽어 동일한 데이터셋/하이퍼파라미터로 실행됩니다.

주요 파일
- `configure.py`: 공통 실험 설정(데이터셋, 클라이언트 수, 에포크, 옵티마이저)
- `common.py`: 데이터 로딩, 모델 초기화, FedAvg, 재현성 유틸리티
- `pso_utils.py`: PSO 파티클과 병렬 평가 유틸리티
- `only_pso.py`: PSO 기반 연합 학습
- `only_dqn.py`: 수동 특징 DQN + FedAvg
- `main.py`: 수동 특징 DQN + PSO
- `oort.py`: Oort 기반 클라이언트 선택 + FedAvg
- `fed_avg.py`: 기본 FedAvg
- `fedprox.py`: FedProx
- `scaffold.py`: SCAFFOLD (클라이언트 드리프트 보정)
- `fednova.py`: FedNova (로컬 업데이트 정규화)

실행 예시
```bash
python fed_avg.py
python fedprox.py
python only_pso.py
python only_dqn.py
python main.py
python oort.py
python scaffold.py
python fednova.py
```

설정 변경
1. `configure.py`에서 `DATASET_NAME`을 선택합니다.
2. `NUMOFCLIENTS`, `EPOCHS`, `CLIENT_EPOCHS`, `SELECT_CLIENTS` 등을 조정합니다.
3. `IID`를 설정하면 결과가 `results/<DATASET_NAME>/(iid|non-iid)`로 저장됩니다.
