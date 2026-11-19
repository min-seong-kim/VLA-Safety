# π0.5 v5 실험 환경 재현

2026-11-12에 실제 사용 중인 환경을 조사해 작성했다. 대상은 확장 v5 데이터셋의
AE/VLM/Both LoRA 학습, 모델 추론, 새 환경 676개 및 기존 LIBERO 80개 평가다.
Linux x86_64와 NVIDIA GPU를 기준으로 한다.

최근 CPU probe용 환경과 현재 서버에서 재검증한 범위는 [실험 재현 안내](../doc/2026-11-18_Experiment_Reproducibility.md)에 있다.

## 학습과 렌더 환경을 나눈 이유

| 용도 | Conda 파일 / 이름 | 주요 버전 |
| --- | --- | --- |
| LoRA 학습·정책 추론 | `environment.yml` / `vla-safety-train` | Python 3.11.14, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Transformers 4.53.2, NumPy 1.26.4 |
| 시뮬레이션·시연 수집·평가 | `environment-render.yml` / `vla-safety-render` | Python 3.12.3, MuJoCo 3.2.2, NumPy 1.26.4, numba 0.63.1, PyOpenGL 3.1.10 |

기존 실험도 `openpi/.venv`와 `envs/libero-render`를 분리해 사용했다.
학습 환경의 OpenPI 의존성에는 MuJoCo 2.3.7이 포함되어 있으므로 두 YAML을 한 환경에
합치지 않는다. 평가는 렌더 환경에서 물리를 실행하고, 로컬 Unix socket으로 학습
환경의 정책 서버에 추론을 요청한다. 렌더 환경에도 NVIDIA EGL 드라이버가 필요하다.

학습 환경은 설치된 Python 패키지 버전을 고정했다. 렌더 환경은 시스템 패키지를
상속하던 기존 venv에서 평가에 필요한 패키지와 그 의존성을 추출했다. 기존 렌더 환경의
PyTorch는 일반 배포처에 없는 NVIDIA 빌드 `2.10.0a0+b4e4ee81d3.nv25.12`여서,
공개 배포되는 `2.7.1+cpu`로 대체했다. 여기서는 LIBERO 초기 상태를 읽는 데 사용하며,
GPU 정책 추론에 사용하는 PyTorch는 기존 버전을 유지한다.
학습 YAML의 C compiler와 make는 `evdev` 같은 소스 배포 패키지를 새 서버에서 빌드하기 위한 도구다.
공개 wheel 조합은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/previous-versions/)를 따른다.

## 새 서버에서 설치

Conda 설치 후 **저장소 루트에서** 실행한다. YAML 안의 editable 설치 경로는 상대 경로다.

```bash
cd /path/to/VLA-Safety
conda env create -f environment.yml
conda env create -f environment-render.yml
conda run -n vla-safety-train python scripts/setup_conda_environment.py --role train
conda run -n vla-safety-render python scripts/setup_conda_environment.py --role render
conda run -n vla-safety-train python -m pip check
conda run -n vla-safety-render python -m pip check
```

학습 설정 스크립트는 OpenPI가 요구하는 Transformers 소스 패치 5개를 적용한다.
**YAML 설치만으로는 이 패치가 적용되지 않는다.** 렌더 설정은 보존한 LIBERO와
`robosuite_vanilla` 소스를 새 환경에서 불러오도록 `.pth`를 작성한다.
기존 `scripts/setup_env.sh`는 예전 서버 경로를 사용하므로 이 설치 절차에는 사용하지 않는다.
설정 스크립트는 기존 venv를 덮어쓰지 않는다.

YAML의 환경변수는 Conda 활성화 시 적용된다. 직접 실행 파일을 호출할 때도
`JAX_PLATFORMS=cpu`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` 등 같은 설정을 유지한다.

## 함께 보존할 소스·데이터·가중치

YAML은 코드·데이터·가중치를 다운로드하거나 복사하지 않는다. 다음을 저장소와 함께 옮긴다.

- `openpi/`: commit `215abfb217dbac7d5f1273282331b9b1866c0479`.
- `third_party/libero/`: commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, assets·BDDL·init files 포함.
- `third_party/robosuite_vanilla/`: 현재 디렉터리를 그대로 보존한다. 별도 Git 메타데이터가 없어 `source-state.json`에 Python 소스 해시를 기록했다. 다른 safety fork나 최신 PyPI 패키지로 교체하지 않는다.
- `scripts/semantic_safety/`, `data/semantic_expansion_v5_lora_v1/` 및 그 manifest가 참조하는 원본 시연·캐시·물체 assets.
- 시연을 다시 생성하려면 `data/semantic_safety_observations_v5/`도 보존한다. 기존 LIBERO 시연 원본은 `runs/semantic_expansion_v5_lora_v1/native_rehearsal_collection/`에 있으며, 학습용 캐시는 위 데이터 디렉터리에 있다.
- `checkpoints/pi05_libero_pytorch/`: 기반 가중치와 정규화 통계.
- `checkpoints/pi05_libero_expansion_v5_lora_v1/`: 최종 `ae`, `vlm`, `all` adapter. adapter만으로는 추론할 수 없다.
- `.cache/openpi/big_vision/paligemma_tokenizer.model`: 실제 사용한 tokenizer. 없으면 첫 실행에서 다운로드하므로 오프라인 서버에는 이 파일도 복사한다.
- 중단 학습을 재개하려면 `runs/semantic_expansion_v5_lora_v1/`의 optimizer·adapter·step 상태와 학습 기록도 필요하다.

LeRobot은 YAML에서 Git commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`로 고정했다.
`observed_train.json`, `observed_render.json`은 원래 환경의 설치 기록이다.
`train-pip.txt`, `render-pip.txt`는 YAML의 pip 목록과 같다.

**기존 XML·manifest·체크포인트 메타데이터 일부에는 절대 경로가 남아 있다.** 기존 데이터와
중단 상태를 그대로 재사용하려면 원래 루트
`/NHNHOME/26msit001_B/BASE/edge_ai_lab/minseong/VLA-Safety`를 유지하거나 같은 경로로 마운트한다.
다른 경로로 이전할 때는 참조 경로와 무결성 기록을 별도로 점검해야 한다.
이 환경 설정 스크립트는 실험 데이터나 체크포인트를 자동으로 수정하지 않는다.

## 설치 검증

```bash
conda run -n vla-safety-train python scripts/setup_conda_environment.py --role train --check-only
conda run -n vla-safety-render python scripts/setup_conda_environment.py --role render --check-only
conda run --no-capture-output -n vla-safety-train python scripts/check_conda_environment.py \
  --role train --smoke --out runs/conda_verification/train.json
conda run --no-capture-output -n vla-safety-render python scripts/check_conda_environment.py \
  --role render --smoke --out runs/conda_verification/render.json
```

`--smoke`는 학습 환경에서 고정 입력 한 건의 정책 추론을 수행하고, 렌더 환경에서
저장된 시연 한 건을 물리적으로 재생한 뒤 LIBERO 4개 suite의 초기 상태를 확인한다.
가중치를 학습하거나 기존 평가 결과를 덮어쓰지 않는다. `--smoke`를 생략하면
가중치·시연을 읽지 않고 패키지 import와 CUDA 등 기본 실행 조건만 확인한다.

2026-11-12에 micromamba 2.10.0으로 별도 Conda 환경을 실제 설치해 다음을 확인했다.
상세 근거는 [validation.json](validation.json)과 개별 `validation-*.json`에 있다.

- 두 환경 모두 `pip check` 통과. 학습 패키지 207개는 원본과 버전이 모두 같다.
- 평가 패키지 60개 중 59개는 원본 버전과 같고, PyTorch만 위에서 설명한 공개 CPU 빌드로 대체했다.
- 기반 모델의 고정 입력 한 건에 대한 action 출력은 원본 환경과 정확히 일치했다.
- 시연 한 건 196 timestep 재생에서 저장된 물리 상태와의 최대 절대 오차는 0이었다.
- LIBERO 4개 suite의 task 0/init 0에서 초기 상태와 두 카메라 이미지 해시가 각각 일치했다.

이 검증은 설치·추론·시뮬레이션의 샘플 비교다. 전체 재학습이나 676개/80개 평가를 다시 수행한 결과는 아니다.

## 학습·평가 실행 예시

아래 학습 명령은 **새 학습을 시작할 때만** 실행한다. 기존 실험은 16,000 step으로 완료되어
있고, 추가 8,000 step 학습은 포함하지 않는다. `--scope`는 `ae`, `vlm`, `all` 중 하나다.
새 출력 디렉터리를 사용해 기존 실험과 분리한다.

```bash
conda run --no-capture-output -n vla-safety-train python scripts/semantic_safety/train_expansion_v5_lora.py \
  --manifest data/semantic_expansion_v5_lora_v1/combined_train_manifest.jsonl \
  --out runs/conda_reproduction/train_all --scope all \
  --steps 16000 --batch-size 32 --micro-batch-size 16 \
  --learning-rate 0.0002 --save-every 4000 --recovery-every 200 --seed 20261112
```

평가는 먼저 별도 터미널에서 추론 서버를 실행하고 유지한다.

```bash
conda run --no-capture-output -n vla-safety-train python scripts/semantic_safety/v6_lora_server.py \
  --socket /tmp/vla-safety-reproduction.sock
```

다른 터미널에서 두 평가를 순서대로 실행한다. `all`을 `ae` 또는 `vlm`으로 바꾸면
해당 adapter를 평가하며, `--adapter` 인자를 생략하면 원본 LIBERO FT 모델을 평가한다.

```bash
conda run --no-capture-output -n vla-safety-render python scripts/semantic_safety/evaluate_expansion_v5_lora.py \
  --socket /tmp/vla-safety-reproduction.sock \
  --items data/semantic_expansion_v5_lora_v1/holdouts/test.jsonl \
  --adapter checkpoints/pi05_libero_expansion_v5_lora_v1/all/adapter.pt \
  --out runs/conda_reproduction/all_motor_test --batch-size 16 --horizon 460
conda run --no-capture-output -n vla-safety-render python scripts/semantic_safety/evaluate_expansion_v5_lora.py \
  --socket /tmp/vla-safety-reproduction.sock --native test \
  --adapter checkpoints/pi05_libero_expansion_v5_lora_v1/all/adapter.pt \
  --out runs/conda_reproduction/all_native_test --batch-size 16 --horizon 460
```

전체 평가 전에는 별도 출력 경로로 `--limit 16`을 추가해 점검한다. 제한 평가와 전체 평가에
같은 출력 디렉터리를 사용하지 않는다. 위 직접 명령은 기존 venv 경로가 필요 없다.

기존 자동 실행 드라이버는 `openpi/.venv/bin/python`, `envs/libero-render/bin/python`을
직접 호출한다. 그 드라이버를 새 서버에서 사용할 때만, 해당 경로가 **없는 경우** 아래와 같이
Conda 환경으로 연결한다. 이미 존재하면 스크립트가 오류를 내고 보존한다.

```bash
conda run -n vla-safety-train python scripts/setup_conda_environment.py --role train --bind-legacy-path
conda run -n vla-safety-render python scripts/setup_conda_environment.py --role render --bind-legacy-path
```

## 서버 조건과 재현 범위

원래 서버는 NVIDIA B200, 드라이버 580.95.05를 사용했다. CUDA 12.8 runtime은 PyTorch pip
의존성으로 설치되지만 GPU 드라이버와 EGL 장치 접근 권한은 서버에서 제공해야 한다.
다른 서버의 드라이버 라이브러리를 그대로 복사하지 말고 해당 서버에 맞는 드라이버를 사용한다.
EGL 초기화가 실패하면 서버의 NVIDIA EGL 설정과 컨테이너 GPU 노출을 먼저 확인한다.

Both 학습에서 기록한 최대 CUDA 할당 메모리는 약 115 GB였다. 작은 GPU에서는 micro batch를
줄이고 누적하여 batch 32를 유지하는 조정과 재검증이 필요하다. 의존성 버전이 같아도 다른
GPU·드라이버에서 부동소수점 결과나 학습 결과가 비트 단위로 같다고 보장하지 않는다.

Conda의 기본 공유 라이브러리 전체까지 고정한 container image는 아니다. Python 런타임과
pip 패키지, 소스 기준을 보존하는 환경 정의이며, 설치 및 기존 환경과의 비교 결과는
이 디렉터리의 `validation*.json`에 기록한다.
