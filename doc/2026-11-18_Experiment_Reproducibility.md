# π0.5 실험을 다른 서버에서 다시 실행하는 방법

**저장된 특징으로 probe를 다시 학습하는 과정은 독립 Python 환경에서 재현했다.** 공식 체크포인트에서 특징을 다시 추출하는 과정과 LIBERO 첫 장면 생성은 아래 표본에서 검증했다. LoRA 학습은 재실행에 필요한 설정과 파일을 확인했으며, 전체 학습에 대한 반복 검증은 별도로 필요하다.

| 단계 | 무엇을 비교했나 | 결과 |
|---|---|---|
| CPU probe 학습·평가 | 새 Python 3.11.14 venv에 버전이 고정된 패키지를 설치했다. Contrast의 두 공식 모델 × 두 트랙 × VLM·AE 주요 위치 × 5 fold를 임시 디렉토리에서 다시 학습했다. | 기존 test logit 1,828개가 모두 일치했다. 최대 절댓값 차이 0. |
| Benign 혼합 probe | 새로 학습한 40개 baseline으로 0% 조건을 검증하고 10%·30% probe 80개도 다시 학습했다. | 원래 fold별 지표, 전체 집계, 개별 test 예측 CSV 3개가 바이트 단위로 일치했다. |
| GPU 특징 추출 | 기존 학습 환경에서 공식 Base와 LIBERO FT를 각각 다시 로드하고, 보관한 LIBERO 입력 중 처음 12개를 기존 배치 크기 12로 추론했다. | 두 체크포인트 모두 입력·가중치 hash가 일치했다. 저장된 78개 위치의 특징과 최대 절댓값 차이 0. |
| LIBERO 첫 장면 | Spatial, Object, Goal, Long의 task 0·초기 상태 0을 현재 서버에서 다시 생성했다. 기존 평가와 같이 초기 안정화 10 step 직후를 비교했다. | 4개 모두 영문 지시, 로봇 상태, 두 카메라 RGB가 일치했다. 전체 80개 에피소드의 성공률까지 다시 측정한 검사는 아니다. |
| LoRA 학습 | 확장 v5와 Benign 각각 AE-only·VLM-only·Both의 학습 config에 기록된 코드·학습 manifest SHA-256을 현재 파일과 비교했다. | 6개 모두 일치했고 복구용 상태 파일도 있다. 새 서버에서 16,000 step을 처음부터 학습해 최종 가중치까지 일치하는지는 검증하지 않았다. |

현재 검증은 NVIDIA B200, 드라이버 590.44.01에서 했다. 11월 12일의 별도 검증에서는 드라이버 580.95.05에서 새 Conda 학습·렌더 환경을 설치하고 고정 입력 1건의 action과 196-step 물리 재생을 비교했다. 이번 CPU 검증은 새 venv에 YAML의 pip 버전을 설치한 결과다. 이번에 probe용 Conda solver까지 새로 실행하지는 않았다. [이번 검증 수치](../environments/validation-replay-2026-11-18.json) · [이전 Conda 설치·재생 검증](../environments/validation.json)

## Probe만 재현할 때

저장소 루트에서 실행한다. 입력 특징, manifest, 비교 대상 예측 파일이 필요하다. 체크포인트를 로드하거나 GPU를 할당할 필요는 없다.

```bash
sha256sum --quiet -c environments/probe-replay.sha256
conda env create -f environment-probe.yml
conda run -n vla-safety-probe python -m pip check
conda run -n vla-safety-probe python scripts/semantic_safety/check_probe_reproduction.py
```

검사기는 빈 임시 폴더에서 학습하므로 기존 결과를 읽고 끝내거나 덮어쓰는 일이 없다. 원본 probe의 logit은 수치 허용 오차로 검사하고 최대 차이를 출력한다. 혼합 probe의 CSV는 SHA-256을 비교하므로 미세한 부동소수점 차이도 검출한다. 다른 머신에서 이 검사가 실패하면 먼저 특징·manifest의 hash를 확인한 뒤 점수의 수치 차이를 확인한다. 체크섬 파일에는 이 CPU 검사에 필요한 파일 경로도 들어 있다. [probe 환경](../environment-probe.yml) · [입력·기준 결과 체크섬](../environments/probe-replay.sha256)

## Seed와 보존해야 할 설정

- **LoRA** — 확장 v5는 `20261112`, Benign 적응은 `20261116`이다. 미니배치 샘플링은 `seed + step × 100003`, diffusion noise/time은 `seed + step`으로 고정한다. 두 실험 모두 rank 16, alpha 32, batch 32, micro-batch 16, peak LR 0.0002, 16,000 optimizer updates다. 정확한 대상 모듈·스케줄·초기화·데이터 목록은 각 run의 `train_{ae,vlm,all}/config.json`을 따른다. Benign 최종 평가에 쓴 adapter는 AE 8k, VLM 12k, Both 8k이므로 마지막 체크포인트와 구분한다.
- **LIBERO** — 표준 suite·task ID와 배포된 init 파일을 함께 고정한다. 기존 native 평가에서 train rehearsal은 상태 2–5, selection은 7, test는 0·1이었다. 환경 초기 seed는 0이고 행동 추론 seed는 suite·task·상태·control step에 따라 정해진다. 카메라 방향, 초기 안정화, action horizon, 재추론 간격도 유지한다.
- **Probe** — 이전 trajectory/family 분할의 seed는 `20261112`, 첫 장면 contrast·split 비교는 `20261116`이다. Contrast 첫 장면은 공통 10 × 32 초기 noise를 사용한다. 기존 manifest의 행 순서, group/fold 목록, 학습 가중치, scaler, C 선택, 판정 기준을 보존한다. Benign 혼합에서는 LIBERO task 묶음과 임계값 선택 규칙까지 run에 저장돼 있다. 난수 seed 하나만으로 이 결정을 복원할 수는 없다.

[Benign LoRA 학습·평가](2026-11-17_Semantic_Safety_Benign_Adaptation_v1_Results.md) · [첫 장면 probe](2026-11-16_Semantic_Safety_Contrast_Initial_Linear_v1_Results.md) · [혼합 probe](2026-11-18_Semantic_Safety_LIBERO_Benign_Mix_Probe_v1_Results.md) · [분할 비교](2026-11-17_Semantic_Safety_Split_Comparison_v1_Results.md)

## 서버를 옮길 때

1. **코드와 로컬 자료를 함께 보존한다.** 이번 작업 트리에는 아직 커밋되지 않은 스크립트·문서·환경 정의가 있다. Git에서 제외하는 `data/`, `checkpoints/`, `runs/`, `openpi/`, `third_party/`, `.cache/openpi/big_vision/paligemma_tokenizer.model`은 별도 보관이 필요하다. 시뮬레이션은 XML·mesh·texture·BDDL·init 파일과 심볼릭 링크 대상까지 필요하다. 일부 파일이 절대 경로를 쓰므로 같은 경로에 마운트하는 편이 가장 간단하다.
2. **GPU 학습·추론과 렌더 환경을 분리한다.** 기존 Conda 설치 안내의 train/render 설정 스크립트까지 실행한다. 학습 환경에는 Transformers 패치 5개가 필요하다. 렌더 Conda는 원래 호스트의 NVIDIA PyTorch 빌드를 CPU용 PyTorch로 대체해 검증했다. 이번 서버에서는 이전 580 드라이버용 `.persist/nvidia-egl`을 강제하면 EGL 초기화에 실패했고, 현재 시스템의 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`과 해당 드라이버 라이브러리로 통과했다. 일부 supervisor는 예전 경로를 내부에서 지정하므로 새 서버에서는 확인된 EGL로 개별 실행 명령을 쓰거나 supervisor의 해당 설정을 맞춘다. [설치·검사 명령](../environments/README.md) · [외부 소스 버전·해시](../environments/source-state.json)
3. **같은 실험을 새로 실행할 때는 출력 위치를 분리한다.** 학습 폴더에 `training_state.pt`가 있으면 optimizer·adapter·step을 읽고 이어서 학습한다. 처음부터 비교하려면 기존 manifest를 사용하되 빈 출력 폴더를 지정한다. 경로가 고정된 supervisor는 완료된 run을 재사용할 수 있으므로 독립 반복은 별도 작업 사본에서 진행한다. 같은 seed를 써도 GPU, 드라이버, 병렬 연산 라이브러리가 바뀌면 긴 학습의 최종 가중치나 접촉이 포함된 rollout 결과까지 동일하다고 보장할 수 없다.
