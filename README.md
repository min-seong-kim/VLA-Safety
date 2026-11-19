# VLA Semantic Safety

π0.5의 VLM backbone과 Action Expert에서 안전·위험 지시의 정보를 선형적으로 읽을 수 있는지 분석하는 연구 저장소다. 첫 장면·trajectory representation 분석, 안전한 물체·조작 적응, 분할 단위 비교를 보존한다.

## 핵심 연구 문서

1. [기존 안전·위험 trajectory representation 분석](doc/2026-11-13_Semantic_Safety_Representation_Linear_v1_Results.md)
2. [새 대조 데이터셋의 첫 장면 Linear Probe](doc/2026-11-16_Semantic_Safety_Contrast_Initial_Linear_v1_Results.md)
3. [안전한 물체·조작 적응 설계](doc/2026-11-16_Semantic_Safety_Benign_Adaptation_v1_Design.md) · [학습·평가 결과](doc/2026-11-17_Semantic_Safety_Benign_Adaptation_v1_Results.md)
4. [Family·연결 group·개별 contrast 분할 비교](doc/2026-11-17_Semantic_Safety_Split_Comparison_v1_Results.md)

이전 확장 v5 LoRA 학습, 장면 구성, RoboCasa 물체 가져오기, 자동 보고의 관련 문서는 `doc/`에 함께 남겨두었다. 이전 VLM 벤치마크, π0·OpenVLA 평가, 초기 운반·실패 진단 실험은 현재 보존 범위에서 제외했다.

## 디렉토리

| 경로 | 내용 |
|---|---|
| `scripts/semantic_safety/` | 보존한 실험 실행·feature 추출·probe 학습·시연 생성·평가와 공통 모듈 |
| `doc/` | 연구 방법·결과·그래프 |
| `data/semantic_expansion_v5_lora_v1/` | 기존 338조건의 시연·학습 목록·LIBERO rehearsal 캐시 |
| `data/semantic_safety_representation_teacher_v1/` | 기존 데이터의 Teacher trajectory |
| `data/semantic_safety_contrast_v1/` | 새 87개 대조 묶음·336개 의미 조건과 초기 관측 |
| `data/semantic_benign_adaptation_v1/` | 안전한 물체·조작 적응 시연 |
| `runs/semantic_expansion_v5_lora_v1/` | 기존 적응 모델 학습·선택·평가 및 Both 참조 trajectory·LIBERO 시연 원본 |
| `runs/semantic_safety_representation_linear_v1/` | 기존 trajectory probe 결과 |
| `runs/semantic_safety_contrast_initial_linear_v1/` | 새 데이터의 공식 base·LIBERO FT probe 결과 |
| `runs/semantic_benign_adaptation_v1/` | Benign 적응·선택·평가·probe 결과 |
| `runs/semantic_safety_split_comparison_v1/` | 두 데이터셋·세 분할·다섯 모델의 비교 결과 |
| `checkpoints/` | 관련 π0.5 원본·변환 체크포인트, 두 적응 실험의 adapter, tokenizer |
| `openpi/`, `third_party/libero/`, `third_party/robosuite_vanilla/` | 실제 사용한 framework·시뮬레이터 소스와 학습 환경 |

`data/safety_sources/`와 `data/robocasa_libero_import_v1/`에는 공유 mesh·texture와 원본 정보가 있다. `data/semantic_rollout_v3/assets/`, `data/semantic_safety_two_object_v4/assets/`, `data/semantic_safety_transport_v6/assets/`도 현재 자산 생성 코드의 의존성이므로 보존한다. 일부 `v6_*` 스크립트 역시 현재 코드가 import하는 공통 모듈이며, 해당 이름의 과거 독립 실험을 보존한다는 뜻은 아니다.

## 실행 환경

CPU probe 환경과 단계별 재현 검증은 [실험 재현 안내](doc/2026-11-18_Experiment_Reproducibility.md)와 [environment-probe.yml](environment-probe.yml)에 정리했다.

| 용도 | 현재 interpreter |
|---|---|
| π0.5 추론·LoRA 학습 | `openpi/.venv/bin/python` |
| 시뮬레이션·시연·평가 | `envs/libero-render/bin/python` |
| CPU linear probe·결과 집계 | `envs/representation-probe/bin/python` |
| 데이터 검토 HTML의 browser 검사 | `envs/html-preview/`, `envs/v4-preview-check/` |

새 서버의 학습·렌더 환경 설치는 [환경 재현 안내](environments/README.md), [environment.yml](environment.yml), [environment-render.yml](environment-render.yml)를 따른다. 현재 runtime을 지탱하는 `envs/uv-python/`과 tokenizer cache `.cache/openpi/`도 보존한다.

이 서버에서는 로컬 프로세스를 사용한다. 과거 Hangang/Slurm job과 오래된 bootstrap 스크립트는 제거했다. 현재 실행 절차는 이 README와 위 연구 문서를 기준으로 확인한다.

## 분할 비교 실행과 확인

저장소 루트에서 실행한다. 완료 상태에서는 새 학습·평가를 시작하지 않는다.

```bash
envs/representation-probe/bin/python scripts/semantic_safety/split_comparison_v1.py supervise --workers 24
```

설정을 바꾼 실험은 새 run으로 분리한다. 고정한 manifest, fold, 원본 지표·feature와 체크포인트를 덮어쓰지 않는다. 다른 실험의 실행 명령과 학습 recipe는 각 문서에 있다.

```bash
PYTHONPATH=scripts/semantic_safety envs/representation-probe/bin/python -m unittest representation_probe_test split_comparison_v1_test experiment_notifications_test
envs/libero-render/bin/python scripts/semantic_safety/benign_evaluation_oracle_test.py
```

## 데이터와 모델 보관

`.gitignore`는 `data/`, `runs/`, `checkpoints/`, 환경·cache·외부 framework를 제외한다. Git 저장소만으로 원본 데이터와 가중치가 복원되지는 않으므로, 관련 경로를 별도로 보관한다. 일부 XML·manifest는 현재 루트의 절대 경로를 사용한다.

2026-11-17 정리의 삭제 목록과 보존·실행 검증 기록은 `runs/repository_cleanup_*/`에 있다. 핵심 네 문서와 실험 코드는 내용 변경 없이 보존했다.
