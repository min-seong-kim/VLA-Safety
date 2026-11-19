# LIBERO 안전 표본을 섞으면 선형 probe의 오탐이 줄어드는가?

**줄었다.** 검증 데이터에서 위험 탐지율을 80% 이상으로 맞춘 비교에서, 안전 클래스의 학습 가중치 중 30%를 LIBERO에 주면 8개 모델·모듈·트랙 조건 모두에서 새 LIBERO task의 오탐이 줄었다. 기존 위험 요청을 얼마나 놓치는지는 조건에 따라 달랐다.

## 무엇을 바꿨는가?

- 모델은 공식 π0.5 Base와 LIBERO FT에서 고정했다. 첫 장면의 VLM 17층 residual, AE 17층 residual(flow 10)을 각각 사용했다. 기존과 같은 전처리, 공통 초기 noise, token 평균 pooling을 사용했다. 각 트랙의 양성(위험)은 기존 대조 데이터에서만 가져왔다.
- LIBERO의 안전 학습 후보는 40개 task의 기존 rehearsal 시연 154개(초기 상태 2–5)의 첫 정책 관측이다. 문장을 그대로 사용하고 두 카메라를 정방향으로 맞췄다. 시연을 성공했다고 안전 라벨을 부여한 것이 아니라, 요청이 표준 장면에서 요구하는 동작을 기준으로 판단했다. 다만 이 시연 저장소 자체는 이전에 성공 시연을 선별한 자료다. 가열기를 직접 켜는 두 task는 맥락에 따라 판단이 달라질 수 있어 해당 8개를 제외했다. 학습·검증에 쓸 수 있는 고유 LIBERO 관측은 38개 task, 146개다.
- “30% 혼합”은 **안전 클래스 안에서의 비율**이다. 전체 가중치가 기존 위험 50%, 기존 안전 35%, LIBERO 안전 15%가 된다. 10% 혼합에서는 각각 50%, 45%, 5%다. 기존 대조 표본은 원래의 그룹·클래스 가중치를 유지한 뒤 안전 쪽 총량만 줄였다. LIBERO는 과제 묶음에 같은 총량을 주고, 묶음 안에서는 task별로, task 안에서는 초기 상태별로 균등 배분했다. 총 가중치 합은 비율에 관계없이 기존 train 행 수로 고정했다.
- 0%는 저장된 기존 scaler와 probe를 그대로 사용했다. 10%와 30%는 각 fold의 혼합 train으로 가중 표준화기와 L2 logistic regression을 새로 학습했다. 원래 방식대로 C = 0.01, 0.1, 1 중 기존 대조 validation AUROC가 가장 높은 것을 선택했다. 동률이면 작은 C를 택했다.

## 어떤 데이터를 보지 않고 평가했는가?

기존 대조 데이터는 원래 연결 그룹 5-fold를 유지했다(A 336개 중 위험 108개, B 121개 중 위험 58개). 별도로 LIBERO 40개 task를 지시 내용으로 7개 묶음에 나눴다. 같은 task의 상태만 바꿔 train과 test에 넣지 않았고, 다른 suite에 속해도 식품을 바구니에 넣는 작업처럼 가까운 변형은 함께 보류했다. 묶음은 정답 예측을 이용해 구성하지 않았다. 큰 묶음부터 task 수가 작은 bin에 배정한 뒤 test bin과 다음 validation bin을 제외해 학습했다.

| Fold | LIBERO test 과제 묶음 | 혼합 시 LIBERO train / validation 장면 | LIBERO test 장면 |
|---:|---|---:|---:|
| 0 | 식품을 바구니에 넣기 | 43 / 52 | 26 |
| 1 | 그릇·컵을 접시에 배치하기와 그 변형 | 73 / 21 | 26 |
| 2 | 캐비닛·서랍 수납, 병 배치 | 115 / 10 | 12 |
| 3 | 스토브 주변 조작 | 124 / 12 | 10 |
| 4 | 책 정리, 음식→그릇, 컵→전자레인지(별도 3개 묶음) | 83 / 51 | 6 |

LIBERO test는 이전에 평가한 초기 상태 0·1의 80개로 통일했다. 이 80개는 이미 이전 연구 단계에서 본 벤치마크다. 이번 혼합 학습에서는 각 test task를 해당 fold의 train·validation에서 제외했다. 한 장면당 해당 묶음을 보지 않은 **한 개 fold**의 결과를 사용하므로 이전의 5-probe 다수결 수치와는 집계가 다르다. 예를 들어 A / LIBERO FT VLM의 기존 다수결 오탐은 35개, 이번 0%의 고정 임계값 오탐은 40개다.

## 주 비교 · Validation 위험 탐지율을 맞춘 경우

0%, 10%, 30% 각각에 대해 기존 대조 validation의 위험 표본 중 적어도 80%가 양성이 되는 가장 높은 logit을 임계값으로 정했다. 표본 수가 작아 실제 validation 탐지율은 80–100%였다. **Test에서 80%를 보장하거나 test를 보고 임계값을 조정한 것이 아니다.** 이때 0%가 임계값만 조정한 대조군이다.

각 칸은 **LIBERO 오탐 수 / 기존 위험 탐지 수**다. 첫 숫자는 80개 중 작을수록 좋고, 두 번째는 A 108개 또는 B 58개 중 클수록 좋다. 각 표본을 정확히 한 번 세는 5-fold 합산이다.

| 트랙 | 체크포인트 | 위치 | 0% | 10% | 30% | 검증에서 비율 선택 |
|---|---|---|---:|---:|---:|---:|
| A | Base | VLM | 6 / 61 | 2 / 61 | 2 / 62 | 6 / 61 |
| A | Base | AE | 53 / 68 | 21 / 73 | 19 / 72 | 19 / 73 |
| A | LIBERO FT | VLM | 41 / 80 | 4 / 77 | 2 / 81 | 2 / 78 |
| A | LIBERO FT | AE | 9 / 76 | 2 / 73 | 0 / 73 | 7 / 73 |
| B | Base | VLM | 32 / 44 | 16 / 38 | 13 / 43 | 16 / 38 |
| B | Base | AE | 30 / 44 | 12 / 42 | 9 / 42 | 12 / 44 |
| B | LIBERO FT | VLM | 40 / 49 | 7 / 42 | 5 / 40 | 5 / 40 |
| B | LIBERO FT | AE | 25 / 42 | 2 / 45 | 0 / 44 | 1 / 45 |

마지막 열은 부가 분석이다. 각 fold에서 위 임계값을 적용한 뒤 보류한 LIBERO validation 묶음의 오탐률이 가장 낮은 혼합비를 선택했다. 동률이면 낮은 비율을 선택했다. A / LIBERO FT VLM은 fold 순서대로 10·30·10·10·10%를 선택해 **오탐 2/80, 기존 위험 탐지 78/108**을 기록했다. 30% 고정 열은 사전에 정한 후보의 결과이며 이 표에서 승자를 골라 독립 검증한 수치는 아니다.

## 같은 장면의 판정은 어떻게 달라졌나?

- **A / LIBERO FT VLM.** 0%→30%에서 LIBERO 오탐 41개 중 39개를 바로잡았고 새 오탐은 없었다. 남은 두 장면은 ‘오렌지 주스를 바구니에 넣기’의 초기 상태 0과 ‘책을 정리함 뒤쪽에 놓기’의 초기 상태 1이었다. 가열기를 켜는 두 task를 test에서도 제외하면 오탐은 38/76→2/76이다. 기존 위험에서는 새로 7개를 탐지하고 6개를 놓쳐 총 탐지는 80→81개였다. 기존 안전의 오탐도 40/228→37/228이었다.
- **B / LIBERO FT VLM.** LIBERO 오탐은 40→5개로 줄었지만, 기존 위험 58개 중 0%에서 잡던 10개를 놓치고 1개를 새로 잡아 탐지는 49→40개가 됐다. 반면 같은 모델의 **B / AE**에서는 오탐 25→0개, 탐지 42→44개였다. 0/80은 이 표본에서 관측한 값이다.
- **순위와 임계값의 영향.** A / LIBERO FT VLM의 기존 대조 test AUROC(fold 평균)는 0.888→0.883, B / LIBERO FT VLM은 0.839→0.837이었다. B VLM의 순위 지표가 비슷해도 validation에서 정한 임계값을 test에 옮기면 탐지 수가 달라질 수 있다.

## 비교 · 원래 임계값 0을 고정한 경우

아래도 **LIBERO 오탐 수 / 기존 위험 탐지 수**다. 마지막 열의 혼합비는 위 validation 규칙으로 선택한 것을 유지했다. 임계값만 0으로 바꿔 판정했다.

| 트랙 | 체크포인트 | 위치 | 0% | 10% | 30% | 검증에서 비율 선택 |
|---|---|---|---:|---:|---:|---:|
| A | Base | VLM | 6 / 79 | 4 / 72 | 4 / 76 | 6 / 78 |
| A | Base | AE | 43 / 69 | 11 / 64 | 9 / 68 | 10 / 65 |
| A | LIBERO FT | VLM | 40 / 79 | 4 / 81 | 3 / 85 | 3 / 84 |
| A | LIBERO FT | AE | 6 / 73 | 1 / 72 | 0 / 77 | 3 / 71 |
| B | Base | VLM | 43 / 40 | 6 / 37 | 6 / 39 | 6 / 37 |
| B | Base | AE | 24 / 39 | 9 / 39 | 4 / 39 | 9 / 38 |
| B | LIBERO FT | VLM | 39 / 48 | 8 / 44 | 7 / 47 | 7 / 46 |
| B | LIBERO FT | AE | 18 / 36 | 0 / 36 | 0 / 37 | 0 / 36 |

## 해석 범위

목표가 일반 LIBERO 작업의 불필요한 차단을 줄이는 것이라면, 비슷한 task까지 묶어 보류한 조건에서도 benign 혼합이 효과를 보였다. 다만 LIBERO 학습의 예시는 모두 안전이므로, probe가 LIBERO 장면의 공통 외관을 안전 단서로 사용하는 가능성은 남는다. 위험 탐지율은 **기존 대조 데이터**에서 측정한 것이다. LIBERO 스타일의 위험 요청에서도 모두 탐지할 수 있다는 근거는 아니다. 두 초기 상태는 같은 지시의 반복이고 test 묶음 크기도 달라, 장면들을 독립 표본으로 취급해 통계적 유의성을 주장하지 않는다.

## 파일과 재실행

[실험 코드](../scripts/semantic_safety/mix_libero_benign_probes.py) · [설정](../runs/contrast_libero_benign_mix_v1/experiment_protocol.json) · [전체 집계](../runs/contrast_libero_benign_mix_v1/summary.csv) · [fold별 임계값·C·검증 및 test 점수](../runs/contrast_libero_benign_mix_v1/fold_metrics.csv) · [표본별 test 예측](../runs/contrast_libero_benign_mix_v1/test_predictions.csv) · [혼합 probe 체크포인트](../runs/contrast_libero_benign_mix_v1/probes/) · [검증 기록](../runs/contrast_libero_benign_mix_v1/verification.json)

0% 원본 probe 40개의 기존 contrast와 LIBERO logit이 재현됐고, 두 모델의 전처리 입력 hash가 일치했다. 각 fold에서 원래 입력과 LIBERO 입력을 함께 놓고 train·validation·test 간 동일 전처리 입력 중복이 없음을 검사했다. 학습한 혼합 probe 80개(8조건 × 2혼합비 × 5fold)를 저장했다.

저장소 루트에서 아래 순서로 재현한다. 마지막 단계만 실행하면 저장된 특징으로 probe를 다시 학습한다.

```bash
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
export PYTHONPATH="$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src"
script=scripts/semantic_safety/mix_libero_benign_probes.py
envs/representation-probe/bin/python "$script" prepare
openpi/.venv/bin/python "$script" base
openpi/.venv/bin/python "$script" libero
envs/representation-probe/bin/python "$script" fit
```
