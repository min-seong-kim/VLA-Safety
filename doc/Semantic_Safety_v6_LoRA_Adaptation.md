# v6 π0.5 모터 적응: VLM / AE / 공동 LoRA

상태: **학습 및 최종 평가 완료**. 공식 LIBERO FT π0.5에서 출발한 세 LoRA 모델을 비교했다. 아래 모터 성공은 **집기 → 들어올리기 → 목적지 부근 운반**의 성공이다. 최종 붓기·절단·타격이나 위해 발생을 뜻하지 않으며, semantic safety awareness 자체는 아직 측정하지 않았다.

## 최종 수행 결과

Checkpoint는 방향 균형 validation 66개로 선택한 뒤 고정했다. Test 결과로 다시 학습하거나 checkpoint를 바꾸지 않았다.

| 모델 | 새 배치 test 운반 | 원본 v6 운반 | Native LIBERO 성공 | 선택 step |
|---|---:|---:|---:|---:|
| 기본 π0.5 | 38/264 (14.4%) | 19/132 (14.4%) | 16/16 (100.0%) | — |
| AE-only | 109/264 (41.3%) | 57/132 (43.2%) | 16/16 (100.0%) | 7,000 |
| VLM-only | 188/264 (71.2%) | 88/132 (66.7%) | 15/16 (93.8%) | 8,000 |
| 공동 LoRA | 179/264 (67.8%) | 93/132 (70.5%) | 16/16 (100.0%) | 6,000 |

Native는 Spatial 8개 task × 초기 상태 2개인 16-episode 회귀 검사이다. 전체 LIBERO benchmark 성능으로 일반화하지 않는다. 새 test도 동일한 물체 종류의 새로운 배치이며, 미관측 물체/위험 종류 평가가 아니다.

| 모델 | Test 지속 운반 | Test XY-only 운반 | 확인용 validation 운반 | Test 양쪽 성공 쌍 (상황 수) |
|---|---:|---:|---:|---:|
| 기본 π0.5 | 35/264 (13.3%) | 55/264 (20.8%) | 9/66 (13.6%) | 11/132 (7/33) |
| AE-only | 106/264 (40.2%) | 128/264 (48.5%) | 22/66 (33.3%) | 33/132 (19/33) |
| VLM-only | 188/264 (71.2%) | 193/264 (73.1%) | 43/66 (65.2%) | 80/132 (28/33) |
| 공동 LoRA | 177/264 (67.0%) | 189/264 (71.6%) | 41/66 (62.1%) | 73/132 (23/33) |

| 모델 | 안전 지시 test 운반 | 위험 지시 test 운반 | 시연 있는 63조건 | 시연 없는 3조건 |
|---|---:|---:|---:|---:|
| 기본 π0.5 | 25/132 (18.9%) | 13/132 (9.8%) | 38/252 (15.1%) | 0/12 (0.0%) |
| AE-only | 58/132 (43.9%) | 51/132 (38.6%) | 109/252 (43.3%) | 0/12 (0.0%) |
| VLM-only | 93/132 (70.5%) | 95/132 (72.0%) | 188/252 (74.6%) | 0/12 (0.0%) |
| 공동 LoRA | 85/132 (64.4%) | 94/132 (71.2%) | 179/252 (71.0%) | 0/12 (0.0%) |

| 모델 | Test 집기 관측 | Test 4cm lift 관측 | 기본 모델 대비 운반 추가 성공 / 상실 |
|---|---:|---:|---:|
| 기본 π0.5 | 201/264 (76.1%) | 140/264 (53.0%) | — |
| AE-only | 224/264 (84.8%) | 178/264 (67.4%) | +80 / −9 |
| VLM-only | 243/264 (92.0%) | 215/264 (81.4%) | +155 / −5 |
| 공동 LoRA | 244/264 (92.4%) | 216/264 (81.8%) | +143 / −2 |

집기와 lift는 episode 중 한 번 이상 해당 상태가 관측됐는지를 센다. 추가 성공/상실은 동일한 test 장면에서 기본 모델의 실패가 성공으로, 또는 성공이 실패로 바뀐 수이다. 각 모델의 운반 성공은 현재 grasp와 연속 운반 조건까지 별도로 충족해야 한다.

동일 장면의 개선/퇴보 수는 [paired_model_comparisons.csv](../runs/semantic_motor_v6_lora_v2/paired_model_comparisons.csv)에 있다. 합계뿐 아니라 안전/위험 쌍을 모두 수행했는지도 집계했다. 본 비교는 각 조건을 한 번씩 학습한 결과이며 독립적인 반복 실행의 분산은 측정하지 않았다.

## 학습 데이터

- 신규 시연 **3,132개 / 673,508 control steps**. 66개 조건 중 63개에서 최소 48개씩 확보했다. 조건 단위 안전/위험 수는 32/31이다.
- Teacher의 모든 action은 실제 물리 step으로 실행했다. 채택한 시연은 독립 환경에서 전 action을 재실행하고 상태 및 성공을 확인했다. 이미지·state·action 캐시는 원본 배열과 정확히 일치한다.
- 모터 조건 → 시연 → 10-step 시작점을 균등 추출하며, 네 가지 지시 표현은 위치/방향과 독립적으로 추출한다.
- 학습 샘플의 10%는 공식 모델의 native 성공 시연 50개에서 뽑는다. Spatial 10개 task의 초기 상태 2–6을 사용하며, 새 native 회귀 평가의 초기 상태 0–1은 이 추가 학습에서 제외했다. 이는 공식 checkpoint의 과거 LIBERO 학습에서도 미관측이었다는 뜻은 아니다.
- 학생의 실제 모델 계산에는 main/wrist RGB와 지시가 사용된다. Privileged 물체 pose와 이진 안전 label은 입력하지 않는다. 공식 π0.5 정규화 통계를 유지했다.
- Train, validation 132개, test 264개는 별도 seed로 만들었으며 정확한 배치 중복을 배제했다. 평가 장면은 teacher/model 성공 여부로 필터하지 않았다.
- 버튼 배터리→컵, 나무 그릇→매트, 나무 그릇→촛불의 **3개 조건은 성공 시연을 확보하지 못했다**. 이 조건도 평가에는 남아 있다. Teacher 탐색 실패를 물리적 불가능의 증거로 해석하지 않는다.
- 앞쪽 목적지의 일부 train 배치는 도달 가능성을 고려해 조정했다. 지갑 시연 48개에는 세운 초기 자세를 사용했다. 원본 v6와 holdout 배치는 변경하지 않았다.

전체 조건별 확보 수와 검증 기록: [데이터 안내](../data/semantic_motor_v6_lora_v2/README.md), [coverage CSV](../data/semantic_motor_v6_lora_v2/condition_coverage.csv), [data audit](../data/semantic_motor_v6_lora_v2/training_data_audit.json).

## 모델과 학습 설정

모든 원본 가중치는 고정하고 선택한 모듈의 LoRA만 학습했다. Rank 16, alpha 32, dropout 0, batch 16, AdamW, 최대 learning rate 2e-4, warmup/cosine schedule, gradient clip 1, seed 20261111이다. 세 조건은 동일 manifest와 샘플링/flow noise seed 설정을 사용한다.

| 조건 | LoRA 범위 | 학습 파라미터 | 학습 steps | 원본 가중치 고정 |
|---|---|---:|---:|---|
| AE-only | Action transformer·조건부 norm 선형층·action/time projection | 9,454,592 | 6,000 + 2,000 | 통과 |
| VLM-only | PaliGemma 영상·언어·멀티모달 선형층 | 28,358,144 | 6,000 + 2,000 | 통과 |
| 공동 LoRA | VLM 및 AE의 위 범위 | 37,812,736 | 6,000 + 2,000 | 통과 |

VLM validation 운반이 4k의 38/66에서 6k의 43/66으로 증가해, test를 열기 전에 세 조건 모두에 동일한 추가 2,000-step 단계를 정했다. 각자의 6k adapter에서 시작하며, 새 AdamW optimizer, 최대 learning rate 2e-5, warmup 200, cosine 최저 learning rate 2e-6, seed 20271111을 사용했다. 기존 optimizer 상태를 이어받은 연속 재개는 아니다. 본 비교의 총 학습량은 각 8,000 updates이고, 선택 후보는 누적 2k/4k/6k/7k/8k이다. 추가 단계 adapter 내부의 `metadata.step`은 해당 단계 안의 step이며 export의 `training_history.json`에 누적 step을 기록했다.


AE-only에서는 동일 입력의 VLM hidden state도 학습 전후 정확히 같았다. VLM/공동 학습에서는 differentiable prefix KV를 통해 action loss가 영상·언어 adapter에 도달함을 확인했다. 마지막 VLM layer의 K/V 외 Q/O/MLP adapter는 action loss에 직접 연결되지 않아 gradient가 없는 점을 기록했다. 파라미터 수가 다른 실험이므로 동일 파라미터 예산의 ablation은 아니다.

## Checkpoint 선택

33개 safety/unsafety 쌍을 유지하면서 상황 그룹 번호의 짝/홀로 배치를 고른 66개 장면을 사용했다. 운반 방향은 17쌍/16쌍이다. 선택 순서는 운반 성공 수 → 지속 성공 수 → 더 이른 step이며, 나머지 validation 66개는 선택 후 확인에 사용했다. 초기 한 방향 subset은 학습 모델 평가 전에 수정했고 기록을 보존했다.

| 조건 | Step | 운반 | 지속 운반 | XY-only |
|---|---:|---:|---:|---:|
| AE-only | 2,000 | 13/66 | 13/66 | 21/66 |
| AE-only | 4,000 | 24/66 | 24/66 | 28/66 |
| AE-only | 6,000 | 24/66 | 24/66 | 32/66 |
| AE-only | 7,000 | 26/66 | 26/66 | 32/66 |
| AE-only | 8,000 | 26/66 | 25/66 | 31/66 |
| VLM-only | 2,000 | 37/66 | 37/66 | 43/66 |
| VLM-only | 4,000 | 38/66 | 38/66 | 43/66 |
| VLM-only | 6,000 | 43/66 | 43/66 | 45/66 |
| VLM-only | 7,000 | 43/66 | 42/66 | 46/66 |
| VLM-only | 8,000 | 47/66 | 47/66 | 50/66 |
| 공동 LoRA | 2,000 | 35/66 | 35/66 | 38/66 |
| 공동 LoRA | 4,000 | 42/66 | 42/66 | 44/66 |
| 공동 LoRA | 6,000 | 48/66 | 48/66 | 50/66 |
| 공동 LoRA | 7,000 | 40/66 | 40/66 | 44/66 |
| 공동 LoRA | 8,000 | 47/66 | 45/66 | 48/66 |

![학습 곡선](../runs/semantic_motor_v6_lora_v2/learning_curves.png)

곡선의 6,000-step 점선은 추가 학습 단계의 시작을 표시한다.

## 평가 기준과 검증

모터 episode는 최대 360 control steps이며 10-step action chunk와 10회 flow denoising을 사용한다. 원래 운반 기준은 현재 양쪽 finger grasp, 4cm 이상 lift, 연속/순 이동 각각 10cm 이상, 목표 XY 반경 5cm 및 Z 오차 ±4cm이다. 4-step 지속 성공도 별도 보고한다. XY-only는 이 중 Z 조건만 제외한 진단 지표이며 목적지 위 배치나 위해 발생을 증명하지 않는다.

원본 v6에는 이미지·지시가 같지만 그룹에 따라 목표 높이가 다른 네 입력 그룹이 있다. 망치→못 박힌 나무는 약 10.7cm, 물 컵→대야는 약 8.9cm 차이가 난다. 원본 기준은 유지했으므로 이 모호성을 수행 실패의 해석에 고려해야 한다.

채택 시연의 독립 물리 replay, 원본 가중치 고정, gradient 범위, 공식 policy.infer와의 출력 일치, 평가 입력/action/event 정렬, video frame 수, 모델 사이 초기 입력·지시·noise seed를 검증했다. 132개 전체 trajectory에서 batch 8/32의 모든 action/state/정책 입력 이미지가 일치했다. 빠른 모터 영상은 2Hz이며 모든 20Hz action/state/event를 별도 저장한다. 마지막 작은 batch의 크기는 비교 시 유지한다.

## 해석 범위와 산출물

이 실험은 새로운 물체와 운반 동작의 수행 능력을 준비하는 단계이다. 성공 trajectory가 늘었다고 VLM이 위험을 인식하거나 AE에 경고가 전달됐다고 결론 내릴 수 없다. 성공한 쌍만 고른 분석에는 선택 편향과 물체/상황 차이가 남을 수 있다. t-SNE, LDA, linear probe 및 내부 warning 전달 분석은 아직 수행하지 않았다.

AE-only의 VLM 함수는 고정 입력에 대해 유지되지만, 정책의 action이 달라지면 이후 관찰 이미지도 달라진다. 따라서 서로 다른 rollout의 같은 timestep feature 차이를 모듈 변경 효과로 바로 해석해서는 안 된다. 후속 분석에는 동일 관찰 입력 재생이나 행동 단계 정렬을 함께 고려해야 한다.

- [세 adapter와 로딩 방법](../checkpoints/pi05_libero_v6_lora_v2/README.md)
- [동일 장면 영상 비교](../runs/semantic_motor_v6_lora_v2/review.html)
- [영상 비교 묶음 ZIP](../runs/semantic_motor_v6_lora_v2/review_bundle.zip)
- [비교 CSV](../runs/semantic_motor_v6_lora_v2/comparison_summary.csv)
- [66개 조건별 test 성능과 학습 시연 수](../runs/semantic_motor_v6_lora_v2/test_condition_summary.csv)
- [성공한 safety/unsafety 쌍과 trajectory 경로](../runs/semantic_motor_v6_lora_v2/representation_candidate_pairs.json)
- [환경과 실제 upstream source 기록](../runs/semantic_motor_v6_lora_v2/environment.json)
- [세 학습 조건의 설정·가중치·gradient 검증](../runs/semantic_motor_v6_lora_v2/training_comparison_audit.json)
- [이전 실패 진단](Semantic_Safety_v6_Failure_Diagnosis.md)

원본 v6는 이미 진단/teacher 설계에 사용한 개발 benchmark이다. 새 시연은 별도로 생성했지만 이 점 때문에 원본 v6 결과를 미관측 benchmark 성능으로 해석하지 않는다.

## 후속 representation 분석용 후보

안전·위험 양쪽 운반에 성공한 쌍을 모델별로 저장했다. 모델 사이 비교에서 공통으로 사용할 수 있는 쌍의 수는 다음과 같다. 아래 수는 각 모델에서 두 episode 모두 운반에 성공한 경우의 교집합이며, 지속 성공은 후보 파일에서 추가로 구분할 수 있다.

| 평가 집합 | 세 LoRA 모델 공통 | 기본 모델까지 네 모델 공통 |
|---|---:|---:|
| confirmation | 5 | 2 |
| test | 25 | 6 |
| v6 | 16 | 2 |

이 후보 목록에는 feature가 아직 추출되어 있지 않다. 물체/상황 식별과 안전 의미의 구분을 분리하려면 후속 probe의 학습·평가도 상황 단위로 나눠야 한다.

## 재실행 예시

저장소 루트에서 아래와 같이 새 출력 디렉터리에 학습한다. `--scope`는 `ae`, `vlm`, `all` 중 하나이다. 현재 v2 manifest는 v1의 불변 시연·캐시 파일을 공유하므로 해당 원본 데이터도 필요하다.

```bash
PYTHONPATH="openpi/src:openpi/packages/openpi-client/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
openpi/.venv/bin/python scripts/semantic_safety/train_v6_scoped_lora.py \
  --manifest data/semantic_motor_v6_lora_v2/combined_train_manifest.jsonl \
  --out runs/reproduce_v6_vlm --scope vlm --steps 6000 --batch-size 16 \
  --learning-rate 0.0002 --seed 20261111 --save-every 2000
```

추가 단계는 같은 manifest와 scope로 새 출력 디렉터리를 지정하고 `--resume <해당 scope의 adapter_step_06000.pt> --steps 2000 --learning-rate 0.00002 --seed 20271111 --save-every 1000`을 사용한다. `--resume`은 adapter 가중치만 불러오며 optimizer는 새로 만든다.

학습 데이터 보완: 스푼의 집기 높이를 calibration에서 8mm 높여 새 시연 48개를 확보했다. 이전 실행은 보존하고, 본 비교의 세 모델은 모두 보완한 동일 데이터로 공식 base부터 다시 학습했다. 물체 asset과 원본 평가 목표는 바꾸지 않았다.
