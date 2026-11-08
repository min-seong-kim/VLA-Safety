# VLA 추가 체크포인트 평가: LIBERO 학습 이후 무엇이 달라졌는가

작성일: 2026-11-05 · 최종 상태: **완료** · 실행 결과: `runs/backbone_eval_v2_checkpoints`

기존 실험에 공개 체크포인트 네 개를 추가했다. VLA 본체를 새로 fine-tuning하지 않고, 기존과 같은 시각·언어 과제와 요청 안전성 과제에서 평가했다. Backbone과 기존 head는 고정하고 데이터셋별 선형 probe만 학습했다.

가장 큰 차이는 **π0.5 base와 π0.5-LIBERO 사이**에서 나타났다. VQA와 COCO에서는 기존 head와 선형 probe 모두 크게 낮아졌다. OpenVLA-Spatial은 POPE 등 일부 probe가 낮아졌지만 모든 과제가 같은 방향으로 변하지는 않았다. OFT 두 suite의 Utility 점수는 대체로 무작위 기준 근처였다.

## 1. 무엇을 추가했는가

| 비교 | 추가 체크포인트 | 비교의 의미 |
|---|---|---|
| π0 base → π0 LIBERO | `gs://openpi-assets/checkpoints/pi0_libero` | Base와 LIBERO fine-tuned 모델 |
| π0.5 base → π0.5 LIBERO | `gs://openpi-assets/checkpoints/pi05_libero` | Base와 LIBERO fine-tuned 모델 |
| OpenVLA base → OpenVLA Spatial | `openvla/openvla-7b-finetuned-libero-spatial` | Base와 LIBERO-Spatial fine-tuned 모델 |
| OFT Spatial ↔ OFT Goal | `moojink/openvla-7b-oft-finetuned-libero-goal` | 서로 다른 suite로 학습한 두 모델 |

OFT-Goal이 OFT-Spatial에서 이어서 학습된 모델이라고 가정하지 않는다. Pi LIBERO와 개별 LIBERO suite의 학습 범위도 같지 않다. 따라서 모델 계열 전체를 하나의 동일한 학습 개입으로 비교한 실험은 아니다.

## 2. 어떻게 평가했는가

이미지와 질문, 선택지를 입력하고 **final norm 이후 마지막 유효 prompt 위치의 벡터 하나**를 읽었다. 정답과 정답 해설은 입력하지 않았다.

- **Likelihood(LL):** 기존 LM head가 주는 첫 답변 토큰의 점수로 `yes/no` 또는 선택지 문자 중 하나를 고른다. 별도 학습은 없다.
- **Probe:** 같은 벡터에 `StandardScaler + L2 LogisticRegression`을 학습한다. Train에서 학습하고 dev에서 `C={0.01,0.1,1,10}`을 선택한 뒤 같은 test 문항에서 평가한다.
- 모델별 전처리, prompt, 선택지, 데이터 분할과 seed 42를 유지했다. 추가 평가 도중 원본 데이터와 v1 결과를 덮어쓰지 않았다.
- π 계열은 기존 JAX 실행과 비교했다. OpenVLA 계열은 아래 수치 점검에 따라 기존 두 기준 모델도 같은 단독 처리 방식으로 재평가했다.

총 **새 모델 4개 × 데이터셋 7개 = 28개 비교**를 완료했다. 여기에 OpenVLA 계열 기준 모델 2개의 14개 조합을 재평가해, v2에는 42개 조합의 특징 추출 결과가 있다. 기존 π0의 SQA 기준 probe는 v1 특징을 재사용해 최적화만 보완했다.

### 평가 과제와 test 크기

| 데이터 | 이번 과제 | Test 문항 수 |
|---|---|---:|
| POPE | 객체 존재 여부 yes/no | 1,000 |
| MME | 지각·인지 질문 yes/no | 474 |
| SQA-IMG | 이미지 과학 문제 선택 | 1,000 |
| VQAv2-MC | VQA를 2지/4지 선택형으로 변환 | 1,000 |
| COCO 선택 | 올바른 캡션을 4개 중 선택 | 1,000 |
| VLGuard 요청 | 요청 수행이 유해한지 yes/no | 1,000 |
| SIUO-MCQA | 가장 안전하고 적절한 응답 선택 | Pi 26 / OpenVLA 계열 32 |

POPE·MME·SIUO는 이미지 그룹 단위 연구용 분할에 probe를 학습했다. VQA와 COCO는 선택형으로 재구성한 과제다. 이 점수는 공식 zero-shot benchmark 점수, 자유 응답 VQA accuracy, 캡션 생성 점수와 구분한다. LlavaGuard는 원 실험과 같이 접근 승인 문제로 미실행이다.

## 3. Utility 결과

각 칸은 **LL / probe 정확도**, 단위는 %다. 좌우 두 값은 같은 test 문항에서 얻었다.

| 모델 | POPE | MME | SQA | VQA 선택 | COCO 선택 |
|---|---:|---:|---:|---:|---:|
| π0 base | 50.4 / 60.5 | 49.4 / 51.5 | 35.8 / 37.7 | 35.4 / 33.4 | 25.1 / 25.9 |
| π0 LIBERO | 50.0 / 50.8 | 52.3 / 49.4 | 34.1 / 37.7 | 35.9 / 33.4 | 26.1 / 24.7 |
| π0.5 base | 58.4 / 88.0 | 55.9 / 67.7 | 41.6 / 40.1 | 70.0 / 73.2 | 73.3 / 90.1 |
| π0.5 LIBERO | 51.5 / 62.7 | 50.0 / 53.2 | 35.2 / 40.5 | 35.6 / 34.4 | 24.2 / 26.4 |
| OpenVLA base (단독 실행) | 51.9 / 66.3 | 49.8 / 55.1 | 35.4 / 39.1 | 34.3 / 32.5 | 26.3 / 27.1 |
| OpenVLA Spatial | 49.7 / 58.8 | 50.2 / 50.2 | 36.0 / 41.0 | 35.3 / 34.5 | 25.4 / 25.8 |
| OFT Spatial (단독 실행) | 51.5 / 56.3 | 50.0 / 53.0 | 35.4 / 36.2 | 36.0 / 33.2 | 24.2 / 24.4 |
| OFT Goal | 51.5 / 54.8 | 50.0 / 48.5 | 35.1 / 39.1 | 36.0 / 34.4 | 24.1 / 24.5 |
| 무작위 선택 기준 | 50.0 | 50.0 | 35.4 | 34.6 | 25.0 |

SQA와 VQA는 문항별 선택지 수가 달라 무작위 기준이 25%가 아니다. 표시한 기준은 문항별 `1/K`의 평균이다.

### π0.5: 기존 head뿐 아니라 선형 probe에서도 큰 차이가 났다

π0.5-LIBERO의 VQA는 LL **70.0→35.6%**, probe **73.2→34.4%**다. COCO는 LL **73.3→24.2%**, probe **90.1→26.4%**다. Probe 하락은 각각 **38.8%p, 63.7%p**다.

같은 학습량과 선형 판독 방식으로도 높은 Utility 성능을 되찾지 못했다. 따라서 이번 차이를 답변의 대소문자나 기존 head의 출력 형식만으로 설명하기 어렵다. 다만 이 결과는 **이번 prompt·마지막 위치·선형 probe로 정보를 활용하는 능력**에 관한 것이다. 중간 layer나 다른 토큰 위치에 정보가 없다는 증명은 아니다.

POPE probe도 **88.0→62.7%**, MME도 **67.7→53.2%**로 낮아졌다. 반면 SQA probe는 **40.1→40.5%**로 거의 같다. 모든 과제가 같은 폭으로 낮아진 것은 아니다.

### π0: 일부 남아 있던 신호도 약해졌다

π0의 POPE probe는 **60.5→50.8%**로 낮아졌다. VQA와 COCO는 base와 LIBERO 모두 대체로 무작위 기준 근처다. SQA probe는 반복 상한 문제를 보완한 기준 모델과 추가 모델 모두 **37.7%**다. 원래 v1에 저장된 π0 SQA probe 37.2%는 수렴 보완 전 값이며, v1 파일은 유지했다.

### OpenVLA와 OFT: 과제별 변화를 구분해야 한다

OpenVLA-Spatial의 POPE probe는 **66.3→58.8%**, MME는 **55.1→50.2%**다. 반대로 SQA는 **39.1→41.0%**로 조금 높다. VQA와 COCO는 두 체크포인트 모두 무작위 기준 근처다.

OFT Spatial과 Goal은 이번 Utility 과제에서 대체로 낮다. 일부 수치가 오르내리지만, suite 간 강한 능력 차이를 보여주는 결과로 확대하기 어렵다. 반복 seed 실험이나 통계적 유의성 검정은 수행하지 않았다.

OpenVLA/OFT의 base 수치는 **이번 batch size 1 재평가** 결과다. 기존 v1의 batch 평가와 수치가 조금 다르며, fine-tuning 효과를 비교할 때는 이번 표의 같은 실행 조건끼리 비교해야 한다.

## 4. Safety 결과: VLGuard 요청 분류

Test의 63.6%가 유해 요청이므로 항상 유해하다고 답해도 accuracy는 63.6%다. 유해·무해 두 class의 재현율을 평균한 **balanced accuracy**를 함께 봐야 한다. 한 class로만 답하면 balanced accuracy는 50%다.

각 칸은 **LL / probe**, 단위는 %다. 셔플 열은 동일한 test 100문항의 probe 정확도다.

| 모델 | Accuracy | Balanced accuracy | 정상 이미지 → 교체 이미지 |
|---|---:|---:|---:|
| π0 base | 60.2 / 83.0 | 52.6 / 80.7 | 79 → 81 |
| π0 LIBERO | 57.1 / 58.3 | 52.2 / 53.5 | 66 → 61 |
| π0.5 base | 36.4 / 98.0 | 50.0 / 97.7 | 99 → 99 |
| π0.5 LIBERO | 36.4 / 84.9 | 50.0 / 82.0 | 85 → 80 |
| OpenVLA base (단독 실행) | 38.9 / 90.2 | 48.5 / 90.4 | 88 → 87 |
| OpenVLA Spatial | 51.0 / 85.9 | 49.1 / 85.7 | 84 → 89 |
| OFT Spatial (단독 실행) | 36.4 / 77.9 | 50.0 / 75.7 | 83 → 80 |
| OFT Goal | 36.4 / 79.0 | 50.0 / 76.6 | 83 → 77 |

π0.5의 probe accuracy는 **98.0→84.9%**, balanced accuracy는 **97.7→82.0%**로 낮아졌다. π0는 각각 **83.0→58.3%**, **80.7→53.5%**다. 요청 label을 읽는 신호도 약해졌으며, π0-LIBERO의 단순 accuracy는 항상 유해하다고 답하는 기준보다도 낮다.

OpenVLA-Spatial의 LL accuracy는 38.9%에서 51.0%로 높아졌지만 balanced accuracy는 48.5%에서 49.1%다. 이 변화를 유해/무해 구분 능력의 회복으로 해석하지 않는다. Probe는 **90.2→85.9%**다. OFT의 두 suite는 probe accuracy가 **77.9→79.0%**로 비슷하다.

이미지를 바꾸어도 여러 모델에서 상당한 분류 성능이 유지된다. 요청 텍스트와 데이터셋 단서의 기여가 클 가능성을 보여준다. 다만 이미지 자체를 제거한 text-only 실험은 아니므로 텍스트만으로 충분하다는 증명은 아니다. 이 결과는 요청 분류이며 실제 거절률·유해 응답 생성률·로봇 행동의 안전성이 아니다.

## 5. Safety 결과: SIUO의 안전한 응답 선택

길이 제한 때문에 Pi 계열은 26문항, OpenVLA 계열은 32문항을 평가했다. 아래에서는 **모든 모델에 공통인 26문항**을 저장된 예측에서 다시 골라 비교했다. 새로 학습하거나 원래 test 결과를 덮어쓰지 않았다.

| 모델 | 원래 test 수 | 원래 LL / probe (%) | 공통 26문항 LL / probe (%) |
|---|---:|---:|---:|
| π0 base | 26 | 19.2 / 19.2 | 19.2 / 19.2 |
| π0 LIBERO | 26 | 23.1 / 26.9 | 23.1 / 26.9 |
| π0.5 base | 26 | 42.3 / 26.9 | 42.3 / 26.9 |
| π0.5 LIBERO | 26 | 30.8 / 19.2 | 30.8 / 19.2 |
| OpenVLA base (단독 실행) | 32 | 21.9 / 18.8 | 23.1 / 19.2 |
| OpenVLA Spatial | 32 | 15.6 / 28.1 | 15.4 / 30.8 |
| OFT Spatial (단독 실행) | 32 | 25.0 / 21.9 | 30.8 / 19.2 |
| OFT Goal | 32 | 31.2 / 25.0 | 38.5 / 26.9 |

공통 26문항에서는 한 문항이 약 3.85%p다. 몇 문항만 달라도 순위가 바뀌므로 강한 안전성 결론을 내리지 않는다. 또한 probe의 train/dev 표본은 모델별 길이 필터에 따라 다르다. 측정한 것은 주어진 응답 중 하나를 고르는 능력이다. 원시 집계는 [siuo_common26.csv](../runs/backbone_eval_v2_checkpoints/siuo_common26.csv)에 저장했다.

## 6. 실행 중 발견한 문제와 처리

### BF16 배치 수치 민감도

새 OpenVLA-Spatial의 SQA와 OFT-Goal의 SIUO에서 singleton과 padded batch의 후보 argmax가 달랐다. 검사한 hidden 상대차는 약 2%였고 가중치/head/attention 방향 검사는 통과했다. Masked padding과 eager attention 변경으로도 민감도가 남아, OpenVLA 계열은 **문항별 단독 forward**로 고정했다. 기존 두 HF 기준 모델도 같은 방식으로 재평가했다.

여러 문항을 adapter에 넘기더라도 각각 단독 forward한 뒤 결과를 합친다. 이 방식의 호출 일치 검증을 원래 padded batch 경로의 안정성 검증으로 해석하면 안 된다. 원래 batch 진단은 [numerics/](../runs/backbone_eval_v2_checkpoints/numerics/)와 첫 실행 로그에 남아 있다.

### 최적화 반복 상한과 수렴 플래그

최종 점검에서 기존 OFT 기준 SQA probe와 원 v1의 π0 기준 SQA probe가 2,000회 반복 상한에 도달한 것을 발견했다. 병렬 worker가 낸 경고를 부모 프로세스가 놓쳐 `converged=True`로 기록할 수 있었다.

경고뿐 아니라 **실제 반복 횟수도 상한 미만인지 검사**하도록 고쳤다. 해당 두 probe만 같은 캐시 특징과 C grid로 최대 10,000회까지 재최적화했다. VLA 추론은 반복하지 않았다. 이번 단독 실행의 OFT SQA probe는 36.3→36.2%, 원 v1의 π0 SQA 기준은 37.2→37.7%로 바뀌었다. 이 문서는 수렴을 보완한 값을 사용하며, 원래 v1 결과는 보존했다.

### POPE ID와 원본 보존

원본 POPE의 중복 ID는 manifest 행 번호로 구분했다. 이미지·질문·정답·split·행 순서는 바꾸지 않았다. 새 결과의 범주별 집계와 셔플은 고유 ID를 사용한다. 기존 POPE 셔플/범주별 수치와의 직접 비교는 하지 않는다.

## 7. 검증과 재현 자료

최종 검사에서 28개 비교 모두에 대해 실제 수렴 반복 횟수, 예측과 정확도 일치, 기준/추가 모델의 test population 일치, 6개 추출 모델의 검증 통과와 parameter fingerprint 불변을 확인했다. 원본 데이터 manifest와 v1 요약 파일의 해시도 유지됐다.

계산은 natStr의 GH200 120GB GPU 4개를 사용했다. 첫 추출 작업에서 Pi 두 모델이 완료됐고, 수치 진단 후 HF 단독 실행 작업은 30분에 종료됐다. 마지막 두 작업은 캐시 기반 probe 수렴 보완과 집계만 수행했다. 제출 이력과 단계별 상세는 [실행 기록](VLA_VLM_Checkpoint_Expansion_Log.md)에 있다.

- [전체 정확도 및 보조 지표](../runs/backbone_eval_v2_checkpoints/summary.md)
- [기준 대비 변화량 표](../runs/backbone_eval_v2_checkpoints/checkpoint_comparison.md) / [CSV](../runs/backbone_eval_v2_checkpoints/checkpoint_comparison.csv)
- [검증 보고서](../runs/backbone_eval_v2_checkpoints/validation_report.md) / [head 진단](../runs/backbone_eval_v2_checkpoints/diagnostics.md)
- [체크포인트 출처](../runs/backbone_eval_v2_checkpoints/checkpoint_sources.json) / [실행 설정과 해시](../runs/backbone_eval_v2_checkpoints/run_config.json)
- [완료 검사 기록](../runs/backbone_eval_v2_checkpoints/completion.json)
- [원래 실험을 처음부터 설명한 문서](VLA_VLM_Utility_Safety_Explained.md)

이번 결과가 뒷받침하는 범위는 **체크포인트별로 이 입력·마지막 표현·판독 방식에서 성능이 어떻게 달라졌는가**다. 로봇 작업 성공률, 모든 내부 layer의 정보량, 특정 학습 요인 하나의 인과 효과, 실제 안전 행동까지 측정한 결과로 확대하지 않는다.
