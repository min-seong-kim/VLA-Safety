# 확장 v5 π0.5 LoRA 적응

상태: **세 모델 학습·checkpoint 선택·최종 평가 완료**.

## 결과

| 모델 | 새 배치 모터 proxy | Native LIBERO | 양쪽 성공 대조쌍 | 선택 step |
|---|---:|---:|---:|---:|
| 기본 π0.5 LIBERO | 122/676 (18.0%) | 80/80 (100.0%) | 28/338 | 0 |
| AE-only LoRA | 296/676 (43.8%) | 78/80 (97.5%) | 98/338 | 16,000 |
| VLM-only LoRA | 468/676 (69.2%) | 78/80 (97.5%) | 199/338 | 16,000 |
| 공동 LoRA | 480/676 (71.0%) | 78/80 (97.5%) | 189/338 | 16,000 |

모터 성공은 실제 집기와 3.5 cm lift, 연속으로 잡고 이동한 XY 경로 12 cm 이상 및 순변위 10 cm 이상을 먼저 확인한 뒤,
도구를 잡은 상태의 접촉·수신자 옆 전달·목적지 위 50도 이상 기울이기 중 해당 조건을 연속 충족한 것이다.
놓기·제공·덮기·채우기 지시는 검증된 운반 후 목적지와 접촉하면 놓은 상태도 인정한다. 도구 사용 지시는 마지막까지 잡은 상태를 요구한다.
전자레인지는 원본 LIBERO Long 환경에서 전면 접근 또는 접촉을 확인한다. 실제 절단·문 닫기·가열·액체 이동·위해 발생의 성공률이 아니다.

## 시연 생성과 검증

- v5의 **169개 상황, 338개 안전/위험 조건**을 유지했다. 물리 teacher가 물체 pose를 이용해 실제 `env.step` action을 생성했다.
- 신규 학습 시연 **10,816개 / 2,567,389 control steps**, 시연 확보 조건 338/338. 미확보 조건: 없음.
- 채택한 모든 시연을 별도 환경에서 action 단위로 재생하고 상태 및 성공을 검증했다. 기록 중 물체를 순간 이동시키지 않았다. 불안정한 물리 rollout은 채택하지 않았다.
- 위치 변화에 취약한 집기는 학습 배치에서 추가로 자세를 탐색했다(192개 시연). 물리 환경·controller·채점 기준은 유지했으며 이전 실패 기록도 보존했다. 평가 배치는 이 탐색에 사용하지 않았다.
- 수집 완료 후 모든 파일 hash, RGB·state·action의 시간 정렬, 유효한 action 범위, 관찰 shape·dtype, episode별 시작·중간·종료 카메라 관찰을 검사했다. `trajectory_quality_audit.jsonl`에 결과를 기록하고 실제 학습 배치의 표본 관찰을 `training_quality_review.html`에 제공한다. 전체 영상을 사람이 검수했다는 뜻은 아니다.
- 칼은 세운 상태로 절삭면을 아래로 수정했다. 이동 물체에 free joint를 추가하고 목적지는 고정했다. 정적 자산의 관성 보완, 보이는 집기 받침, 장갑 scan의 충돌 형상 분할, 작은 유리 조각의 12 g 질량 설정은 학습 환경 변환 기록에 남겼다.
- 모델에는 공식 π0.5 입력 경로의 RGB 두 시점과 원래 v5 지시를 사용했다. privileged pose, 안전 label, 근거 설명은 모델에 전달하지 않았다.

## LIBERO 재학습

기본 π0.5의 성공 rollout **154개**, 40개 task를 독립 재생 검증해 사용했다.
Spatial/Object/Goal/Long의 초기 상태 2–5에서 수집했다. 이는 원본 사람 시연 파일을 재사용한 것이 아니라 공식 모델의 성공 시연이다.
전체 **학습 샘플의 10%**를 이 데이터에서 추출한다. 시연 개수의 10%라는 뜻은 아니다. 측정된 첫 학습 단계의 비율은 10.04%이다.

## 학습과 선택

공식 LIBERO FT π0.5에서 세 조건을 각각 시작했다. 원본 가중치는 모두 고정하고 VLM, AE, 두 모듈의 LoRA만 각각 학습했다.
Rank 16, alpha 32, dropout 0, AdamW, 최대 LR 2e-4, 유효 batch 32이다. GPU 메모리를 위해 microbatch 16 두 번의 gradient를 누적한다.
동일 manifest, 예제·flow noise seed, optimizer update 수를 사용했다. 모듈별 학습 파라미터 수는 달라 파라미터 수를 맞춘 비교는 아니다.
복구용 optimizer·LoRA 상태를 200 updates마다 저장한다. 중단 시 저장된 단계부터 동일 예제와 noise로 재개하며 미저장 구간의 중복 로그는 별도로 보존한다.
VLM 마지막 층의 일부 LoRA는 AE에 전달되는 KV 이후의 연산에 있어 action loss의 gradient가 없다. 실제 gradient와 해당 tensor 목록을 감사 파일에 기록했다.

초기 16,000 updates의 4k/8k/12k/16k를 비교했다. 사용자의 시간 제약에 따라 8,000-update 추가 학습은 보류했다. 이 변경은 최종 test 실행 전에 기록했으며, 각 모델의 16,000-step 이하 체크포인트에서 선택했다.
checkpoint는 native validation 성능 유지 조건을 우선 적용하고 모터 성공 수로 선택했다. 선택 결과를 고정한 후 test를 실행했다.
추가 학습 여부와 모든 후보 점수는 `extension_decision.json`, `checkpoint_candidates.json`, `frozen_selection.json`에 있다.

## 평가 집합과 해석 범위

- 모터 selection: 172조건. 나머지 validation 166조건은 선택 후 확인용이다.
- 모터 test: **676 episode = 169상황 × 안전/위험 × 새로운 배치 2개**. 동일 물체·상황의 새 배치이며 미관측 위험 종류의 평가가 아니다.
- Native validation: 4개 suite × 10 task × 초기 상태 7번 = 40 episode.
- Native test: 4개 suite × 10 task × 초기 상태 0–1번 = 80 episode. 모든 LIBERO 초기 상태를 평가한 공식 benchmark 점수로 일반화하지 않는다.
- train과 추가 학습의 평가 배치는 분리했다. 공식 checkpoint의 과거 학습에서도 미관측이었다는 뜻은 아니다.
- 평가 조건은 teacher나 모델의 성공 여부로 걸러내지 않았다. 독립 seed의 반복 학습 분산은 측정하지 않았다.

이 실험은 지시를 수행할 수 있는 모터 능력을 적응시키는 단계다. 안전/위험 양쪽 행동을 학습했으며 거부 능력이나 위험 인식 자체를 평가한 결과가 아니다.
representation 비교에는 원본 모델 대조, 고정 관찰 입력 재생, 상황 단위 probe split을 함께 고려해야 한다.
`representation_candidate_pairs.json`은 성공한 쌍의 경로 목록이다. 이후 완료한 공통 입력 기반 분석은
[안전·위험 representation Linear probe 결과와 해석](2026-11-13_Semantic_Safety_Representation_Linear_v1_Results.md)에 정리했다.

## 산출물

- 학습 데이터: `data/semantic_expansion_v5_lora_v1/combined_train_manifest.jsonl`, `condition_coverage.csv`, `training_data_audit.json`.
- 실제 학습 시연 검토: `data/semantic_expansion_v5_lora_v1/training_quality_review.html`.
- adapter: `checkpoints/pi05_libero_expansion_v5_lora_v1/{ae,vlm,all}/adapter.pt`.
- 결과와 영상: `runs/semantic_expansion_v5_lora_v1/final/`, `comparison_summary.csv`, `skill_suite_summary.csv`, `review.html`.
- 학습 곡선: `learning_curves.png`. 검사: `training_comparison_audit.json`.

원본 v4와 기존 v6 실험·모델은 유지했다.
