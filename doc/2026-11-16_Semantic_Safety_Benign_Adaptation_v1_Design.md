# 안전한 물체·조작 적응 후 Linear Probe 실험 설계

## 1. 확인하려는 질문

**새 물체와 조작을 안전한 과제로 익히면, π0.5 내부에서 선형적으로 읽히는 안전·위험 정보는 유지·강화·약화되는가?**

출발점은 공식 LIBERO FT π0.5다. VLM-only, AE-only, Both 세 LoRA 모델을 별도로 학습한다. 비교 기준은 최근 첫 장면 probe 실험의 공식 LIBERO FT 모델이며, π0.5 base 결과도 참고값으로 제시한다.

첫 장면 probe 자체에 작업 수행 성공이 필수인 것은 아니다. 이번 FT는 새로운 물체·조작 적응의 영향을 보는 추가 비교 조건이다.

이전 FT도 안전·위험 정답을 가르친 분류 학습은 아니었고 action 모방 학습이었다. 이번에는 평가에 사용하는 위험 문맥과 목표 조합을 정책의 시연 분포에서 분리한다. 과거 성능 저하를 “representation이 엉켰기 때문”이라고 단정하지 않는다. 일반 모델에서 FT 후 사전학습 feature의 변화와 분포 밖 성능 저하를 관찰한 연구는 있지만, π0.5의 LoRA에서도 같은 원인인지 별도 검증이 필요하다. [Kumar et al., ICLR 2022](https://openreview.net/forum?id=UYneFzXSJWh)

## 2. 시연 과제 구성

물리 검증을 통과한 **69종 물체, 89개 benign 목표 조합**을 사용한다. 과제 정의와 성공 시연을 따로 검증한 후 학습 manifest를 확정한다.

| 동작 | 예시 | 검증할 실제 움직임 |
|---|---|---|
| place | 칼을 새 공구 정리 매트로 옮기기 | 잡기→들기→운반→접촉→gripper 열기 |
| place | 신선한 빵을 새 식품 준비 매트로 옮기기 | 같은 기준; 사람에게 제공하는 문맥 제외 |
| wipe | 면 패드로 세척 연습용 보드를 닦기 | 잡고 운반한 뒤 접촉하면서 3 cm 이상 수평 이동 |
| cut | 칼로 교체 가능한 고무 연습 시트에 절단 동작 | 칼의 접촉·긋기; 실제 절단 효과는 구현하지 않음 |
| sand / saw / grate | 교체 가능한 공작용 연습 재료에 작업 동작 | 접촉·긋기; 재료 제거·변형은 구현하지 않음 |
| cover | 천·수건·종이로 작은 빈 연습 프레임 덮기 | 개구부와 충분히 겹쳐 놓고 gripper 열기 |
| press | 전원이 연결되지 않은 연습 패드 누르기 | 잡고 운반한 뒤 일정 시간 접촉 |
| drill | 드릴로 교체 가능한 석고 연습 타일에 작업 동작 | 도구의 접촉 유지; 실제 구멍 생성은 구현하지 않음 |
| pour | 물병·차가운 주전자를 새 계량 용기 위에서 기울이기 | 잡고 운반한 뒤 용기 위에서 50도 이상 기울임; 액체 이동은 구현하지 않음 |

같은 물체를 운반하는 시연만으로 cut/wipe/cover를 배웠다고 주장하지 않는다. 각 동작의 시연과 채점 기준을 따로 둔다. 시뮬레이터의 rigid asset으로 실제 절단·세척·천 변형·액체 이동을 재현한 것으로 해석하지 않는다.

종이로 프레임을 덮는 후보 1개는 물리 시연의 안정성이 부족해 학습 과제에서 제외했다. 종이의 이동, 천·수건의 덮기 과제는 유지한다. 69종 물체와 원래 안전·위험 probe 조건은 모두 유지한다.

## 3. 기존 probe 데이터와의 관계

- 물체 어휘와 시각 asset은 의도적으로 공유한다. 이것이 적응시키려는 대상이다.
- 기존 probe 조건의 동일한 전체 지시문과 물체–대상 identity 조합은 재사용하지 않는다. 단순한 위치·표현 변화만으로 별도 의미 조건이라고 세지 않는다.
- 새 정리 매트, 세척 보드, 고무 연습 시트, 빈 연습 프레임 등 목적지·작업 표면을 새로 만든다.
- 기존 benchmark의 안전 항목을 그대로 추출해 정책 FT train set으로 사용하지 않는다.
- 인체·영아 위해, 생물 대상 오염, 변질 식품 제공, 통전·가열 손상 시연을 넣지 않는다. 신선한 식품과 차갑거나 전원이 꺼진 변형을 사용한다.
- 안전 판단 라벨·이유·특수 safety token은 정책의 입력·학습 목표에 넣지 않는다.

물체와 기술에 노출된 뒤 새로운 위험 문맥에서 평가하는 실험이다. 미관측 물체에 대한 일반화나, 모든 의미적 유사성이 제거되었다고 주장하지 않는다.

## 4. 시연 생성과 검증

Scripted teacher가 시뮬레이터의 물체 pose를 읽어 Panda의 연속 action을 생성한다. 학생 정책은 두 카메라 RGB, robot state, 작업 지시만 받는다.

Robot state는 pipeline에 전달되지만 이 LIBERO π0.5 설정에는 별도 state token이 없다. π0.5의 AE에도 별도 state embedding을 추가하지 않는다. 물체 pose는 학생의 입력에 포함하지 않는다.

1. 각 과제에서 성공하는 grasp profile과 동작을 찾는다.
2. 실제 `env.step`으로 시연을 기록한다. 기록 중 물체 teleport는 허용하지 않는다.
3. 별도 환경에서 action을 다시 재생해 상태와 성공 기준을 검증한다.
4. 관측 `t`와 action `t`의 정렬, 유한한 state/action, 카메라 형태, 충분한 운반과 접촉을 검사한다.
5. train/validation/test 초기 배치 seed를 분리한다. 정책 평가는 teacher 성공 여부로 선별하지 않는다.

운반은 실제 손가락 grasp와 lift를 확인한다. 얇은 종이처럼 순간적으로 한 손가락 접촉이 끊기는 경우에는, 닫힌 gripper와 물체의 상대 위치가 2 cm 이내로 유지되면서 들어 올려져 12 cm 이상 함께 이동하는지도 검사한다. 단순한 목적지 도착만으로 운반 성공을 인정하지 않는다.

과제당 train 시연 32개를 목표로 한다. 초기 배치는 validation/test에 각각 2개씩 둔다. Grasp 탐색과 물리 조정은 모델 평가 전에 끝내고 검증 후 정의를 동결한다.

정적 관측용 asset의 동적 사용에는 별도 물리 설정이 필요하다. 칼은 시각 mesh를 유지하면서 carrier 질량·접촉 강성과 받침을 조정한다. 새 환경은 gripper/물체/연습 표면의 마찰을 각각 2.0/0.3/0.2로 명시한다. 네 비교 정책과 teacher가 같은 설정을 사용한다. 원본 probe 장면은 수정하지 않는다.

천 공·천 책·놀이 블록은 Panda gripper의 약 8 cm opening에 맞춘 학습용 크기 변형을 사용한다(원래 크기의 0.70/0.60/0.75배). 같은 시각적 형태를 축소한 것으로 원본 probe asset의 크기는 보존한다. 칼의 회전각 단위는 LIBERO와 동일한 radians로 명시하고 동적 배치의 바닥 높이를 다시 정렬한다.

## 5. 세 모델의 동일 학습 조건

| 항목 | 설정 |
|---|---|
| 학습 범위 | VLM-only / AE-only / Both |
| VLM | PaliGemma의 vision·language·multimodal linear 모듈; 생성 head 제외 |
| AE | expert Transformer와 action·time projection |
| LoRA | rank 16, alpha 32, dropout 0; 원래 가중치 freeze |
| 손실 | action flow-matching MSE |
| Optimizer | AdamW, weight decay 0.01, gradient clip 1 |
| Learning rate | peak 2e-4, warmup 300 updates, cosine 감소 |
| 유효 batch | 32 = microbatch 16 × accumulation 2 |
| 학습 길이 | 각 16,000 updates, 4k/8k/12k/16k 저장 |
| Rehearsal | minibatch 샘플의 10% LIBERO; 나머지 90% 새 시연 |
| 새 시연 내부 비율 | 50% 이동 / 50% 나머지 verb를 균등하게 뽑은 뒤 과제·episode·시점을 샘플링 |
| Action chunk | 10 control timesteps |
| 추가 학습 | 자동 8,000-update 연장 없음 |

LIBERO rehearsal은 이전에 별도 replay 검증한 공식 LIBERO 정책 성공 시연 154개를 재사용한다. 네 suite의 40개 task, init state 2–5에 해당한다. 에피소드 개수를 90:10으로 맞춘다는 뜻은 아니다.

새 시연의 RGB는 기존 첫 장면 probe와 같은 upright convention을 쓴다. LIBERO rehearsal과 유지 평가는 이전 공식 평가 pipeline의 180도 convention을 유지한다. 이 차이는 데이터별로 명시적으로 처리한다.

## 6. 체크포인트 선택과 평가

체크포인트는 사전에 고른 benign motor validation과 LIBERO validation으로 선택한다. LIBERO validation은 init state 7을 사용한다. 기존 안전·위험 probe의 test AUROC를 checkpoint 선택에 사용하지 않는다.

이후 네 정책(공식 LIBERO 및 세 적응 모델)을 별도 benign motor test 초기 배치와 LIBERO 80개(4 suites × 10 tasks × 2 초기 상태)에서 평가한다. LIBERO 80개 결과는 전체 공식 benchmark와 구분한다.

## 7. Linear Probe의 비교 조건

- 기존 첫 장면 A 336개/B 121개, 이미지·지시·state, 연결 group과 5-fold 분할을 그대로 사용한다.
- VLM/AE 0–17 layer residual 및 AE flow 1/5/10의 velocity projection 전후, 총 78개 위치를 추출한다.
- 추출 시 가중치를 freeze하고 기존과 같은 noise·10회 denoising·token 평균 pooling을 사용한다. 환경 행동은 실행하지 않는다.
- 각 모델마다 scaler와 linear probe를 새로 학습한다. 이전 probe를 그대로 옮기면 representation의 좌표 변화까지 영향을 주므로 이번 주 비교와 구분한다.
- 기존 weighting, C=0.01/0.1/1의 validation 선택, logit≥0 판정 기준을 유지한다.
- AUROC, balanced accuracy, accuracy, confusion matrix, 위험 판정 비율을 함께 보고한다.
- AE-only에서는 동일 입력의 VLM feature가 공식 LIBERO 모델과 정확히 같아야 한다. 18개 layer 전체에서 이를 검사한다.

BF16에서는 batch 크기가 달라져도 수치 반올림 차이가 발생할 수 있다. 본 추출은 기존 reference와 동일한 순서·batch 12·마지막 batch 크기를 사용한다. 16개 smoke run에서는 같은 batch 형태인 첫 12개를 bit-exact 비교하고, 본 추출에서는 457개 전체를 비교한다.

정책의 조작 성능과 probe 분류 성능은 별도로 해석한다. 높은 probe 성능은 위험 관련 라벨 정보가 선형적으로 읽힌다는 근거다. 위험 경고의 인과적 전달이나 행동에서의 사용을 입증하지 않는다. benign FT에서도 feature는 달라질 수 있고, 낮은 probe 성능만으로 위험 지식 자체가 사라졌다고 결론 내릴 수 없다.

## 8. 파일과 실행

- 새 시연: `data/semantic_benign_adaptation_v1/`
- 진행·체크포인트·probe: `runs/semantic_benign_adaptation_v1/`
- 최종 adapter: `checkpoints/pi05_libero_benign_adaptation_v1/{ae,vlm,all}/adapter.pt`
- 최종 보고서: `doc/2026-11-17_Semantic_Safety_Benign_Adaptation_v1_Results.md`
- 기존 결과와 데이터는 보존한다. 중복 실행은 lock으로 차단하고 학습 복구 상태를 저장한다.

```bash
openpi/.venv/bin/python scripts/semantic_safety/run_benign_adaptation_v1.py --supervise
```

이 문서는 설계 설명이며 결과 완료를 뜻하지 않는다. 실제 데이터 규모와 성공률은 최종 검증 기록과 결과 보고서에서 확정한다.
