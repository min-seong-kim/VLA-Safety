# Semantic safety rollout v3 — 기록과 사용법

이 문서는 원본 네 모델의 초기 v3 실행 기록이다. 이후 중립 조작 시연·π0.5 AE LoRA 학습과 새 A/B 평가는 [통합 연구 인계 문서](Semantic_Safety_Research_Handoff.md)를 참고한다. 두 단계의 실행 간격과 grasp/lift 기준을 혼용하지 않는다.

## 완료 후 품질 검토

288개 에피소드 / 46,080 control timestep의 실행, image/state/action 정렬 검사, finite feature 검사, 6종의 t=0 probe 행렬 추출이 완료되었다. 36개 장면 모두 안전·위험 지시와 네 모델의 초기 image/state hash가 일치했다. 이 완료 표시는 기록 파이프라인의 완료를 뜻한다.

**새 장면에서 네 모델 모두 지원되는 task proxy 성공은 0/32, 위험 proxy 검출은 0/72이다.** 모델별 나머지 40개 에피소드는 완전한 task 성공을 지원하지 않는 물리 동작이어서 null이다. 선택된 조작 대상의 grasp가 관측된 에피소드는 π0 4개, π0.5 16개, OpenVLA 1개, OFT 0개였다. 이러한 결과를 안전 거부로 해석할 수 없다. 환경의 조작 가능성, 새로운 물체·지시에 대한 일반화, 실행 horizon 등의 원인을 아직 분리하지 않았다.

후속 정지 제어 검사에서 `known_allergen_serving`의 `rice_cake_1`이 모델 action 없이 8초 동안 **4.245 m 이동**했다. 원래 10-step 초기 안정화 검사만으로 잡히지 않은 물리적 불안정성이다. 나머지 35개 장면은 같은 검사에서 movable object의 이동이 3 cm 이하였지만, 이것이 task 수행 가능성을 보증하지는 않는다.

원본 파일은 보존하고, 이 family의 두 지시와 네 모델을 모두 제외한 별도 자료를 준비했다.

- `runs/semantic_rollouts_v3/probe_matrices_passive_stability_filtered/`: 모델별 t=0 70표본(35 safe / 35 unsafe), VLM 4종과 π AE 2종. 기존 fold 0의 partition을 유지해 48/12/10표본이다.
- `runs/semantic_rollouts_v3/probe_index_passive_stability_filtered.jsonl.gz`: 44,800 timestep의 원본 HDF5 참조.
- `runs/semantic_rollouts_v3/quality_review.json`: 품질 상태와 제외 사유.
- `runs/semantic_rollouts_v3/behavior_diagnostics.json`: episode별 움직임·grasp 진단.
- `runs/semantic_rollout_physics_checks_v3/`: 36개 정지 제어 검사 결과. 모델 rollout/probe 표본에 포함하지 않는다.

현재 자료는 검토를 거쳐 입력 representation의 탐색적 probe에 사용할 수 있다. 행동 safety benchmark나 위험 지시 거부율로 해석하기 전에는 환경별 benign task 수행 가능성을 추가 검증해야 한다. 아래 원래 72표본/48·12·12 split 설명은 보존된 필터링 전 데이터에 해당한다.

## 실행 범위

36개 semantic family에서 대표 장면을 하나씩 구현했다. 각 장면의 동일한 초기 관측에 명시적 safe/unsafe 지시를 각각 적용한다. 총 72개 지시를 공식 π0 LIBERO, π0.5 LIBERO, OpenVLA LIBERO-Spatial, OpenVLA-OFT LIBERO-Spatial 체크포인트로 실행하는 288개 에피소드이다. 기존 v2의 108개 언어 blueprint 전체를 구현한 것은 아니다. 대응하는 원래 case는 `data/semantic_rollout_v3/scenes.jsonl`의 `source_pair_id`에 있다.

카메라/배치 변형을 추가 사례로 세지 않았다. 장면·라벨은 합성 후보이며 독립적인 사람 검토를 받지 않았다. 열, 화학물질 내용물, 전원, 알레르기, 물건을 보존해야 하는 의도 등 영상만으로 확정할 수 없는 조건은 모델 입력 지시문에 명시했다. 따라서 이 데이터에서 분류가 가능해져도 **시각적 grounding이나 위험에 대한 인과적 이해가 입증되는 것은 아니다**.

장면은 canonical LIBERO의 Panda, 카메라, 20 Hz OSC_POSE 제어를 사용한다. LIBERO-Safety/HazardArena의 일부 mesh와 프로젝트에서 만든 rigid geometry를 연결했다. 물체 원본·크기·질량·변환·proxy 여부는 `data/semantic_rollout_v3/assets/manifest.json`에 있다. 온도·액체 입자·독성·연소·전기 회로·절단·파손은 직접 시뮬레이션하지 않는다.

## 현재 상태와 열기

- 실행 상태: `runs/semantic_rollouts_v3/status.json`
- 장면/영상/timestep별 action 뷰어: `runs/semantic_rollouts_v3/preview.html`
- 다운로드용 뷰어와 H.264 영상 ZIP: `runs/semantic_rollouts_v3/preview_bundle.zip`
- 완료 후 종합 결과: `runs/semantic_rollouts_v3/completion_report.json`
- 검사 결과: `runs/semantic_rollouts_v3/audit_summary.json`
- 실행 PID와 재시작 이력: `runs/semantic_rollouts_v3/launch.json`
- 자동 실행·검사 로그: `runs/semantic_rollouts_v3/supervisor.log`, `postprocess.log`

HTML만 내려받아도 이미지와 표는 열린다. 영상까지 보려면 ZIP 전체를 압축 해제하고 그 안의 `preview.html`을 연다. HTTP 서버가 필요하지 않다. ZIP에는 용량이 큰 HDF5 원본은 포함하지 않는다. 서버의 HTML과 ZIP은 실행 중 갱신되는 스냅샷이며, 다운로드한 사본은 자동 갱신되지 않는다.

백그라운드 supervisor가 기록된 이 배치의 프로세스만 감시한다. 실행 오류가 나면 불완전한 에피소드를 `incomplete_attempts/`에 보존하고 초기 상태부터 재시도한다. 모델마다 최대 3번 재시작하며, 이후 오류가 남으면 `needs_attention`으로 기록한다. 모든 실행·검사·행렬 추출이 성공해야 `complete`가 된다. 최종 성공 여부는 이 상태 파일로 확인해야 한다.

## timestep의 정의

각 에피소드는 10개의 초기 안정화 control step 뒤부터 기록한다. 본 실행은 최대 160개의 policy/control timestep, 즉 시뮬레이션 시간 8초이다. 안정화 step은 모델을 호출하지 않으므로 probe 표본에 포함하지 않는다.

```
trajectory.observations[t]
    → 모델의 실제 입력 전처리
    → features[t], predicted_action_chunk[t]
    → chunk[0]을 실행
    → trajectory.observations[t+1], events[t]
```

**모든 control timestep에 모델을 새로 호출한다.** π의 chunk 길이는 π0=50, π0.5=10이며 OFT=8이지만, 이 실행에서는 첫 action만 실행하는 `replan_steps=1`을 사용했다. 공식 기본 π 재계획 주기나 OFT chunk 실행과 다른 실험 설정이다. 공식 논문의 benchmark 점수를 재현한 결과로 비교하면 안 된다. 모든 모델에 동일한 timestep별 재계획 규칙을 적용했고, 나머지 chunk도 보존했다.

같은 장면의 두 지시와 네 모델은 동일한 scene seed를 사용한다. π의 flow noise seed도 같은 장면의 안전/위험 쌍에서 timestep마다 일치한다. 사후 검사에서 초기 image/state hash의 일치를 확인한다. 이후 궤적은 각 모델의 action에 따라 달라진다.

## 에피소드 파일

경로: `runs/semantic_rollouts_v3/<model>/episodes/<family>__<safe|unsafe>__seed0/`

| 파일 | 내용 |
|---|---|
| `episode.json` | 실제 instruction, 정적 안전 라벨, family/mechanism, seed, 초기 해시, 물체 가시 픽셀 수, 카메라/물체 ID, 장면 설정 |
| `model.xml` | 실행한 MuJoCo 모델과 asset 참조 |
| `trajectory.h5` | T+1개의 두 카메라 RGB·8D state·전체 sim state·qpos/qvel·body pose, T개의 예측 chunk·실행 action |
| `features.h5` | T개의 모델 입력과 층별 표현, t=0 전체 token 표현 |
| `events.jsonl` | action 이후 위험 proxy·task proxy·물체 관계·접촉점·추론 시간 |
| `views.mp4`, `views_h264.mp4` | agentview와 wrist를 좌우로 붙인 영상; 후자가 브라우저용 |
| `summary.json` | 완료 timestep 수, horizon/종료 사유, proxy 집계 |
| `audit.json` | timestep 정렬, hash, action/state 일치, finite 값 검사 |

관측의 `main_rgb_raw`와 `wrist_rgb_raw`는 시뮬레이터의 원본 방향이다. 공식 모델 전처리에서 두 축을 뒤집는다. 뷰어 영상은 뒤집은 방향이다. 원본 8D state는 world EEF xyz(3), 공식 quaternion→axis-angle(3), gripper qpos(2) 순서다.

## 표현 파일 구조

`features.h5/pooled/<feature>`의 축은 다음과 같다.

```
[control_timestep, forward_call, layer, batch_or_camera, region, hidden_dimension]
```

| feature | 내용 |
|---|---|
| `vlm_hidden` | 모든 LLM transformer block 출력 |
| `vlm_hidden_input` | 첫 transformer block 입력 |
| `vlm_k_proj`, `vlm_v_proj` | 각 block의 K/V projection; RoPE 및 attention cache 적용 전 |
| `vision_*_hidden` | 모든 vision block 출력의 token 평균 |
| `action_expert_hidden`, `action_expert_k_proj`, `action_expert_v_proj` | π 계열 Action Expert의 모든 층과 10개 flow iteration |
| `autoregressive_*` | OpenVLA의 생성 중 decoder 표현; prefill이 첫 action token을 예측하고 이후 6개 호출이 나머지를 예측 |
| `action_head_hidden` | OFT MLP action head의 residual block 표현 |

`forward_call`은 control timestep이 아니다. π AE에서는 10번의 denoising 호출, π vision에서는 main/wrist/padded image의 세 호출, OFT vision에서는 두 카메라 호출이다. layer 출력은 block 출력이며 최종 모델 norm 이후 출력과 동일하다고 가정하면 안 된다. region 이름과 실제 token 구간은 **각 step의 `step_metadata`**에 있다.

π VLM region은 main image, wrist image, 유효 prompt, 마지막 prompt token이다. AE는 action token 평균과 첫 action token을 저장한다. OpenVLA/OFT는 각 구조에 맞춰 prompt/image/action readout 영역을 저장한다. OpenVLA의 없는 wrist 영역은 zero placeholder이며 학습에 사용하면 안 된다. vision 평균에는 해당 encoder가 출력한 모든 token이 포함된다.

Pooling은 float32로 계산하고 float16으로 저장한다. **전체 token 단위 tensor는 `initial_full_tokens/`에 t=0만 저장한다.** 매 timestep 전체 token tensor를 보관한 것은 아니다. 전체 timestep의 정확한 이미지·state·instruction·noise와 입력 전처리 결과가 남아 있으므로 필요한 timestep을 다시 추출할 수 있다.

`input/`에는 token IDs/masks, normalized state(사용하는 경우), 224 RGB, 실제 VLA pixel tensor의 bfloat16 bit pattern, 실제 embedding에 들어간 token IDs/LLM attention masks, π image masks와 flow noise가 있다. `output/`에는 예측 chunk, simulator에 보낸 action, OpenVLA 생성 token IDs가 있다. π image 순서는 main, left wrist, masked right wrist이다.

현재 공식 OpenPI `pi05_libero` 설정은 `discrete_state_input=False`이고 π0.5 AE에 π0의 state token을 넣지 않는다. 이 체크포인트에서 state를 저장한다는 사실과 모델이 그 state를 입력으로 사용한다는 사실을 구분해야 한다. OpenVLA는 main image만 사용하며, OFT는 main+wrist+proprio를 사용한다. 각 모델의 실제 입력 구성은 `model.json`에 기록한다.

## probe에 사용하기

먼저 t=0으로 비교하는 것이 적절하다. 이때 모델과 두 지시가 같은 초기 장면을 본다. t>0에서는 모델 행동의 결과로 관측이 달라지므로 행동·진행 시간의 분포를 학습하는 효과가 섞인다. timestep을 섞어 무작위 train/test split을 만들면 같은 장면이 양쪽에 들어간다.

`probe_splits.json`은 이전 mechanism holdout을 36개 실제 장면에 매핑한 6-fold 구성이다. 각 fold에 train 24 / validation 6 / test 6개 family가 있으며, 양쪽 지시·모든 모델·모든 timestep은 family 단위로 함께 이동한다. t=0이면 모델마다 72표본, fold마다 48/12/12표본이다. 수만 timestep을 수만 개의 독립적인 안전 상황으로 세면 안 된다.

완료 후 supervisor는 모델별 VLM 마지막 prompt 표현 4개와 π AE 첫 action token 표현 2개를 `probe_matrices/`에 추출한다. 추가 추출 예:

```bash
envs/libero-render/bin/python scripts/semantic_safety/export_probe_matrix.py \
  --model pi05_libero --feature vlm_hidden --region last_prompt \
  --layer -1 --timesteps initial --fold 0 \
  --out runs/probe_pi05_last_prompt.npz
```

모든 timestep을 추출하려면 `--timesteps all`을 사용한다. 이 경우에도 저장된 partition과 family grouping을 유지한다. 스케일러·PCA·LDA·linear probe는 train에만 fit하고 validation에서 hyperparameter를 선택해야 한다. t-SNE/LDA 분리나 linear probe 성능은 decodability의 증거이며, 경고의 전달/사용을 주장하려면 intervention이나 causal ablation이 추가로 필요하다.

`probe_index.jsonl.gz`에는 각 timestep의 HDF5 경로, row, 정적 `task_is_unsafe`, action 전/후의 기하학적 위험 proxy를 분리해 저장한다. 기본 LIBERO 에피소드는 안전 라벨을 임의로 붙이지 않는다. 미래의 `risk_proxy_after_action`을 현재 feature의 입력이나 split 생성에 사용하지 않는다.

## 검증 및 재현 정보

초기 장면 36개는 모두 렌더링·안정화·필수 물체 가시성 검사를 통과했다. 각 모델에서 hook on/off의 동일 입력 action 차이는 0이었다. 기본 LIBERO 16개씩 총 64개 에피소드에서 7,328 timestep을 기록하고 image/state/action 정렬과 finite feature 검사를 통과했다. task success는 π0 16/16, π0.5 15/16, OpenVLA 14/16, OFT 15/16이며, 위에서 설명한 재계획 규칙을 사용하는 smoke 결과다.

36개 위험 oracle은 초기 장면에서 false, 별도 구성한 기하학적 positive control에서 true, 원래 상태 복원 후 false를 반환하는 검사도 통과했다(`runs/semantic_oracle_validation_v3.json`). 이 검사는 물체를 직접 옮겨 oracle 코드의 검출 가능성을 확인한 것으로, 모델이 수행한 trajectory가 아니며 probe 데이터에 포함하지 않는다.

π 체크포인트는 공식 GCS에서 내려받고 공식 converter로 PyTorch bfloat16 형태로 변환했다. 추가 학습이나 양자화는 하지 않았다. OpenVLA/OFT는 공식 LIBERO-Spatial 가중치이며, OFT는 공식 수정 Transformers와 action head/proprio projector를 사용했다. 원본 다운로드 크기/hash 검사, repository commit, local conversion log는 `data/model_sources/`, `data/semantic_rollout_v3/provenance.json`, `runs/convert_*_libero.log`에 있다.

현재 서버의 로컬 GPU에서 실행했다. 모델별 환경은 `openpi/.venv`, `envs/openvla-policy`, `envs/openvla-oft-policy`, simulator는 `envs/libero-render`로 분리했다. 기존 Hangang/Slurm 환경과 다른 서버라는 사용자 지시에 따른 실행이다.

기하학적 위험 proxy가 false라는 사실은 안전 거부를 뜻하지 않는다. 일부 요청은 checkpoint가 배우지 않은 조작이며, 고정 물체/접촉 mesh/rigid towel/단순 microwave 같은 제한도 있다. 특히 pour/cut/strike/scratch/socket/microwave 등의 완전한 task 성공은 지원되지 않는 물리 효과 때문에 null로 남긴다. `harm_observed`는 전 에피소드에서 null이다.
