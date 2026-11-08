# VLA 내부 VLM의 Utility·Safety 평가 결과 (backbone_eval_v1)

작성: 2026-11-05 · 계획: `../VLA_VLM_Utility_Safety_Experiment_Plan.md` · 실행 기록: `VLA_VLM_Utility_Safety_Experiment_Log.md`
원본 산출물: `runs/backbone_eval_v1/{summary.md, summary.csv, validation_report.md, diagnostics.md, models_manifest.json}`, `data_eval/data_manifest.json`

## 1. 무엇을 측정했나

추가 학습 없이, 모델마다 **prompt-only forward 한 번**으로 두 가지를 읽었다.

- **Likelihood**: 모델의 기존 언어 출력 head로 첫 답변 토큰(`yes/no` 또는 `A~E`)의 전체 vocab log-softmax를 구하고 후보 중 argmax. 학습·threshold 없음.
- **Linear probe**: 같은 forward의 마지막 유효 prompt 토큰 벡터(final norm 이후)에 `StandardScaler + L2 LogisticRegression`을 데이터셋별 train split로 학습, dev로 C 선택, **같은 test id**에서 평가. backbone·head·adapter는 전부 고정.

| 모델 | 체크포인트 | 언어 head | attention | 상태 |
|---|---|---|---|---|
| pi0 | openpi `pi0_base` | Gemma tied embedding | PaliGemma prefix-LM(양방향) | 완료 |
| pi05 | openpi `pi05_base` | Gemma tied embedding | PaliGemma prefix-LM | 완료 |
| paligemma | HF `google/paligemma-3b-pt-224` | Gemma tied embedding(HF lm_head) | HF 추론 mask: 전체 양방향 | 완료 (토큰 확보 후 추가; `paligemma_lb` = openpi letterbox 전처리 조건) |
| openvla | `openvla/openvla-7b` | 별도 Llama `lm_head` | causal (transformers 4.40.1) | 완료 |
| openvla_oft | `moojink/openvla-7b-oft-finetuned-libero-spatial` | 별도 Llama `lm_head` | **OFT fork 양방향**(실측 확인) | 완료 (single image, proprio/action 미사용) |
| prismatic | TRI `prism-dinosiglip-224px+7b` | 별도 Llama `lm_head` | causal | 완료 (OpenVLA와 동일 구조·초기화 계열) |

데이터: POPE-YN, MME-YN(research_split), SQA-IMG-MC(official), VQAv2-MC-adapted·COCO-CaptionChoice-adapted(official train/val, 단일 토큰 선택형으로 변환), SIUO-MCQA(research_split, n_test 26~32). VLGuard-request-adapted(official train/test)는 토큰 확보 후 추가. LlavaGuard는 데이터셋 접근 양식 승인 전까지 **blocked**(403). 분할 세부와 제외 사유는 `data_eval/data_manifest.json`.

## 2. 주 결과 (test split, accuracy)

`chance` = mean(1/K), `maj` = train 다수 class를 test에 적용. Likelihood와 probe는 같은 test id.

| 모델 | POPE-YN | MME-YN | SQA-IMG-MC | VQAv2-MC | COCO-CapChoice | SIUO-MCQA |
|---|---|---|---|---|---|---|
| chance / maj | 0.50 / 0.49 | 0.50 / 0.50 | 0.35 / 0.35 | 0.35 / 0.36 | 0.25 / 0.25 | 0.25 / 0.25–0.27 |
| **prismatic** ll / probe | **0.903 / 0.916** | **0.698 / 0.835** | **0.695 / 0.690** | **0.845 / 0.847** | **0.986 / 0.988** | 0.375 / 0.344 (n=32) |
| **openvla** ll / probe | 0.520 / 0.660 | 0.508 / 0.540 | 0.348 / 0.385 | 0.346 / 0.319 | 0.264 / 0.277 | 0.219 / 0.156 (n=32) |
| **openvla_oft** ll / probe | 0.515 / 0.526 | 0.500 / 0.525 | 0.354 / 0.364 | 0.360 / 0.336 | 0.242 / 0.234 | 0.250 / 0.219 (n=32) |
| **pi05** ll / probe | 0.584 / **0.880** | 0.559 / 0.677 | 0.416 / 0.401 | 0.700 / 0.732 | 0.733 / **0.901** | 0.423 / 0.269 (n=26) |
| **pi0** ll / probe | 0.504 / 0.605 | 0.494 / 0.515 | 0.358 / 0.372 | 0.354 / 0.334 | 0.251 / 0.259 | 0.192 / 0.192 (n=26) |

N(train/dev/test): POPE 3000/500/1000, MME 1426/474/474, SQA 3000/500/1000, VQA 2880/500/1000, COCO 2880/500/1000, SIUO 79–95/27–33/26–32
(SIUO는 프롬프트 길이 제한으로 PaliGemma 계열 35건, Llama 계열 7건 제외).

### 2.1 이진 데이터셋 보조 지표 (ll / probe)

| 모델 | 데이터 | balanced acc | macro-F1 | recall(yes) | FPR(no→yes) |
|---|---|---|---|---|---|
| prismatic | POPE | 0.901 / 0.915 | 0.902 / 0.916 | 0.827 / 0.889 | 0.025 / 0.058 |
| prismatic | MME | 0.698 / 0.835 | 0.681 / 0.835 | 0.932 / 0.810 | 0.536 / 0.139 |
| pi05 | POPE | 0.571 / 0.880 | 0.481 / 0.880 | 0.142 / 0.889 | 0.000 / 0.128 |
| pi05 | MME | 0.559 / 0.677 | 0.538 / 0.677 | 0.346 / 0.671 | 0.228 / 0.316 |
| openvla | POPE | 0.511 / 0.659 | 0.461 / 0.660 | 0.196 / 0.641 | 0.175 / 0.322 |
| openvla_oft | POPE | 0.500 / 0.528 | 0.340 / 0.525 | 0.000 / 0.581 | 0.000 / 0.526 |
| pi0 | POPE | 0.506 / 0.605 | 0.503 / 0.605 | 0.567 / 0.598 | 0.555 / 0.388 |

### 2.2 이미지 셔플 통제 (test 최대 100개, 다른 이미지 그룹의 이미지로 교체, 질문·선택지·label 유지)

| 모델 | POPE probe | COCO probe | VQA probe | POPE ll | COCO ll |
|---|---|---|---|---|---|
| prismatic | 0.92 → 0.53 | 1.00 → 0.17 | 0.82 → 0.53 | 0.94 → 0.52 | 1.00 → 0.20 |
| pi05 | 0.88 → 0.51 | 0.93 → 0.20 | 0.69 → 0.43 | 0.57 → 0.52 | 0.80 → 0.23 |
| openvla | 0.61 → 0.57 | 0.27 → 0.23 | 0.34 → 0.37 | 0.45 → 0.48 | 0.33 → 0.25 |
| openvla_oft | 0.57 → 0.61 | 0.30 → 0.18 | 0.36 → 0.30 | 0.49 → 0.49 | 0.22 → 0.22 |
| pi0 | 0.53 → 0.51 | 0.23 → 0.22 | 0.34 → 0.37 | 0.45 → 0.50 | 0.28 → 0.32 |

prismatic과 pi05의 점수는 이미지를 바꾸면 chance로 떨어진다(시각 입력에 의존). 나머지 세 모델은 원래부터 chance 근방이라 변화가 없다.

### 2.3 추가 실행 (HF 토큰 확보 후, 잡 1218325): 원본 PaliGemma와 VLGuard

**원본 PaliGemma-3B-pt-224** (Pi 계열의 초기화 모델; `paligemma` = HF 기본 전처리, `paligemma_lb` = Pi와 같은 letterbox 전처리)

| 모델 | POPE ll / probe | MME | SQA | VQAv2-MC | COCO-CapChoice | SIUO (n=26) |
|---|---|---|---|---|---|---|
| paligemma | **0.884 / 0.905** | 0.542 / 0.772 | 0.350 / 0.433 | 0.342 / 0.365 | 0.320 / 0.760 | 0.115 / 0.192 |
| paligemma_lb | 0.874 / 0.886 | 0.521 / 0.768 | 0.360 / 0.432 | 0.341 / 0.377 | 0.301 / 0.760 | 0.154 / 0.269 |
| (참고) pi05 | 0.584 / 0.880 | 0.559 / 0.677 | 0.416 / 0.401 | 0.700 / 0.732 | 0.733 / 0.901 | 0.423 / 0.269 |
| (참고) pi0 | 0.504 / 0.605 | 0.494 / 0.515 | 0.358 / 0.372 | 0.354 / 0.334 | 0.251 / 0.259 | 0.192 / 0.192 |

- 두 전처리 조건의 결과가 거의 같다(차이 ≤0.03). Pi 계열과의 격차는 전처리 때문이 아니다.
- PaliGemma-pt는 소문자 `yes/no`를 그대로 내므로(cand_mass 1.00) POPE likelihood 0.88이 나오지만, MME에서는 94%를 `yes`로 답해 0.54에 그친다(probe 0.77).
- **선택지 문자 과제는 pt 모델이 잘 따르지 못한다.** 1위 토큰이 소문자 `b`, `d` 등으로 흩어지고 후보 질량이 0.2~0.55라 likelihood는 chance 수준이다. 그러나 probe는 COCO 0.76, SQA 0.43으로 표현에는 정보가 있다.
- pi0.5는 VQAv2-MC·COCO에서 원본 PaliGemma-pt보다 **높다**(ll 0.70/0.73 vs 0.34/0.32; probe 0.73/0.90 vs 0.37/0.76). π0.5 사전학습의 web VLM·instruction 데이터 공동학습이 선택지 응답 능력을 더해 준 것과 일치한다. 반대로 pi0는 PaliGemma-pt에 비해 선형으로 읽히는 신호가 거의 사라졌다(POPE probe 0.61 vs 0.89, COCO 0.26 vs 0.76).

**VLGuard-request-adapted** (요청이 유해한지 yes/no; test 1000, yes 63.6%; train 2366 / dev 500)

| 모델 | ll acc | ll bal-acc | ll yes-recall / no-FPR | probe acc | probe bal-acc | probe yes-recall / no-FPR | 이미지 셔플 probe |
|---|---|---|---|---|---|---|---|
| prismatic | 0.665 | 0.540 | 0.998 / 0.918 | **0.995** | 0.995 | 0.995 / 0.005 | 1.00 → 0.98 |
| pi05 | 0.364 | 0.500 | 0.000 / 0.000 | 0.980 | 0.977 | 0.989 / 0.036 | 0.99 → 0.99 |
| paligemma | 0.368 | 0.502 | 0.009 / 0.005 | 0.978 | 0.975 | 0.986 / 0.036 | 0.99 → 0.98 |
| openvla | 0.386 | 0.483 | 0.127 / 0.162 | 0.905 | 0.894 | 0.936 / 0.148 | 0.91 → 0.88 |
| pi0 | 0.602 | 0.526 | 0.805 / 0.753 | 0.830 | 0.807 | 0.892 / 0.277 | 0.79 → 0.81 |
| openvla_oft | 0.364 | 0.500 | 0.000 / 0.000 | 0.769 | 0.748 | 0.825 / 0.330 | 0.82 → 0.74 |

- Likelihood는 어떤 모델도 요청 안전성을 읽지 못한다. pi05·paligemma·OFT는 거의 전부 `no`, prismatic은 거의 전부 `yes`로 답해 balanced accuracy가 0.50 근방이다.
- Probe는 모든 모델에서 높지만, **이미지를 바꿔도 거의 변하지 않는다**(0.98→0.98 등). 즉 이 데이터셋의 safe/unsafe 요청은 요청 **문장만으로** 선형 분리되며, probe가 읽는 것은 시각 신호가 아니라 텍스트 표현이다. 계획 문서가 지적한 대로 이는 "요청 분류" 성능이며 이미지 기반 안전 판단이나 거절 행동이 아니다. 모델 간 차이(prismatic 0.995 → pi0 0.83 → OFT 0.77)는 마지막 위치 표현에 요청 문장의 의미가 얼마나 선형으로 남아 있는지의 차이로 읽는 것이 적절하다.

## 3. 언어 head가 실제로 내놓는 토큰 (`diagnostics.md`)

`cand_mass` = 후보 토큰들의 확률 합(전체 vocab softmax), `top1∈cands` = 전체 vocab 1위가 후보인 비율.

| 모델 | 데이터 | cand_mass | top1∈cands | 가장 흔한 top-1 토큰 |
|---|---|---|---|---|
| prismatic | POPE / SQA / COCO | 0.20 / 1.00 / 1.00 | 0.01 / 1.00 / 1.00 | `▁No`·`▁Yes`(대문자) / `▁A`·`▁B` / 선택지 문자 |
| pi05 | POPE / SQA / COCO | 0.012 / 0.78 / 0.91 | 0.00 / 1.00 / 1.00 | `No`(78%)·`Yes` / `B`·`C` / `C`·`B` |
| openvla | 전부 | 0.000 | 0.00 | action bin 토큰(id 31744–31999, 예 `ක`, `渡`, `飛`) |
| openvla_oft | 전부 | 0.000 | 0.00 | 단일 토큰 id 31872 (`Ÿ`, action bin 영역) 100% |
| pi0 | 전부 | 0.000 | 0.00 | `GraphicsUnit`, `serviceWorker`, `KEYCODE` 등 무의미 토큰 |

- 이진 과제에서 prismatic·pi05는 **대문자 `Yes/No`**를 가장 선호한다. 계획대로 소문자 후보 `yes/no`로 채점했으므로 cand_mass는 낮지만, 두 후보 간 argmax는 그대로 의미가 있다. 참고로 전체 vocab top-1 토큰을 대소문자 무시로 읽은 정확도(진단용)는 pi05 POPE 0.731, MME 0.563, prismatic POPE 0.907, MME 0.717이다.
- OpenVLA·OFT의 head는 **모든 입력에 대해 action 토큰만** 1위로 낸다. 특히 OFT는 입력과 무관하게 한 토큰(31872)으로 고정된다.

## 4. 해석 (계획 §9의 문구 범위 안에서)

1. **action 학습이 없는 기준 모델(prismatic)은 두 readout이 거의 일치하고 높다.** 파이프라인(프롬프트, 토큰 경계, 위치, head)이 정상 동작한다는 내부 통제이기도 하다.
2. **OpenVLA(OXE action 학습)**: 기존 head로는 한 토큰 정답을 전혀 읽지 못한다(cand_mass 0, 1위는 항상 action 토큰, 정확도 chance). 선형 probe는 POPE에서만 chance 위(0.66, 균형 정확도 0.66)이고 그 외는 chance다. 즉 마지막 prompt 위치의 표현에 선형으로 읽히는 신호가 **일부(객체 존재 여부) 남아 있으나 작고**, VQA·캡션·SQA 수준의 정보는 이 위치·이 선형 readout으로는 읽히지 않는다. 같은 초기화 계열인 prismatic과의 격차가 크지만, 이것을 action 학습 하나의 인과 효과로 단정하지는 않는다(학습 데이터·단계·목표가 모두 다르다).
3. **OpenVLA-OFT(LIBERO fine-tune, 양방향 attention, L1 head)**: head는 상수 토큰, probe도 전 데이터셋 chance. 이번 입력(single image, action placeholder 없음)·마지막 위치·선형 readout에서는 판독 불가. 이는 native 양방향 설정에 원 LM head를 적용한 VLM-only readout 결과이며, 중간 layer나 다른 위치에 정보가 없다는 증명은 아니다.
4. **pi0**: head 출력 분포가 무너졌고(무의미 토큰), likelihood는 정확히 chance. probe는 POPE 0.605를 제외하면 chance. 이전 보고서(sequence likelihood)와 일치한다.
5. **pi0.5**: head로 읽은 정확도는 MC에서 유의하게 chance 위(VQA 0.70, COCO 0.73, SQA 0.42), 이진에서는 약함(POPE 0.58, 강한 "no" 편향: yes recall 0.14). **같은 벡터에 선형 probe를 얹으면 POPE 0.88, COCO 0.90, MME 0.68**로 크게 오른다. → "기존 readout 밖에 선형으로 판독 가능한 신호가 더 있다"는 결과. 이진 과제의 낮은 likelihood는 대문자 `No/Yes` 선호와 `no` 편향이라는 **readout 보정 문제**가 큰 부분을 차지하며, 이를 semantic alignment 손상으로 단정하지 않는다.
6. **Safety(SIUO-MCQA)**: test 26~32개로 탐색적 결과다. prismatic 0.375/0.344, pi05 ll 0.423(probe 0.269), 나머지 chance 근방. 안전한 응답을 *고르는* 능력이며 실제 생성의 안전성이 아니다. LlavaGuard·VLGuard는 blocked여서 이미지 안전 rating·요청 분류는 측정하지 못했다.
7. 계열 간 비교 주의: Pi 계열(PaliGemma, letterbox 224, Gemma tokenizer)과 Llama 계열(DINO+SigLIP, resize-naive 224, Llama-2 tokenizer)은 backbone·전처리·학습 단계가 다르다. 계열 **내부** 비교(prismatic→openvla→oft, pi0 vs pi05)가 해석의 중심이다.

## 5. 검증 요약 (`validation_report.md`)

- head: Llama 계열은 `lm_head`와 `embed_tokens`가 다른 storage·다른 값임을 확인. Pi 계열은 tied embedding(설계상 동일).
- attention 방향: 마지막 토큰을 바꿨을 때 앞 위치의 hidden이 변하는지로 측정. openvla·prismatic = 변하지 않음(causal), openvla_oft = 변함(양방향, fork 파일 sha 기록), pi = 변함(prefix-LM).
- 단일 토큰: 모든 후보가 실제 경계에서 1 토큰(PaliGemma `yes`=3276 `no`=956 `A..E`; Llama `▁yes`=4874 `▁no`=694 `▁A..▁E`). joint/separate 인코딩 교차 확인.
- padding 불변성: Pi(JAX, 고정 shape)는 차이 0. Llama 계열은 bf16 + SDPA 커널 경로 차이로 h 상대차 0.4~1.6%, 후보 logp 차 ≤0.43이나 후보 argmax는 모두 동일(수치 기록).
- 입력 누출: 프롬프트는 body만의 함수. label을 바꿔도 token id 동일. MC body에 정답 표시 없음(빌드 시 assert).
- freeze: 추출 전후 파라미터 fingerprint 동일. probe는 캐시된 npz에서 sklearn으로만 학습.
- 데이터: 이미지 그룹이 split을 넘지 않음(assert). SQA는 test와 이미지가 겹치는 train 1,128·dev 378건 제외. 예시 10개씩 `data_eval/<ds>/examples.md`.
- Prismatic 로딩: TRI 체크포인트는 projector+llm만 포함(vision은 Prismatic 학습에서 고정). timm 사전학습 DINOv2/SigLIP(timm 0.9.10 기본 소스)로 채워 strict 로드, 누락 키 0.

## 6. 한계와 다음 단계

- LlavaGuard는 HF 데이터셋 접근 양식 승인이 필요하다(토큰만으로는 403). 승인 후 `prepare_eval_data.py --stage plan_llavaguard` → dm01에서 이미지 URL 수집 → `--stage build --datasets llavaguard_rating` → 잡 재실행으로 추가된다.
- SIUO는 n이 작아 신뢰구간이 넓다(±0.17 수준).
- 이진 likelihood는 소문자 후보 고정이라 대문자 선호 모델에 불리하다. 대소문자 합산(`p(yes)+p(Yes)`)은 후속 실험에서 후보 변형으로 추가할 수 있다(이번에는 진단용 top-1 판독만 보고).
- VQAv2-MC·COCO-CapChoice는 단일 토큰 선택형 재구성이며 공식 VQA accuracy·CIDEr가 아니다. MME는 research_split이며 공식 total score가 아니다.
- 중간 layer, token pooling, 비선형 probe, causal 복원 ablation(OFT)은 계획대로 제외했다.

## 7. 재현

```bash
cd /scratch/t512a07/minseong/VLA-Safety
openpi/.venv/bin/python scripts/backbone_eval/prepare_eval_data.py --stage plan      # 공유 COCO 학습 이미지 목록
ssh dm01 'cd $PWD && python3 scripts/dm_fetch_images.py data_raw/coco_images/needed.tsv data_raw/coco_images/train_images.tar'
openpi/.venv/bin/python scripts/backbone_eval/prepare_eval_data.py --stage build     # data_eval/
sbatch jobs/backbone_eval.sbatch                                                     # 4 GPU lane, smoke 게이트 → 추출 → probe → 요약
sbatch --export=ALL,ONLY="openvla_oft prismatic",EXTRA="--overwrite" jobs/backbone_eval.sbatch   # 일부 모델 재실행
openpi/.venv/bin/python scripts/backbone_eval/diagnostics.py                         # diagnostics.md
```

Slurm: 1215471(1차, 15m28s; OFT·Prismatic smoke 실패, pi SIUO segfault), 1216779(2차 재실행, 20m40s, 전부 완료), 1218325(PaliGemma 2조건 + VLGuard 추가, 11m11s). `runs/submitted-jobs.tsv`.
