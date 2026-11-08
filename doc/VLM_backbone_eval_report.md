# π0 / π0.5 VLM Backbone 평가 보고서

작성일: 2026-10-31 · 환경: KISTI 한강 (`natStr`, GH200 120GB) · 프로젝트: `/scratch/t512a07/minseong/VLA-Safety`

## 1. 목적

VLA(π0, π0.5)는 PaliGemma-3B(SigLIP So400m + Gemma-2B)를 VLM backbone으로 쓰고, 그 위에 action expert(300M)를 붙여
flow-matching으로 action을 학습한다. 이 과정에서 backbone의 **vision-language 능력이 얼마나 남아 있는지**를 네 체크포인트에 대해 측정한다.

| 이름 | 체크포인트 | 설명 |
|---|---|---|
| `pi0_base` | `gs://openpi-assets/checkpoints/pi0_base` | π0 사전학습 모델 |
| `pi0_libero` | `gs://openpi-assets/checkpoints/pi0_libero` | π0를 LIBERO로 full fine-tuning (30k step) |
| `pi05_base` | `gs://openpi-assets/checkpoints/pi05_base` | π0.5 사전학습 모델 |
| `pi05_libero` | `gs://openpi-assets/checkpoints/pi05_libero` | π0.5를 LIBERO로 full fine-tuning (30k step) |

## 2. 문제: action 학습된 backbone은 정상적인 text를 못 낸다 — 그런데 어떻게 측정했나

### 2.1 logit을 꺼내는 방법

openpi의 Gemma 구현(`openpi/models/gemma.py`)은 입력 embedding 테이블을 LM head로 공유(tied embedding)한다.
`Embedder.encode(x) = E[x]`, `Embedder.decode(h) = h · Eᵀ`. 즉 **별도 LM head가 없어도 backbone의 마지막 hidden state에 Eᵀ를 곱하면
257,152 vocab에 대한 next-token logit이 나온다.** π0/π0.5는 추론 시 이 decode를 전혀 쓰지 않지만(action은 action expert가 출력), 가중치는 그대로 들어 있다.
우리는 `scripts/vlm_backbone.py`에서 다음만 사용한다.

- `PaliGemma.img` (SigLIP): 224×224 이미지 → 256 image token
- `PaliGemma.llm` 의 **첫 번째 expert(PaliGemma)만** 실행 (`llm([x, None], ...)`), action expert는 실행하지 않음
- 최종 hidden state × `embedder.input_embedding`ᵀ → logit → `log_softmax`

### 2.2 시퀀스 배치

PaliGemma 학습 방식(prefix-LM)과 openpi의 `embed_prefix`와 동일하게 맞췄다.

```
[ image 256 tokens | BOS prompt... "\n" | answer tokens ... ]
   양방향 attention      양방향 attention        causal
```

위치 i의 logit은 토큰 i+1을 예측하므로, `"\n"` 위치의 logit이 answer 첫 토큰을 예측한다.
프롬프트 토큰화는 openpi `PaligemmaTokenizer`의 π0 형식(`encode(prompt, add_bos=True) + encode("\n")`)과 같다.

### 2.3 두 종류의 측정

**(A) Likelihood 기반 — 생성이 필요 없다.**
정답 후보 문자열을 answer 자리에 넣고 `Σ log p(answer token | image, prompt)`를 계산해 **후보 간 비교**한다.
모델이 "yes"에 −45, "no"에 −60을 주면(둘 다 쓰레기 수준이어도) argmax는 "yes"다. 따라서 생성이 완전히 깨져도 수치는 나오며,
backbone이 **무엇을 아는지**(정답 후보를 오답 후보보다 선호하는지)를 측정한다. 아무것도 모르면 정확히 chance가 나온다.

**(B) 생성 기반 — greedy decoding.**
`"\n"` 이후를 argmax로 한 토큰씩 이어 붙여(EOS 또는 max token까지) 문자열을 만들고 보통의 벤치마크 metric으로 채점한다.
backbone이 **아는 것을 실제로 말로 꺼낼 수 있는지**를 측정한다. (A)와 (B)의 격차가 "알지만 표현 못 함"의 크기다.

**(C) 보조 지표 — 정답의 NLL/token.**
정답 문자열의 평균 음의 로그우도. 후보 비교와 달리 절대값이라서 "텍스트 분포에서 얼마나 멀어졌나"를 모델 간에 바로 비교할 수 있다.
PaliGemma급 VLM이면 1~4, 완전히 무너지면 20~40 수준이다.

## 3. 지표 정의 (표의 열)

| 열 | 벤치마크 | 정의 |
|---|---|---|
| `POPE ll-acc` | POPE (9,000) | `log p("yes"<eos>)` vs `log p("no"<eos>)` 중 큰 쪽을 예측으로 한 정확도. chance 0.5 |
| `POPE ll-acc(noEOS)` | POPE | 같은 비교를 EOS 없이 (`log p("yes")` vs `log p("no")`, lm-eval-harness 방식) |
| `POPE ll-F1` | POPE | 위 예측의 "yes" 클래스 F1 (POPE 공식 지표) |
| `POPE gen-acc` | POPE | greedy 생성(최대 4 토큰) 첫 단어가 yes/no로 파싱되고 정답과 일치한 비율 |
| `POPE gen-valid` | POPE | 생성 결과가 yes/no로 파싱된 비율 (생성이 멀쩡한 정도) |
| `MME ll-acc`, `(noEOS)`, `gen-acc` | MME (2,374) | POPE와 동일 정의. 14개 카테고리 평균 |
| `SQA ll-acc` | ScienceQA-IMG test (2,017) | 선택지 텍스트별 `Σ log p / 토큰 수`(길이 정규화)가 최대인 선택지 = 정답인 비율. chance 0.355 (선택지 수 2~5) |
| `SQA gen-acc` | ScienceQA-IMG | greedy 생성(16 토큰)이 정답 선택지와 정규화 후 완전 일치한 비율 (참고용) |
| `VQA NLL/tok` | VQAv2 val 앞 3,000 | 다수결 정답(`multiple_choice_answer`)의 평균 NLL/token. 낮을수록 좋음 |
| `VQA gen-acc` | VQAv2 | 공식 VQA accuracy: `min(1, 일치한 annotator 수/3)`, 소문자·구두점·관사 정규화 |
| `COCO NLL/tok` | COCO val2017 앞 2,000 | 이미지당 reference caption 5개의 평균 NLL/token |
| `COCO CIDEr`, `BLEU4` | COCO | greedy 생성(32 토큰)에 대한 CIDEr-D, BLEU-4 (`pycocoevalcap`, 자체 소문자/구두점 제거 토큰화) |

프롬프트 템플릿:

| 템플릿 | VQA/yes-no/MC | caption | 비고 |
|---|---|---|---|
| `paligemma` | `answer en {q}` | `caption en` | PaliGemma 사전학습 원본 형식 |
| `plain` | `{q}` | `Describe the image.` | 자연어 지시 |
| `paligemma_strip` | `answer en {q}` 에서 MME 말미의 "Please answer yes or no." 제거 | — | MME 전용 추가 실험 |

## 4. 결과

### 4.1 `paligemma` 템플릿 (주 결과)

| model | POPE ll-acc | POPE ll-acc(noEOS) | POPE ll-F1 | POPE gen-acc | POPE gen-valid | MME ll-acc | MME gen-acc | SQA ll-acc | VQA NLL/tok | VQA gen-acc | COCO NLL/tok | COCO CIDEr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pi0_base | 0.509 | 0.504 | 0.523 | 0.000 | 0.000 | 0.520 | 0.000 | 0.349 | 27.1 | 0.000 | 29.7 | 0.000 |
| pi0_libero | 0.502 | 0.499 | 0.546 | 0.000 | 0.000 | 0.502 | 0.000 | 0.356 | 36.3 | 0.000 | 34.5 | 0.000 |
| **pi05_base** | **0.804** | **0.804** | **0.771** | **0.816** | 0.996 | 0.515 | 0.524 | **0.419** | **1.15** | **0.710** | **3.39** | 0.009 |
| pi05_libero | 0.497 | 0.500 | 0.553 | 0.008 | 0.013 | 0.497 | 0.049 | 0.370 | 10.4 | 0.000 | 11.9 | 0.000 |

chance: POPE/MME 0.5, SQA 0.355.

### 4.2 `plain` 템플릿

| model | POPE ll-acc | POPE gen-acc | POPE gen-valid | MME ll-acc | MME gen-acc | SQA ll-acc | VQA NLL/tok | VQA gen-acc | COCO NLL/tok | COCO CIDEr | COCO BLEU4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pi0_base | 0.510 | 0.000 | 0.000 | 0.499 | 0.000 | 0.360 | 26.6 | 0.000 | 28.9 | 0.000 | 0.000 |
| pi0_libero | 0.506 | 0.000 | 0.000 | 0.495 | 0.000 | 0.360 | 36.7 | 0.000 | 34.7 | 0.000 | 0.000 |
| **pi05_base** | **0.710** | 0.635 | 0.806 | 0.514 | 0.551 | **0.400** | **1.10** | **0.715** | **3.26** | 0.013 | 0.080 |
| pi05_libero | 0.503 | 0.118 | 0.204 | 0.505 | 0.055 | 0.354 | 8.1 | 0.010 | 10.0 | 0.000 | 0.000 |

### 4.3 세부

**POPE 카테고리별 ll-acc (paligemma)** — pi05_base: adversarial 0.79 / popular 0.81 / random 0.81. 나머지 세 모델: 모두 0.49~0.52.

**VQAv2 타입별 gen-acc (paligemma)** — pi05_base: yes/no 0.88, other 0.62, number 0.56. 나머지: 0.

**MME (pi05_base)** — 원문 프롬프트 0.515 → "Please answer yes or no." 제거 시 **0.626** (gen-acc 0.593, valid 0.956).
카테고리별(strip): existence 0.95, color 0.82, scene 0.82, count 0.70, landmark 0.64, position 0.63, artwork 0.60, commonsense 0.57,
OCR 0.53, celebrity 0.51, numerical 0.50, text_translation 0.50, posters 0.49, code_reasoning 0.42.
EOS 포함/미포함 정확도는 같지만 yes 비율이 0.906(EOS 포함) ↔ 0.058(미포함)으로 뒤집힌다 — "yes" 뒤에는 EOS가 잘 오고 "no" 뒤에는 이어지는 토큰이 오는 경향.

**생성 예시**

| model | 프롬프트 | 생성 | 정답 |
|---|---|---|---|
| pi05_base | `answer en Where is he looking?` | `up` | down |
| pi05_base | `answer en What is the person doing?` 류 | `watching` | watching |
| pi05_base | `Describe the image.` | "The image features a cat sitting on a wooden floor. The cat appears to be eating or interacting with a bird. The bird is positioned towards the center of the…" | "A cat eating a bird it has caught." |
| pi05_base | `caption en` | `no`, `plate`, `ground` (1~2단어) | COCO 캡션 |
| pi05_libero | 아무 프롬프트 | `立媄媄䊐數`, `፪媄媄媄` | — |
| pi0_base | 아무 프롬프트 | `GraphicsUnitGraphicsUnit…`, `Савезне Савезне…`, `Pozri` | — |

## 5. 해석

1. **π0.5-base의 backbone은 VLM으로 정상 동작한다.** POPE 0.80, VQAv2 0.71(PaliGemma-3B-pt 수준), SQA chance+6.4pt, 정답 NLL 1.1~3.4.
   plain 프롬프트에서는 길고 grounded한 묘사를 생성한다. COCO CIDEr가 0에 가까운 것은 (i) `caption en`에는 짧은 VQA식 답(평균 1.9단어)을 내고,
   (ii) `Describe the image.`에는 LLaVA 스타일 긴 묘사(평균 27.9단어, 32토큰 제한으로 잘림)를 내서 COCO의 짧은 캡션과 **스타일/길이가 안 맞기 때문**이지
   언어가 깨져서가 아니다(reference NLL 3.3/token은 정상 범위). π0.5 사전학습에 VQA·캡션 등 web VLM 데이터와 텍스트 target이 포함된 것과 일치한다.
2. **π0-base, π0-LIBERO는 backbone의 텍스트 출력 분포가 완전히 무너졌다.** 정답 NLL 27~37/token, 생성은 무작위 토큰, likelihood 비교도 정확히 chance.
   flow-matching loss만으로 학습된 π0는 tied-embedding LM head 기준으로 VLM 능력이 남아 있지 않다. fine-tuning(`pi0_libero`)은 이를 더 악화시킨다(27→36).
3. **LIBERO fine-tuning은 π0.5의 VLM 능력을 사실상 지운다** (POPE 0.80→0.50, VQA 0.71→0.00, NLL 1.1→8~12). 단 π0만큼 멀어지지는 않았고
   SQA 0.37~0.38로 미세하게 chance 위라서, "완전 파괴"보다 "심한 drift"에 가깝다. 표현(hidden state) 자체가 얼마나 남았는지는 LM head와 분리해 봐야 한다(6절).
4. **MME 저점수는 절반이 프롬프트 형식 탓.** MME 질문 끝의 지시문을 제거하면 0.52→0.63. 남은 약점(OCR·텍스트·코드·포스터 chance)은 224px PaliGemma-pt의 일반적 한계와 일치한다.
5. **likelihood 지표와 생성 지표는 서로 보완적이다.** pi05_base POPE에서는 둘이 거의 같고(0.804 vs 0.816), 캡션에서는 likelihood는 정상인데 생성 metric은 0이다.
   깨진 모델에서는 둘 다 chance/0이라, "표현만 못하는 것"이 아니라 "LM head 경로 자체가 무너진 것"으로 해석된다.

## 6. 한계와 다음 단계

- **원본 PaliGemma-3B-pt-224 기준선이 없다.** openpi가 쓰는 `gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz`는 익명 접근 403이고
  HF `google/paligemma-3b-pt-224`는 gated라 HF 토큰이 필요하다. 이것이 들어가야 "얼마나 잃었나"가 절대 수치로 나온다.
- **표현 vs LM head 분리:** image token / 최종 hidden state에 linear probe(또는 k-NN)를 붙여 분류·검색 성능을 재면, π0·π0.5-LIBERO의 backbone이
  "시각 표현은 유지했지만 텍스트로 못 꺼내는지" 아니면 "표현도 무너졌는지"를 가를 수 있다.
- **캡션 생성 길이:** pi05_base는 `max_new_tokens=64` 이상과 짧은 캡션용 프롬프트로 다시 측정하면 CIDEr가 의미 있게 나올 것이다.
- **ScienceQA:** 2,017개 중 67개(3%)는 hint가 길어 프롬프트 앞부분을 잘랐다(정답 선택지는 잘리지 않음).
- VQAv2는 validation 앞 3,000개, COCO는 val2017 앞 2,000개만 사용했다.
- 모든 forward는 bf16, 이미지는 openpi의 `resize_with_pad`와 동일하게 224×224 letterbox.

## 7. 재현

```bash
cd /scratch/t512a07/minseong/VLA-Safety
bash scripts/setup_env.sh                                   # openpi/.venv (uv, Python 3.11, JAX 0.5.3 CUDA, aarch64)
bash scripts/download_ckpt.sh pi0_base pi0_libero pi05_base pi05_libero
openpi/.venv/bin/python scripts/prepare_data.py             # data/<bench>/{items.jsonl,images/}
sbatch --export=ALL,CKPTS="pi0_base pi0_libero pi05_base pi05_libero",EXTRA="--templates paligemma plain --batch 32" jobs/eval_backbone.sbatch
openpi/.venv/bin/python scripts/summarize.py                # runs/summary_table.md
```

산출물: `runs/<model>/summary.json`(지표), `runs/<model>/<bench>.<template>.jsonl`(항목별 log-prob·생성문), `runs/summary_table.md`.

Slurm 작업 기록 (`runs/submitted-jobs.tsv`):

| Job | 내용 | 결과 |
|---|---|---|
| 1182129 | GPU probe | COMPLETED (16s) |
| 1182194 | smoke (POPE 16개) | COMPLETED |
| 1184311 | 전체 평가 1차 | FAILED — SQA 긴 프롬프트 truncation 버그, 수정 |
| 1184338 | 전체 평가 (paligemma + plain) | COMPLETED (39m 32s, 4 GPU) |
| 1184503 | POPE/MME no-EOS 지표 추가 | COMPLETED (17m) |
| 1184556 | MME strip (패치 전 제출) | CANCELLED (본인 작업) |
| 1184557 | MME strip, pi05_base | COMPLETED (2m) |
