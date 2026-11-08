# π0 / π0.5 VLM Backbone 평가 — 수행 절차 상세

작성일: 2026-11-03 · 대상 코드: `scripts/vlm_backbone.py`, `scripts/eval_backbone.py`, `scripts/prepare_data.py`, `jobs/eval_backbone.sbatch`
결과 해석은 `VLM_backbone_eval_report.md`를 참조. 이 문서는 **"어떤 과정을 거쳐 숫자를 얻었는가"**만 다룬다.

---

## 0. 한눈에 보는 파이프라인

```
[A] 환경 구축        scripts/setup_env.sh        → openpi/.venv (uv, Python 3.11, JAX 0.5.3 CUDA, aarch64)
[B] 체크포인트       scripts/download_ckpt.sh    → checkpoints/<name>/params/   (GCS 공개 버킷, HTTPS)
[C] 벤치마크 준비    scripts/prepare_data.py     → data/<bench>/items.jsonl + images/*.png (224×224 letterbox)
[D] 평가 실행        jobs/eval_backbone.sbatch   → srun 4 task, GPU 1장 = 체크포인트 1개
      └ scripts/eval_backbone.py
            └ scripts/vlm_backbone.py (BackboneLM)
                  ├ score()    : log p(answer | image, prompt)      ← likelihood 지표
                  └ generate() : greedy decoding                    ← 생성 지표
[E] 집계             scripts/summarize.py        → runs/summary_table.md
```

산출물: `runs/<name>/summary.json`(지표), `runs/<name>/<bench>.<template>.jsonl`(항목별 기록), `logs/eval-<name>-<jobid>.log`.

---

## 1. 평가 대상과 전제

| 이름 | 소스 | `--pi05` |
|---|---|---|
| `pi0_base` | `gs://openpi-assets/checkpoints/pi0_base` | 아니오 |
| `pi0_libero` | `gs://openpi-assets/checkpoints/pi0_libero` | 아니오 |
| `pi05_base` | `gs://openpi-assets/checkpoints/pi05_base` | 예 |
| `pi05_libero` | `gs://openpi-assets/checkpoints/pi05_libero` | 예 |

네 체크포인트 모두 구조는 **PaliGemma-3B(SigLIP So400m/14 + Gemma-2B) + action expert(Gemma-300M 변형)** 의 mixture transformer다.
우리가 묻는 질문은 "action 학습 후에도 PaliGemma 부분이 VLM으로서 동작하는가"이므로, **action expert는 로드만 하고 실행하지 않는다.**

토크나이저는 `checkpoints/tokenizer/paligemma_tokenizer.model`(PaliGemma sentencepiece, vocab 257,152) 하나를 모든 모델에 공통으로 쓴다.
특수 토큰: `PAD_ID = 0`, `EOS_ID = 1`, BOS는 sentencepiece `add_bos=True`로 삽입.

---

## 2. 데이터 준비 (`scripts/prepare_data.py`)

### 2.1 소스와 서브셋

| bench | HF 데이터셋 | split | 사용량 | task |
|---|---|---|---|---|
| `pope` | `lmms-lab/POPE` | test | 전체 9,000 | yes/no |
| `mme` | `lmms-lab/MME` | test | 전체 2,374 | yes/no |
| `scienceqa` | `lmms-lab/ScienceQA-IMG` | test | 전체 2,017 | 다지선다 |
| `vqav2` | `lmms-lab/VQAv2` | validation | **앞 3,000개** | 자유 응답 VQA |
| `coco` | `lmms-lab/COCO-Caption2017` | val | **앞 2,000개** | 캡션 |

`streaming=True`로 받아 순서대로 잘라 쓴다. 무작위 샘플링이 아니므로 재현 가능하다.

### 2.2 이미지 전처리

모든 이미지를 **224×224 resize-with-pad(letterbox)** 로 저장한다. 종횡비를 유지한 채 긴 변을 224에 맞추고 검은색으로 가운데 패딩한다.
이는 openpi `openpi.shared.image_tools.resize_with_pad`와 동일한 규칙이다. 즉 VLA가 학습 때 본 것과 같은 전처리를 통과한 이미지로 평가한다.
SigLIP은 224×224 입력을 14×14 패치로 나누어 **256개 image token**을 만든다.

### 2.3 항목 포맷 (`items.jsonl`)

공통 필드: `id`, `task`, `question`, `answers`(리스트), `category`, `image`(상대경로).
- `scienceqa`: 추가로 `choices`, `answer_idx`. `hint`가 있으면 `"{hint} {question}"`으로 질문 앞에 붙인다.
- `vqav2`: 추가로 `majority`(= `multiple_choice_answer`), `answers`는 annotator 10명 응답, `category`는 `answer_type`(yes/no, number, other).
- `coco`: `answers`에 reference caption(최대 5개 사용), `question`은 빈 문자열.
- `pope`/`mme`: `answers[0]`은 소문자 `"yes"`/`"no"`.

---

## 3. 모델 로딩 (`BackboneLM.__init__`)

1. `pi0_config.Pi0Config(pi05=<flag>)` 생성. `pi05=True`이면 openpi가 π0.5 구조(discrete state 입력, adaRMSNorm 등)로 모델을 만든다. 체크포인트 계열과 반드시 일치해야 한다.
2. `_model.restore_params(ckpt_dir/"params", dtype=bfloat16)` 로 전체 파라미터를 **bf16**으로 복원. 네 체크포인트 모두 동일한 처리.
3. `cfg.load(params)` 로 nnx 모델 객체 생성.
4. `params["PaliGemma"]["llm"]["embedder"]["input_embedding"]` 을 따로 꺼내 `[V=257152, D=2048]` bf16 배열로 보관한다. 이것이 **LM head 대용**이다.
5. `nnx.split(model)` 로 graphdef/state를 분리하고, forward 함수를 `jax.jit`으로 컴파일한다. 입력 shape은 항상 `[b, 224, 224, 3]` 이미지와 `[b, 192]` 토큰으로 고정되어 재컴파일이 없다.

### 왜 embedding 테이블이 LM head인가

openpi `gemma.py`의 `Embedder`는 하나의 테이블로 양방향을 처리한다.

```python
def encode(self, x): return self.input_embedding_table[(x,)] * sqrt(embed_dim)   # token → vector
def decode(self, x): return jnp.dot(x, self.input_embedding_table.T)             # vector → vocab logit
```

Gemma 계열은 별도 출력 projection이 없는 tied embedding 구조다. π0/π0.5 추론 코드는 `decode`를 전혀 호출하지 않지만(action은 action expert가 내보냄),
테이블 자체는 프롬프트 임베딩 용도로 계속 학습·저장되어 있다. 따라서 **backbone의 마지막 hidden state에 Eᵀ를 곱하면 next-token logit이 복원**된다.
주의: 여기서 E는 원본 PaliGemma의 것이 아니라 **각 체크포인트에 저장된, action 학습을 거친 E**다.

---

## 4. Forward 경로 (`fwd` 함수)

입력: `images [b,224,224,3]`, `tokens [b,L]`, `tok_mask [b,L]`, `ar_mask [b,L]`, `targets [b,L]` (L = 192)

```
① img_tokens = PaliGemma.img(images, train=False)           # SigLIP → [b, 256, D]
② txt_tokens = PaliGemma.llm(tokens, method="embed")        # E[tokens]·sqrt(D) → [b, L, D]
③ x = concat([img_tokens, txt_tokens], axis=1)              # [b, 256+L, D]
④ in_mask = [1]*256 ++ tok_mask                              # 패딩 위치 False
   ar      = [0]*256 ++ ar_mask                              # 이미지·프롬프트 0, 정답 토큰 1
   attn_mask = make_attn_mask(in_mask, ar)                   # big_vision 방식 prefix-LM 마스크
   positions = cumsum(in_mask) - 1                           # 패딩 제외 0-based 위치 (RoPE용)
⑤ (out, _), _ = PaliGemma.llm([x, None], mask=attn_mask, positions=positions)
                                                             # expert 0(PaliGemma)만 실행, expert 1(action)은 None → 건너뜀
                                                             # 반환값은 final RMSNorm까지 적용된 hidden state
⑥ h = out[:, 256:, :]                                        # 텍스트 위치만 [b, L, D]
⑦ logits = einsum("bld,vd->blv", h, E, preferred_element_type=float32)   # [b, L, V], fp32 누적
⑧ logp = log_softmax(logits)
   target_logp = logp[..., targets]                          # 위치 i가 토큰 i+1을 맞힐 log-prob
   argmax = argmax(logits)                                   # greedy 후보
```

### 4.1 Attention 구조

`make_attn_mask`는 `ar_mask`의 누적합이 자기보다 작거나 같은 위치만 attend하게 만든다. 결과:

```
[ image 256 | BOS p₁ p₂ … pₙ "\n" | a₁ a₂ … aₘ EOS | PAD … ]
   ar=0          ar=0                 ar=1              mask=0
   ◀──── 서로 완전 양방향 attention ────▶  ◀─ causal ─▶   무시
```

- 이미지 토큰과 프롬프트 토큰은 서로 전부 볼 수 있다(prefix).
- 정답 토큰은 prefix 전체와 자기 앞의 정답 토큰만 본다(causal).
- 이는 PaliGemma 사전학습의 prefix-LM 설정, 그리고 openpi `Pi0.embed_prefix`가 이미지+프롬프트에 `ar_mask=False`를 주는 방식과 동일하다.

### 4.2 위치와 타깃의 정렬

위치 i의 출력은 토큰 i+1을 예측한다. `targets = roll(tokens, -1)`로 한 칸 당긴 뒤 `take_along_axis`로 해당 log-prob을 읽는다.
따라서 **`"\n"` 토큰 위치의 logit이 정답 첫 토큰 a₁을 예측**하고, aₘ 위치의 logit이 EOS를 예측한다.

---

## 5. 토크나이즈와 인코딩 (`BackboneLM.encode`)

### 5.1 프롬프트

openpi `PaligemmaTokenizer`의 π0 포맷을 그대로 재현한다.

```python
cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
prefix_ids = tok.encode(cleaned, add_bos=True) + tok.encode("\n")
```

- BOS를 앞에 붙이고, **개행 토큰을 별도로 인코딩해 "정답 시작" 구분자**로 쓴다.
- π0.5의 `Task: …, State: …;\nAction:` 포맷은 쓰지 않는다. 우리는 action이 아니라 텍스트 정답을 묻기 때문에 PaliGemma 원래 포맷에 맞춘다.

### 5.2 정답(suffix)

```python
suffix_ids = tok.encode(answer.strip()) + ([EOS_ID] if add_eos else [])
```

`add_eos`는 yes/no 과제에서 두 변형을 모두 계산하기 위해 노출되어 있다(7.1절).

### 5.3 고정 길이 배치

- `max_len = 192`. `tokens`는 PAD(0)로 채우고, `mask`는 실제 토큰 위치만 True, `ar_mask`는 suffix 위치만 True.
- 길이 초과 시 **정답은 절대 자르지 않고 프롬프트 본문만 자른다.** BOS와 마지막 `"\n"`은 유지한다.
  ```python
  keep = max_len - len(suffix) - 2
  prefix = [BOS] + prefix[1:-1][:keep] + ["\n"]
  ```
  ScienceQA에서 hint가 긴 67건(3%)이 여기에 걸렸다. 1차 전체 평가(job 1184311)는 이 처리가 없어 실패했고, 추가 후 재실행했다.
- `prefix_len[i]`, `suffix_len[i]`를 기록해 score 단계에서 어느 구간을 합산할지 결정한다.

---

## 6. 두 가지 측정 함수

### 6.1 `score(images, prompts, answers, add_eos=True)` → `(sum_logp [b], n_tokens [b])`

```python
lo = prefix_len - 1            # "\n" 위치: a₁을 예측
hi = prefix_len + suffix_len - 1   # 마지막 정답 토큰 위치: EOS(또는 마지막 토큰) 예측
sum_logp[i] = target_logp[i, lo:hi].sum()
```

반환되는 합이 `Σ_j log p(a_j | image, prompt, a_<j)` 즉 **정답 문자열의 로그우도**다. `n_tokens`는 EOS 포함 토큰 수로, NLL/token 계산에 쓴다.
한 번의 forward로 배치 전체를 처리하므로 후보 K개를 비교하려면 forward를 K번(또는 K배 배치) 돌린다.

### 6.2 `generate(images, prompts, max_new_tokens)` → `list[str]`

1. 정답 없이 인코딩(`answers=None`). `cur = prefix_len`.
2. 매 step: 고정 shape 전체 forward → `argmax[i, cur[i]-1]`을 다음 토큰으로 선택 → `tokens/mask/ar_mask[i, cur[i]]` 갱신 → `cur += 1`.
3. EOS가 나오거나 `max_new_tokens`에 도달하거나 `max_len`에 닿으면 해당 샘플 종료. 전부 종료되면 루프 탈출.
4. `tokens[prefix_len:cur]`에서 EOS/PAD를 제거하고 sentencepiece로 디코딩.

KV cache 없이 매 step 전체를 다시 계산하는 단순 구현이다. 느리지만 shape이 고정되어 jit 재컴파일이 없고, score와 완전히 같은 forward를 쓴다.
샘플링 없이 **순수 greedy**다.

---

## 7. 벤치마크별 채점 절차 (`eval_backbone.py`)

공통: `items`를 `--batch`(기본 16, 전체 실행은 32) 단위로 잘라 이미지를 `uint8`로 로드하고, 내부에서 `[0,255] → [-1,1]`로 정규화한다.
템플릿의 `{q}`에 질문을 채워 프롬프트를 만든다(8절).

### 7.1 yes/no — POPE, MME (`run_yesno`)

네 번의 `score` 호출로 후보 4개를 얻는다.

| 호출 | 의미 |
|---|---|
| `score(…, ["yes"]*b)` | log p("yes" EOS) |
| `score(…, ["no"]*b)` | log p("no" EOS) |
| `score(…, ["yes"]*b, add_eos=False)` | log p("yes") |
| `score(…, ["no"]*b, add_eos=False)` | log p("no") |

- `pred_ll = "yes" if logp_yes > logp_no else "no"` → `ll_acc`, `ll_f1_yes`(POPE 공식 지표), `ll_yes_rate`, `ll_acc_by_category`.
- `pred_ll_noeos` → `ll_noeos_acc`, `ll_noeos_yes_rate`. lm-eval-harness 관례와의 비교용. EOS 유무에 따라 yes 비율이 크게 뒤집힐 수 있어 둘 다 기록한다.
- 생성: `generate(max_new_tokens=4)` 후 정규식 `^\W*(yes|no)\b`로 첫 단어 파싱. 매칭 실패는 `"invalid"`.
  → `gen_acc`(invalid은 오답), `gen_valid_rate`.
- MME는 `paligemma_strip` 템플릿일 때 질문에서 `"Please answer yes or no."`를 제거한 뒤 템플릿에 넣는다.

### 7.2 다지선다 — ScienceQA (`run_mc`)

- 항목마다 선택지 수(2~5)만큼 `(image, prompt, choice_text)` 쌍을 평탄화하고 `score`로 각 선택지 텍스트의 로그우도를 구한다.
- `pred_sum = argmax Σ log p`, `pred_mean = argmax (Σ log p / 토큰 수)`. 보고서 표의 `SQA ll-acc`는 **길이 정규화한 `ll_acc_mean`**이다.
- `chance = mean(1/선택지 수)` = 0.355.
- 생성: `generate(max_new_tokens=16)` 결과를 소문자·구두점·관사 제거 정규화 후 정답 선택지 텍스트와 **완전 일치**하면 정답(`gen_acc_exact`). 참고용.

### 7.3 자유 응답 VQA — VQAv2 (`run_vqa`)

- `score(…, majority)`로 다수결 정답의 `nll_majority = -Σ log p / 토큰 수` → `nll_majority_per_token`(모델 간 절대 비교용).
- 생성: `generate(max_new_tokens=8)`. 공식 VQA accuracy `min(1, #일치 annotator / 3)`을 정규화 후 계산 → `gen_vqa_acc`, `gen_vqa_acc_by_type`.

### 7.4 캡션 — COCO (`run_caption`)

- 이미지당 reference caption 최대 5개를 평탄화해 `score`로 각 reference의 NLL/token을 구하고 이미지별 평균 → `ref_nll_per_token`.
- 생성: `generate(max_new_tokens=32)`. `pycocoevalcap`의 `Cider`, `Bleu(4)`로 채점. 토큰화는 소문자·구두점 제거 자체 구현(PTBTokenizer 미사용). 빈 생성은 `"empty"`로 대체.
  추가로 `gen_mean_len_words`, `gen_empty_rate` 기록.

### 7.5 공통 정규화 (`norm_answer`)

소문자 → 구두점 제거 → `a/an/the` 제거 → 공백 정리. VQA와 ScienceQA 생성 채점에 공통 적용.

---

## 8. 프롬프트 템플릿

| 템플릿 | yes/no · VQA · MC | caption | 비고 |
|---|---|---|---|
| `paligemma` | `answer en {q}` | `caption en` | PaliGemma 사전학습 원본 형식. 주 결과 |
| `plain` | `{q}` | `Describe the image.` | VLA가 본 bare instruction에 가까운 형식 |
| `paligemma_strip` | `answer en {q}` (MME 말미 지시문 제거) | `caption en` | MME 전용 추가 실험 |

같은 체크포인트에 대해 `--templates paligemma plain`처럼 여러 템플릿을 한 실행에서 순차 평가하며, 결과 키는 `"<bench>/<template>"`이다.

---

## 9. 실행 (`jobs/eval_backbone.sbatch`)

```
#SBATCH --partition=natStr --nodes=1 --ntasks-per-node=4 --gres=gpu:4 --exclusive --time=04:00:00
```

1. `cd $SLURM_SUBMIT_DIR`, 경로가 `/scratch/$USER/` 아래인지 확인, `module purge`, `source scripts/hangang-env.sh`
   (venv 활성화, 모든 캐시를 프로젝트 `.cache/`로 고정, `XLA_PYTHON_CLIENT_PREALLOCATE=false`).
2. `srun --ntasks=<체크포인트 수> --gpus-per-task=1 bash -c '…'`
   각 task는 `SLURM_PROCID`로 `CKPTS` 배열에서 자기 체크포인트를 고르고, 이름이 `pi05*`면 `--pi05`를 붙인다.
   **GPU 1장 = 체크포인트 1개 = 독립 프로세스.** DDP가 아니다.
3. `--kill-on-bad-exit`를 쓰지 않아 한 체크포인트가 실패해도 나머지는 계속 돈다.
4. 각 task의 stdout/stderr는 `logs/eval-<name>-<jobid>.log`로 분리.

실제 제출 이력(`runs/submitted-jobs.tsv`):

| Job | 내용 | EXTRA | 결과 |
|---|---|---|---|
| 1182129 | GPU probe (JAX가 4 GPU를 보는지) | — | COMPLETED |
| 1182194 | smoke | `--limit 16` | COMPLETED |
| 1184311 | 전체 1차 | `--templates paligemma plain` | FAILED (SQA truncation) |
| 1184338 | 전체 재실행 | `--templates paligemma plain --batch 32` | COMPLETED, 39m 32s |
| 1184503 | POPE/MME noEOS 지표 추가 | `--benches pope mme` | COMPLETED, 17m |
| 1184557 | MME strip, pi05_base만 | `--benches mme --templates paligemma_strip` | COMPLETED, 2m |

---

## 10. 산출물과 집계

### 10.1 `runs/<name>/summary.json`

```json
{"name": "...", "ckpt_dir": "...", "pi05": true, "limit": null,
 "results": {"pope/paligemma": {...}, "mme/paligemma": {...}, "mme/paligemma_strip": {...}, ...}}
```

벤치 하나가 끝날 때마다 즉시 덧쓰기 저장하므로 중간 실패에도 완료분은 남는다.
`--limit`이 없으면 기존 `summary.json`의 `results`를 먼저 읽어 **키 단위로 병합**한다. 그래서 1184503, 1184557처럼 일부 벤치/템플릿만 다시 돌려 열을 추가할 수 있었다.

### 10.2 `runs/<name>/<bench>.<template>.jsonl`

항목별 raw 기록. yes/no는 네 가지 log-prob과 생성문, MC는 선택지별 `(sum_logp, n_tokens)`, VQA/COCO는 NLL과 생성문, reference를 담는다.
표에 없는 분석(카테고리별, 길이별, 사례 추출)은 전부 이 파일에서 다시 계산할 수 있다.

### 10.3 `scripts/summarize.py`

`runs/*/summary.json`을 모아 템플릿별 markdown 표를 출력한다. 열 정의는 스크립트의 `COLS`에 고정되어 있다.
`SQA ll-acc`는 `ll_acc_mean`(길이 정규화)을 쓴다.

---

## 11. 실험 설계상 결정과 그 이유

| 결정 | 이유 |
|---|---|
| likelihood 지표를 주 지표로 | 생성이 무너진 모델(π0)도 수치가 나오고, 모르면 정확히 chance가 나와 해석이 명확 |
| 생성 지표 병행 | likelihood와의 격차가 "알지만 표현 못 함"의 크기 |
| NLL/token 보조 지표 | 후보 비교와 달리 절대값이라 모델 간 "텍스트 분포에서 얼마나 멀어졌나" 직접 비교 가능 |
| EOS 포함·미포함 둘 다 | yes/no 뒤에 EOS가 올 확률이 모델·답변별로 달라 결과가 뒤집힐 수 있음 |
| MC는 길이 정규화 | 선택지 길이가 다르면 합산 log-prob이 짧은 쪽에 유리 |
| 224×224 letterbox | VLA 학습 전처리와 동일. 분포 shift를 평가 변수에서 제거 |
| 프롬프트 BOS … `"\n"` 포맷 | PaliGemma 사전학습 및 openpi π0 토크나이저와 동일 |
| action expert 미실행 | 질문이 "backbone이 VLM으로 동작하는가"이고, action expert는 텍스트 출력 경로에 관여하지 않음 |
| bf16 forward, fp32 logit | 메모리·속도 대비 정밀도. logit은 257k vocab softmax라 fp32 누적 |
| 체크포인트별 독립 프로세스 | 네 평가가 서로 독립. 하나가 죽어도 나머지 유지 |

---

## 12. 재현 시 주의할 점

- `--limit`이 있으면 기존 `summary.json`을 **병합하지 않고 새로 쓴다.** smoke 테스트는 반드시 `--out_dir runs/smoke_<name>`을 지정해 본 결과를 덮지 않게 한다.
- `--pi05`는 체크포인트 계열과 일치해야 한다. sbatch는 이름 접두사로 자동 결정하지만, 수동 실행 시 직접 넣어야 한다.
- 데이터 서브셋(VQAv2 3,000, COCO 2,000)과 letterbox 규칙을 바꾸면 기존 `runs/`와 비교할 수 없다.
- 원본 PaliGemma-3B-pt-224 기준선은 아직 없다(`checkpoints/paligemma_pt_224/` 비어 있음). 같은 파이프라인에 넣으려면 `Pi0Config` 대신 PaliGemma 단독 파라미터 트리를 `BackboneLM`이 받도록 로더만 추가하면 된다.
- 생성은 KV cache 없는 greedy라 `max_new_tokens`를 늘리면 비용이 토큰 수의 제곱에 가깝게 늘어난다.

## 13. 파일 지도

| 파일 | 역할 |
|---|---|
| `scripts/vlm_backbone.py` | `BackboneLM`: 로딩, forward, `encode/score/generate` |
| `scripts/eval_backbone.py` | 벤치 로더, 템플릿, 4개 runner, 요약 병합·저장 |
| `scripts/prepare_data.py` | HF → letterbox PNG + `items.jsonl` |
| `scripts/summarize.py` | `summary.json` → markdown 표 |
| `scripts/download_ckpt.sh` | GCS HTTPS 다운로드(크기 비교로 재개) |
| `scripts/setup_env.sh`, `scripts/hangang-env.sh` | venv 구축 / Slurm 내부 환경 활성화 |
| `jobs/eval_backbone.sbatch`, `jobs/gpu_probe.sbatch` | 평가 제출 / GPU 점검 |
| `openpi/src/openpi/models/{pi0.py,gemma.py,tokenizer.py,model.py}` | 재사용한 upstream 구성요소(`make_attn_mask`, `Embedder`, 토크나이저 포맷, `restore_params`) |
