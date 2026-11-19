# π₀.₅ AE 출력층의 Null Space: Codex용 소규모 실험 계획

- 제안 실험 ID: `semantic_safety_ae_output_nullspace_pilot_v1`
- 문서 상태: **실험 계획이며, 아래 실험은 아직 실행하지 않았다.**
- 목적: 안전 라벨을 읽을 수 있는 AE 표현과, 현재 행동 출력에 직접 영향을 주는 표현의 관계를 진단한다.
- 범위: **체크포인트 하나, 추출 위치 하나, 마지막 flow 호출 하나부터 시작한다.** 새로운 VLA 학습이나 대규모 layer 탐색은 하지 않는다.

> **Codex에 요청할 작업**
>
> 기존 데이터·분할·probe 코드를 재사용해 아래의 표현 분해, 성분별 probing, 고정 probe 검사, 소수 입력의 출력 불변성 검사를 구현하고 실행한다. 기존 결과는 덮어쓰지 않는다. 행동 steering은 핵심 진단과 분리된 선택 실험으로 구현한다. 결과가 예상과 달라도 그대로 기록하며, probe 점수 변화를 행동 안전성 개선으로 해석하지 않는다.

---

## 1. 왜 이 실험을 하는가?

기존 실험에서는 동결한 VLM과 Action Expert(AE)의 representation에서 안전·위험 라벨을 선형적으로 분류할 수 있었다. 그러나 probe는 연구자가 별도로 학습한 분류기이며, 원래 VLA가 이 probe를 사용해 행동을 결정하는 것은 아니다. [R1 §2, §12]

동일한 AE 표현을 두 함수가 서로 다르게 읽는다.

```text
                         ┌─ action_out_proj ─→ flow velocity ─→ 행동 후보 갱신
AE representation h ────┤
                         └─ 별도 linear probe ─→ 안전·위험 분류 점수
```

따라서 이번에 확인하려는 질문은 세 가지다.

| 질문 | 대응 실험 |
|---|---|
| 현재 출력에 직접 반영되는 성분과 반영되지 않는 성분에서 각각 안전 라벨이 읽히는가? | `h`, `h_out`, `h_null`에 각각 probe 학습 |
| 출력에 반영되지 않는 성분을 제거하면 기존 probe의 판단은 달라지는가? | **기존 scaler와 probe를 고정**하고 입력만 변경 |
| 그때 실제 velocity와 최종 action은 유지되는가? | 실제 추론 경로에 제한적으로 개입해 출력 비교 |

이후 선택적으로, **위험 라벨 점수를 높이는 방향에 개입했을 때 실제 행동도 멈추는가**를 탐색한다. 이것은 별도 가설이다. 위험 관련 표현의 강화가 거절이나 정지로 이어진다고 전제하지 않는다.

### 기존 결과가 제공하는 동기

Benign adaptation의 Both 모델에서 flow 10의 AUROC는 Track A에서 projection 전 0.854, 후 0.579이고, Track B에서는 전 0.748, 후 0.558이다. 이 차이는 분석의 동기이지, 출력층이 안전 지식을 제거했다는 증거는 아니다. 차원, pooling, 표준화와 규제의 영향도 있다. [R2 §7]

또한 보고서의 benign motor 성공률은 유해 실행률이 아니다. 기존 probe 실험도 환경에서 action을 실행하지 않았다. 이번에 행동 효과를 확인하려면 별도의 rollout이 필요하다. [R2 §5–6]

---

## 2. 개념을 작은 예시로 이해하기

설명용으로 representation이 `h = (2, 0)`이라고 하자. 행동 출력은 두 숫자의 **합**, probe는 두 숫자의 **차이**를 읽는다고 가정한다.

| 표현 | 행동 출력: `h₁ + h₂` | Probe 점수: `h₁ − h₂` |
|---|---:|---:|
| `(2, 0)` | 2 | 2 |
| `(0, 2)` | 2 | −2 |

표현을 바꿨더니 probe 점수는 바뀌지만 행동은 그대로다. 두 좌표가 각각 안전·운동을 뜻하는 것이 아니라, **두 함수가 같은 좌표를 다른 조합으로 읽는 것**이다.

이때 원래 표현은 다음처럼 나눌 수 있다.

```text
h = (2, 0)
  = (1, 1) + (1, −1)
      ↑          ↑
    h_out      h_null

행동 출력에서:
(1, 1)의 기여 = 2
(1, −1)의 기여 = 0
```

**Null space는 벡터 자체가 0인 공간이 아니라, 해당 출력층에 넣었을 때 변화가 0으로 상쇄되는 방향들의 공간이다.**

이번 실험에서의 용어는 다음과 같다.

| 용어 | 의미 | 의미하지 않는 것 |
|---|---|---|
| `h_out` | 현재 선형 출력에 영향을 줄 수 있는 성분 | 순수한 운동 지식, 안전한 행동 방향 |
| `h_null` | 현재 선형 출력에는 직접 영향을 주지 않는 성분 | 순수한 안전 지식, 모델 전체에서 쓸모없는 정보 |
| Safety-relevant | 주어진 데이터에서 안전·위험 라벨과 관련되어 읽히는 신호 | 위험을 인과적으로 이해하거나 거절에 사용하는 정보라는 확정 |

**우리는 의미를 기준으로 안전·운동 정보를 분리하는 것이 아니라, 출력 행렬을 기준으로 분해한 뒤 각 성분에서 무엇이 읽히는지 조사한다.**

---

## 3. 첫 실행의 범위와 기존 자료

### 3.1 기본 설정 제안

| 항목 | 첫 실험의 설정 |
|---|---|
| 체크포인트 | Benign adaptation의 **Both**, 기존 선택 step 8,000 |
| 선정 이유 | 새 benign motor 능력이 보고되어 있고, projection 전후 probing 차이를 조사하기 적합한 pilot 대상. 최적 모델이라는 주장은 아님 |
| 추출 위치 | **`velocity_pre`**, 최종 normalization 이후 `action_out_proj` 입력 |
| Flow 호출 | **`flow_10`**, 마지막 denoising 호출 |
| 환경 시점 | 초기 장면만 |
| 특징 | Action token 10개를 평균한 1,024차원 표현 |
| 출력 행렬 | 해당 checkpoint의 LoRA 효과를 포함한 **유효 출력 행렬** |
| 내부 출력 / 실행 action | 32차원 / 보고서상 앞의 7차원 |
| 데이터 | 기존 A 336개, B 121개를 **별도 트랙으로** 평가 |
| 분할 | 기존 연결 그룹 20개와 5-fold train/validation/test 그대로 재사용 |
| 추론 | 기존 전처리, `eval`, 고정 가중치, 기존 seed `20261116`의 초기 noise 재사용 |

이 설정의 근거는 [R1 §3–7], [R2 §4–7]이다. 체크포인트 선택 파일과 실제 로더를 확인하고, 보고서와 다른 부분이 있으면 기록한다. 처음부터 다른 모든 모델·layer·flow로 확대하지 않는다.

### 3.2 Codex가 먼저 확인할 기존 파일

아래는 보고서에 기록된 **저장소 상대경로**다. 실제 작업 저장소에서 존재 여부와 파일 내부 구조를 확인해야 한다. 보고서만으로 HDF5 key나 적응 모델의 feature 경로를 추정해 하드코딩하지 않는다.

```text
scripts/semantic_safety/contrast_initial_linear.py
scripts/semantic_safety/run_benign_adaptation_v1.py
runs/semantic_safety_contrast_initial_linear_v1/manifest.json
runs/semantic_safety_contrast_initial_linear_v1/experiment_protocol.json
runs/semantic_benign_adaptation_v1/frozen_selection.json
runs/semantic_benign_adaptation_v1/experiment_protocol.json
checkpoints/pi05_libero_benign_adaptation_v1/all/adapter.pt
```

기존 `velocity_pre / flow_10`의 feature·scaler·probe를 찾아 재사용한다. **AE layer 17 residual의 probe와 바꿔 사용하지 않는다.** 두 위치 사이에는 final normalization이 있다. [R1 §5]

필요한 파일이 없으면 누락 항목을 보고하고 가능한 offline 분석까지만 진행한다. 기존 실험 폴더나 checkpoint를 변경하지 않는다.

---

## 4. 분해 방법: 필요한 수식만

이 절에서는 벡터를 열벡터로 쓴다. Action token 하나의 projection 직전 표현을 `h`라고 하면 다음과 같다.

\[
v = Wh + b,
\qquad W\in\mathbb R^{32\times1024}.
\]

현재 checkpoint에서 실제 사용하는 `W`를 SVD로 분해하여 row space의 직교기저 `Q`를 구한다. `Q`의 크기는 `1024 × r`이고 `r`은 실제 수치 rank다.

\[
P_{\mathrm{out}}=QQ^\top,
\qquad h_{\mathrm{out}}=P_{\mathrm{out}}h,
\qquad h_{\mathrm{null}}=h-h_{\mathrm{out}}.
\]

정확한 산술에서는 다음이 성립한다.

\[
h=h_{\mathrm{out}}+h_{\mathrm{null}},
\qquad Wh_{\mathrm{null}}=0,
\qquad Wh_{\mathrm{out}}+b=Wh+b.
\]

PyTorch처럼 token을 행으로 저장한 `H[..., 1024]`에서는 개념적으로 `H_out = (H @ Q) @ Q.T`, `H_null = H - H_out`이다. 큰 `1024 × 1024` 행렬을 만들 필요는 없다.

### 최소 구현 확인 사항

- **Raw hidden 좌표에서 분해한 뒤** probe용 표준화를 한다. `W`와 표준화된 좌표를 섞지 않는다.
- LoRA가 활성화된 경우 원래 `weight`만 사용하지 않는다. `eval`에서 실제 projection이 `Wh+b`로 재현되는지 소수 입력으로 확인한다. 확인되지 않으면 null-space 불변성을 주장하지 않는다.
- SVD는 FP32 또는 FP64로 계산하고, 사용한 rank와 cutoff를 기록한다. 결과가 좋아지도록 작은 singular value를 임의로 버리지 않는다.
- Bias는 그대로 유지한다. Null 조건은 `Wδ=0`이지 `Wδ+b=0`이 아니다.

평균 pooled feature도 같은 방식으로 분해할 수 있다. 선형 연산이므로 평균과 projection의 순서를 바꾸어도 수학적으로 같다. 다만 **평균 feature를 수정한 것은 offline 분석이지, 실제 token별 action을 수정한 것이 아니다.**

---

## 5. 핵심 실험: 세 단계면 충분하다

### 실험 A. 성분별로 새 probe를 학습한다

**질문: 각 성분에서 안전·위험 라벨을 얼마나 잘 읽을 수 있는가?**

| 조건 | Probe 입력 | 학습 |
|---|---|---|
| Full | 원래 pooled `h` | 기존 해당 위치의 probe 재현·재사용 |
| Output-related | `h_out` | 해당 특징의 train에서 scaler와 probe를 새로 학습 |
| Null | `h_null` | 해당 특징의 train에서 scaler와 probe를 새로 학습 |
| Output 기준선 | 평균 `velocity_post` | 기존 해당 위치 결과 재사용 가능 |

기존 recipe를 유지한다. Train-only StandardScaler와 그룹·클래스 보정 가중치, L2 logistic regression, `C ∈ {0.01, 0.1, 1.0}`, validation AUROC로 C 선택, `logit ≥ 0` 판정을 재사용한다. A와 B는 합치지 않는다. [R1 §7]

보고할 값은 **5-fold test AUROC와 balanced accuracy**, fold별 값, OOF confusion matrix면 충분하다. Fold SD를 신뢰구간이라고 부르지 않는다.

해석상의 두 제한을 함께 기록한다.

1. `h_null`의 자유도가 `h_out`보다 훨씬 크므로 점수 차이를 **안전 정보량의 비율**로 해석하지 않는다. 이번 pilot은 접근성을 비교한다. 정밀한 차원 통제는 후속 과제로 남겨도 된다.
2. `h_out`과 `Wh`는 선형적으로 서로 복원 가능하다. 두 probe의 성능 차이는 유한 표본·좌표별 표준화·규제 등의 영향일 수 있으며, 다른 정보량을 증명하지 않는다.

### 실험 B. 기존 probe를 고정하고 `h_null`을 제거한다

**질문: 원래 probe의 판정은 출력에 직접 반영되지 않는 성분에 얼마나 의존했는가?**

실험 A와 달리 **원래 full `h`에서 학습한 scaler와 probe를 그대로 고정**한다. 각 test sample에 그 sample의 그룹을 학습하지 않은 fold의 probe를 적용한다.

```text
변경 전: original_probe(original_scaler(h))
변경 후: original_probe(original_scaler(h_out))

h_out = h - h_null
```

여기서는 scaler·probe를 다시 학습하지 않는다. 다시 학습하면 다른 질문을 평가하게 된다.

기록할 값은 sample별 logit 변화, 안전↔위험 판정 전환 수, AUROC·balanced accuracy 변화다. 위험 점수가 내려갈 것이라고 가정하지 않는다. 올라가거나 거의 바뀌지 않을 수도 있다.

> **`h_null`을 제거한다는 것은 `h_out`을 남긴다는 뜻이다. 정책에 `h_null`만 넣으면 원래 출력을 보존하지 못한다.**

변경 후 feature는 원래 학습 분포 밖일 수 있다. 따라서 점수 변화는 기존 probe의 의존성과 개입 민감성을 보여주며, 자연스러운 안전 판단의 개선을 뜻하지 않는다.

### 실험 C. 실제 추론에서도 출력이 유지되는지 소수 입력으로 확인한다

**질문: Probe 점수가 바뀌더라도 실제 projection 출력과 최종 action은 유지되는가?**

처음에는 안전·위험을 포함한 train/validation 입력 8–16개를 ID 기준으로 고정해 사용한다. 결과가 흥미로운 사례만 고르지 않는다. 모델 가중치는 바꾸지 않는다.

같은 입력·같은 noise로 다음을 비교한다.

| 실행 | 처리 |
|---|---|
| Baseline | Hook 없이 추론 |
| Identity hook | 값을 바꾸지 않는 hook으로 추론 |
| Null removal | **마지막 flow의 `velocity_pre`에서만** 각 action token `h_j`를 `h_out,j`로 대체 |

Hook은 `action_out_proj`의 **입력**을 바꾼다. Final normalization 이전이나 projection 출력에 개입하는 실험과 혼동하지 않는다. 호출 횟수와 flow time을 확인해 한 정책 호출의 마지막 flow에만 적용한다.

비교할 항목은 다음과 같다.

- 고정 probe의 개입 전후 logit.
- Token별 32차원 velocity의 최대 절대 차이와 RMSE.
- 최종 32차원 action chunk 및 후처리한 실제 7차원 action의 차이.

수학적으로는 출력 불변을 예상하지만, BF16·LoRA 연산 순서·반올림 때문에 실제 차이가 생길 수 있다. FP32/64의 대수적 검사와 실제 추론 dtype의 검사를 구분한다. Test를 보고 허용 오차를 늘리지 말고, identity-hook 차이와 dtype을 기준으로 수치 기준을 먼저 기록한다. 설명되지 않는 큰 차이가 나면 위치·LoRA·좌표 처리를 확인하고 steering으로 넘어가지 않는다.

**이 단계에서는 환경 rollout이 필수는 아니다.** 우선 최종 action이 유지되는지까지 확인하면 된다. 출력 불변은 예상된 대수적 성질이고, 실제로 조사하는 것은 그 상태에서 기존 probe의 판정이 얼마나 달라지는가다.

---

## 6. 선택 실험: 위험 관련 표현을 강화하면 실제로 멈추는가?

이 절은 핵심 진단 이후의 작은 탐색이다. **Null-space projection 자체가 거절을 학습시키지는 않는다.** 아래 두 접근을 분리해서 평가한다.

### 6.1 자연적인 내부 연결 탐색: 위험 점수를 높이는 방향에 개입

기존 full probe가 다음처럼 작동한다고 하자.

\[
s(h)=w^\top\left((h-\mu)\oslash\sigma\right)+c.
\]

`unsafe = 1`이므로 점수를 높이는 raw 좌표 방향의 계수는 다음이다.

\[
a=w\oslash\sigma.
\]

`coef_`를 그대로 raw activation에 더하지 않는다. 저장된 실제 scaler의 `scale_`를 사용한다. 이때 단순히 위험 class 점수를 높이는 것이며, 의미적 안전 이해를 강화했다고 부르지 않는다.

방향을 다음처럼 나누면 탐색의 목적이 명확해진다.

\[
a_{\mathrm{out}}=P_{\mathrm{out}}a,
\qquad a_{\mathrm{null}}=a-a_{\mathrm{out}}.
\]

| 방향 | 역할 |
|---|---|
| `a_null` | Probe 점수는 바뀔 수 있지만 현재 projection 출력은 유지되어야 하는 대조 |
| `a_out` | 현재 projection 출력을 실제로 바꿀 수 있는 위험 점수 방향 |

방향의 norm이 너무 작으면 정규화해 증폭하지 말고 해당 조건을 생략하고 기록한다. 우선 `a_out`을 단위 길이로 만든 `d_out`에 대해 다음 작은 개입만 시험한다.

\[
h'_j=h_j+\alpha d_{\mathrm{out}}.
\]

**첫 범위:** 같은 `velocity_pre / flow_10`에서 `α=0`, 작은 양수, 같은 크기의 음수와 동일 norm의 무작위 row-space 방향을 비교한다. 강도는 train feature 크기를 기준으로 정하고 validation에서 action 크기를 점검한다. 예를 들어 train pooled-feature norm 중앙값의 1% 정도를 작은 pilot 시작값으로 사용할 수 있지만, 보편적으로 안전한 값이라는 뜻은 아니다. 큰 강도·여러 layer·모든 flow 탐색으로 확장하지 않는다.

양수 방향은 위험 라벨 점수를 높일 뿐이다. 실제 행동은 정지할 수도, 변화가 없을 수도, 더 위험해질 수도 있다. 음수와 무작위 방향을 함께 보는 이유는 **의미적 거절과 일반적인 행동 교란을 구분하기 위해서**다.

#### 중요한 해석 제한

동일 fold에서 모든 sample에 같은 `αd`를 더하면, 고정 linear probe의 logit에는 같은 상수가 더해진다. 따라서 판정·확률은 달라져도 이 순수한 점수 이동만으로는 AUROC가 개선되지 않는다.

또한 마지막 선형층 직전의 고정 방향 개입은 `Δv = αWd`라는 출력 편향이다. 동작이 줄어들었다고 곧바로 내부 reasoning이나 거절 메커니즘을 활성화했다고 해석하지 않는다. 다른 상태·정상 대응 작업에서도 선택적인 효과가 있는지 봐야 한다.

### 6.2 작은 행동 검증: 분류 점수가 아니라 실행 결과를 본다

실행은 **시뮬레이션에서만** 수행한다. 초기 상태에서 안전한 유지 제어가 가능한 task로 한정하고, 우선 정상·위험 대응쌍 5–10개 정도만 사전에 선정한다. 이는 feasibility pilot이며, 전체 benchmark 성능으로 제시하지 않는다.

기준 정책이 해당 정상 counterpart를 수행할 수 있는지 확인한다. 선택 기준과 제외 사례를 기록하고, steering 성공 사례만 남기지 않는다. 이미 본 test 결과로 강도를 고르지 않는다. 탐색용 validation과 평가용 held-out 조건을 구분하고, 각 조건에서 해당 그룹을 학습하지 않은 probe 방향을 사용한다.

먼저 첫 호출의 action 변화만 확인한다. Rollout으로 확장할 경우에는 **매 VLA 재추론의 마지막 flow에 같은 규칙을 적용한다**고 명시한다. 환경 control step마다 flow가 실행된다고 가정하지 않는다. 초기 장면으로 학습한 probe가 중간 상태에도 유효한지는 아직 검증되지 않았다는 한계를 남긴다.

주요 관측은 다음 네 가지면 충분하다.

| 측정 | 확인할 내용 |
|---|---|
| 정상 counterpart 성공률 | 기존 skill이 유지되는가? |
| 금지 상호작용 진입 여부 | 유해 목표를 향한 진행이 줄어드는가? |
| 검증된 정지·유지 여부 | 단순 실패가 아니라 정의한 중단 상태인가? |
| Probe 점수와 action 변화 | 점수 변화가 실제 실행 변화와 일치하는가? |

금지 상호작용은 task별로 관측 가능한 predicate를 사전에 정한다. 기존 motor success predicate를 그대로 안전 판정으로 사용하지 않는다. 판정기가 없다면 action 변화와 영상 관찰까지만 보고하고, 안전 차단률을 만들어내지 않는다. 실제 손상 대신 시뮬레이션의 대리 사건을 측정한다는 한계를 유지한다. [R2 §3]

### 6.3 별도 실행 연결: 원래 probe → HOLD

내부 방향 강화와 별도로, 가장 간단한 실행 연결을 기준선으로 둔다.

```text
개입 전 representation → 원래 probe
                           ├─ 허용 → 원래 action 실행
                           └─ 위험 → 잔여 action queue 취소 → 검증된 HOLD
```

HOLD는 특별한 모델 token이 아니라 **현재 제어 인터페이스에서 자세·그리퍼 상태를 안정적으로 유지하고 기존 목표를 중단하는 동작**이다. 초기 안정 상태에서만 시작하고, 물리 시뮬레이션을 계속 진행하여 정해진 구간 동안 유지되는지 확인한다. 시뮬레이터를 즉시 종료하거나 시간을 멈춘 것을 성공으로 세지 않는다.

정규화된 action을 전부 0으로 만들거나 flow velocity를 0으로 만드는 것을 HOLD로 간주하지 않는다. 후자는 현재 행동 후보의 갱신을 멈출 뿐이다. Queue와 실제 제어 명령을 확인해야 한다.

Gate에는 **개입 전** 점수를 사용한다. 위험 방향을 강화해서 스스로 gate를 발동시키면, “원래 내부 연결 때문에 멈췄다”와 “외부 규칙으로 멈췄다”를 혼동하게 된다. 정답 label은 사후 평가용이며, test에서 gate나 개입 여부를 정하는 입력으로 쓰지 않는다.

`probe → HOLD`가 작동하면 기존 정보에 실행 제한 경로를 연결한 결과다. 내부 steering이 작동하지 않아도 정보가 없었다는 뜻은 아니다. 반대로 gate의 성공은 원래 VLA에 거절 회로가 있었다는 증거가 아니다.

---

## 7. 결과를 어떻게 해석할 것인가?

| 결과 | 가능한 결론 | 피해야 할 결론 |
|---|---|---|
| `h_null`의 새 probe가 잘 작동 | 출력에 직접 반영되지 않는 성분에서도 라벨이 읽힘 | 안전 지식 전체가 null space에 있음 |
| Null 제거 후 고정 probe 판정만 변화, action 유지 | 해당 readout과 현재 action 출력이 분리될 수 있음 | 모델 전체가 안전 정보를 한 번도 사용하지 않음 |
| Null 제거 후 판정이 거의 유지 | 기존 probe가 제거 성분에 덜 의존했거나 정보가 다른 성분에도 있음 | 안전 정보가 없음 |
| 위험 방향 강화 후 score만 상승 | 위험 class 점수를 높이는 데 성공 | 안전 이해·거절 능력 향상 |
| 위험·정상 task 모두 수행 불능 | 일반적인 출력 교란 가능성 | 의미적 안전성 개선 |
| 위험 실행 감소, 정상 counterpart 유지 | 선택적 제어 가능성을 후속 검증할 근거 | 작은 pilot만으로 일반적인 안전 보장 |
| 원래 probe → HOLD만 작동 | 읽힌 정보를 새 실행 제어 경로에 연결할 수 있음 | 원래 AE가 자발적으로 거절함 |

안전·위험 라벨은 요청 수준의 후보 라벨이며, 물체·언어 shortcut과 의미 이해가 완전히 분리된 상태는 아니다. 이번 분석으로 기존 데이터의 한계가 없어지지는 않는다. [R1 §12.3; R2 §8]

---

## 8. Codex의 최소 산출물과 완료 기준

새 폴더 `runs/semantic_safety_ae_output_nullspace_pilot_v1/` 아래에 다음 정도만 남긴다. 새로운 대형 framework는 만들지 않는다.

| 산출물 | 내용 |
|---|---|
| 설정 기록 | Checkpoint·위치·split·noise·유효 W·rank·수치 기준·원본 파일 식별 정보 |
| Probe 결과 CSV | 성분별 새 probe와 기존 probe 고정 실험을 구분한 fold 지표·sample 예측 |
| 수치 검사 JSON | 분해 복원, `W h_null`, projection 재현, hook 전후 velocity/action 차이 |
| `results.md` | 질문별 결과 표, 쉬운 해석, 한계, 재실행 명령 |
| 선택 steering 결과 | 수행한 경우만 action 비교·rollout 지표·대표 영상; 미수행이면 이유 명시 |

**핵심 완료 기준:** 성분별 probing과 고정 probe 검사를 구분했고, `h_null` 제거의 출력 보존 여부를 실제 추론에서 확인했으며, 무엇이 읽히는지와 무엇이 행동을 바꾸는지를 혼동하지 않고 보고했다.

코드·환경의 실제 실행 명령은 구현 후 검증한 명령만 기록한다. 기존 결과 폴더를 덮어쓰거나 전체 VLA를 재학습하지 않는다. 핵심 실험이 끝났다고 자동으로 모든 모델·layer·flow·동적 위험·RL까지 확대하지 않는다.

### 이번 pilot이 최종적으로 답할 질문

> **AE에서 안전 라벨을 읽게 해 준 신호는 현재 출력층과 어떤 관계가 있으며, 그 신호의 판정 변화와 실제 행동 변화는 어디까지 분리되는가?**
>
> 그다음의 별도 질문은 **“이 정보를 어떤 실행 제어 경로에 연결해야 정상 skill을 보존하며 위험 행동을 멈출 수 있는가?”**다.

---

## 근거 문서

- **[R1]** `Semantic_Safety_Contrast_Initial_Linear_v1_Results.md` — §2: frozen VLA와 probe, §4–5: 입력·추출 위치, §6–8: 분할·학습·집계, §12: 해석 제한, §14: 기존 코드·결과 경로.
- **[R2]** `Semantic_Safety_Benign_Adaptation_v1_Results.md` — §3: motor 판정 한계, §4–5: LoRA 범위·선택 checkpoint·능력 평가, §6–7: probe 프로토콜·projection 전후 결과, §9: 파일 경로.

위 문서에서 확인한 기존 설정·결과와, 이 계획에서 새로 제안한 실험을 구분한다. 이 문서는 두 보고서를 읽어 작성한 계획이며, 원시 feature·checkpoint를 분석하거나 새 행동 결과를 측정한 보고서가 아니다.
