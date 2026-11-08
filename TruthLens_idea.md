**기술적으로 적용 가능한 방향이다. 다만 첫 실험의 목표를 “VLA가 입력받은 요청의 위험성을 special token으로 탐지하는 것”으로 한정하고, 이후 “실제로 생성한 행동의 위험성을 탐지하는 것”으로 확장하는 구성이 적절하다.**

두 문제는 서로 다르다. 예를 들어 `이미지: 칼과 아기 / 지시: 칼을 아기에게 줘라`에서 요청 자체가 위험하다는 판단은 action trajectory를 생성하기 전에도 가능하다. 반면 정상적인 물건 정리 요청에서 로봇 팔이 사람과 충돌할지는 구체적인 행동 정보까지 확인해야 한다.

TruthLens는 객체 존재 여부를 special-token score에 정렬하는 방법이지, 로봇의 위험 행동을 검증한 연구는 아니다. 따라서 아래에서는 **원논문에서 가져올 수 있는 메커니즘과 VLA에 맞춰 새로 설계해야 하는 부분을 구분**하겠다. 원논문에서 직접 가져오는 것은 “기존 vocabulary의 드문 token을 scalar 출력 통로로 사용하고, 그 점수를 supervision으로 학습한다”는 구조다. ECCV_2026_TruthLens Object Hall…

# 1. 먼저 무엇을 탐지할 것인지 정의해야 한다

## 1.1 첫 번째 목표: 입력된 요청의 위험성

가장 먼저 구현할 대상으로 다음 문제를 제안한다.

\[
\boxed{
R_{\mathrm{request}}(I,X)
=
\text{현재 장면에서 지시 }X\text{를 수행하는 목표가 위험한가?}
}
\]

여기서 \(I\)는 이미지이고, \(X\)는 instruction이다. 로봇 상태가 필요한 설정에서는 실제로 readout이 접근할 수 있는 상태 정보까지 입력에 포함할 수 있다.

예를 들면 다음을 구분하는 것이다.

| 이미지 | Instruction | 요청 수준의 판단 |
|---|---|---|
| 칼과 아기 | 칼을 아기에게 줘라 | 위험한 목표 |
| 같은 이미지 | 칼을 아기가 접근할 수 없는 잠금 보관함에 넣어라 | 안전을 위한 목표 |
| 칼과 아기 | 왼쪽 물건을 아기에게 줘라 | 왼쪽 물건이 칼이라면 위험 |
| 큰 봉제 인형과 아기 | 왼쪽 물건을 아기에게 줘라 | 정의된 실험 조건에서 허용 가능한 목표 |

이 표의 label은 **요청의 목표에 대한 판단**이다. 잠금 보관함에 넣으라는 지시가 허용 가능하더라도, 실제 로봇이 이동 중 충돌하지 않는다는 의미는 아니다.

여기서 중요한 것은 `칼` 또는 `아기`라는 객체 자체를 위험 label로 학습하지 않는다는 점이다.

\[
\text{위험성}
=
f(\text{객체},\text{대상},\text{관계},\text{지시된 행동})
\]

를 학습해야 한다.

따라서 TruthLens의 object-token supervision을 그대로 복사하기보다는, **이미지와 전체 instruction을 모두 처리한 위치에 하나의 supervision을 부여하는 방식**이 적절하다.

## 1.2 두 번째 목표: 실제 생성된 행동의 위험성

별도의 문제로 다음을 정의할 수 있다.

\[
\boxed{
R_{\mathrm{action}}(I,X,s,A)
=
\text{후보 행동 }A\text{를 실행하면 안전 제약을 위반하는가?}
}
\]

\(s\)는 로봇 상태, \(A\)는 다음 action 또는 action chunk다.

이 문제에서는 동일한 이미지와 instruction에서도 행동에 따라 label이 달라질 수 있다.

\[
R_{\mathrm{action}}(I,X,s,A_{\mathrm{safe}})
\neq
R_{\mathrm{action}}(I,X,s,A_{\mathrm{unsafe}})
\]

반면 행동을 입력으로 보지 않는 request detector는 두 후보를 구분할 수 없다.

**현재 제시한 칼–아기 사례는 첫 번째 문제부터 시작하기에 적합하다. 하지만 연구 결과를 “위험한 trajectory를 탐지했다”라고 해석하려면 두 번째 문제까지 별도로 검증해야 한다.**

# 2. 공통 설계: object 위치 대신 “행동 결정 직전 위치”에서 점수를 읽는다

## 2.1 Caption을 생성할 필요가 없다

TruthLens에서는 caption을 생성한 뒤 객체 위치를 찾았다. VLA의 요청 위험 탐지에서는 그 과정을 생략할 수 있다.

제안하는 흐름은 다음과 같다.

```text
이미지 + 원래 로봇 instruction
              ↓
       VLA의 기존 forward
              ↓
전체 입력을 처리한 hidden state h_dec
              ↓
      기존 LM vocabulary projection
              ↓
지정된 special token의 log-probability
              ↓
          요청 위험 점수
```

여기서 \(h_{\mathrm{dec}}\)는 다음을 만족하는 위치에서 읽는다.

> **이미지와 전체 instruction을 볼 수 있고, 실제 action을 생성하기 직전의 문맥을 나타내는 위치.**

원본 OpenVLA에서는 첫 action token을 예측하는 prompt boundary가 자연스러운 후보다. π 계열에서는 image–language prefix의 마지막 유효 위치가 첫 후보가 된다. 다만 padding, prompt template, attention mask를 확인하여 해당 위치가 실제로 어떤 입력을 볼 수 있는지 검증해야 한다.

**`칼`이라는 단어가 등장한 위치에만 loss를 주는 방식은 첫 실험으로 권하지 않는다.** 그 위치가 전체 지시를 보지 못하는 구조에서는 “칼을 아기에게 전달”과 “칼을 안전하게 보관”을 충분히 구분하지 못할 수 있기 때문이다.

## 2.2 Special-token risk score

모델의 기존 LM head를 \(W_{\mathrm{LM}}\), 선택한 special token을 \(z_r\)라고 하자.

\[
\ell_\theta(c)=W_{\mathrm{LM}}h_{\mathrm{dec},\theta}(c)
\]

\[
\boxed{
R_\theta(c)
=
\beta
\left[
\log\operatorname{softmax}\bigl(\ell_\theta(c)\bigr)_{z_r}
-
c_{\mathrm{ref}}
\right]
}
\]

여기서 \(c\)는 readout이 실제로 접근하는 입력 문맥이다.

학습 label은 다음처럼 정의하면 된다.

\[
y_{\mathrm{risk}}=
\begin{cases}
1 & \text{위험한 요청}\\
0 & \text{허용 가능한 요청}
\end{cases}
\]

따라서 loss는

\[
\mathcal L_{\mathrm{risk}}
=
\mathbb E_{(c,y)}
\left[
w_y\bigl(R_\theta(c)-y\bigr)^2
\right]
\]

로 시작할 수 있다. \(w_y\)는 class imbalance를 보정하는 가중치다.

이때 원논문에서는 `Real=1`이었지만, 여기서는 설명의 편의를 위해 **`Unsafe=1`로 label 방향을 바꾼 것**이다. 메커니즘 자체에는 문제가 없다. 원논문의 핵심 역시 special-token log-probability를 변환한 값에 binary target을 회귀시키는 것이다. ECCV_2026_TruthLens Object Hall…

그리고 다음 세 가지를 명확히 해야 한다.

**Special token을 실제로 출력하지 않는다.** 기존 vocabulary 출력의 해당 차원만 읽는다.

**별도의 reward model은 필요하지 않다.** 이 단계는 위험 label에 대한 supervised regression이며, 강화학습을 요구하지 않는다.

**\(R=0.8\)이 위험 확률 80%라는 뜻은 아니다.** 이 점수는 우선 회귀 출력이다. 확률적 해석이나 실행 threshold는 별도 validation으로 정해야 한다.

## 2.3 무엇을 학습할 것인가?

첫 구현은 앞서 논의한 TruthLens 설정에 가깝게 다음과 같이 구성하는 것이 좋다.

\[
\boxed{
\text{기존 가중치와 LM head는 고정}
+
\text{VLM attention의 LoRA만 학습}
}
\]

TruthLens 공개 학습 코드는 `q_proj`, `k_proj`, `v_proj`, `o_proj`를 LoRA 대상으로 지정한다. 따라서 VLA에서도 이에 대응하는 attention projection만 우선 학습하는 조건을 만들 수 있다. [GitHub](https://github.com/wyqstan/TruthLens/blob/main/train/train.py)

그러면

\[
\Delta\theta_{\mathrm{LoRA}}
\rightarrow
h_{\mathrm{dec}}\text{ 변화}
\rightarrow
P(z_r)\text{ 변화}
\]

를 통해 위험 점수가 학습된다.

단, 이것은 **기존에 존재하던 위험 정보를 읽기만 하는 실험은 아니다.** Backbone의 표현 자체가 바뀔 수 있다. 기존 safety 정보의 존재 여부는 frozen-backbone linear probe로 따로 확인해야 한다.

# 3. 모델별 적용 방법

단순히 “action expert가 있는 모델/없는 모델”로만 나누기보다 다음 세 구조를 구분하는 것이 정확하다.

| 모델 | 행동 출력 구조 | 우선 제안하는 risk readout |
|---|---|---|
| 원본 OpenVLA | LM vocabulary에서 discrete action token 생성 | 첫 action token 예측 위치의 LM 출력 |
| OpenVLA-OFT의 대표적인 L1 설정 | Backbone hidden state에서 연속 action head로 action chunk 예측 | Prompt boundary 또는 action-position hidden state의 vocabulary readout |
| π₀ / π₀.₅의 flow-matching 설정 | VLM prefix와 별도 action expert | VLM prefix의 hidden state에서 vocabulary readout |

원본 OpenVLA는 vocabulary token을 action discretization에 사용하지만, OpenVLA-OFT는 대표적으로 parallel decoding, continuous action representation, L1 regression을 사용한다. 따라서 **OpenVLA-OFT에는 π 계열과 같은 별도 Transformer action expert는 없더라도, 별도의 action head는 있다.** [GitHub](https://raw.githubusercontent.com/openvla/openvla/main/prismatic/vla/action_tokenizer.py)

## 3.1 원본 OpenVLA: 가장 직접적으로 적용할 수 있는 경우

원본 OpenVLA에서는 다음 구조를 사용할 수 있다.

\[
(I,X)
\rightarrow
h_{\mathrm{dec}}
\rightarrow
W_{\mathrm{LM}}h_{\mathrm{dec}}
\]

이 vocabulary 출력에서 두 가지를 동시에 얻는다.

\[
\begin{aligned}
\text{Action 출력} &: \text{기존 action token 확률}\\
\text{Risk 출력} &: \log P(z_r)
\end{aligned}
\]

따라서 첫 action token을 예측하는 시점에서 요청 위험 점수를 읽고, 높은 경우 행동 명령을 로봇에 전달하지 않는 구조가 가능하다.

이 모델에서는 **special token 선택이 특히 중요하다.** OpenVLA의 action tokenizer는 vocabulary의 사용 빈도가 낮은 뒤쪽 token들을 action 표현으로 재사용한다. 그러므로 “드물게 사용되는 token이면 아무거나 선택한다”는 방식은 action token과 충돌할 수 있다. 선택한 \(z_r\)가 action token ID 범위와 겹치지 않는지 확인해야 한다. [GitHub](https://raw.githubusercontent.com/openvla/openvla/main/prismatic/vla/action_tokenizer.py)

또한 점수는 action decoding을 위한 vocabulary mask나 sampling processor를 적용하기 **이전의 raw logits**에서 읽는 것이 적절하다. Action token만 허용하도록 masking한 분포에서는 \(z_r\)의 확률이 0으로 제거될 수 있기 때문이다.

첫 실험에서는 `<unk>`를 후보로 검사할 수 있지만, 모델별 token ID와 출력 처리 과정을 확인하기 전에는 고정하지 않는 편이 좋다.

## 3.2 π₀ / π₀.₅: VLM prefix에 risk readout을 추가한다

π 계열에서는 요청 위험 탐지와 행동 생성을 다음처럼 분리할 수 있다.

```text
이미지 + instruction
        ↓
    VLM prefix
        ├─ 최종 hidden state → vocabulary projection → risk score
        │
        └─ layer별 K/V → action expert → action chunk
```

공개 `openpi` 구현은 action sampling 전에 prefix forward를 수행하여 KV cache를 만든다. 이후 action expert는 이 cache와 noisy action을 이용해 flow-matching 출력을 계산한다. 이 과정에서 prefix의 최종 hidden output은 기본 sampling 코드에서는 사용하지 않고 버리므로, **그 출력을 보존하여 risk readout에 사용하는 수정**이 가능하다. [GitHub](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0.py)

또한 Gemma 구현에는 입력 embedding table의 전치로 vocabulary projection을 수행하는 구조가 있다. 따라서 해당 checkpoint의 기존 table을 사용하여 readout을 구성할 수 있다. 이 경우 “LM head를 freeze한다”는 것은 **공유된 embedding/readout matrix도 고정한다**는 의미로 구현해야 한다. [GitHub](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/gemma.py)

### 중요한 제한: prefix는 구체적인 action을 보지 않는다

공개 구현에서 image–language prefix는 뒤의 state/action suffix를 보지 않도록 구성된다. 따라서 prefix에서 읽는 점수는 특정 sampled trajectory의 위험 점수가 아니라, prefix에 주어진 문맥에 대한 위험 점수다. [GitHub](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0.py)

이 구조에서 다음은 성립한다.

\[
R_{\mathrm{prefix}}(c)
\quad\text{는 같은 }c\text{에서 action noise를 바꾸어도 동일하다.}
\]

즉, 동일한 요청에서 서로 다른 action chunk가 생성되더라도 prefix score만으로 둘을 비교할 수 없다.

**칼을 아기에게 전달하라는 요청을 차단하는 데는 적합하지만, 같은 요청에서 생성된 두 trajectory 중 어느 것이 충돌하는지 비교하는 데는 부족하다.**

추가로 π₀와 π₀.₅는 상태 입력 위치도 다르다. 공개 설정에서 π₀는 연속 state를 suffix에 넣는 반면, π₀.₅는 state를 discrete language token 쪽에 포함하는 설정을 사용한다. 따라서 두 모델의 prefix detector가 동일한 정보를 보고 있다고 자동으로 가정하면 안 된다. [GitHub](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0_config.py)

### “추론 비용이 0”이라고 주장해서는 안 된다

π 계열의 일반적인 action 추론은 full vocabulary logits를 필요로 하지 않는다. 따라서 risk score를 위해 vocabulary projection과 `logsumexp`를 추가하면 비용이 발생한다.

\[
\log P(z_r)
=
w_{z_r}^{\top}h
-
\log\sum_{v\in\mathcal V}\exp(w_v^\top h)
\]

에서 보듯, **정확한 log-probability는 special token의 logit 하나만 계산해서 얻을 수 없다.**

단일 logit \(w_{z_r}^{\top}h\)만 사용하는 저비용 변형도 비교할 수 있지만, 그것은 원래 TruthLens의 log-probability 방식과 구분해야 한다.

## 3.3 OpenVLA-OFT: LM 출력과 action head를 분리해서 다룬다

OpenVLA-OFT의 L1 경로에서는 action 위치의 hidden state를 추출한 뒤, 이를 action head에 넣어 연속 행동을 예측한다. 공개 코드에서도 `actions_hidden_states`와 `action_head.predict_action(...)`이 구분되어 있다. [GitHub](https://raw.githubusercontent.com/moojink/openvla-oft/main/prismatic/extern/hf/modeling_prismatic.py)

따라서 두 가지 readout 후보를 비교할 수 있다.

**요청 위험 탐지:** 전체 이미지와 instruction을 처리한 prompt-boundary hidden state.

**행동 생성 관련 위험 탐지:** Action head에 전달되는 action-position hidden state.

첫 번째는 다른 모델들과 공통 비교하기 좋고, 두 번째는 실제 행동 생성에 더 가까운 표현을 검사한다.

다만 action-position hidden state를 사용했다는 이유만으로 해당 점수가 이미 “실행 결과의 위험성”을 의미하는 것은 아니다. 그 의미를 부여하려면 **생성된 행동의 안전성 label로 학습·평가**해야 한다.

# 4. 가장 중요한 문제: detector를 학습하면서 원래 행동이 바뀔 수 있다

이 부분은 실험 설계의 핵심이다.

> **Action expert 또는 action head의 weight를 freeze해도, 그 입력 표현을 만드는 VLM을 바꾸면 행동은 달라질 수 있다.**

π 계열을 단순화하면 다음과 같다.

\[
A_\theta
=
G_{\phi_0}\left(
KV_{\theta_0+\Delta\theta_{\mathrm{LoRA}}}(c),
s,\xi
\right)
\]

여기서 expert parameter \(\phi_0\)는 고정되어도, 입력되는 KV가 달라지므로 \(A_\theta\)는 달라질 수 있다. 이는 prefix KV를 action expert가 사용하는 공개 구조에서 직접 도출되는 결과다. [GitHub](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0.py)

따라서 risk MSE만으로 학습하는 조건과, **행동 보존 제약을 함께 사용하는 조건**을 비교해야 한다.

## 4.1 모델별 action-preservation loss

원논문의 language-distribution KL을 그대로 사용하는 것보다, 각 모델의 실제 행동 출력 구조에 맞는 보존항을 제안한다.

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{risk}}
+
\lambda_{\mathrm{keep}}\mathcal L_{\mathrm{keep}}
}
\]

| 모델 | 제안하는 행동 보존항 |
|---|---|
| 원본 OpenVLA | 같은 문맥에서 원래 모델과 학습 모델의 action-token 분포 사이 KL |
| OpenVLA-OFT L1 설정 | 원래 모델과 학습 모델의 예측 action chunk 사이 L1 또는 L2 차이 |
| π₀ / π₀.₅ | 같은 noisy action과 flow timestep에서 velocity prediction 차이 |

π 계열의 예시는 다음과 같다.

\[
\mathcal L_{\mathrm{keep}}^{\pi}
=
\mathbb E_{c,x_\tau,\tau}
\left[
\left\|
v_{\theta}(x_\tau,\tau;c)
-
v_{\mathrm{ref}}(x_\tau,\tau;c)
\right\|_2^2
\right]
\]

이것은 **새로 제안하는 distillation 제약**이다. 원논문이 VLA에서 검증한 loss는 아니다.

보존항에는 정상적인 작업 replay 데이터를 사용하는 것이 적절하다. 위험 요청에서 원래 모델이 생성하는 행동까지 무조건 모방하도록 만드는 것은 목표와 충돌할 수 있다.

또한 student 쪽 frozen action expert를 `no_grad`로 감싸거나 입력을 detach하면, 보존항의 gradient가 VLM LoRA까지 전달되지 않을 수 있다. **파라미터 고정과 gradient 경로 차단을 구분**해야 한다.

이 loss가 작다고 행동이 완전히 보존된다고 보장할 수도 없다. 최종적으로 정상 task rollout에서 성능을 확인해야 한다.

## 4.2 OpenPI의 기본 LoRA 옵션을 그대로 쓰면 안 되는 이유

현재 공개 Gemma LoRA 설정에는 attention뿐 아니라 FFN LoRA도 포함된다. 또한 freeze filter는 선택한 variant에 따라 expert 쪽 학습 가능 여부가 달라진다. 따라서 단순히 `gemma_2b_lora`를 선택했다고 해서 “VLM의 Q/K/V/O만 학습하고 나머지는 모두 freeze”한 조건이 되는 것은 아니다. [GitHub](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/gemma.py)

첫 실험에서는 명시적으로 다음 조건을 확인해야 한다.

**Vision encoder, embedding/readout matrix, MLP, action expert/head, action projection은 고정하고, 지정한 VLM attention adapter만 학습하는 조건.**

JAX 구현에서는 Q/K/V가 별도 `q_proj` 이름이 아니라 결합된 einsum parameter로 구현될 수도 있으므로, 모듈 이름을 그대로 복사하기보다 실제 대응 연산을 확인해야 한다. [GitHub](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/gemma.py)

## 4.3 행동 변화와 detector 효과를 완전히 분리하는 대조군

유용한 대조군은 **risk adapter와 실행 policy를 분리하는 방식**이다.

Risk score는 adapter를 켠 상태에서 계산하되, 실제 행동은 원래 policy로 생성한다. 그러면 detector 학습이 행동 자체를 바꾸는 효과와, detector가 실행을 차단하는 효과를 분리할 수 있다.

다만 이때는 원래 policy의 forward 또는 prefix cache를 별도로 계산해야 한다. **Risk adapter가 만든 KV cache를 재사용하면서 expert만 원래 상태로 두는 것은 원래 policy를 유지한 것이 아니다.**

이 방식은 계산 비용이 늘지만, 초기 연구에서 해석이 명확한 대조군이 된다.

# 5. 데이터는 “위험한 객체”보다 “위험한 관계와 요청”을 학습하도록 구성해야 한다

## 5.1 첫 실험에는 위험 행동 demonstration이 필요하지 않다

Request detector의 학습 데이터는 다음으로 충분하다.

\[
\mathcal D_{\mathrm{request}}
=
\{(I_i,X_i,y_i)\}
\]

따라서 위험한 요청을 실제 로봇으로 수행할 필요가 없다. 이미지와 instruction에 대해 정의된 안전 규칙에 따라 label을 부여하면 된다.

TruthLens의 caption sampling이나 object extraction도 필수적이지 않다. 이 적용에서는 **각 입력 쌍 자체가 하나의 평가 대상**이기 때문이다.

## 5.2 Counterfactual pair를 중심으로 구성한다

단순히 위험한 문장만 수집하면 모델이 이미지 없이도 분류할 수 있다. 특히 “칼을 아기에게 줘라”는 문장 자체가 강한 신호다.

따라서 다음 두 유형이 중요하다.

**같은 이미지, 다른 instruction.** 위험한 전달과 안전한 보관을 구분하도록 한다.

**같은 instruction, 다른 이미지.** “왼쪽 물건을 아기에게 줘라”처럼 시각적 grounding이 필요한 지시를 사용하고, 물건의 종류를 바꾼다.

이렇게 해야 다음 shortcut을 검사할 수 있다.

\[
\text{“칼이 보인다”}\Rightarrow\text{Unsafe}
\]

또는

\[
\text{“아기라는 단어가 있다”}\Rightarrow\text{Unsafe}
\]

이 아니라, 실제로 이미지와 instruction의 조합에 반응하는지 확인할 수 있다.

## 5.3 분할은 원본 장면 단위로 해야 한다

동일 이미지에서 만든 safe/unsafe instruction pair나 이미지 편집본은 같은 split에 넣는 것이 적절하다. 같은 장면이 train과 test에 동시에 들어가면 위험 관계를 일반화한 것인지 장면을 기억한 것인지 구분하기 어렵다.

추가 평가에서는 학습에 없던 객체–행동 조합, instruction 표현, 배경, 카메라 시점을 분리해 볼 수 있다.

이미지에서 위험 판단에 필요한 정보가 보이지 않는 경우도 무조건 safe로 처리하지 말아야 한다. 가림, 지시 대상의 모호성, 관측되지 않은 상태는 별도의 불확실 사례로 관리하는 편이 적절하다.

실제 행동 평가는 시뮬레이션과 위험 요소를 제거한 대체 환경에서 시작해야 한다. **아동과 실제 날카로운 도구를 함께 두는 실험은 필요하지 않다.**

# 6. Linear probe와의 비교가 연구의 설득력을 결정한다

Special token 방식은 흥미롭지만, 다음 질문을 반드시 받아야 한다.

> “그냥 hidden state 위에 작은 linear classifier를 붙이면 되는 것 아닌가?”

이 질문에 답하지 못하면 “새 출력 채널을 사용했다” 이상의 이점을 보여주기 어렵다.

첫 비교에는 다음 조건을 제안한다.

| 조건 | 학습 대상 | 확인하는 질문 |
|---|---|---|
| 학습하지 않은 special-token score | 없음 | 원래 출력 차원에 유효한 신호가 있는가? |
| Frozen backbone + linear probe/LDA | 작은 판별기만 | 기존 표현에 위험 정보가 있는가? |
| Frozen LM head + attention LoRA + special-token score | VLM LoRA | 위험 신호를 기존 vocabulary 출력에 학습시킬 수 있는가? |
| 같은 attention LoRA + 일반 scalar head | LoRA와 작은 head | 향상이 special-token 방식 때문인가, 표현을 학습한 효과인가? |

이미 계획했던 Yes/No likelihood 평가도 함께 사용할 수 있다. 다만 safety 질문을 추가하는 평가와 원래 action prompt에서 그대로 점수를 읽는 평가는 입력 조건이 다르므로 구분해서 보고해야 한다.

### 특히 중요한 해석

Frozen linear probe가 이미 높은 성능을 보인다면, 원래 representation에서 위험 정보가 추출 가능하다는 근거가 된다.

반대로 frozen probe는 낮고 LoRA 이후에만 높다면, 이를 곧바로 “이미 있던 safety knowledge를 노출했다”고 해석할 수 없다. **Fine-tuning 과정에서 위험 정보를 새롭게 학습했을 가능성도 있다.**

또한 별도 대형 모델이 필요 없다는 장점만으로는 linear probe보다 우월하다고 할 수 없다. Linear probe 역시 기존 hidden state를 재사용하면 추가 backbone forward가 필요하지 않으며, 특히 π 계열에서는 full-vocabulary log-probability보다 계산이 저렴할 수 있다.

따라서 special-token 방식의 장점은 **정확도, 일반화, 행동 보존, 실제 latency**로 검증해야 한다.

# 7. 탐지 성능과 실제 안전성 개선을 분리해 평가한다

## 7.1 Detector 평가

Unsafe를 positive class로 두고 AUROC와 AUPRC를 보고하되, 안전 적용을 위해서는 높은 위험 탐지율에서의 정상 요청 오차단율도 중요하다.

예를 들어 validation set에서 목표 recall에 맞춰 threshold를 정하고, 독립 test set에서 위험 요청 누락률과 정상 요청 차단율을 측정할 수 있다. 모델별 score scale이 다를 수 있으므로 같은 숫자 threshold를 강제로 공유할 필요는 없다.

Text-only, image-only, 전체 입력 조건을 비교하고, 앞서 만든 counterfactual pair에서 score가 올바른 방향으로 바뀌는지도 평가해야 한다.

## 7.2 실행 차단 평가

학습된 score를 행동 안전성으로 연결하려면 명시적인 실행 정책이 필요하다.

\[
A_{\mathrm{executed}}
=
\begin{cases}
\text{검증된 중단·보류 절차}, & R_\theta(c)>\mu\\
A_{\mathrm{policy}}, & R_\theta(c)\leq\mu
\end{cases}
\]

이때 점수가 낮다는 것은 안전 보장이 아니라 **detector가 위험으로 분류하지 않았다는 의미**다.

또한 “모든 action을 0으로 만든다”를 일반적인 안전 동작으로 가정해서는 안 된다. 물체를 잡고 있는 상태인지, 정지 시 낙하 위험이 있는지 등 로봇과 작업에 맞는 fallback이 필요하다.

실험에서는 최소한 다음 세 조건을 분리하는 것이 좋다.

| 조건 | 의미 |
|---|---|
| 원래 VLA | 기본 행동과 위험 수준 |
| Risk score를 학습한 VLA, gate 없음 | Detector 학습 자체가 행동을 바꾸었는가? |
| Risk score를 학습한 VLA, gate 있음 | 탐지에 따른 실행 차단이 얼마나 기여하는가? |

여기에 원래 VLA와 frozen linear-probe gate를 결합한 조건을 넣으면 special-token 방식의 실용적 이점도 비교할 수 있다.

평가 결과는 위험 실행률뿐 아니라 정상 task 성공률, 정상 요청 오차단율, 추가 추론 지연을 함께 보여야 한다. 모든 행동을 막으면 위험 실행률은 낮아지지만 유용한 policy라고 할 수 없기 때문이다.

**이 실험은 “위험을 탐지하도록 학습했다”와 “위험한 행동을 자율적으로 피하도록 학습했다”를 구분하는 데도 중요하다.** Score만 학습했다고 후자가 자동으로 성립하지는 않는다.

# 8. 다음 단계: Action-conditioned special-token monitor로 확장한다

첫 단계가 성공하면 연구적으로 더 중요한 질문으로 확장할 수 있다.

> **VLM이 판단한 요청 위험과 실제 action generator가 만들어내는 행동 위험은 일치하는가?**

## 8.1 원본 OpenVLA

첫 action token 생성 전에 읽는 score는 요청 기반 예측이다.

특정 action 후보를 평가하려면 생성된 action token들을 입력 문맥에 포함시킨 후 score를 읽는 방식을 추가할 수 있다.

\[
R_{\mathrm{action}}
=
R(I,X,a_1,\ldots,a_K)
\]

중요한 점은 **평가하려는 후보 행동이 실제로 입력 문맥에 포함되어야 한다**는 것이다. 후보 생성 직전의 동일한 prefix score를 여러 후보에 복사하는 것은 후보별 위험 평가가 아니다.

## 8.2 OpenVLA-OFT

Action head에 전달되는 hidden state를 이용하는 monitor를 먼저 만들 수 있다. 이후 필요하면 실제 예측 action vector까지 함께 조건화한다.

이는 request readout과 같은 모델 내부에서 비교할 수 있다.

\[
R_{\mathrm{request}}(h_{\mathrm{prompt}})
\quad\text{vs.}\quad
R_{\mathrm{action}}(H_{\mathrm{action}},A)
\]

두 점수의 label은 분리해야 한다. 전자는 요청의 위험성, 후자는 후보 행동의 안전 제약 위반 여부다.

## 8.3 π 계열

π 계열에서는 action expert hidden state를 검사해야 한다. 다만 **expert hidden state에 VLM의 LM head를 그대로 곱할 수 있다고 가정하면 안 된다.**

공개 기본 설정에서 VLM Gemma와 action expert는 hidden width가 다르고, vocabulary embedding/readout은 첫 번째 expert인 VLM에 연결된다. 따라서 action expert에 동일한 special-token readout을 적용하려면 추가 연결부가 필요하다. [GitHub](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/gemma.py)

한 가지 확장안은 다음과 같다.

\[
g_A=\operatorname{Pool}(H_{\mathrm{expert}})
\]

\[
\tilde h
=
h_{\mathrm{request}}+B_\psi g_A
\]

\[
R_{\mathrm{action}}
=
\beta\left[
\log\operatorname{softmax}
(W_{\mathrm{LM}}\tilde h)_{z_r}
-c_{\mathrm{ref}}
\right]
\]

여기서 \(B_\psi\)는 expert 표현을 VLM readout 공간으로 연결하는 작은 projection이다.

이 설계는 policy 전체를 freeze한 채 \(B_\psi\)만 학습하는 조건으로도 검사할 수 있다. 그러나 이것은 **원래 TruthLens를 그대로 적용한 것이 아니라, action-conditioned 보조 readout을 새로 추가한 변형**이다. 별도 대형 reward model은 아니지만 추가 학습 모듈은 존재한다.

또한 flow 중간 표현을 이용한 점수는 계획된 행동의 위험 예측이지, 환경에서 실제 결과가 안전하다는 증명은 아니다. 시뮬레이션 rollout이나 명시적인 제약 검사로 label을 만들어 검증해야 한다.

이 단계의 가장 중요한 평가는 다음이다.

\[
\boxed{
\text{동일한 관측·지시에서 생성된 서로 다른 행동들의 위험도를 구분할 수 있는가?}
}
\]

이를 해야 request detector와 action-conditioned detector의 차이가 실제로 드러난다.

# 최종적으로 권하는 진행 순서

**첫 단계에서는 π₀, π₀.₅, OpenVLA, OpenVLA-OFT 모두에 동일한 요청 위험 데이터셋을 적용하고, native action 입력에서 추출한 마지막 유효 문맥 표현으로 frozen linear probe와 special-token LoRA를 비교하는 것이 적절하다.** Object-token 식별이나 caption 생성은 넣지 않는다.

**두 번째 단계에서는 행동 보존 제약과 실행 gate를 추가하여, 정상 task 성능을 유지하면서 위험 요청 실행을 줄일 수 있는지 확인한다.** Detector의 성능과 gate의 기여를 분리한다.

**세 번째 단계에서는 action expert 또는 action-position 표현을 활용하여, 요청 위험과 실제 행동 위험 사이의 차이를 분석한다.**

이 연구의 핵심을 단순히 “special token으로 위험을 분류한다”로 두기보다는 다음 질문에 두는 편이 더 강하다.

> **VLA의 기존 표현에서 읽을 수 있는 위험 정보와, 실제 행동 생성에 반영되는 위험 정보 사이에는 어떤 차이가 있으며, 이를 낮은 비용의 내부 readout으로 탐지하고 실행 전에 활용할 수 있는가?**

현재 제시한 아이디어는 이 질문을 검증하는 구체적인 출발점이 된다. 다만 **special token의 효과, 표현을 fine-tuning한 효과, 외부 실행 차단의 효과를 분리하는 실험**이 연구의 설득력을 결정할 것이다.