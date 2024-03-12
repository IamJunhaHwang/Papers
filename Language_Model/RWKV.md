## :page_facing_up: RWKV: Reinventing RNNs for the Transformer Era

### Info

* `publication`: EMNLP 2023 - Long paper FINDINGS
* `author`: Bo Peng, Eric Alcaide, Quentin Anthony et al.
* `url`: https://aclanthology.org/2023.findings-emnlp.936/

### 1\. Abstract & Introduction

Transformer는 거의 모든 NLP task에서 좋은 성능을 보였지만, 시퀀스 길이의 제곱만큼의 크기의 메모리와 계산 복잡도가 필요하다는 단점을 가지고 있다. 반면, RNN의 경우는 선형 규모의 메모리와 계산 복잡도를 필요로하지만 vanishing gradient 문제와 parallelization, scalability 한계가 존재한다.

이러한 문제들을 해결하기 위해, 우리는 핵심 단점들을 피하면서 RNN과 Transformer의 강점을 조합한 **Receptance Weighted Key Value (RWKV)** 모델을 제안한다. RWKV는 효율적인 linear scaling을 통해 Transformer의 expressive properties를 유지하면서 메모리 병목과 quadratic scaling을 완화한다. 자세히는, RWKV는 dot-product attention을 더 효과적인 channel-directed attention으로 바꾸는 것으로 attention 메커니즘을 다시 만든다. 이를 통해 낮은 계산, 메모리 복잡도를 가질 수 있다.

RWKV의 motivation은 계산 효율과 신경망의 표현력 간의 균형을 맞추는 것이며, 본 논문의 contribution은 다음과 같다.

1. RNN과 Transformer의 한계는 완화하면서 장점을 조합한 RWKV의 제안

2. 실험을 통해 large-scale model을 위한 벤치마크 데이터에서 RWKV의 성능과 효율성을 증명함.
3. 169M ~ 14B의 `Pile`로 학습한 사전 학습 모델을 공개함.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/c0e815fb-11b3-4cae-bf07-8025b6174168" width="35%"></img></div>

<br></br>

### 2. RWKV

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/0bb14395-c831-4c27-bfe0-a155ea2b76d4" width="80%"></img></div>

- RWKV 모델은 `time-mixing` 과 `channel-mixing block` 에 사용되는 4가지 요소로 정의된다.

  - R : Receptance, 과거 정보를 수용하는 역할을 하는 벡터

  - W : Weight, positional weight decay vector, trainable parameter

  - K : Key, 전통적인 어텐션 메커니즘에서의 Key 역할과 동일
  - V : Value, 전통적인 어텐션 메커니즘에서의 Value 역할과 동일

#### 2-1. Architecture

RWKV는 residual block들이 쌓인 형태로 구성되며 각 block은 `time-mixing`과 `channel-mixing` sub-block으로 구성된다. layer normalization과 time-dependent softmax 작업이 포함된 특별한 attention-like score update 과정은 안정적인 학습과 vanishing gradient 문제를 완화해준다.

##### Token Shift

모든 linear projection vectors(R, K, V, R', K')는 현재 input과 이전 input간의 linear interpolation으로 계산된다.

- time-mixing

  - $r_t = W_r \cdot (\mu_r \odot x_t + (1-\mu_r) \odot x_{t-1})$
  - $k_t = W_k \cdot (\mu_k \odot x_t + (1-\mu_k) \odot x_{t-1})$
  - $v_t = W_v \cdot (\mu_v \odot x_t + (1-\mu_v) \odot x_{t-1})$

- channel-mixing

  - $r_t' = W_r' \cdot (\mu_r' \odot x_t + (1-\mu_r') \odot x_{t-1})$
  - $k_t' = W_k' \cdot (\mu_k' \odot x_t + (1-\mu_k') \odot x_{t-1})$

##### WKV Operator

WKV 연산은 `Attention Free Transformer(AFT)`와 유사하지만, **W** 가 pairwise matrix인 AFT와 다르게 우리는 **W** 를 상대적인 위치에 의해 바뀌는 channel-wise vector로 취급했다. 이러한 재귀적인 방식은 다음과 같이 WKV vector의 time-dependent update로 정의된다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/03a29f36-2285-43bf-84bd-75246c6aa518" width="40%"></img></div>

W에 대한 잠재적인 성능 저하를 피하기 위해 현재 토큰에 attend하는 vector U를 도입했다. ($e^{u+k_i}$)

##### Output Gating

Output gating은 time-mixing과 channel-mixing block 모두에 구현이 되어 있다.

- time-mixing

  - $o_t = W_o \cdot (\sigma(r_t) \odot wkv_t)$

- channel-mixing

  - $o_t' = \sigma(r_t') \odot (W_v' \cdot max(k_t', 0)^2)$

  - squared ReLU 활성화 함수 적용

<br></br>

### 3. Evaluation

- 이 파트에서는 다음 2가지 질문에 집중한다.

  - 같은 계산량을 가질 때, RWKV가 transformer 모델과 경쟁할 만한 성능이 나오는가?

  - context length를 늘렸을 때, RWKV는 좀 더 좋은 language modeling loss를 만들어내는가?
    - 대부분의 transformer 모델은 그러지 못했음

비교에 사용한 모델은 비슷한 토큰 수로 학습된 비슷한 크기의 모델들이다(Pythia, OPT, BLOOM). 모든 RWKV 모델들은 Pile로 1 epoch (330B tokens) 훈련하였다.

사용한 데이터 : ARC, BoolQ, COPA, HeadQA, HellaSwag, LAMBADA, OpenBookQA, PIQA, ReCoRD, SciQ, Winogrande

+) RWKV가 `ChatGPT / GPT-4`에 비해 프롬프트에 민감하다는 것을 밝혀냈다(Appendix L). GPT와 다르게 RNN 기반인 RWKV는 이전 지시사항을 되짚어볼 수 없기 때문으로 생각되며, RNN에 적합한 프롬프트로 조정했을 때, F1 성능이 44.2% -> 74.8% 로 증가하였다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/ddcb6a72-d2a0-472d-b6f7-d95502c302ce" width="75%"></img></div>

#### 3-1. Extended Context Finetuning

Transformer와 다르게 RNN은 sequence length를 미리 정해놓지 않지만, 효율적인 연산을 위해 training data를 똑같은 길이로 미리 전처리를 하였다. 또한, 우리는 sequence length를 점점 늘리면서 fine-tuning하는 것으로 상당히 큰 batch size를 모델이 효율적으로 처리하도록 가르칠 수 있다는 것을 찾아냈다.

처음 sequence length를 1024 -> 2048로 하여 10B 토큰 fine-tuning을 진행하고, 다음으로 다시 2배 늘려 2048 -> 4096으로 늘려 100B토큰, 4096 -> 8192으로 늘려 100B 토큰 fine-tuning한다.

아래 그림과 같이 context length를 증가시키는 것은 Pile에서 학습할 때의 loss를 줄여준다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/888ab9cd-0695-4842-9314-9b842ece27c3" width="35%"></img></div>

#### 3-2. Inference Experiments

텍스트 생성에 걸리는 시간과 메모리 필요량을 여러 모델들과 비교하였을 때 아래와 같은 결과를 얻었다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/2512b375-b5a5-4856-921c-9101e44932ce" width="35%"></img></div>
