## :page_with_curl: Attention Is All You Need

### Info

* url:https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
* authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
* NIPS2017

### 1. Abstract & Introduction

시퀀스 변환(벡터로의) 모델은 인코더-디코더 구조를 가지는 RNN이나 CNN 모델이 지배적이었다.

또한 어텐션 메커니즘은 주로 RNN과 함께 쓰여졌다. [이는 아직 장기 기억 의존성을 가진 것]

본 논문은 간단한 recurrent나 convolution 없이 어텐션 메커니즘만을 사용한 모델 구조를 제안하고 질적인 향상과 훈련 시간 또한 단축시킴을 증명한다.

### 2. Model Architecture

인코더는 input_sequence( $x_1,...,x_n$ )를 연속적인 symbol representation sequence( $z_1,...,z_n$ )로 매핑하며,   
디코더는 주어진 z에 대해 한 번에 하나씩 output_sequnce ( $y_1,...,y_m$ )를 만들어 냄.

각 스텝에서 텍스트를 만들 때, 모델은 auto-regressive 모델이다. [이전 단계에 만든 symbol을 추가 입력으로 넣음]

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/212308693-5d9e2f2b-ccae-4bd0-9007-dc3f2301ef2c.png" width="35%"></img></div>

#### 2.1 Encoder and Decoder Stacks

- **Encoder**

  - $N=6$ 개의 동일한 layer를 쌓은 형태로 각 layer에는 2개의 sub-layer가 있음. [multi-head self-attention, position-wise FCNN]
  
  - 2개의 sub-layer 각각에 `residual connection`을 적용한 후 Layernormalization을 적용한다. [즉, $LayerNorm(x + Sublayer(x))$ ]
  - 위의 layer normalization을 위해 embedding layer를 포함한 모든 sub-layer의 출력 차원을 512로 설정함. [ $d_{model} = 512$ ]

- **Decoder**

  - Decoder 또한 $N=6$ 개의 동일한 layer를 쌓은 형태

  - encoder와 비슷한 구조를 가지지만 두 개의 sub-layer 사이에 encoder stack의 출력에 대한 multi-head attention을 수행하는 것을 넣는다.

  - 그리고 self-attention sub-layer를 변형한다. [각 position에서 후의 position을 참조하지 못하게 막음]

  - 이러한 **masking**은 position $i$ 는 $i$ 이전의 값에서만 참조할 수 있게 함. [output embedding은 하나의 position만을 출력( offset으로 나타냄)]

#### 2.2 Attention

`Attention`연산은 query와 key-value 쌍을 출력으로 매핑(변환)시키는 것이다. (query, key-value는 모두 벡터)

출력(output)은 값들의 가중합으로 계산되고 각 값들의 가중치는 query와 이에 대응하는 key의 호환성 함수(compatibility function)으로 계산된다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/212308733-52ec6704-1c04-4331-8acf-7cd6e93c3223.png" width="55%"></img></div>

- **Scaled Dot-Product Attention**

  - input은 $d_k$ 의 차원을 가지는 queries & keys, $d_v$ 의 차원을 가지는 value로 구성. [query는 행렬 Q, key는 행렬 K, value는 행렬 V로 계산하게 된다]
 
  - query 하나와 모든 key들을 내적한 후 $\sqrt{d_k}$ 로 나눈다. 그리고 value의 가중치를 얻기 위해 `softmax`를 적용한다.

  - 사실 query, key, value는 랜덤하게 초기화되는 가중치 행렬에 입력 행렬을 곱해 나온 것이다. [즉, Q의 shape = `input_dim` x `d_k`]

  - $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

  - additive attention보다 dot-product attention이 실제로 더 빠르고 공간효율이 좋지만 $d_k$가 커지게 되면 내적 값도 커지게 되어 공간이 너무 커져서 softmax가 너무 작은 gradient 갖게 된다. (그래서 additive attention이 더 좋은 성능을 보였음)

  - 즉, $d_k$ 가 커지면 분산도 커지므로 softmax 값 각각이 매우 지엽적인 값을 갖게 될 수 있음.
  
  - 위를 막기 위해  $\sqrt{d_k}$로 스케일링함

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/212309052-3856b0a0-d328-43e5-b22d-791395a34c53.png" width="55%"></img> 출처: 구글 BERT의 모든 것 </div>

- **Multi-Head Attention**

  - $d_{model}$ 의 차원을 가지는 query, key, value로 단일 attention 연산을 하는 것이 아닌 각각 $d_k$, $d_k$, $d_v$ 차원으로 $h$ 개를 학습 시킴.
    - 즉, 각 차원에 맞는 가중치와 input embedding의 내적 값이 scaled dot product의 Q, K, V가 됨.
    
    
    - 다른 h개의 표현 공간으로부터 정보를 보는 관점이 여러 개가 됨.

  - 이렇게 되면 각 attention을 병렬로 수행하고 $d_v$ 차원의 output 값이 생기는데 이를 concat한 후, 다시 `linear layer`를 통과시켜 마지막 값을 얻는다.

  - $MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$ $\\ where \  head_i = Attention(Q{W_i}^Q, K{W_i}^K, V{W_i}^V$
    - 실제로는 Q, K, V에 input embedding을 넣음. ==> $Attention(X{W_i}^Q, X{W_i}^K, X{W_i}^V)$

  - $h = 8$ 을 적용했고, $d_k = d_v = d_{model} / h = 64$ [512 / 8 = 64]


- **Applications of Attention in our Model**

  - `encoder와 decoder 사이의 어텐션`에서는 query는 디코더에서, key와 value는 encoder의 결과를 사용
    - 디코더가 입력의 모든 부분(위치)을 볼 수 있음

  - `encoder 내부의 어텐션`은 이전 encoder의 출력에서의 query, key, value로 계산. encoder의 각 위치는 이전 encoder의 모든 위치를 볼 수 있음.

  - `decoder 내부의 어텐션`에는 `auto-regressive`특성을 보존하기 위해 마스킹을 이용해 다음의 출력값을 미리 attention할 수 없게 만듦.

- **Position-wise Feed-Forward Networks**

  - encoder와 decoder의 각 layer는 FCNN을 포함하며 두 개의 linear 층과 RELU 활성화 함수로 구성된다.

  - $FFN(x) = max(0, xW_1 + b_1) W_2 + b_2$

  - 선형 변환(linear 층)은 다른 위치에 똑같이 적용되지만 layer가 달라지면 파라미터도 달라진다. [ $d_ff = 2048$ 사용 ]


- **Embeddings and Softmax**

  - input, output token을 벡터로 바꾸기 위해 embedding을 학습하게 함.

  - decoder의 output이 다음 토큰의 확률을 출력하게 하기 위해 선형변환과 softmax 사용

  - 두 임베딩 layer와 softmax 전 단계의 선형변환에서 같은 가중치를 사용


- **Positional Encoding**

  - 시퀀스의 순서 정보를 주기 위해 상대적 or 절대적 위치 정보를 줌. (이 값은 imput embedding에 더해짐, $d_{model}$의 차원을 가짐)

  - 다른 주기를 갖는 sine & cosine 함수를 사용.

  - $PE_{(pos, 2i)} = sin(pos / 10000^{2i / d_model})$
  - $PE_{(pos, 2i + 1)} = cos(pos / 10000^{2i / d_model})$

  - `pos`는 위치이고 `i`는 차원

  - 즉, 각 차원은 sin곡선에 대응됨.

  - 학습된 위치 임베딩과 위의 방법을 비교했을 때, 동일한 성능이 나왔고 위의 방법이 training에서 보다 큰 시퀀스가 추론 시에 등장 했을 때 문제 없이 작동하므로 위 방법을 채택.

<br></br>

### 3\. Training

- Data: WMT2014의 English -> German(4.5M), English -> French(36M)

- Vocab: 각각 37000 tokens BPE, 32000 word-piece

- 하나의 batch는 약 25000token 쌍들을 포함.

- base model은 12시간, big model은 3.5일간 훈련

- optimizer: Adam( $\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$ )

- warmup_steps = 4000

- learning rate = ${d_{model}}^{-0.5} \cdot min(step_num^{-0.5}, step_num \cdot warmup_steps^{-1.5})$

  - 처음에는 선형적으로 lr을 증가시키다 warmup_step을 넘어가면 step number의 inverse square root 만큼 감소 시킴.

- Residual Dropout 적용 ( $P_{drop} = 0.1$ )

  - input+positional embedding의 합과 sublayer input을 더하고 normalize하기 전에 적용함. [즉,  x + `h(sub(norm(x)))`]

- label smoothing 사용 [ $\epsilon_{ls} = 0.1$ ]

<br></br>

### 4. Results

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/212308914-638572f8-27fd-4866-9b71-52ef3c0508c2.png" width="55%"></img></div>

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/212308937-662eb9e9-2bd6-44cd-9178-8db5c81da1cf.png" width="55%"></img></div>

- 본 논문에서 제안한 Transforemr(big)이 SOTA를 달성함.

- transformer의 hyperparameter를 조절하고 성능을 관찰해 봄.

  - head 개수와 key, value의 차원을 계산량을 똑같이 두고 조절해보았는데, head가 많은 것이 오히려 성능을 떨어뜨렸음.

  - key 차원을 줄이는 것은 성능을 떨어뜨렸음

  - model이 커질 수록 성능이 좋아졌으며 dropout으로 over-fitting을 피할 수 있었음.
