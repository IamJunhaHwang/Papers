## :page_facing_up: MiniMax-01: Scaling Foundation Models with Lightning Attention

### Info

* `publication`: arxiv 2025
* `author`: MiniMax et al.
* `url`: https://arxiv.org/pdf/2501.08313
* `code`: https://github.com/huggingface/transformers/pull/35831

### 1. Abstract & Introduction

최근 Large Language Models (LLMs)와 Vision Language Models (VLMs)가 빠르게 발전하였으나 이들의 context 길이는 32K~256K 토큰에 불과하다. 이는 책 전부를 context로 쓴다던지, programming project의 전체 코드를 context로 쓰던가, 여러 example을 이용해 inference를 할 경우 context가 부족할 수 있다.

context window 확장을 위해 여러 방법들이 제안되었지만 transformer 구조에 내재된 $O(n^2)$ 계산 복잡도때문에 한계가 있었다(context 확장시 hardware 능력보다 계산 요구량이 훨씬 많아지게 됨). 이에 따라 여러 attention 메커니즘들이 제안되었지만 commercial-scale model(Large-scale model) 적용에 있어서는 한계가 있었다.

- 우리는 한 차원 더 긴 context window를 제공하면서 commercial-scale model과 동등한 성능을 내는 모델을 개발하였다. 이를 위해 다음 3가지 요소를 신중히 조절하였다

  - network architexture: lightning attention과 I/O-aware implementation of a linear attention variant를 사용한 Hybrid 구조

  - data: 다양한 high-quality corpus에 data cleaning 작업, reward-based quality 향상, data mixture balancing 작업을 수행

  - computation: Mixture of Experts (MoE) 구현, MoE 내의 all-to-all communication을 위해 Expert Parallel (EP) 와 Expert Tensor Parallel (ETP) 사용, context window의 제한없는 확장을 위해 varlen ring attention을 설계하였으며 Linear Attention Sequence Parallelism (LASP)를 개선시킴

    - 추가로 lightning attention inference의 맞춤 CUDA kernels set을 구현

- **Contributions**

  - 표준 academic benchmarks에서 top-tier closed-source model에 버금가는 모델 개발하였으며, 이 모델은 4M token까지의 context input을 지원하여 long-context 평가에서 뛰어난 성능을 보였음

  - linear attention을 이용한 첫 large-scale model 구현하였으며, 알고리즘 설계와 engineering 최적화에 대한 종합적인 상세 설명을 제공

  - 다양한 모델, 데이터셋, 평가, 알고리즘에 대한 탐구를 위한 실용적인 접근 방식과 실험 방법론의 outline 제공

  - 모델 가중치를 공개하고 API 제공

<br></br>

### 2. Model Architecture

<div align="center"><img src="https://github.com/user-attachments/assets/af08cf68-4fed-4177-89fd-db63cfe37e2e" width="40%"></img></div>   

제한된 리소스와 긴 시퀀스를 더 잘 처리하면서 최적의 성능을 달성하기 위해 **Mixture of Experts(MoE) approach와 linear attention** 을 적용했으며, 이를 이용해 최대한 traditional softmax function을 대체하였다.

제안하는 구조는 위의 그림과 같으며, channel mixer(attention block) 과 feature mixer(MLP block)으로 이루어진다.

- 2가지 channel mixer: lightning attention, softmax attention

  - 매 7개의 linear attention 후에 softmax attention을 두었다(이렇게 총 80 layers로 구성).

  - 각 attention은 64 heads를 가지며, 각 head dimension은 128이다.

  - softmax attention은 Group Query Attention(GQA)을 적용하였으며 Group size는 8을 사용했다.

  - Rotary Position Embedding(RoPE)이 attention head가 가지는 차원 반절에 적용하였다(base frequency set=10,000).
  
  - model hidden size = 6144, 각 layer는 32 experts(experts의 hidden size는 9216)를 포함하며 top-2 routing 전략 사용.

- feature mixer: 여러 feed-forward networks를 가지는 MoE

  - MoE block(FFNN block)의 load balancing을 위해 GShard에서 영감을 받은 새로운 load balancing 방법 제안 (global router라 부름)

  - 또한, DeepNorm이 전체 성능 향상을 위해 포함되었다.

이렇게 만들어진 MiniMax-Text-01은 456B 파라미터를 가지며, 45.9B 파라미터가 각 토큰 처리 시 활성화 된다.

#### 2-1. Mixture of Experts

MoE는 여러 FFN experts로 이루어지며, 각 토큰은 1개 이상의 experts로 보내진다. input token $x_t$ 와 이에 대한 output hidden state $h_t$ 는 다음과 같이 계산된다.

- $h_t = \sum^E_{i=1} {Softmax}_i (TopK(x_t \cdot W_g)) \cdot {FFN}_i(x_t)$

  - `E`: experts의 총 개수

  - $W_g$: 게이트(라우터) 가중치

  - `FFN`: i번째 expert

  - `TopK()`: 모든 experts 사이에서의 top-k score, 나머지 score는 $-\infty$

MoE training efficiency를 위해 token-drop strategy를 채택하였다. 각 expert는 처리할 수 있는 최대 토큰 개수(capacity limit)가 지정되고 이에 도달하면 이후의 토큰들은 버려진다.

모델 크기를 키웠을 때, routing collapse를 겪게 되었다. 이를 완화하기 위해, 간단한 global routing strategy를 GShard auxiliary loss에 포함하였다.

- Auxiliary loss: $L_{aux} = \alpha_{aux} \cdot \frac{1}{E} \sum^E_{i=1} f_i \cdot m_i$

  - $\alpha_{aux}$: auxiliary loss의 계수

  - $f_i$: i번째 expert에 할당된 토큰 비율

  - $m_i$: i번째 expert의 평균 라우팅 확률

- Global Router

  - 토큰 분포는 다른 Expert Parallel(EP) 그룹 사이에서 달라지므로, load imbalances를 유발한다. 따라서, global token dispatching strategy를 EP 그룹들 사이에 적용

  - 토큰을 분배하기 전에 각 expert가 처리할 대기 토큰 수를 동기화하기 위해 추가적인 allgather 통신 단계를 도입

#### 2-2. Linear Attention

<div align="center"><img src="https://github.com/user-attachments/assets/6fbefa28-adff-4851-94cc-22014b8a0ccd" width="80%"></img></div>  

Linear Attention은 `right product kernel trick`을 사용해 linear complexity로 변환한다.

위 그림과 같은 TransNormer에서의 예시와 같이 NormAttention은 다음과 같이 쓸 수 있다.

- $O = Norm((QK^T)V)$

  - $Q,K,V \in \mathcal{R}^{n X d}$ : 쿼리, 키, 벨류 행렬

  - `n`: sequence length

  - `d` : feature dimension

- 위 식은 다음과 같이 바뀔 수 있다: $O = Norm(Q(K^T V))$

  - 시퀀스 길이에 관계없이 $𝑂(𝑑²)$의 일정한 계산 복잡도를 보장

  - 이는 $K^T V$ 항을 반복적으로 업데이트하여 전체 어텐션 행렬을 반복 계산할 필요를 없애는 방식으로 이루어짐. 이에 비해, 소프트맥스 어텐션은 추론 중에 $𝑂(𝑛𝑑²)$의 복잡도가 발생됨.

하지만, `casual language modeling`에서 right product의 효과가 떨어지기 때문에, `cumsum` 계산이 필요한데 이는 고효율 병렬 계산에 방해된다. 이에 따라, 최신 LLM들에 linear attention이 채택되지 못했다.

#### 2-3. Lightning Attention

lightning attention은 linear attention의 casual language modeling에서 느린 `cumsum` 연산을 피하기 위해 새롭게 타일링 기법을 제안했다. 

이 방법의 핵심은 attention 계산을 intra-blcok과 inter-block이라는 2개의 개별 요소로 나누는 것에 있다. intra-block 연산에서는 left product attention이 적용되고 inter-block에서는 right product attention이 적용된다. 이렇게 나누는 것이 중요한 이유가 intra block은 크기를 효과적으로 줄일 수 있어, 전체 계산 복잡도는 선형으로 유지시킬 수 있기 때문이다. 

- left product in casual attention은 다음처럼 정의된다: $O = [(QK^T) ⊙ M]V$ 

  - $M_{ts} = 1$ if $t >=s$, otherwise 0

- right product는 다음같이 recursive하게 계산된다: $kv_0 = 0, \ kv_t = kv_{t-1}+ k_tv_t^{\top}, \ o_t^{\top} = q_t^{\top}kv_t$

  - 이는 내재적으로 병렬화불가능하므로, lightning attention은 타일링 기법으로 attention score를 계산한다.

- 이에 따라 Q,K,V 행렬을 2개의 개별 block으로 나눈다: 

  - <img src="https://github.com/user-attachments/assets/5fd9c7e9-1ad2-4a1b-8a96-b03e46f4657f" width="50%"></img>


  - right product를 펼치면 다음과 같이 쓸 수 있다: $kv_s = kv_0 + \sum^s_{j=1}k_jv_j^{\top}, \ s=1,..., m$ $o_s^{\top} = q_s^{\top} kv_s =q_s^{\top}kv_0 + q_s^{\top} \sum^s_{j=1} k_jv_j^{\top}$

  - 이를 block form으로 다시 쓰면:  
  
    - <img src="https://github.com/user-attachments/assets/86c49bf6-792f-47ef-95dc-565f98c07c32" width="50%"></img>

      - intra-block인 $[(Q_1K_1^{\top}) ⊙M)]V_1$ 은 left product를 사용하고, inter-block인 $Q_1KV_0$ 은 right product를 사용한다.

    - intra-block은 다음과 같이 나뉠 수 있다:

      - <img src="https://github.com/user-attachments/assets/6d7cc153-d88b-4dbe-a82b-0ddf375e22ef" width="60%"></img>

  - 2번째 block을 계산하기 위해, $KV_1 = kv_m$ 을 사용하며 다음과 같이 계산된다: $KV_1 = KV_0 + \sum^m_{j=1} k_m v_m^{\top} = KV_0 + K_1^{\top} V_1$

    - $KV_0 = kv_0$

행렬을 여러 개의 block들로 나누는 방법을 반복적으로 적용함으로써, 실제 계산 복잡도를 선형으로 줄일 수 있다. lightning attention의 계산 복잡도는 $O(nd^2 + nBd)$이며, 여기서 `B`는 block size이다.

![image](https://github.com/user-attachments/assets/4e7f7cb0-c942-4c44-8389-221fb352b805)


#### 2-4. Experiments & Results

실험을 위해 softmax(Flash attention-2 적용), lightning attention, hybrid-lightning attention model을 다양한 크기(70M ~ 7B)로 학습하였다. 각 모델은 300B 토큰을 이용해 8192의 context length로 학습하였으며, `Chinchilla`의 학습 방법을 따랐다.   
여기서, hybrid-lightning attention model은 매 8번째 layer에 lightning attention이 아닌 softmax attention을 둔 모델이다.

<div align="center"><img src="https://github.com/user-attachments/assets/d1a535f1-91ec-4e2c-82e8-4a0eb254a4d8" width="80%"></img></div>  


실험결과, `NIAH`를 제외하면 lightning attention은 기존 transformer model과 비슷한 성능을 보였다. 하지만, retrieval task에서는 약한 성능을 보였는데, 이는 hybrid model에서는 오히려 transformer model을 능가하였다.   
따라서, hybrid-lightning attention model은 LLM의 in-context learning에 적합하다 볼 수 있다.

<div align="center"><img src="https://github.com/user-attachments/assets/591732be-d2ca-4a62-9ad9-415f8be5a3ac" width="60%"></img></div>  


3B 모델들의 end-to-end training speed를 평가하기 위해 초당 GPU 토큰 처리율(TGS)를 확인한 결과, lightning attention은 sequence length와 상관없이 변화없는 training speed를 보였으며 FlashAttention-2를 능가하는 유일한 선형 모델이다.

<div align="center"><img src="https://github.com/user-attachments/assets/ce01b169-39e0-4cbc-9944-05c037e025a1" width="80%"></img></div>  



- Module Ablations in MoE

  - Hybrid-lightning Attention VS Softmax Attention: 28B 모델(MoE with 5B)인 softmax attention으로 구성된 base model과 여기에서 매 8번째만 softmax attention으로 두고 나머지는 lightning attention으로 둔 hybrid-lightning attention을 비교했다. 그 결과, lightning attention이 대부분의 benchmark에서 좋은 성능을 보였다.

  - Pre Layer Normalization VS Post Layer Normalization: PostNorm은 gradient 소실 및 폭발 문제로 인해 기존 transformer LLM에서는 PreNorm이 사용되는 추세였다. 하지만 실험 결과 hybrid-lightning attention model에서는 PostNorm이 성능이 더 좋았다(여기에서는 DeepNorm 사용).

<br></br>

### 3. Pre-Training

- Tokenizer: Byte-level BPE

  - resulting vocabulary size is set to 200K

- Data:  academic literature, books, web content, programming code

  - 데이터 질을 높이기 위해 필터링 과정을 거침(rule-based cleaning, deduplication)

  - 여러 데이터 카테고리를 포함하기 위해 샘플링 사용


### 4. Post-Training

- Supervised Fine-Tuning (SFT)

  - `rejection sampling`을 이용해 높은 품질의 대답 생

- Offline and Online Reinforcement Learning (RL)

  - Offline: DPO를 사용해 모델 성능 최적화

  - Online: Group Relative Policy Optimization(GRPO) 를 바꾸어 사용

### 5. Results

<div align="center"><img src="https://github.com/user-attachments/assets/34436d00-d3a9-4c44-9e62-e71337ff3fda" width="60%"></img></div> 


<div align="center"><img src="https://github.com/user-attachments/assets/2dad2002-360d-4c5c-b42f-8e18699c932f" width="60%"></img></div>     
