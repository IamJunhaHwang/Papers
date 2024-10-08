## :page_facing_up: LLAMA-ADAPTER: EFFICIENT FINE-TUNING OF LARGE LANGUAGE MODELS WITH ZERO-INITIALIZED ATTENTION


### Info

* `publication`: ICLR 2024
* `author`: Renrui Zhang et al.
* `url`: https://openreview.net/pdf?id=d4UiXAHN2W

### 1. Abstract & Introduction

<div align="center"><img src="https://github.com/user-attachments/assets/7ac28d28-5aa7-48f7-904e-17dd675ad9e6" width="60%"></img></div>

본 논문에서는 LLaMA의 효율적인 instruction tuning을 위한 lightweight adaption method인 **LLaMA-Adapter** 를 제안한다. LLaMA-Adapter는 1.2M의 학습 가능 파라미터를 LLaMA 7B 모델에 도입하고 fine-tuning에 1시간 이하 정도만을 사용한다.

특히, 학습가능한 zero-gating을 적용해 LLaMA의 self-attention layer에 적응적으로 instruction cues를 주입하는 zero-initialized attention 메커니즘을 제안했다. 이를 통해 안정적인 학습 과정과 좋은 성능에 기여한다. (Alpaca with fully finr-tuned 7B와 견줄만한 quality를 가질 수 있음)   
LLaMA의 위쪽(higher) Transformer layer에 학습 가능한  adaption prompts 세트를 prefix로 단어 토큰에 추가한다. 그리고 훈련 초반부에서의 랜덤으로 초기화된 prompt로부터 오는 noise를 피하기위해, frozen self-attention layers를 learnable gating factor와 함께 사용한다. 이 gating 메커니즘은 0으로 초기화되며, 어텐션 계산 과정에서 프롬프트와 단어 토큰 간의 feature interaction을 제어한다.

또한, image encoder를 포함하는 것으로 본 논문의 approach를 image-conditioned instruction following을 위한 Multi-modal LLM으로 확장할 수 있다. 

더 나아가, 제안된 zero-initialized attention 메커니즘을 전통적인 비전 및 NLP 과제에서 다른 사전 학습된 모델(ViT, RoBERTa, CLIP)을 fine-tuning하는 데에도 적용하여, 해당 방식의 효과성과 범용성을 입증하였다.


* contribution

  - 1.2M parameters : 7B parameters를 full fine-tuning하는 대신, LLaMA를 freeze시키고 1.2M의 zero-initialized attention 메커니즘만을 학습시켰다.

  - one-hour fine-tuning : 위와 같은 lightweight adaption module with zero-initialized gating으로 LLaMA-Adapter는 8개의 A100 GPUs로 1시간 이하의 training cost를 소요한다.

  - plug with expertise : 다양한 시나리오에 따라 LLaMA에 각기 다른 전문가 지식이나 새로운 모달리티 입력을 제공하기 위해 각각의 어댑터를 유연하게 삽입할 수 있다. 따라서 13G 크기의 LLaMA 전체를 복사하는 대신, 각 컨텍스트마다 1.8M 크기의 어댑터만 저장하면 된다.

  - Multi-modal Reasoning : 제안하는 방법은 image encoder를 활용하여 multi-modal LLM을 만들 수 있다. 다른 연구들과 비교하여 LLaMA-Adapter가 높은 tuning 효과를 보였다.

<br></br>

### 2. LLAMA-ADAPTER

#### 2-1. LEARNABLE ADAPTION PROMPTS

- LLaMA의 N개의 transformer layers가 주어지면, 위쪽의 L개의 layer($L \leq N$)에 learnable adaption prompt를 삽입한다.

  - 이 때, prompt는 $\{P_l\}^L_{l=1}$ 이고, $P_l \in \mathbb{R}^{K \times C}$

    - K : 각 layer의 prompt length
    - C : LLaMA의 transformer dimension과 같음

- l번째에 삽입한 layer($l \leq L$)로 예를 들면, 길이가 M인 word tokens을 $T_l \in \mathbb{R}^{M \times C}$ 라 하자. 이는 input instruction을 나타내고 이미 생성된 response이다.

- learnable adaption prompt와 위의 word tokens는 concatenate 되며, 이를 prefix로써 사용한다.

  - $[P_l ; T_l] \in \mathbb{R}^{(K+M) \times C}$

  - 이러한 방법으로 $P_l$에서 배운 instruction knowledge는 transformer block에서의 zero-initialized attention layer을 통해 다음 contextual response를 생성하도록 $T_l$ 을 효과적으로 가이드할 수 있다. 

<div align="center"><img src="https://github.com/user-attachments/assets/14a4fd09-907c-4d12-afe4-72dc393b1a08" width="30%"></img></div>



#### 2-2. ZERO-INITIALIZED ATTENTION

adaption prompts가 random으로 초기화된다면, 초기 학습부분에서 word tokens에 방해를 줄 수 있다. 이를 고려하여 우리는 마지막 L개의 layer의 vanilla self-attention을 zero-initialized variants로 바꾸었다.

l번째 inserted layer에서 $[P_l ; T_l]$ 에 추가적으로 (M+1)번째 단어를 생성한다고 했을 때, 해당 word token을 $t_1 \in \mathbb{R}^{1 \times C}$ 로 정의한다.   

- 어텐션 메커니즘에서, input token을 Queries, keys, values로 바꾸기 위해 linear projection layers가 먼저 적용된다.

  - $Q_l = Linear_q(t_l)$

  - $K_l = Linear_k([P_l ; T_l; t_l])$
  - $V_l = Linear_v([P_l ; T_l; t_l])$

- 새로운 word token와 모든 (K+M+1) tokens간의 attention score는 다음과 같이 계산된다. 

  - $S_l = Q_lK^T_l / \sqrt{C} \in \mathbb{R}^{1 \times (K+M+1)}$

  - $S_l = [S^K_l; S_l^{M+1}]^T$

    - $S_l^K \in \mathbb{R}^{K \times 1}$ : learnable prompt가 다음 토큰인 t를 생성하는 것에 얼마나 많은 정보를 기여하는지를 나타내며, 학습 초기에 방해를 유발할 수 있다.
    - $S_l^{M+1} \in \mathbb{R}^{(M+1) \times 1}$

- 어텐션에서 $S_l^K$ 의 중요성을 적응적으로 제어하기 위해 learnable gating factor인 $g_l$ 을 적용한다.

  - $g_l$ 은 0으로 초기화되어 under-fitted 프롬프트의 영향을 제거한 후, 점진적으로 크기를 증가시켜 LLaMA에 더 많은 instruction semantics를 제공할 수 있다.

  - 따라서, 다음과 같이 softmax를 2가지 구성요소에 적용한다 : $S_l^g = [softmax(S_l^K) \cdot tanh(g_l); softmax(S_l^{M+1})]^T$

    - `tanh()` 는 gating factor를 -1 ~ 1의 값을 가지도록 하기 위해 적용된다.

    - 소프트맥스를 따로 적용함으로써 두 번째 항이 adaption prompt와 무관하도록 보장하며, 사전 학습된 지식이 방해받지 않도록$softmax(S_l^{M+1})$ 에 어떠한 계수도 곱하지 않는다. 이는 사전 학습된 확률 분포를 그대로 유지한다.

  - $g_l$ 이 0에 가까워지면, LLaMA의 사전 학습된 지식을 대부분을 토큰 $t_l$ 로 전달하여 신뢰할 수 있는 생성이 가능하며, 실제로는 multiple $g_l$ 을 적용해 attention의 각 head에서 독립적으로 학습하게 한다. (multi-head attention)

- 마지막으로, l번째 attention layer의 output은 lineaer projection layer를 거치게 된다.

  - $t^o_l = Linear_o(S^g_l V_l) \in \mathbb{R}^{1 \times C}$

우리가 제안한 zero-initialized attention을 통해, adaption prompts가 점차 새로운 instructional signal을 transformer에 주입하도록 할 수 있으며, 동시에 LLaMA의 pre-trained 지식을 포함시킬 수 있게되어 높은 질의 response를 제공한다.

<br></br>

<div align="center"><img src="https://github.com/user-attachments/assets/1b90dc07-1629-43ae-90aa-cd8c96c4ec28" width="70%"></img></div>

#### 2-3. MULTI-MODAL REASONING

language instruction 뿐만 아니라, 간단한 변경을 거치면 LLaMA-adapter는 image input을 기반으로 QA를 가능하게 한다.

- input image에 대해 CLIP과 같은 pre-trained visual encoder를 활용해 multi-scale global features $\{I_m \}^M_{m=1}$ 를 추출한다.

  - $I_m \in \mathbb{R}^{1 \times C_m}$
  - M : scale number

- 그 다음, M-scale features를 channel dimension에 따라 concatenate하고 learnable projection network를 적용해 word embedding space로 변환한다.

  - $I_p = Projection(Concat(\{I_m \}^M_{m=1}))$

    - $I_p \in \mathbb{R}^{1 \times C}$

  - 이는 adaption prompt와 같이 동일한 특징 차원을 가진 전체 이미지 토큰으로 간주된다.

- $I_p$ 를 K번 반복하고 모든 L개의 inserted transformer layers에서 길이가 K인 adaption prompts에 element-wise하게 더한다.

  - l번째 layer의 multi-modal prompt는 다음과 같이 나타낸다 : $P^v_l = P_l + Repeat(I_p) \in \mathbb{R}^{K \times C}$

    - $P^v_l$ : 주어진 image로 부터의 visual 정보를 포함한 prompt

- 이후, zero-initialized attention은 LLaMA에 image-conditional semantic을 점진적으로 불어넣을 수 있도록 학습하게 된다.

##### Training Strategy

language-only instruction tuning을 위한 Alpaca의 데이터를 사용하는 대신 multi-modal instruction data로 LLaMA-Adapter를 fine-tune하고 평가하였다.

- ScienceQA : training set으로 LLaMA-Adapter를 훈련하고, in-domain testing 진행하였다. **image encoder와 LLaMA는 freeze하였으며, projection layer와 zero-initialized attention 메커니즘만 훈련**

- Zero-shot Multi-modal Evaluation : out-of-domain generation 능력을 검증하기 위해, 2단계의 multi-modal training과 3가지 benchmarks(MME, MMBench, LVLM-eHub)에서의 평가를 진행했다.

  - 첫 단계로, LAION-400M의 raw image-caption 데이터로 projection network와 zero-initialized attention 모듈을 tuning하였으며 이는 visual features와 word tokens간의 embedding space의 alignment를 위한 것이다.

  - 두 번째 단계로, projection network는 freeze시키고 Alpaca data와 LLaVA-I의 조합으로 zero-initialized attention을 학습한다. 이는 human instruction을 기반으로 detailed response를 생성하는 능력을 LLM에게 심어주기 위함이다.

<br></br>

### 3. EXPERIMENT

#### 3-1. Instruction-following Evaluation

- Settings

  - Stanford Alpaca를 따라, 52K instruction-following data를 사용하여, LLaMA-Adapter를 8개의 A100 GPUs에 5 Epochs로 학습

  - warmup epochs, batch-size, learning rate, weight decay : 2, 64, 0.009, 0.02

  - pre-trained LLaMA 7B(32 transformer layers)를 이용했으며, `K=10` 인 adaption prompt를 마지막 `L = 30` layer에 삽입
  
  - 똑같이 52K instruction data로 학습한 Alpaca, Alpaca-LoRA와 비교하였으며, GPT-4 평가 벤치마크로 평가하였다. 
    - GPT-4 평가 벤치마크 : GPT-4에게 80개의 질문에대해 다른 두 모델이 만든 response의 품질을 평가하게 하는 것

<div align="center"><img src="https://github.com/user-attachments/assets/87694a0c-0a0f-4810-8c22-56f2c4428b09" width="70%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/eb8c8c98-59a2-40b9-bffc-93f33f8d4138" width="70%"></img></div>


- Performance & Efficiency

  - GPT-4 평가에서 LLaMA-adapter가 다른 모델들에 비해 `win`을 더 많이 얻어냈다.

  - Table 1에서 알 수 있듯이, 제안한 LLaMA-adapter는 뛰어난 효율을 보이고 있다.

#### 3-2. Multi-modal Evaluation

-  Settings

  - CLIP을 visual encoder로 사용하였으며, projector로는 간단하게 MLP를 이용하였다.

  - ScienceQA 데이터에서, 주어진 질문, textual context, options를 하나의 문장으로 concat하였다.

  - zero-shot multi-modal evaluation(MME, MMBench, LVLM-eHub)에서의 비교 모델로는 LLaVA와 MiniGPT-4를 사용했다.

<div align="center"><img src="https://github.com/user-attachments/assets/96c86b73-b248-4d4d-83f2-28520f10853b" width="70%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/f3ee0903-3ae7-46f6-9fe3-87bea2fcebae" width="70%"></img></div>


- Performance

  - single modal인 `LLaMA-Adapter-T`는 78.31%의 정확도로 여러 VQA 모델들을 능가하였으며, visual semantic을 더하는 것으로 +6.88%의 정확도 향상을 얻어내어 GPT 시리즈들보다 좋은 성능을 보였다.

  - Table 3의 multi-modal benchmark에서는 비교 모델들과 견줄만한 성능을 얻었다. 하지만, 우리가 제안한 모델을 비교 모델보다 더 효율적이다. (비교모델들은 full fine-tuning 임)


<div align="center"><img src="https://github.com/user-attachments/assets/b5c7cc4b-1105-45b0-b9fe-7704008abf76" width="70%"></img></div>

#### 3-3. Ablation Study

- Insertion layers

  - insertion layer를 늘릴수록 파라미터가 더 필요해지지만, 큰 성능 향상을 가져왔다(ScienceQA validation set에서).

  - 최적의 개수를 찾을 자원이 제한된 경우, 단순히 모든 트랜스포머 레이어에 삽입하는 것이 일반적으로 좋은 방법

    - 너무 많은 레이어에 삽입하면 입력 단어의 초기 인코딩을 방해할 수 있다.

- Zero-initialized Attention

  - 제안한 zero-initialized 와 random initialized를 비교하였을 때, 제안한 방법이 상당한 향상(+43.08% on ScienceQA validation set)을 가져왔다.

  - 오른쪽의 loss plot을 보면, random-initialized는 느리게 0.15정도에 도달하며 완전히 수렴하지도 않는다.
