## :page_facing_up: The Transfer Neurons Hypothesis: An Underlying Mechanism for Language Latent Space Transitions in Multilingual LLMs

### Info

* `publication`: EMNLP 2025 -  Long

* `author`: Hinata Tezuka et al.
* `url`: https://aclanthology.org/2025.emnlp-main.1618.pdf
* `github`: https://github.com/HinaTezuka/emnlp2025-transfer-neurons


<div align="center"><img src="https://github.com/user-attachments/assets/3983f918-356b-4075-b8a9-fedd9ae7c7a1" width="40%"></img></div>


### 1. Abstract & Introduction

최근 연구에서는, decoder-based multilingual LLM이 input language와 독립된 semantic latent space에서 동작한다고 추정했다: (1) input을 shared latent space로 변환하고, (2) semantic processing을 수행한 후, (3) 다시 results를 input language로 변환하는 식이다(Figure 1 - a).

하지만, 이러한 transformation이 내부에서 어떤 식으로 동작하는지는 underexplored 되었다. 이를 다루기 위해, (1) 내부 representation이 layers간에 어떻게 바뀌는지, middle layers에 shared semantic latent space가 존재하는지 실험하였다.   
(2) 또한, representation transfer mechanism 설명을 위해 _Transfer Neurons Hypothesis_ 를 제안한다.   
이는 특정 neurons(named transfer neurons)가 input language에 sensitive해서 이들의 활성화로 해당되는 value vectors가 residual stream에 더해지고, 이를 통해 representation trasfer가 된다는 가정이다.

- Contributions

  - multilingual LLMs의 internal representations dynamics를 알아냄: 초기 representation은 language-specific하고, 곧이어 shared latent space로 수렴되었다가 다시 갈라짐

  - transfer neurons의 존재를 실험적으로 증명함. 또한, 해당 representation transfer의 필요성을 intervention 실험을 통해 보임

  - 최근 연구들에서 밝혀진 language-specific neurons의 핵심 역할 중 하나는 lantent spaces 간의 이동이며, 이러한 transfer neurons가 downstream tasks에 필수적임을 보임

<br></br>

### 2. The Dynamics of Hidden States

이전 연구의 발견들을 바탕으로, shared semantic latent space는 English-centric이 대부분이라고 가정한다.


#### Spatial Transition Phenomenon

<div align="center"><img src="https://github.com/user-attachments/assets/52e28430-638f-448b-93e6-dd765ec4fc30" width="70%"></img></div>

**Representations Form Language-Specific Latent Spaces and a Shared Latent Space in Hidden State Space.**

multilingual parallel QA dataset인 MKQA dataset을 이용해 각 언어당 1k의 문장들을 encoding하여, 각 layer로 부터 문장 각각의 final token에 해당하는 hidden states를 추출하였다.   
이에 대해, PCA를 적용하여 top-2 주성분으로 plotting하여 위 그림에 나타내었다.

그림을 보면, 초기 layers에서 언어들 각각이 그만의 latent space를 형성하는 것을 볼 수 있으며, 이는 점점 middle layer의 shared latent space로 수렴된다. final layer에서는 다시 구별되는 각각의 latent space를 만든다. (hypothesis를 뒷받침)

또한, English와 비슷한 Dutch, Italian의 경우 initial과 final layer에서 English와 가까운 latent space를 가지며, 이와 반대로 Japanese와 Korean은 English와 구별되는 것을 볼 수 있다(shared latent space의 경우에도 섞여있긴 하지만 완전히 수렴되지는 않음).

이러한 관찰은 representations는 English-centric shared semantic latent space로 수렴되는 경향이 있으며, 언어들간의 수렴 정도는 다양하다는 것을 의미한다.

#### The Existence of a Shared Semantic Latent Space in Middle Layers

<div align="center"><img src="https://github.com/user-attachments/assets/7c80b430-962f-4a1e-91eb-b539dfccd5f7" width="40%"></img></div>

**Sentence Representations with Similar Meanings across Languages Converge to Similar Locations in Middle Layers**

언어와 상관없는 semantic processing이 shared semantic latent space에서 일어나는지 수치적으로 증명하기 위해, language pairs 간의 parallel sentence representations 사이의 `Mutual k-Nearest Neighbor Alignment Metric` 을 측정했다.   
만약, middle layers에서 similarities peak가 있다면, 의미적으로 비슷하거나 동일한 문장 representation이 hidden state space에서 언어와 상관없이 비슷한 위치로 수렴이되는 것으로 볼 수 있다(이는, shared nearest neighbors를 늘림).

`Fig.4` 는 middle layers에서 English-other languages 간의 latent space similarity **peak** 를 나타낸다. 즉, 언어와 상관없는 semantic processing이 있는 것이다.

<br></br>

### 3.  Identifying Transfer Neurons

#### 3-1. Hypothesis:  Specific Activations and Value Vectors Facilitate a Parallel Shift of Representations to the Target Latent Space


- input vector $x \in \mathcal{R}^d$ 가 주어졌을 때, Transformer의 layer l에서 MLP module은 아래와 같이 동작한다:

  - $MLP^l(x) = a(xM^l_{up})M^l_{down}$

- `Geva et al. (2021)`에서 제안한 vector-based key-value store view에서의 MLP module은 다음과 같다:

  - keys: $d_m$ column vectors in $M^l_{up}$

  - values: $d_m$ row vectors in $M^l_{down}$
  - query: x

- 이에 따라, $\alpha^l = a(xM^l_{up})$ 을 key와 query의 relevance scores로써 보면 MLP module은 아래와 같이 쓸 수 있다:

  - $MLP^l(x) = \alpha^lM^l_{down} = \sum^{d_m}_{i=1}\alpha^l_i v^l_i$

    - $v^l_i$ : $M^l_{down}$ 의 i 번째 row vector

    - 본 연구에서는, gate mechanism을 적용함: $\alpha^l = a(xM^l_{gate}) \odot xM^l_{up}$

      - $M^l_{gate} \in \mathcal{R}^{d \times d_m}$: gate projection
      - $\odot$: element-wise multiplication

위 수식에 기반해, 특정 activations와 이에 해당하는 value vectors $\alpha^l_i v^l_i$ 는 language-specific latent spaces 와 shared semantic latent space 사이의 internal representations shifting을 유발한다고 가정한다. 여기에서 activation units를 _Transfer Neurons_ 라 정의한다.

#### 3-2. Preparation: Two Types of Transfer Neurons

- 본 논문에서는 2가지 transfer neurons를 고려한다.

  - **Type-1 Transfer Neurons**: 초기 layers에서 input representations를 language-specific -> shared semantic 으로 바꾸는 Neurons

  - **Type-2 Transfer Neurons**: 마지막 layers에서 representations를 shared semantic -> language-specific 으로 바꾸는 Neurons

- 특정 layers에 퍼져있는 parallel & non-parallel sentence pairs 사이의 representation similarity 차이에 따라, Type-1 Neurons 탐지에 1\~20 layers를 Type-2 Neurons 탐지에 21\~32(마지막 layer까지) layers를 설정 

#### 3-3.  Scoring the Candidate Neurons

우리의 목표는 원하는 latent space에 가깝게 representation을 옮기는 후보 neurons에 높은 점수를 할당하는 것.

- Centroids Estimation for Latent Spaces: 각 representational latent space의 중심 추정으로 거리 계산

  - $h^l_{L2, k}$: 특정 non-English 언어 "L2"에서 l번째 layer의 hidden states에 해당하는 k번째 sample sentence

  - 각 space의 centroids는 다음과 같이 추정

    - $C^l_{L2} = \frac{1}{n} \sum^n_{k=1} h^l_{L2, k}$: l번째 layer의 L2-specific latent space 의 중심

    - $C^l_{shared} = \frac{1}{n} \sum^n_{k=1} mean(h^l_{en, k}, h^l_{L2, k})$: l번째 layer의 shared semantic latent space 중심

      - 계산을 위해 parallel sentence pairs English-L2 pair 사용

    - n: total number of sample sentences


- Scoring Methodology: 다음과 같이 각 cancidate neuron의 점수 측정


  - <img width="477" height="267" alt="image" src="https://github.com/user-attachments/assets/df2a4dbf-829a-4bbc-bca7-d7fafdc6c1e6" />

    - $h^{l-1}_k$: 이전 layer의 hidden state

    - $A^l_k$: l번째(현재) layer의 self-attention output

    - k: index of the sample sentence

    - $(h^{l-1}_k + A^l_k)$: k번째 input에 대해 l번째 layer에서 MLP module 직전의 hidden states

    - `dist`: 두 벡터가 얼마나 가까운지 측정하는 함수 (여기에서는 cosine similarity)

    - $L^l_k$: layer score; MLP 이전의 hidden state가 target latent space의 l번째 레이어 중심과 얼마나 가까운지

    - $N^l_{i, k}$: neuron score; neuron과 이에 해당하는 value vector가 representation을 target latent space로 얼마나 효과적으로 옮기는지

  - l번째 layer에서 i번째 neuron의 점수가 0보다 크다면, neuron과 value vector가 대부분의 sample에 대해 target latent space 중심에 가깝게한 것이고, 반대로 0보다 작다면, 중심과 멀게 만든 것임

  - Type-1 neurons를 찾기 위해 $C^l$ 을 $C^l_{shared}$ 로 설정하고, Type-2 neurons의 경우 $C^l_{L2}$ 로 설정

  - score가 클수록 representations를 target latent space로 더욱 가깝게 옮긴 것이며, 모든 후보 neruons를 내림차순으로 정렬하여 top-n neruons를 추출함

<br></br>

### 4. Detecting and Controlling Transfer Neurons

#### Distribution

<div align="center"><img src="https://github.com/user-attachments/assets/aa40fc9f-4893-48c4-a8e0-e88148217f46" width="40%"></img></div>

위 그림은 layer간의 transfer neurons distribution을 보여준다.   
Type-1 neurons는 첫 layers와 middle layers에 주로 포함되어 있지만, Type-2의 경우 마지막 layers에서 주로 찾을 수 있다.

#### Representation Similarity Measurement while Deactivating Transfer Neurons

<div align="center"><img src="https://github.com/user-attachments/assets/87bb6537-1fdd-489c-81fd-a79ad557b207" width="70%"></img></div>

Figure 5는 parallel & non-parallel English-L2 sentence pairs에서 hidden states와 MLP activation patterns의 similarity를 나타낸다.

baseline은 Type-1 neurons를 랜덤으로 1k를 deactivating한 것이고, 이외는 top-1k를 deactivating한 것이다(전체 neuron의 0.2%).    
아주 작은 neurons가 deactivated 되었음에도 similarity에서 가파른 reduction을 관찰했다.

이러한 결과는 Type-1 neurons가 deactivate되었을 때, input representations를 shared semantic latent space로 mapping할 수 없음을 의미한다.

<br></br>

### 5. The Nature of Transfer Neurons

#### Language Specificity

<div align="center"><img src="https://github.com/user-attachments/assets/d82759e0-1580-487d-a593-9d5805d23e92" width="40%"></img></div>

transfer neurons의 language specificity를 알아내기 위해, neuron activations와 inputs 사이의 correlation ratio을 측정했다.

표를 보면, Type-1 neruons는 일반적으로 correlation이 없고, Type-2의 경우 강한 상관관계를 보인다. 이는 Type-2 neurons가 shared semantic latent space에서 각 language-specific latent space로 강하게 shift한다고 볼 수 있다.

#### Assessing the Importance of Transfer Neurons in Reasoning

reasoning에서 Type-1 neurons의 역할을 알기 위해 inference에서 Type-1 neurons를 deactivate 해보았다.

- 사용 데이터: MKQA

- 결과는 (a) 개입 없을 때의 성능, (b) top-1k Type-1 neurons deactivate, (c) ramdomly 1k Type-1 neurons deactivate로 보였다.

- 실험 결과, Type-1 neurons를 deactivate했을 때, 상당한 degradation을 관찰했다.
