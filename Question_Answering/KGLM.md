## :page_facing_up: Empowering Language Models with Knowledge Graph Reasoning for Open-Domain Question Answering

### Info

* `conf`: Accepted at EMNLP 2022
* `author`: Ziniu Hu et al.

* `url`: https://aclanthology.org/2022.emnlp-main.650.pdf


### 1\. Abstract & Introduction

open-domain QA (ODQA)는 맥락 내 개체에 대한 world knowledge를 필요로 한다. 하지만 PLM이 모든 필요한 knowledge를 저장할 수 없기 때문에 외부의 knowledge source를 이용해 LM을 augment했다. (ODQA는 이와 같이 external source에 접근하는 것을 가정함)

이 논문에서는 knowledge Graph (KG)를 이용한 `knOwledge REasOning empowered Language Model (OREOLM)` 를 제안한다. 이는 기존의 Transformer 기반 LM에 연결될 수 있는 Knowledge Interaction Layer (KIL)로 구성되며 Knowledge Graph Reasoning Module과 상호 작용한다. 제안한 방법을 통해 LM은 KG를 원하는 답으로 이끌도록 하며 찾아낸 knowledge가 LM의 성능을 높여준다.

p.s. 기존의 KG를 이용한 QA는 LM이 올바르게 질문을 이해하기 위해 KG와 상호 작용하지 않았으며, 답으로 내놓는 것들은 node나 edge에 제한적이었음.

제안한 `OREOLM`을 RoBERTa와 T5에 적용하여 상당한 성능 향상을 얻을 수 있음을 관찰하였고 이는 KG resoning의 능력 덕분이다.

- Contributions

  - symbolic knowledge graph resoning과 neural LM의 통합을 위한 `OREOLM`을 제안함.

  - OREOLM을 RoBERTa와 T5로 Wikipedia corpus로 pretrain하였을 때, ODQA에서 상당한 성능 향상이 있었다.
  - OREOLM은 해석 가능한 resoning path와 high-order reasoning rule을 제공한다.

<br></br>

### 2. Methodology

- Preliminary

  - $\mathcal{KG} = (\mathcal{E, R, A} = \{A_r\}_{r\in\mathcal{R}})$

    - $e \in \mathcal{E}$, $r \in \mathcal{R}$ : entity node & relation label

    - $A_r \in \{0, 1 \}^{|\mathcal{E}| X |\mathcal{E}|}$ : 희소 인접 행렬(sparse adjacency matrix); 개체 간의 관계 `r`이 성립하는지 여부를 나타냄

  - `knowledge graph reasoning`은 factoid query `(s,r,?)`에 답하는 것을 목표로 한다. 즉, 대상 개체와 관계 r을 가지는
개체가 무엇인지 찾는 것. (이는, KG가 완벽하다면 인접행렬 A를 통해 간단히 가능하겠지만, KG가 불완전한 경우에는 multi-hop을 돌아 정답을 내는 path-based reasoning 접근법을 채택하였었다.)

- **Overview**

  - `Knowledge Interaction Layer (KIL)`는 LM layer 사이에 위치하여 KG reasoning module과 상호 작용한다.

  - KIL은 먼저, 모든 in-context 개체들에 대해 relation distribution을 예측하고 KG reasoning module은 이 distribution에 따라 그래프에서 움직인다.

  - 각 step의 reasoning 결과는 움직이면서 찾아낸 개체 간의 가중 평균 임베딩으로 요약된다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/7328fe35-f8d0-492b-9f4a-f0bbf5963181" width="70%"></img></div>

#### Notations

- `Q = "The Bauhaus represented Germany’s recovery from which event?"`라는 질문이 주어지면 QA model은 모든 N개의 in-context entity mentions $M = \{m_i \}^n_{i=1}$ 에 대해 외부 지식이 필요하다. (예를 들면, Bauhaus가 설립되었을 때의 독일의 역사)   

  - 이러한 open-domain QA는 다음과 같이 추상화할 수 있다. $P(a | q, M)$

- 각 mentioned entity $m_i$ 에서 시작해, 모델이 관련된 지식을 찾기 위해 그래프를 거닐도록 학습하고 질문에 대답하기 위한 길이 T의 reasoning path를 만들게 한다.

  - entity mention 으로부터 시작한 각 reasoning path를 chain of entities(states) random variables $p_i = \{e^t_i \}^T_{t=0}$ 으로 정의한다. 각 mentioned entity가 initial state이다. i.e. $e^0_i = m_i$

  - 모든 path의 union은 질문에 대답하기 위해 각 mentioned entity로부터의 reasoning path로 구성된 $\mathcal{Q} = \{p_i\}$ 로 정의한다.

- 가능한 path $\mathcal{Q}$ 를 잠재 변수로 포함하는 것으로 $P(a | q, M)$ 은 다음과 같이 볼 수 있다.

  <img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/b3d99e0c-e846-4da9-afec-87d191bb11b0" width="40%"></img>

  - 여기에서, (1) reasoning path들은 각기 다른 entities에서 독립적으로 만들어지며 (2) reasoning path들은 autoregressive하게 생성된다고 `가정`한다.

  - 이에 따라, QA 문제는 (1) autoregressive하게 graph를 거닐며 각 $m_i$ 로부터 path $p_i$ 를 얻는 **KG Reasoning**, (2) 정답 예측을 위한 외부 지식을 얻기위해 reasoning path로 부터의 이득을 취하는 **knowledge-injected LM** 으로 나뉜다.

    - KG Reasoning 단계에서 $p_i$ 는 각 step `t` 에서 다음 entity $e_i^t$ 의 선택을 필요로 하는데, 우리는 이것을 2가지 단계로 나누었다: (1) 현재 상태를 기반으로 next-hop의 relation을 예측하기 위해 LM이 관여하는 **relation prediction**, (2) KG와 예측한 relation을 기반으로 next-hop entity를 예측하는 **non-parametric state transition**

      <img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/9169d4e2-0e4e-4481-b6a2-a04186de2c35" width="50%"></img>

    - 각 step `t` 에서, entity distribution을 확률 벡터 $\pi_i^{(t)} \in \mathcal{R}^{|\mathcal{E}|}$ 를 통해 추적한다. $\pi_i^{(t)} [e]$ 는 entity `e`에 머물 확률이 된다. 즉, $P(e^t_i = e | q, {e_i^{\lt t}})$

- **Model Input**

  1. input question에 존재하는 모든 N entity mentions $\{m_i \}_{i=1}^N$ 과 대응되는 KG entities를 식별한다.

  2. 각 entity mentions 에 대해 다음과 같이 special token을 붙인다. ex) [S-ENT] Bauhaus [REL] [T-ENT]

  3. KIL은 유연하게 LM intermediate layer에 삽입될 수 있으며, 여기에서는 각 N Transformer LM layer에 삽입했다. 따라서, t번째 KIL의 input은 $LM_k^{(t)}$ 로써 각 토큰 k의 contextualized embeddings 이다.

<br></br>

#### 2-1. LM involved KG Reasoning

<img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/4e244420-e5b8-44de-9d91-cda2833bb56b" width="50%"></img>

##### Relation Prediction : 각 $m_i$ 에 대해 어떤 relation이 $r_i^t$ 로 취해져야 하는지 예측하기

relation 확률 벡터 $\mathfrak{r}^{(t)} = P_{rel} (r_i^t | q, e_i^{\lt t}) \in \mathcal{R}^{|\mathcal{R}|}$ 를 정의한다. 해당하는 `[REL]` 토큰을 `REL[i]`로 표시하며, $LM_{REL[i]}^{(t)}$ 는 next relation에 대한 힌트를 주는 관련된 정보를 가지고 있게된다. 또한, 각 relation의 d-차원 임베딩을 저장하고 있는 global key memory $K_{rel} \in \mathbb{R}^{|\mathcal{R}| \times d}$ 가 있다.

- relation probability vector를 구하기 위해 다음과 같은 과정을 거친다.

  <img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/6643f74c-45ff-4fd0-bb65-ee1f11db4143" width="35%"></img>

  - relation 토큰 임베딩을 key memory $K_{rel}$ 과 같은 공간으로 project하기 위해 `Q-Proj`라는 projection head를 거친 후 LayerNorm을 적용한다. 이후, dot-product로 similarity를 구한 후 softmax를 적용한다.

  - 여기에서 relation Queries $LM_{REL[i]}^{(t)}$ 는 context에 따라 모든 mention $m_i$ 와 reasoning step `t` 에서 다르며, relation 분포 $\mathfrak{r}_i^{(t)}$ 는 질문 q에 기반해 contextualized prediction을 준다.

  - 예측된 relation은 state transition을 만들기 위한 지시로써 knowledge graph reasoning module로 전달된다.

##### Contextualized KG Random Walk : state transition

entity `s`에서의 state를 생각하면, target `t`로 walking할 확률은 만약 `A[s, t] = 1`이면 $\frac{1}{deg(s)}$ 이다. 이에 기반해 random walk를 위한 Markov transition matrix를 다음과 같이 정의한다: $M_{rw} = D_A^{-1} A$, Degree matrix  $D_A \in \mathbb{R}^{|\mathcal{E}| \times |\mathcal{E}|}$ 는 대각 행렬로 주대각성분으로 `deg(1),...`를 가지고 있다.   
$M_{rw}$ 를 이용하면 state distribution을 다음과 같이 바꿀 수 있다 : $\pi^{(t)} = \pi^{(t-1)} M$
 
하지만, 위와 같은 random walk는 question `q`에 기반하지 않는다는 한계가 있다. 따라서, 우리는 **Contextualized Random Walk (CRW)** 를 제안한다.

- relation 분포 $\mathfrak{r}^{(t)}$ 에 기반해 edge 가중치를 조절함으로, 다른 weighted adjacency matrix $\tilde{A_i^{(t)}} \in \mathbb{R}^{|\mathcal{E}| \times |\mathcal{E}|}$ 를 계산한다.

  <img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/05dbc3f0-f22b-43ee-9e1b-4e136e989772" width="35%"></img>

  - $w_r$ 은 relation `r`을 위한 학습 가능 중요도 가중치이며, 이는 downstream task를 푸는 것에 도움을 준다.

  - $\mathfrak{r}_{i, r}^{(t)}$ 은 $\mathfrak{r}_i^{(t)}$ 에서 relation `r`과 대응되는 확률이다.

  - $M_{crw, i}^{(t)}$ 를 이용하면 state transition은 다음과 같이 정의 된다 : $\pi^{(t)} = \pi^{(t-1)} M_{crw, i}^{(t)}$   

- CRW는 각 reasoning path $p_i$ 가 자신만의 transition matrix를 가지게 한다. 하지만, entity nodes $|\mathcal{E}|$ 
 의 수가 커지면 인접 행렬을 업데이트하기 어려워진다. 따라서, 우리는 다음과 같은 `scatter-gather pipeline` 을 적용한다.

  - 먼저 각 edge에 대한 entity와 relation 확률을 얻어낸 다음, 해당 확률을 target node에 scatter 한다.

  - 이를 통해, `message passing`과 동시에 모든 $m_i$ 에 대해 바뀐 인접 행렬 $\tilde{A_i^{(t)}}$ 을 만들 수 있다.

  <img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/0985c772-eaa6-4b48-88a5-e42163bf1d75" width="40%"></img>

<br></br>

#### 2-2. Knowledge-Injected LM

Entity distribution 을 업데이트한 후, LM에 이러한 정보를 주입해야 한다.

<img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/e0a403cf-f418-4023-8600-5442a6eeb703" width="40%"></img>

1. global entity embedding value memory $V_{ent} \in \mathbb{R}^{|\mathcal{E}| \times d}$ 를 이용해 entity embedding을 저장하고 있고 각 배치의 subgraph에서 샘플로 뽑은 entities만 고려하므로 후보 entity embeddings의 세트를 검색하기 위한 쿼리로써 entity index list `I`를 얻는다.

2. entity distribution 과 embedding table로 가중합을 하고 `Value Projection block`으로 LM space로 projection한다.
3. 이후 [T-ENT]의 임베딩에 바로 더해 LayerNorm을 거친 후 해당 LM layer의 output이 되도록 한다.
4. 모든 $\hat{LM}_{T-ENT}^{(t)}$ 를 다음 Transformer-based LM layer의 입력으로 사용해 검색된 지식과 in-context words간의 상호작용을 self-attention을 통해 배우게 한다.

<br></br>

#### 2-3. Pre-Train OREOLM to Reason

존재하는 QA dataset에는 knowledge facts의 coverage가 작으므로 대규모 말뭉치에서의 사전 학습이 필요하다.

##### Salient Span Masking (SSM)

Salinet Span Masking (SSM) objective (Guu et al., 2020) 을 적용하는 것이 straightforward한 approach이다. 이 논문에서는 entities를 마스킹하였고 이때, 랜덤 마스킹 대신 entity ID set을 외부에서 샘플링하고 이것들과 이어진 모든 mentions를 마스킹하였다. 또한, 연속적인 token span을 모두 마스킹하였다.

각 SSM token에 대해 cross-entropy loss를 계산하였다.

##### Weakly Supervised Training of KIL

OREOLM에는 entity & relation embedding의 좋은 초기 값이 없다면 KIL은 random prediction을 하게 되고 KG가 질문과 관련 없는 entity를 가져오며 LM이 지식을 무시하도록 학습되게 된다. 이러한 `cold-start problem`을 피하고 좋은 embedding 초기 값을 제공하기 위해 다음과 같은 2가지 외부 시그널을 이용한다.

- **Entity Linking Loss** : entity embedding table의 초기화

  - supervision으로써 마스킹되지 않은 entity 사용

  - `Fevry et al. (2020)`와 비슷하게, 첫번째 KIL 전에 `[S-ENT]` 토큰의 출력 임베딩을 대응되는 entity embedding과 가까워지도록 강제한다.

    <img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/00d2b9ad-18e9-4f1c-a9c0-fe2ec08a140d" width="40%"></img>

- Weakly Supervised Relation Path Loss

  - 각 Wikipedia passage의 entity mention들은 자연스럽게 `WikiData KG`를 기반해 있으므로 몇몇 entities를 마스킹한 후, KG를 통해 다른 in-context entities로 부터 마스킹 entities까지의 모든 reasoning path를 구할 수 있다. 이를 Grounded Dependency Graph `DG`로 정의한다.

  - t번째 홉에서의 entity mention에 대한 모든 edge간의 전체 관계들의 set인 $R_{DG} (m_i, t)$ 를 정의한다. 이를 기반으로 weakly supervised relation label $q_i^{(t)} \in \mathbb{R}^{|\mathcal{R}|}$ 을 한 set의 각 relation에 uniform한 분포인 확률 벡터로써 정의한다.

  - 우리는 list-wise ranking loss를 적용해 KG에서 도달가능한 relation들이 다른 것보다 높은 점수를 가질 수 있도록 한다.

    <img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/e7431d6c-0a31-4209-864f-74380738911e" width="40%"></img>

<br></br>

### 3. Experiments

제안한 KIL layer는 대부분의 Transformer 기반 모델에 적용가능하다. 본 논문에서는 encoder-based LM으로써 RoBERTa-base, encoder-decoder LM으로써 T5에 대해 1~2 KIL layer을 더해 실험하였다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/5d96ef47-d384-46a2-9edf-0a41c84e869e" width="70%"></img></div>

- Closed-Book Generative QA

  - 3가지 single-hop ODQA datasets(NQ, WQ, TQA)와 2가지 multi-hop datasets(Complex WQ, Hotpot QA)에서 T5로 실험했다.

  - 모든 데이터셋에서 `Exact Match accuracy`를 평가 지표로 사용했다.

  - 실험 결과, 모든 데이터 셋에서 OREOLM을 적용한 모델의 성능이 더 좋았으며 이는 KIL layer가 늘어날수록 더 좋아졌다. 이는 OREOLM이 Closed-Book QA에서 매우 효과적이라는 것을 보여준다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/eeff0509-65f3-49de-906a-7fc54d37da15" width="40%"></img></div>
<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/198fbc70-2cda-46f1-bedc-22526bb51c31" width="80%"></img></div>


- Entity Prediction Task

  - Encoder-based LM의 경우 대부분의 Closed-Book QA에 사용할 수 없으므로 question 이후에 `[MASK]`토큰을 추가해 정답을 예측하게 하는 Closed-Book Entity Prediction을 진행했다.

  - WQ-SP, TQA 데이터셋을 사용해 평가하였으며 `KEPLER`를 이용해 실험하였으며 EaE, FILM, KEPLER 모델과 비교하였다.

  - 실험 결과, entity prediction task에 KG memory를 더하는 것은 상당한 효과를 가져왔으며, OREOLM은 SOTA 모델인 FILM보다 성능이 더 좋았다.

- Analyze KG Reasoning Module

  - 이전에 높은 reasoning step(T = 2)가 더 좋은 성능을 보였는데, 이를 우리가 사용한 KG가 많은 one-hop facts missing을 가지고 있으며 high-order reasoning이 이를 완화하도록 도와준다고 가정했다. 이를 실험해 보기 위해 `EntityQuestions` dataset으로 실험하였다.

  - 그 결과, Figure 4의 좌측처럼 `T = 2`가 one-hop이 크게 성능이 떨어진 것에 반해 괜찮은 성능을 보였으며 이에 대한 reasoning path를 우측에 그려본 결과, 모델이 매우 합리적인 추측을 한 것을 볼 수 있었다.

  - `Ablation Studies`로써, KG reasoning component를 없애보거나 pre-training task에 대해 실험을 진행했다. 결과, KG reasoning 과정이 없으면 KG를 쓴 모델보다는 안좋은 성능을 보였으며 각 pre-training task을 사용했을 때 성능이 증가하는 것으로 보아 각각의 task가 모두 성능 향상에 도움이 된다는 것을 알 수 있다.

<br></br>

- Conclusion

  - symbolic KG reasoning과 LM이 함께 동작하는 `OREOLM`을 제안함

  - OREOLM이 Open-domain QA에서 상당한 성능 향상을 가져옴을 실험을 통해 보였음. (encoder-based, encoder-decoder LM 모두에서)

  - 추가적으로, OREOLM은 reasoning path를 제공하여 model prediction을 해석할 수 있도록 도와줄 수 있음.
