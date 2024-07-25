## :page_facing_up: LLM vs Small Model? Large Language Model Based Text Augmentation Enhanced Personality Detection Model

### Info

* `publication`: AAAI 2024

* `author`: Linmei Hu et al.
* `url`: https://arxiv.org/abs/2403.07581


### 1\. Abstract & Introduction

Personality detection task에서의 어려운 점 중 하나는 labeling data가 부족하다는 것이다. 대부분의 연구들은 제한된 labeling data로 PLM을 fine-tuning하는 방식을 사용하며, labeling 또한 one-hot labeling으로 되어 있는 문제가 있다.   

본 논문에서는 LLM의 지식 증류를 이용한 augmentation으로 personality detection small model의 성능을 높이는 방법을 제안한다. 자세하게는 1) LLM에게 의미적, 감정적, 언어적 측면에서 게시물 분석(증강)을 생성하게 하고, contrastive leraning을 이용해 이를 하나의 임베딩 공간에 투영시켜 post representation에서 좀 더 좋은 정보를 포착할 수 있게 한다. 2) 또한, LLM에게 personality label에 대한 설명을 만들게 하여 label에 대한 정보를 풍부하게 한다.

벤치마크 데이터 셋에서의 실험 결과를 통해 우리의 모델이 SOTA보다 좋음을 증명했다. 

- Contributions

  - personality detection model의 성능을 높이기 위해 LLM 기반의 text augmentation을 제안함.

  - LLM으로 semantic, sentiment, linguistic 3가지 측면으로부터 contrastive representation learning을 위한 augmentation을 생성하게 함.

  - 벤치마크 데이터에서의 실험 결과를 통해 제안한 모델이 SOTA를 능가함을 보였음.


<br></br>

### 2. Method

Personality detection은 multi-document multi-label classifiacation task로 정의될 수 있다.

한 명의 유저에 대한 N개의 post $P = \{p_1, p_2, p_n \}$ 가 주어지며, $p_i = [w_{i1}, w_{i2}, ..., w_{im}]$ 는 m개의 토큰으로 이루어진 i번째 post이다. 해당 task는 `P`를 기반으로 한 명의 유저가 가지는 t 차원의 personality traits label인 $Y = \{y_1, y_2, ..., y_T \}$ 를 예측하는 것을 목표로 한다(MBTI의 경우 T=4 이며, label은 one-hot vector임).

본 논문에서는 post augmentation을 위해 LLM에게 의미적, 감성적, 언어적 측면으로의 요약을 생성하게 한다. augmented data는 $X = \{P, P^s, P^e, P^l \}$ 로 나타낸다(P를 제외하고, 차례대로 semantic, sentiment, linguistic을 의미).   
또한, label 부분에서도 LLM에게 의미적, 감성적, 언어적 측면으로의 설명을 생성하게 하여 label의 정보를 풍부하게 한다. t 차원에 대한 label 표현은 다음과 같다 : $\hat{y_t} = \{L_{y_t,0}, L_{y_t,1} \}$, 각 $L_{y_t, j} = \{l^s_{y_t, j},l^e_{y_t, j}, l^l_{y_t, j} \}$


#### 2-1. Generating Knowledgeable Post Augmentations from LLMs

LLM에게 semantic, sentiment, linguistic의 3가지 측면으로 post를 분석하도록 지시하며, 이 결과를 이용해 data augmentation을 진행한다.

이를 위한 prompt는 다음과 같으며, 이에 대한 생성 예시는 밑의 그림과 같다.

`
Your task is to analyze the characteristics of a user based on a piece of text published by the user on the Internet. You are required to analyze it from the perspectives of semantic, sentiments, and linguistics. Note that if the text is incomplete and ends with an ellipsis, it may have been truncated due to external reasons, in which case you should ignore it. post:. . .
`

<div align="center"><img src=https://github.com/user-attachments/assets/46b8e201-80b9-46b7-8906-266c2ea817d2 width="80%"></img></div>

#### 2-2. Contrastive Post Encoder

i번째 post $p_i$ 와 이에 대한 3가지 analysis $P_{pos} = {p^s_i, p^e_i, p^l_i}$ 를 positive pair로 사용하며, 임베딩으로 아래와 같이 마지막 `[CLS]` 토큰에 대한 hidden state를 사용한다.

- `[CLS]` hidden state : 

  - $h_i = BERT(p_i) \in \mathbb{R}^{1 \times d}$

  - $h^+_i = BERT(p^+_i), where \ p^+_i \in \{p^s_i, p^e_i, p^l_i\}$

- original post text와 analysis text 간의 임베딩 분포가 다르므로, 추가적인 MLP를 적용하는 것을 projection head로 사용 :

  - $z_i = \delta(Wh_i + b)$

  - $z^+_i = \delta(Wh^+_i + b)$

    - $\delta$ : Tanh function

- 미니 배치에 대한 post-wise info-NCE loss 적용

  - $\mathcal{L}_{cl} = \frac{1}{M} \sum^M_{i=1} \mathcal{l}^{cl}_i$

  - $\mathcal{l}^{cl}_i = -log \underset{(z_i, z^+_i) \in P_{pos}}{\sum} \frac{e^{sim(z_i, z^+_i)/\tau}}{\sum^M_{j=1} \sum_{(z_j, z^+_j) \in P_{pos}} e^{sim(z_i, z^+_j)/\tau}}$

    - $\tau$ : temperature hyperparameter

    - `sim()` : cosine similarity

#### 2-3. LLM Enriching Label Information

이전 연구들은 personality detection을 위한 label로 one-hot label을 사용했지만, 본 논문에서는 post에서와 같이 LLM을 이용해 semantic, sentiment, linguistic 측면에서의 설명을 만들게 하는 것으로 augmentation을 진행했다.   

- personality label $L_{y_t, j} = \{l^s_{y_t, j}, l^e_{y_t, j}, l^l_{y_t, j} \}$ 에 대한 label representation을 다음과 같다 :

  - $v_{y_t, j} = mean([BERT(l_{y_t}, j)]), where \ l_{y_t, j} \in L_{y_t, j}$

  - 마지막 레이어의 `[CLS]` 토큰 임베딩을 사용 
  
- label representation에 기반해, over-confident issue를 극복하고자 one-hot label을 soft label로 바꾸어준다.

  - 먼저, dimension-wise label similarity distribution을 계산한다 : 
    
    - $y^s_t = softmax(sim(u, V_{y_t}))$

      - $u = mean([h_1, h_2, ..., h_N])$ : user representation obtained by average pooling

  - 다음 original one-hot vector와 조정 하이퍼파라미터 값을 조합한다 :

    - $y^c_t = softmax(\alpha y_t + y^s_t)$

  - 이렇게 되면, user가 상대적으로 중립적인 label을 가질 때 모델의 일반화 능력이 향상된다.

- 마지막으로, T softmax-normalized linear transformations을 적용해 personality traits를 예측하고, KL-divergence를 loss로 사용한다.

  - $\hat{y}^t = softmax(uW^t_u + b^t_u)$

    - 이 논문에서 쓴 label의 경우 MBTI이므로 T=4이고, 각 dimension(ex. E/I, F/T, etc)마다 위 계산을 하는 것으로 보임

  - $\mathcal{L}_{det} = -\frac{1}{B}\sum^B_{i=1}l^{det}_i$

    - $l^{det}_i = \sum^T_{t=1} KL-divergence (y^c, \hat{y}^t)$
    - $= \sum^T_{t=1}\sum_{j=0,1}y^c_j log(\frac{y^c_j}{\hat{y}^t_j})$

      - `B` : batch 내의 sample 수

- Overall training objective : $\mathcal{L} = \mathcal{L}_{det} + \lambda \mathcal{L}_{cl}$

  - $\lambda$ : 두 loss들 간의 균형을 위한 trade-off parameter

<br></br>

### 3. Experiments

#### Experimental setup

<div align="center"><img src=https://github.com/user-attachments/assets/db6c1dcd-a4ee-430f-ac1c-5a41aae07afa width="40%"></img></div>

- Datasets : Kaggle & Pandora

- Baselines : SVM, XGBoost, BiLSTM, BERT, AttRCNN, SN+Attn, Transformer-MD, TrigNet, D-DGCN, ChatGPT

- Augmentation에는 `gpt-3.5-turbo-0301`를 사용하였다.

#### Overall Results

<div align="center"><img src=https://github.com/user-attachments/assets/e009f6ac-5f75-484a-9ce2-30dd84c26a8b width="60%"></img></div>

- 본 논문에서 제안한 `TAE` 가 모든 baseline보다 성능이 좋았으며, 그 이유는 아래와 같다.

  1. LLM으로부터 post augmentation을 만드는 것으로 contrastive post encoder가 personality detection에 좋은 정보를 추출할 수 있게 하였음.

  2. label에 대해 추가적인 explanation을 생성한 것이 detection task에 도움이 되었음.

- 제안한 모델이 best performance를 보였으므로, 데이터가 희소한 task에서 LLM을 이용하는 것이 효과적임을 증명함.

#### Ablation Study

<div align="center"><img src=https://github.com/user-attachments/assets/c9123640-703f-489d-b876-c4834b72d8f8 width="40%"></img></div>

- post augmentation에서의 각 측면에 대한 analysis를 제외해보았을 때, linguistic이 가장 중요했으며, semantic 정보는 덜 영향을 주는 것을 확인했다(이는 이전 연구에서도 보였었음).

- label information enrichment를 제외했을 때, 성능이 약간 안좋아졌으며, 이를 통해 post augmentation과 label information enrichment 작업에서의 이득이 있음을 증명했다.

- 또한, LLM을 이용한 augmentation을 contrastive learning이 아닌 추가 input으로 직접 이용하는 식의 ablation을 진행하였다.

  - `TAE(concat)` : input과 analyses를 각각 encode한 후 concatenate하여 사용

  - `TAE(WS)` : input과 analyses를 각각 encode한 후 weighted sum을 이용한 pooling

  - 위 두 모델이 TAE의 모든 요소들을 제외한 `TAE(w/o all)` 보다는 좋았지만, TAE보다는 낮은 성능을 보였다. --> **contrastive learning이 효과적임**
