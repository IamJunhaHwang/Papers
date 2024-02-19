## :page_facing_up: Self-Attention with Relative Position Representations

### Info

* `publication`: NAACL 2018 - short
* `author`: Peter Shaw, Jakob Uszkoreit, Ashish Vaswani

* `url`: https://aclanthology.org/N18-2074.pdf

### 1\. Abstract & Introduction

RNN이나 CNN은 모델 구조 자체를 통해 position 정보를 계산한다. 하지만 recurrence나 convolution을 하지 않는 Transformer의 경우 이러한 position 정보의 표현에 대한 고려는 대단히 중요하다.

본 논문에서는 Transformer의 self-attention 메커니즘에서 relative position representation을 포함시키는 효과적인 방법을 제시한다. 우리는 absolute position encoding 전부를 교체했음에도 오히려 번역 task에서 좋은 성능을 보였음을 증명한다. 

<br></br>

### 2. Background - Self-Attention

input sequence $x = (x_1, ..., x_n)$, where $x_i \in \mathbb{R}^{d_x}$   
output sequence $x = (z_1, ..., z_n)$, where $z_i \in \mathbb{R}^{d_z}$

- A. 쿼리-키 간의 scaled dot product
  - $e_{ij} = \frac{(x_iW^Q)(x_jW^K)^T}{\sqrt{d_z}}$

- B. 위에서 계산한 값들에 softmax 적용
  - $\alpha_{ij} = \frac{exp e_{ij}}{\sum^n_{k=1}exp e_{ik}}$

- B에서 구한 가중치를 벨류 값에 적용
  - $z_i = \overset{n}{\underset{j=1}{\sum}} \alpha_{ij}(x_jW^V)$

### 3. Proposed Architecture
#### 3-1. Relation-aware Self-Attention

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/8e27bfbe-5696-48ab-9087-d4fe50bd9cb0" width="35%"></img></div>

- input token간의 관계를 directed fully-connected graph로 모델링 하였으며, input token $x_i$, $x_j$ 간의 edge(관계)는 다음과 같은 벡터로 표현한다. [input token간의 상대 위치 차이에 대한 정보를 포착]

  - $a^V_{ij}, a^K_{ij} \in \mathbb{R}^{d_a}$, where $d_a = d_z$

  - 이 벡터들은 attention head간에 공유됨.

- 위 벡터들을 다음과 같이 self-attention 식에 넣어줌.

  - $e_{ij} = \frac{(x_iW^Q)(x_jW^K + a_{ij}^K)^T}{\sqrt{d_z}}$

  - $z_i = \overset{n}{\underset{j=1}{\sum}} \alpha_{ij}(x_jW^V + a_{ij}^V)$

#### 3-2. Relative Position Representations

일정 거리를 넘어가면 정확한 relative position 정보는 유용하지 않다고 가정하고 Maximum relative position은 절대값 `k`까지로 잘라낸다. 이러한 clipping은 training에서 보지 못한 시퀀스길이에 대해서도 모델링할 수 있게 일반화한다.   
따라서, `2k+1`개의 unique edge labels가 있게된다.

- $a_{ij}^K = w^K_{clip(j-i, k)}$

- $a_{ij}^V = w^V_{clip(j-i, k)}$
- $clip(x, k) = max(-k, min(k, x))$

그리고 relative position representations $w^K = (w^K_{-k}, ..., w^K_{k})$ 와 $w^V = (w^V_{-k}, ..., w^V_{k})$ 를 학습한다 ($w^K_i, w^V_i \in \mathbb{R}^{d_a}$).

<br></br>

### 4. Experiments

- evaluation task: WMT 2014 기계 번역 task(English-German, English-French)

  - 각각 약 4.5M, 36M 문장쌍

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/c40bf2bb-0a50-472e-8b35-7792c08f652e" width="65%"></img></div>

`Englih-to-German`에서는 0.3(base), 1.3(big)만큼 BLEU성능 개선이 있었으며, `English-to-French`에서는 0.5(base), 0.3(big)만큼 BLEU성능 개선이 있었다.

- **relative position representation에 sinusoidal position encoding을 추가하는 것은 별 도움이 되지는 않았다**

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/c29cd1f6-ff79-48a1-a088-c875bb182147" width="75%"></img></div>

clipping distance `k`에 대해 실험을 해본 결과, $k \geq 2$ 에서는 BLEU 점수가 크게 차이나지 않았다.

그리고, $a_{ij}^V$ 와 $a_{ij}^K$ 에 대한 ablation 실험을 해본 결과, 이러한 relative position representation은 성능에 영향을 미치는 것을 증명하였다. 또한, $a_{ij}^V$ 없이 $a_{ij}^K$ 만으로도 충분한 성능을 보여줬는데 이것이 다른 task에도 적용이 되는지는 추가 연구가 필요하다.
