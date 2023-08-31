## :page_facing_up: BERT on a Data Diet: Finding Important Examples by Gradient-Based Pruning


### Info

* `conf`: NeurIPS 2022 Second Workshop on Efficient Natural Language and Speech Processing
* `author`: Mohsen Fayyaz, Ehsan Aghazadeh, Ali Modarressi
* `url`: https://arxiv.org/pdf/2211.05610.pdf


### 1\. Abstract & Introduction

오늘날의 사전 학습 언어 모델은 큰 데이터 셋에 의존하지만, 이전 연구들은 데이터 셋의 모든 샘플들이 훈련 기간에서 똑같이 중요하지 않다는 것을 알아냈다. 실제로, test 성능을 유지하면서 훈련 데이터의 상당 부분을 잘라낼(proune) 수 있다.

vision benchmark에는 데이터 셋에서 중요한 샘플을 찾기 위한 2가지 gradient-based scoring metric (GraNd/EL2N)이 있는데, 이 논문에서는 이를 NLP에 적용한 첫 연구이다.

우리는 가장 높은 GraNd/EL2N score 를 가지는 샘플의 작은 부분들을 잘라내는 것으로 test accuracy를 유지할 수 있을 뿐만 아니라 능가할 수도 있음을 보인다.

- **contribution**

  - 데이터 셋에서 중요한 샘플을 찾기 위해 GraNd, EL2N을 NLP domain에 적용.

  - CV와 반대로, early score computation steps는 적절한 subset을 찾기에는 충분하지 않음을 보였다.

  - 가장 높은 GraNd/EL2N score 를 가지는 샘플의 작은 부분들을 잘라내는 것으로 전체 데이터 셋을 사용하는 것보다 더 좋은 성능을 내는 것을 확인했다.

<br></br>

### 2. Background

#### GraNd

$X = \{x_i, y_i\}^N_{i=1}$ 을 분류 task에서 K개의 class를 가지는 훈련 데이터 셋이라 하자. ($x_i$ : sequence of tokens) `Paul et al. (2021)`은 GraNd라 하는 loss gradient norm의 기댓값을 사용해 각 샘플의 중요도를 측정했다.

- $GraNd(x_i, y_i) = \mathbb{E}_w ||g(x_i, y_i)||_2$

  - vector `g`는 모델의 가중치에 따른 loss gradient

  - **샘플이 네트워크 가중치(w)에 미치도록 예상되는 영향은 곧 샘플의 중요도를 나타낸다는 가정**

  - 훈련 데이터를 이에 따라 정렬한 후, top subset을 사용하고 나머지 데이터는 잘라낸다.

`Paul et al. (2021)`과 다르게 PLM을 사용했다. 따라서, PLM 위의 랜덤 초기화된 classifier에 대해서만 GraNd 점수를 계산한다.

#### EL2N

k번째 logit에 대해 $\psi^{k}(x_i) = \bigtriangledown_w f^{k}(x_i)$ 를 정의함으로써, loss gradient(g)는 다음과 같이 쓸 수 있다.

- $g(x_i, y_i) = \underset{k=1}{\overset{K}{\sum}} \bigtriangledown_{f^{k}} \mathcal{L}(f(x_i), y_i)^T \psi^{(k)}(x_i)$

  - $\mathcal{L}(f(x_i), y_i)$ 는 Cross-Entropy Loss이기 때문에, 아래와 같이 쓸 수 있다.
  
  - $\bigtriangledown_{f^{k}} \mathcal{L}(f(x_i), y_i)^{\top} = p(x_i)^{(k)} - y_i^{(k)}$ , where $p(x_i)$ : output probability vector of model

  - 훈련 데이터 셋에서 $psi$ 에 대해 직교성과 logit 사이의 균일한 크기를 가정하면, `GraNd`는 $||p(x_i) - y_i||_2$과 거의 비슷하게 된다. 즉, GraND의 추정치는 아래와 같이 정의되는 EL2N이다.

    - $EL2N(x_i, y_i) = \mathbb{E}||p(x_i) - y_i||_2$

<br></br>

### 3. Experiments

- `Paul et al. (2021)`의 연구와 같이 실험을 설정했다.

  - 모델은 높은 GraNd, EL2N 점수를 가지는 데이터의 subset으로 훈련됐다.

  - GraNd와 EL2N의 식에서 다중 가중치 초기화에 대한 기댓값을 기반으로 하므로, 점수를 5번의 독립적인 training run으로 평균내었다.

  - 랜덤 프루닝을 baseline으로 이용하고 프루닝이 없는 전체 데이터 세트 훈련의 정확도를 목표 성능으로 이용했다.

- 두 가지 크게 다른 점은 아래와 같다.

  - 우리는 사전 학습 모델인 BERT를 사용했다.

  - fine-tuning은 few epoch를 필요로 하기 때문에, 세분화된 단계(fine-grained steps)를 통해 메트릭을 계산했다

- `Dataset` : MNLI - NLI, AG's News - topic classification

<br></br>

### 4. Results

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/bdc52891-d980-4bfd-9733-2ca06b3d4a8c" width="80%"></img></div>

위 그림은 top 70%, 50%의 샘플들에 대한 fine-tuning 성능을 나타낸 것이다. early score computation step은 랜덤 샘플링보다 더 낮은 성능을 얻을 정도였다. 실제로, 아래와 같이 score computation step 전반에 걸쳐 EL2N이 선택한 상위 예제의 레이블 분포를 보았을 때, 처음에는 레이블이 불균형한 것을 볼 수 있었다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/2b4d9390-843b-46a2-b9fc-54128b77b8e2" width="25%"></img></div>

게다가, 두 데이터 셋은 서로 다른 양상을 보여주었다. `AG's News`는 500 step 이후에 괜찮은 정도의 score가 나왔지만 `MNLI`는 더 오래 걸렸다. 따라서, score computation step으로 1 epoch을 사용했다. (AG : 3,750, MNLI : 12,272)

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/0366cf65-f3cc-4034-862e-fe94a8e68f8c" width="80%"></img></div>

- Preserved fraction : 훈련 데이터 셋의 크기가 얼마나 성능에 영향을 주는지 실험. (전체 데이터 중 k%를 사용해 fine-tuning)

  - 위 그림은 EL2N/GraNd score를 정렬했을 때, 상위 몇 k% 사용하냐에 따른 성능 차이를 보여주는 그림이다.

  - subset이 작을 수록, 전체 데이터셋보다 성능이 좋지 못했다.

  - `AG's News`와 다르게 `MNLI`에서는 60%까지는 random subset보다 성능이 좋지 못했지만, 그 이상에서는 더 성능이 좋아졌다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/350eccab-5050-4e48-a04d-fc285a4cdf98" width="80%"></img></div>


- Noise examples : 노이즈가 많은 예제에 해당할 수 있는 EL2N 및 GraNd에서 가장 높은 점수를 얻은 예제를 제거하여 최종 성능을 향상시킬 수 있는지 확인하기 위해 실험을 수행

  - 위 그림은 데이터의 top 70% 데이터로 훈련한 후에 top k%를 제거한 성능이다.

  - 두 데이터 셋에서 모두 처음에는 성능 향상을 보였고 특히, MNLI 데이터 셋에서는 전체 데이터를 사용하는 것보다 더 높은 성능을 달성할 수 있었다.

  - 반면에 `AG's News`에서는 성능 향상을 내지 못했다.

<br></br>

### 5. Conclusions

- CV에서 사용된 pruning metric인 `EL2N`과 `GraNd`를 NLP에 적용했다.

- 두 score 모두 적어도 1번 이상의 fine-tuning epoch을 거쳐야되는 것을 알아냈다.

- MNLI 데이터 셋에서는 노이즈와 관련될 수 있는 가장 높은 점수 예제를 프루닝하여 전체 데이터 셋에서 훈련할 때보다 더 높은 성능을 달성 할 수 있었다.
