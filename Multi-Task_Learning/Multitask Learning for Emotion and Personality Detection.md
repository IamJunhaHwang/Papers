## :page_facing_up: Multitask Learning for Emotion and Personality Detection

### Info

  * journal: IEEE TRANSACTION ON AFFECTIVE COMPUTING, VOL. 1, NO. 1, JANUARY 2021
    
  * author: Yang Li, Amirmohammad Kazameini, Yash Mehta, Erik Cambria

  * url: https://arxiv.org/pdf/2101.02346.pdf

### 1. Abstract & Introduction

많은 연구자들이 SNS의 디지털 흔적들이 성격 특성을 예측하는 것에 사용될 수 있고 성격 특성과 감정(emotions)간에 상관관계가 있음을 증명했다. 이를 근거로, 동시에 이 둘을 예측하는 `SoGMTL`이라는 CNN 기반의 MTL framework를 제안한다. 또, 이 두 task 사이의 다른 정보 공유 메커니즘에 대해 논의할 것이다. 마지막으로, 모델 최적화를 위해 MAML과 비슷한 방법을 제안한다.

우리 모델은 유명한 personality & emotion dataset들에서 SOTA를 달성했다.

- MTL main task: personality trait detection

  - auxiliary task: emotion prediction

<br></br>

### 2. Model Description

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/218925246-00bc6dfd-69e0-4cff-acce-2756608a6fbf.png" width="40%"></img></div>

위 그림에서 아래 부분이 공유하는 부분이다. 그런데 여기에서 둘은 다른 task이어서 최적화하기가 어려우므로 task 간의 정보를 교환하기 위해 information flow pipeline을 디자인했다.

### 2.1 Preliminary

- sentence $X$ = [$x_1, x_2, ... , x_n$] 으로 표현 됨.

- Embedding layer을 거치면 각 단어는 $x_i \in R^d$ 인 벡터가 됨. ( `d = embed dim` )

- CNN은 `convolutional layer`, `pooling layer`, `dense layer`로 구성됨.

  - convolution operation: $c_i = f(W_k \cdot X_{i:i+h-1} + b)$

    - filter: $W_k \in R^{hxd}$

  - 이 과정을 통해 feature map인 $c = [c_1, c_2, ... , c_{n-h+1}]$ 을 얻을 수 있고 max-pooling layer를 거쳐 $\hat c = max c$ 를 얻을 수 있음.

  - 마지막으로, dense layer를 통해 feature와 class들을 매핑.

- task 간의 정보 공유를 위해 gate mechanism을 적용함. [convolutional, max-pooling, dense layer 이후에 적용]

### 2.2 Information Sharing Gate

`Information Sharing`이 잘 일어나는지 평가하기 위해 gate 이후의 hidden vector간의 `cosine similarity`를 확인한다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/218925266-9b4f7266-04b5-4cf8-9797-67ac24141156.png" width="60%"></img></div>

위 그림과 같이 세 개의 gate mechanism을 이야기할 것임. 

- SiG (Sigmoid Gate)

  - 다른 네트워크의 정보를 간단하게 Sigmoid 함수를 통과시킨 후 다른 네트워크로 보내는 것. [식은 아래와 같다]

  - 이는 중요하거나 중요하지 않은 정보 상관 없이 모두를 넘겨버리므로 optimization comflict가 발생하기 쉽다.

    $m_i^{t1} = \sigma(c_i^{t1})$   
    $h_i^{t2} = c_i^{t2} + m_i^{t1}$ , `t1 = task1, t2 = task2`

- CAG (across Attention Gate)

  - 문맥 정보를 두 task 모두의 추가 특징으로 간주함.

  - 따라서, 두 task 간의 cross attention을 진행함. [Linear Layer + softmax]

  - 하지만 CAG 연산에는 시간이 많이 걸린다는 단점이 있음.

    $m_{i, j} = c_i^{t1} W_c c_j^{t2}$   
    $\alpha_{i, j} = \frac{exp m_{i, j}}{\sum_k exp m_{i, k}}$   
    $h_i^{t2} = c_i^{t2} + (\sum_j \alpha_{i, j}c_j^{t2})$

- SiLG (Simoid weighted Linear Gate)

  - SiG와 비슷한 단점이 있음.
 
    $m_i^{t1} = \sigma(c_i^{t1}) \cdot c_i^{t1}$   
    $h_i^{t2} = c_i^{t2} + m_i^{t1}$

- **SoG (Softmax weight Gate)**

  - 이를 해결하기 위해 Sigmoid를 Softmax로 바꿨음.

  - 결과적으로 다른 메커니즘에 비해 연산이 더 빨랐으며 효과적이었음.

    $m_i^{t1} = softmax(c_i^{t1}) \cdot c_i^{t1}$   
    $h_i^{t2} = c_i^{t2} + m_i^{t1}$

### 2.3 Meta Multitask Training

personality trait은 `multilabel prediction task`이며 emotion prediction task는 `multiclass task`이다. 따라서, 각각의 Loss function은 soft margin loss, cross entropy를 적용했으며 두 loss를 단순히 더한 것을 총 loss로 사용했다.

각 task마다 다른 dataset을 가지기 때문에 이에 맞는 adaptive training framework가 필요하다. [이 논문에서는 k-shot MAML 사용]

- k-shot MAML

  - 하나의 task에서 k이상의 batch를 선택해 training data pair를 만듦. [한 task에서는 하나의 batch + 다른 task의 여러 batch들의 조합들을 data pair로 사용]

  - MAML 방식처럼 parameter update.

  <img src="https://user-images.githubusercontent.com/46083287/218925485-8f006db5-d259-4d81-b921-b577036bca1e.png" width="40%"></img>

<br></br>

### 3. Experiments

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/218925373-94ec84d4-64e0-4e81-9839-3aaa62cd2487.png" width="70%"></img></div>

데이터를 8:2 비율로 `train: test` 로 나누었다. 또, 종합적인 평가를 위해 Precision, Recall, F1 지표를 사용했다.

baseline에서 CNN이 좋은 성능을 보였기에 우리 모델로도 CNN을 prototype model로 선정했다. `maml`이 붙은 것은 전 파트에 말했던 것이고 안 붙은 것은 `Adam`을 사용한 것이다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/218925397-05662f90-db14-4e93-af68-b6fdbc968c51.png" width="55%"></img></div>

- `SOGMTL + MAML`이 baseline의 best모델보다 좋은 성능을 보였음. [두 데이터셋에서의 모델 모두]

- `TEC` dataset이 `ISEAR` dataset 보다 좋은 성능을 보였는데 이는 `TEC`가 3배 더 크기가 크기 때문일 것임.

- MAML 방법은 거의 모든 case에서 성능을 높여주었으며 `SoG`는 적절한 gate mechanism이었음.

### 3-1. Ablation Study

`SoG`를 다른 level(conv, max-pool, dense)에 각각 하나씩 적용한 모델을 비교하고 정보 전달량을 평가하기 위해 코사인 유사도로 평가해봄.

위를 바탕으로 `SoG`가 conv layer에서 가장 큰 성능 향상을 일으키는 것을 알 수 있음.
