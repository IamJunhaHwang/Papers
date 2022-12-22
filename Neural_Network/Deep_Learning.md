## :page_facing_up: Deep Learning

### *Info*

* authors: Yann Lecun, Yoshua Bengio, Geoffrey Hinton
* journal: Nature 2015
* link: https://www.nature.com/articles/nature14539

<br></br>

### *Abstrct*

* 딥러닝은 다중 차원으로 구성된 데이터의 표현을 배우기 위해 여러 개의 처리 층으로 구성된 계산 모델을 가능하게 함. (여러 태스크에서의 비약적인 발전을 가져옴)

- 딥러닝은 `backpropagation` 알고리즘을 이용해 큰 데이터 셋에서 복잡한 구조를 찾아낸다.

  - `backpropagation`: 이전 층의 표현에서 다음 층으로의 표현을 계산하는데 쓰이는 내부 파라미터들을 어떻게 업데이트 시킬 것인지를 알려주는 것.

- `Deep convolutional nets(CNN)`는 이미지, 비디오, 담화, 오디오 데이터를 처리하는데 돌파구가 되고 있으며 `Recurrent nets(RNN)`는 텍스트와 담화같은 `sequential data`에 빛을 비추고 있다.

<br></br>

### *Introduction*

- `Machine-Learning`기술은 현대 사회의 많은 측면에 영향을 주고 있음. (ex. 추천시스템, 이미지 인식, 트랜스크립트, 등)

- 전통적인 `Machine-Learning`은 자연 데이터를 `raw form`으로 처리하는 능력에 한계가 있었음.

  - 이미지의 `pixel`을 내부 표현이나 feature vector로 변환해주는 과정 필요.

  - 신중한 엔지니어링 + 상당한 도메인 지식 필요.

- `Representation Learning`은 `raw data`를 받아, 자동으로 탐지나 분류에 필요한 표현을 찾아내는 것.

  - `딥러닝`은 여러 단계의 표현층을 가지는 `representation-learning`임.

  - 한 단계에서 다음으로 진행될 수록 조금씩 추상화된 표현으로 변환하는 비선형 모듈로 구성. (충분하게 많은 단계로 구성한다면 복잡한 함수도 배울 수 있음.)

- `딥러닝`의 핵심은 `feature`를 사람이 디자인하는 것이 아닌 데이터로부터 일반적인 학습 절차로 배워간다는 것.

  - 수작업이 필요한 일이 줄어들기 때문에 계산량과 데이터량 측면에서 큰 도움이 될 것이다.

<br></br>

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209088259-80a710ba-d408-4c25-be2a-611dbe0d61fd.png" width="70%"></img></div>

- 이미지에서의 예 (위의 그림은 `Alexnet`으로, 논문에 그림이 등장하지는 않음)

  - 이미지는 `pixel` 값의 배열 형태로 입력을 받는다.

  - 첫 번째 층에서 `feature`를 학습한다.

    - 이미지의 특정 방향과 위치에 있는 `edge`의 존재 여부를 나타내는 역할.

  - 두 번째 층에서 무늬(motif)를 탐지한다. (edge위치의 작은 변형과 상관 없이 특정 edge의 구조를 발견함으로써)

  - 세 번째 층에서 찾은 motif들을 익숙한 물체의 부분과 일치하는 큰 조합으로 모은다.

  - 다음 층에서 이 부분의 조합으로 물체를 찾을 수 있을 것.

<br></br>

### *Supervised Learning*

- 딥러닝이든 아니든, 가장 흔한 머신러닝은 `supervised learning(지도 학습)`.

- 그 과정은 아래와 같다. (집, 차, 사람, 동물을 분류하는 문제라 하자.)

  - 먼저, <집, 차, 사람, 동물>을 포함하는 사진 데이터를 모아야 한다. (각자의 카테고리로 라벨링 되어있어야 함)

  - 훈련 시간동안, 컴퓨터는 이미지를 보고 점수를 벡터형태로 출력해낼 것이다. (ex. 집: 80 , 차: 10, 사람: 5, 동물: 5)

  - 원하는 카테고리가 모든 카테고리 중 가장 높은 점수가 되길 바라고 이는 `훈련(학습)` 전에는 불가능한 일이다. (`집`을 나타낸 사진을 `집`으로 높은 확률로 분류하게 만드는 것)

    - 훈련에서는 `error`를 측정하기 위해 **목적 함수**를 계산하며 이를 통해 컴퓨터가 내부의 조절 가능한 `파라미터`를 `error`를 줄이기 위해 바꾼다.

    - 전형적인 딥러닝 모델은 이러한 가중치와 훈련에 이용할 라벨링된 예시들을 수억 개를 가지고 있을 것이다.

- 가중치를 적절하게 조정하기 위해, 학습 알고리즘은 기울기 벡터를 계산함. (목적 함수의 기울기)

  - 기울기 벡터는 각각의 가중치를 조금 변경시키는 것으로 `error`가 얼마나 증감하는지를 가리킨다.

  - 가중치를 기울기 벡터의 반대 방향으로 조절하면 `error`를 줄일 수 있을 것이다.

- 목적함수는 가중치의 차원 공간에서의 언덕(산)과 같이 보일 수 있음.

  - 음의 기울기 벡터는 여기에서의 가장 가파른 하강 방향을 가리킨다. (출력 오차가 평균적으로 낮은 값인 최소점에 근접)

  - 실무자들은 `Stochastic Gradient Descent`를 사용함.

    - 이는 몇 개의 입력 벡터만 보고 출력과 오차를 계산하고 기울기의 평균을 계산해 이에 따른 가중치를 갱신하는 것이다.

    - 이는 목적 함수의 평균이 감소하는 것을 멈출 때 까지 지속된다.

    - `stochastic(확률적)`이라 불리는 이유는 입력에서 몇 개의 작은 `set`을 뽑는 것이 노이즈를 주기 때문이다. (모든 데이터를 보는 것이 아니므로)

    - 이런 방법은 빠르고 좋은 가중치 집합을 찾아낸다.

- 훈련을 끝마치면 `test set`인 다른 예시들로 모델의 성능을 측정해 일반화 능력을 확인한다. (훈련 집합이 아닌 새로운 입력에 대해 좋은 출력을 내는가)

<br></br>

### *Why deep Learning?*

- 많은 현재의 머신러닝을 적용한 응용들은 직접 만든 feature에 `Linear classifier`를 사용해 feature vector의 가중치 합을 계산하고 있음.

  - 가중합이 일정치를 넘어서면 특정 카테고리로 분류되는 식

  - 1960년도에 초평면으로 분리된 반공간으로 나뉘는 매우 작은 영역으로만 입력을 분할할 수 있음을 알았다.

  - 하지만 입력에서 상관없는 변형이 크게 영향을 주게되는 경우가 있어 이미지나 음성 인식에서 문제가 되었다. (이미지에서 배경만 다른 경우, 음성에서 악센트가 다른 경우)

- 따라서, 위와 같은 `얕은` 분류기의 경우 좋은 `feature extractor`가 필요했음. (많은 도메인 전문 지식이 필요한)

  - 이를 해결하는 하나의 방법은 `kernel method`에서 처럼 일반적인 비선형 특징을 사용하는 것인데, 이는 훈련 세트에서만 잘 동작할 뿐 다른 예시에서는 성능이 좋지 않다. (Gaussian kernel)

  - 전통적인 방법은 손수 좋은 `feature extractor`를 설계하는 것.

- 하지만 다목적 학습 절차를 사용한다면 자동으로 좋은 특징들이 학습될 수 있다. (직접 `feature extractor` 설계를 하지 않고)

  - **이것이 딥러닝의 큰 장점.**

- 딥러닝은 간단한 모듈이 다층으로 구성되는 구조를 가지고 있으며 각 층은 학습되고 입력을 비선형 출력으로 변환하는 계산을 한다.

  - 예를 들어, 5~20층 깊이의 비선형 층이 있다면 극도로 복잡한 함수를 구현할 수 있을 것. (세상의 거의 모든 것을 학습 가능)

  - `Layer`를 지날수록 핵심적인 특징에 민감하게 반응하며 상관없는 특징은 무시할 수 있게

<br></br>

### *Backpropagation to train multilayer architectures*

- 모듈(Layer)이 입력과 내부 가중치에 대해 `smooth(연속)`한 함수를 가진다면, 역전파(Backpropagation)를 사용해 기울기를 구할 수 있음.

  - 가중치에 따른 목적함수의 기울기를 계산하기 위한 `역전파`는 연쇄 법칙(chain rule)을 적용한 것에 불과하다.

  - 핵심은 Layer의 입력에 따른 목적 함수가 **Layer의 output(or 다음 Layer의 input)에 따른 기울기로부터 역순으로 계산될 수 있다**는 것.

    <img src="https://user-images.githubusercontent.com/46083287/209088364-8fa937cd-09f8-46fd-b1ae-e5e3b23bda6a.png" width="20%"></img>

    - x에 대한 z의 변화량 = `y에 대한 z의 변화량` * `x에 대한 y의 변화량`

    - 예시로 `z = t^2`, `t = x + y` 하자.
    - dz/dx = `dz/dt` * `dt/dx` = `2(x + y)`
    - `dz/dt = 2t`, `dt/dx = 1`

- 간단한 경사하강법은 `local minima`에 빠질 위험이 있다고 하지만 충분히 큰 모델에서는 문제가 되지 않는다.

  -  몇 개의 아래 방향만 있는 말안장점(saddle point)이 여러 개 존재하지만, 거의 대부분은 목적함수의 값과 비슷한 것으로 나타났다.

  - 따라서, 문제가 되지 않는다.
   
<br></br> 

### *Convolutional Neural Networks*

- `ConvNets`는 다차원 배열의 형태로 입력을 받아 데이터를 처리하게 설계되었음.

  - 4가지 핵심 아이디어: `local connection`, `shared weight`, `pooling`, `use of many layers`
    - `local connection`: 각 영역을 나누어 따로 가중치를 학습 시킴 (R,G,B에 각각 필터를 만듦. 총 3개의 필터가 됨) [참고 자료](https://prateekvjoshi.com/2016/04/12/understanding-locally-connected-layers-in-convolutional-neural-networks/)

- Convnets 구조

  <img src="https://user-images.githubusercontent.com/46083287/209088421-0b906219-e04d-4e36-9d84-00f56489ef1d.png" width="70%"></img>

  - 첫 단계는 `Convolution Layer & Pooling Layers`이다.
  
    - `Convolution Layer`: `feature map`으로 구성되며 그 안의 각 유닛들은 `filter bank`라 불리는 가중치 집합을 통해 전 Layer의 `feature map`의 지역적인 부분과 연결된다. (전 Layer의 feature에 대한 지역합을 탐지하는 역할)

    - 이후, 구해진 지역 가중합은 `ReLU`를 통과한다.

    - `feature map`안의 각 유닛은 같은 가중치를 공유한다. (`feature map`이 다르면 다른 가중치 사용) 

      - 이미지나 값의 부분 집합의 경우 큰 상관관계를 띠기 때문에 쉽게 탐지되는 고유한 local motif 형성함.

      - 이미지나 다른 신호의 지역적 통계는 위치와 상관이 없음. (한 번 등장하면 어느 곳에서도 등장 가능)


    - `Pooling Layer`: 의미적으로 비슷한 특징들은 하나로 합치는 역할.

      - 전형적인 pooling은 `feature map`에서 유닛의 지역 부분의 최댓값을 계산하는 것.

<br></br>

### *Distributed representations and language processing*

- 맥락에서 각 단어는 one-hot vector로 표현됨.

- 첫 번째 층에서 각 단어는 단어 벡터로 바뀌며, 다른 층에서 이 벡터를 다음 단어를 예측하기 위한 출력 벡터(해당 단어의)로 바꾸기 위해 학습 한다.

  - 확률적으로 다음 단어에 어떤 것이 나올지를 사용.

  - 이 모델은 단어의 구별되는 특징을 해석할 수 있는 단어 벡터를 배운다.

- 이러한 표현 방법의 이슈로 `logic기반` 과 `neural-Nets 기반` 사이에 논쟁이 일어난다.

  - `logic기반`은 하나의 심볼이 유일성을 갖는 것으로 보고 신중히 만든 `규칙`에 의해 다루어진다. (분산 표현이 아님)
    - 출현빈도에 기반하므로. (N-gram)

  - `neural-Nets 기반`은 벡터, 가중치, 비선형 스칼라를 사용한다.
    - 실수 값을 가진 벡터로 구성되므로 의미적으로 비슷한 단어를 알 수 있게 됨.

<br></br>

### *Recurrent neural networks*

- RNN은 한 번에 입력 시퀀스의 하나의 원소만 보며 진행이 된다.

  - 그리고 시퀀스의 전 원소들의 정보를 포함하는 `state vector`라는 hidden unit을 유지시킨다.

  - 하지만 여기에는 학습 시의 문제가 생기는데, 역전파된 기울기가 시간이 지남에 따라 폭발하거나 사라진다는 것.
    - LSTM 등으로 해결

- LSTM은 전통적인 RNN보다 성능이 좋다.

  - 메모리 셀이라는 특별한 유닛이 사용됨.

  - 여전히 기계 번역의 인코더 디코더 부분에서 자주 사용됨.

<br></br>

### *The future of deep learning*

- 해당 리뷰에서 다루진 않았지만 `Unsupervised Learning`이 중요하게 될 것.

  - 사람과 동물은 이와 같이 학습하므로

- `NLP`에서 deep learning이 점점 더 많이 쓰일 것.

  - 문장이나 전체 문서를 더 잘 이해시키게 될 것.
