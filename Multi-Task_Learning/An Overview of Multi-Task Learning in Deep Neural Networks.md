## :page_with_curl: An Overview of Multi-Task Learning in Deep Neural Networks

### Info

  * not published
  * author: Sebastian Ruder
  * url: https://arxiv.org/abs/1706.05098

### 1. Introduction & Motivation

- `Multi-Task Learning(MTL)`은 원래의 task에 대한 성능을 일반화하기 위해 관련된 task들 사이의 표현(ex. weight)을 공유하는 것

  - main task에 대한 성능을 일반화하기 위해 보조(auxiliary) task를 이용하자.

  - 사람이 학습을 할 때, 전에 배운 사전 지식(하려는 것과 관련된)을 이용한다는 점을 이용한 아이디어

- 머신러닝 관점에서 `inductive bias`를 도입하는 역할

### 2. Two MTL methods for Deep Learning

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/214227261-32ff4409-d2bc-4432-8c41-d5b7d977a6a8.png" width="90%"></img></div>

- Hard parameter sharing: hidden layer를 공유하고 그 위에 task-specific output layer를 각각 쌓은 것

  - 지금도 많이 쓰임.

  - overfitting 방지 효과 [다양한 task 모두를 아우르는 표현 vector를 찾으려 하기 때문]

  - task의 난이도를 높이는 효과(?) [내 생각]

- soft parameter sharing: 각 task가 각자의 model과 각자의 parameter를 가지고 있으며 parameter를 정규화 시키는 방법

  - 모델의 파라미터 사이의 거리를 정규화함.

<br></br>

### 3. Why does MTL work?

- Implicit data augmentation

  - task마다 각기 다른 noise pattern을 가지고 있으므로, 두 가지 task를 동시에 학습한다면 좀 더 일반적인 표현을 얻을 수 있을 것

  - overfit 방지

- Attention focusing

  - task가 noisy하거나 데이터가 한정적이고 high-dimensional하다면 model이 관련 있는 feature를 찾기 어려움.

  - MTL을 이용한다면 model이 관련 있는 feature에 더 집중할 수 있음.

- Eavesdropping

  - feature G 가 task B를 학습하기는 쉽지만 task A는 어려울 수 있다.

  - 이 때, MTL을 이용하면 task B를 이용해 A에 대해 수월하게 학습하게 할 수 있음.

- Representation bias & Regularization

  - MTL을 task 전체에 대해 적절한 표현을 찾게 되므로 좀 더 일반화시키는 효과가 있음

  - 또, `inductive bias`를 도입하는 것으로 정규화의 역할을 해줌.

  - Overfit 방지

### 4. MTL in non-nueral models

두 가지 main idea: norm 정규화를 통한 task 간의 sparsity 강제, task 간의 관계 modelling

- Block-sparse regularization

  - 많은 방법들이 parameter에 관해 sparsity 가정을 만듦.

  - ex. lasso, ridge, norm regularization

- Learning task relationships

  - MTL에서 관련된 task가 아니라면 negative transfer 효과를 가질 수 있음.

  - 따라서, task들이 서로 관련되어 있는지를 찾아야 함.

  - ex. clustering constraint 적용, SVM, cluster regularization 등

<br></br>

### 5. Recent work on MTL for Deep Learning

- Deep Relationship Networks

  <img src="https://user-images.githubusercontent.com/46083287/214227350-aa81a0b9-34eb-4c0f-8d08-357d043ea2cc.png" width="80%"></img>

  - CNN layer를 공유하면서 task-specific FCNN을 놓은 구조

  - shared and task-specific layer 구조를 위해 matrix priors를 두었음. [모델과 task간의 관계 학습]

    - 베이지안 모델과 비슷

- Cross-stitch Networks

  - task-specific network가 전 layer output에 대한 linear combination을 배우는 것으로 다른 task의 지식을 확대 시킴.

  <img src="https://user-images.githubusercontent.com/46083287/214227384-d86140a3-ea4b-452e-93df-c48e2aa3c45e.png" width="60%"></img>

- Joint Many-Task Model

  - 미리 정의한 계층 구조를 이용한 MTL 모델

  <img src="https://user-images.githubusercontent.com/46083287/214227394-60849a24-a8ae-454b-9a5f-fe7e9605a17b.png" width="50%"></img>

- Weighting losses with uncertainty

  <img src="https://user-images.githubusercontent.com/46083287/214227417-edecd036-af72-4f80-b950-2327f49262b8.png" width="70%"></img>

  - 각 task의 불확실성은 고려하는 방법

  - Gaussian likelihood를 최대화해야하는 multi-task loss function으로 각 task의 관련된 weight를 조절

- Sluice Networks

  <img src="https://user-images.githubusercontent.com/46083287/214227435-116dc5ab-1f9a-4ed0-a391-483ce22f8af3.png" width="50%"></img>

  - 어떤 layer나 subspace들이 공유되어야하는지를 학습하며 input 시퀀스의 best representation을 학습

- what should i share in my model?

  - hard parameter sharing은 지금도 사용되는 방식임.

  - 하지만 task가 관련이 없다면 이는 오히려 성능에 악영향을 주므로 현재는 어떤 것을 공유해야하는지에 집중하고 있음.

<br></br>

### 6. Auxiliary tasks

- Related task: 관련된 task를 사용하는 것은 예전부터 사용했던 방법. [Caruana, 1998]

- Adversarial: 반대되는 task도 사용할 수 있음. [maximize training error를 통해]

- Hints: feature가 학습하기 힘들 시에 hint를 사용할 수 있음. [비교적 쉬운 task를 함께 학습시키는 것]

- Focusing attention: 모델이 feature를 무시하는 부분에 집중하게 할 수 있음. [ex. 안면 인식에 facial landmark 위치 task를 같이 사용하는 것]

- What auxiliary tasks are helpful?

  - 보조 task를 선택하는 것은 이것이 main task와 관련이 있다고 가정하는 것으로 시작한다.

  - 하지만 선택한 보조 task가 정말로 관련이 있는 task인지 확인할 방법이 필요하다.

  - 아직 논의되고 있는 부분임.
