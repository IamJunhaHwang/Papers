## :page_facing_up: Recommending What Video to Watch Next: A Multitask Ranking System

### Info

* `conf`: Recsys 2019 (ACM Conf)

* `author`: Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar,
Maheswaran Sathiamoorthy, Xinyang Yi, Ed Chi


* `url`: https://daiwk.github.io/assets/youtube-multitask.pdf

### 1. Abstract & Introduction

추천 시스템은 유저가 현재 보고 있는 영상이 주어지면 다음에 볼 것 같은 영상을 추천하는 것이다.

전통적인 추천 시스템은 크게 `후보 생성(candidate generation)`과 `ranking` 의 설계로 나뉘는데 본 논문은 `ranking`에 집중한다.

- 현재 추천 시스템에는 아래 두 가지 어려움이 있다.

  - 최적화하려는 목표(objective)가 상충할 수 있다. 예를 들어, 유저들에게 높은 점수(좋아요)를 받거나 공유를 하는 영상을 추천하고 싶을 수 있음.  => 여러 목표가 존재함.

  - 시스템에 내재된 편향이 종종 있다. 예를 들어, 점수가 높다는 이유(추천 영상으로 떴다는)로만 유저가 단순히 영상을 클릭해 봤을 수 있다. [Feedback loop effect] ==> 편향을 최대한 줄여야 함.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/224585225-cc56757e-62cb-4c29-8a89-f21b117a24e4.png" width="80%"></img></div>


#### 위와 같은 문제를 다루기 위해 `multi-task neural network architecture`를 제안한다.

- 이는 `multi-task learning`을 위한 `Multi-gate Mixture-of-Experts (MMoE)`을 적용해 [Wide & Deep](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454) 모델을 확장한 것이다.

- 여러 목표(objective)들을 크게 두 가지로 묶었다. [Multi-Task]

  1) 유저의 클릭, 추천 영상의 시청 시간과 같은 `참여 목표(engagement objective)
`

  2) 영상을 좋아하는지, 점수를 남기는 것과 같은 `만족 목표(satisfaction objective)`

  - 이런 유저들의 행동들을 학습하고 평가하기 위해 `MMoE`를 사용했다.
  
    - 이는 잠재적으로 상충하는 목표들을 공유하기 위한 파라미터를 자동으로 학습함.

    - `Mixture-of-Experts` 구조는 `input layer`를 전문가로 모듈화함. ==> 각각이 input의 다른 측면에 집중

      - 이는 복잡한 특징 공간으로부터 표현(representation)을 효과적으로 학습할 수 있음.

    - 마지막으로 `Multiple gating network`를 사용해 각 목표가 공유하거나 공유하지 않기 위한 `expert`를 고를 수 있다.

- 또, 선택 편향을 모델링하고 제거하기 위해 `shallow tower`를 도입했다.
  
  - 선택 편향과 관계있는 input을 취해서 메인 모델의 마지막 예측을 위한 `bias term(scalar)`으로 제공

  - Wide & Deep model에서 Wide 파트에 쓰임.

- 이 모델은 training data의 label을 두 가지 파트로 분해한다.

  1) main model에서 학습한 편향되지 않은 `user utility`

  2) shallow tower에서 학습 `추정된 경향 점수(estimated propensity score)`

<br></br>

### 2. Model Architecture

쿼리, 후보, 컨텍스트가 주어졌다면, 랭킹 모델은 '클릭, 시청, 좋아요, 노관심' 처럼 사용자가 취할 수 있는 행동들의 확률을 예측한다.(learning-to-rank) 
이 때, 각 후보를 예측을 하는 방법로 Point-wise 접근법을 취한다. (모델을 실제 적용시킬 때는 빠른 시간에 추론이 가능해야 하므로)

학습을 위한 `training label`으로 유저의 행동을 사용한다. 위에서 설명했듯이 Objectives를 `참여 목표`와 `만족 목표`로 나누었는데 이것은 `classification`, `regression` task의 조합으로 모델링한다.  => **Multi-Task Learning**

최종적으로는 `combined score`를 만들어 낸다. (combination function이 존재하며 이를 위한 가중치는 수동으로 조절했음)


<div align="center"><img src="https://user-images.githubusercontent.com/46083287/224585243-03b120bc-9daf-4424-8040-f5160f9ff700.png" width="80%"></img></div>

`hard-parameter sharing(좌)`은 흔하게 사용되지만 task 간의 상관 관계가 적을 때 학습을 헤친다.

따라서, `task conflicts & relation`을 모델링하기 위해 `MMoE(우)`를 설계했다. => `soft-parameter sharing`

- **Multi-gate Mixture-of-Experts**

  - `task difference`를 포착하기 위해 설계되었음. [왼쪽 사진의 shared ReLu층을 MoE 층으로 바꾸고 gating network 추가]

  - `shared hidden layer` 바로 위에 `expert layer`를 추가함. => multimodal feature space를 잘 모델링할 수 있음.

  - `expert layer == FFNN with ReLU`이며 MMoE를 공식화 하면 아래와 같다.

    - $y_k = h^k(f^k(x))$, `h`는 마지막 은닉 층(task layer)이다.

    - $f^k(x)) = \displaystyle \sum^k_{i=1}g^k_{(i)}(x)f_i(x)$

    - `f(x)`는 expert이고 `g(x)`는 gate network이다. `x`는 shared hidden layer embedding

    - $g^k(x) = softmax(W_{g^k}x), $W_{g^k} \in \mathcal{R}^{n x d}$, `n`: expert 개수

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/224585268-2be0c166-c235-4835-bbc7-3a942366d234.png" width="60%"></img></div>

- Selection Bias

  - shallow tower를 통해 편향을 반영하기 위해 학습한다. 여기에는 position feature, device info 등이 들어감.

  - 훈련 시에는 position feature를 반영했다가 서빙 시에는 이를 missing value로 취급한다.

<br></br>

### 3. Experiment

오프라인 테스트 단계에서는 task별 AUC, squared error를 모니터링했고 A/B 테스트와 오프라인 테스트의 결과를 종합하여 하이퍼 파라미터를 튜닝하였다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/224585277-44456d90-f592-4a23-9d4b-4bf134dd2be7.png" width="60%"></img></div>

위 사진은 MMoE를 적용한 것과 안한 것을 비교한 것, engatement metric: 추천한 영상에 소요한 시간

- MMoE가 효과가 있었다.
![image](https://user-images.githubusercontent.com/46083287/224585282-0ab9a7fb-0da3-4930-9f4c-394d61dc8fea.png)

<div align="center"><img src="/uploads/d8b25f7ddb088f6be6b9527086ea78f1/image.png" width="60%"></img></div>

Engagement Task는 여러 experts와 공유했지만 Satisfaction Task는 적은 Expert가 높게 이용했다.

`gating network`를 sharing layer 위가 아닌 input layer 위에 적용해보았지만 큰 차이가 없었다.

- position bias

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/224585289-ff51a3ea-f427-457f-8744-d2a685c8dec0.png" width="80%"></img></div>

예상했던대로 가장 위에 노출되는 추천 영상이 클릭 수가 높았으며 뒤로 갈 수록 낮아졌다. 이에 따라 낮은 위치일 수록 더 적게 bias를 배웠다.

Shallow Tower가 가장 bias를 줄일 수 있었다.
