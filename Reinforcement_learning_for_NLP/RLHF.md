## :page_facing_up: Learning to summarize from human feedback

### Info

* `publication`: NeurIPS 2020
* `author`: Nisan Stiennon et al.
* `url`: https://arxiv.org/pdf/2009.01325.pdf

### 1\. Abstract & Introduction

LLM pre-training은 여러 NLP task에서 높은 성능을 달성하였으며, 특정 task에 대해 fine-tuning하는 식으로 적용하는 방식이 만연해있다(human demonstration에 대한 log probability 최대화).   
하지만, 이런 방법은 성능을 크게 올려줄 수 있지만, 우리의 목표와는 괴리가 있다. 예를 들어, 요약 모델이 사람이 쓴 텍스트 label을 보고 이 likelihood를 최대화하는 것과 모델이 높은 질의 output을 만들어내는 것은 다르다.

본 논문에서는 아래와 같이 위에서 이야기했던 output의 질을 높이는 방법을 연구한다(reward learning을 사용해 human feedback으로 부터 LM fine-tuning 연구 기반).

  1. 요약문 쌍 사이에 human preferences 데이터 셋을 모은다.
  2. 사람이 더 선호하는 요약이 무엇인지 판단하도록 지도 학습을 통해 **reward model (RM)**을 학습한다.
  3. RM이 준 점수를 최대화하기 위해 강화 학습을 통해 policy를 훈련한다(policy는 각 "time step"별로 텍스트 토큰을 만들고 PPO 알고리즘을 이용해 업데이트된다).

- main contributions

  - human feedback을 이용한 모델 학습이 영어 요약 task에서 strong baselines 성능을 능가함

  - 새로운 도메인에서 human feedback 모델은 지도 학습 모델보다 더 일반화가 잘 되어 있음을 보임
  - 우리의 policy와 reward model의 경험적인 분석을 제공
  - human feedback dataset을 공개

<br></br>

### 2. Method and experiment details

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/568da5f5-61a7-4da2-9837-f5442645f8e7" width="70%"></img></div>

#### 2-1. High-level methodology

우리의 방법은 원하는 데이터셋에 대한 지도 학습을 통해 초기 policy를 fine-tune하는 것으로 시작되며, 이후에는 다음의 3가지 단계를 반복한다.

1. 현재 존재하는 policies로부터 sample들을 모으고 사람에게 비교를 위해 보낸다(현재 policy, initial policy, original summaries and 다양한 baseline이 포함된 다양한 source로부터 요약문을 샘플링한 후 사람에게 best summary를 판단하게 함).

2. 사람에 의해 판단된 best summary로부터 reward model을 학습한다(post와 summary 후보들이 주어지면, reward model이 log odds를 예측하도록 훈련).

3. reward model에 가까이 policy를 최적화한다(reward model의 logit output을 reward로 취급하여 PPO 알고리즘을 이용해 강화학습).


#### 2-2. Datasets and task

- Datasets : TL;DR 요약 데이터셋(3M reddit post + post 상에 쓰여져있는 TL;DR)

  - 데이터 질을 높이기 위해 필터링 과정을 거쳤으며, 24~48 tokens를 가지는 사람이 작성한 요약문만을 포함하도록 하였다.
  - 최종 데이터 셋은 123,169 posts로 구성되며 이 중 5%를 validation set으로 가져갔다.

- Task : 최대한 좋은 질을 가지는 48 tokens 이하의 요약문을 생성하는 모델을 만드는 것

  - 요약문의 질은 `독자가 post를 읽지 않고 요약문만을 본다했을 때, 이 독자들에게 원본 post에 대한 내용을 얼마나 신뢰도 있게 전달할 수 있는가?` 로 판단했다.

#### 2-3. Models

- 사용한 모든 모델은 GPT-3이다.

- Pretrained models : large text corpus에서 Next token Prediction으로 사전학습된 모델. 이런 모델들은 `zero-shot` baselines로 사용함.

- Supervised baselines : 위의 pretrained model을 필터링한 TL;DR 데이터 셋을 이용해 요약문을 예측하도록 지도 학습을 통해 fine-tune함.

  - 이 모델은 여러 요약문 비교를 모으기 위한 초기 요약문을 샘플링하고 초기 policy와 reward model을 설정하기 위해 사용한다.

- Reward models : 위의 지도 학습 모델에 scalar value를 뽑아내기 위한 랜덤하게 초기화된 linear head를 더한 모델.

  - 주어진 post $x$ 에 대해, 사람이 판단하기에 어떤 것이 더 좋은지 예측함. $y \in {y_0, y_1}$

  - RM loss : $loss(r_{\theta}) = - E_{(x, y_0, y_1, i) \sim D}[log(\sigma(r_{\theta}(x, y_i) - r_{\theta}(x, y_{1-i})))]$

    - $r_{\theta}(x, y)$ 는 post x와 summary y의 reward model이 뱉는 scalar output

    - 해석 참고 : https://ai.stackexchange.com/questions/38779/instructgpt-what-is-the-sigma-in-the-loss-function-and-why-log-cdot-is-bei

- Human feedback policies : 높은 질의 output을 만드는 policy를 위해 위의 reward model을 사용

  - 전체 summary에 대한 reward model의 output을 reward로 취급하여 PPO 알고리즘을 이용한 강화 학습 사용

  - 초기 policy는 TL;DR에 fine-tune된 모델을 사용

  - reward : $R(x, y) = r_{\theta}(x, y) - \beta log[\pi_{\phi}^{RL}(y|x)/\pi^{SFT}(y|x)]$

    - 파라미터 $\phi$ 를 가진 학습된 RL policy $\pi_{\phi}^{RL}$ 과 original supervised model $\pi^{SFT}$ 간의 KL divergence에 패널티를 주는 term을 포함함.

      - KL term은 entropy bonus처럼 작용하며, 보상 모델이 훈련 중에 본 출력과 너무 다른 출력을 생성하지 않도록 함.

    - PPO value function의 경우 Transformer와 분리된 파라미터로 구성함.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/a89d3377-6697-4af1-981b-0683aa791d8b" width="50%"></img></div>


<br></br>

### 3. Results

#### 3-1. Summarizing Reddit posts from human feedback

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/4aadd43d-7068-4600-8e21-4c46126b98b7" width="40%"></img></div>

- 위 그림은 policy의 질을 해당 정책이 생성한 요약 중 인간이 데이터셋의 참조 요약보다 얼마나 선호되는지 백분율로 측정한 결과이다.

  - human feedback으로 훈련된 우리의 policies가 supervised baselines를 능가하였으며 모델 크기가 커짐에 따라 성능이 증가하였다.

- 우리 모델이 생성한 요약문의 품질에 대해 더 알아보기 위해, reference summaries와 supervised baselines들이 만든 요약문에 대해 human labeler에게 4가지 기준으로 평가하도록 하였다.

  - 4가지 기준 : coverage (원본 post로부터의 중요한 정보들을 얼마나 담고 있는가), accuracy (요약문의 내용과 post의 내용간의 일치하는 정도), coherence (요약문 자체가 얼마나 쉽게 읽히는지), 전체 품질

  - 결과로, human feedback 모델이 모든 기준에서 supervised baselines를 능가하였다.

#### 3-2. Transfer to summarizing news articles

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/1a187af0-3561-4720-a8d5-450a41a223bd" width="70%"></img></div>

- Figure (a) : human feedback 모델이 추가적인 학습 없이 CNN/DM news article 요약에서 훌륭한 성능을 보였다(pre-trained only, supervised on TL;DR 모델들 능가). 그리고 CNN/DM 데이터로 fine-tune된 모델과 거의 비슷한 성능을 보였다.

- Figure (b) : human feedback 모델을 CNN/DM 데이터로 전이하게 되면, CNN/DM으로 훈련된 모델과 `요약 길이`에 대한 분포에서 약간의 차이를 보이기 때문에 직접적으로 비교하기 어렵다. 따라서, 3.1과 마찬가지로 4가지 기준으로 평가한 결과, human feedback 모델이 보다 긴 요약을 생성할 때 더 좋은 성능을 보여주었다.

#### 3-3. Understanding the reward model

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/a026b345-8636-4324-a45a-154b75cd9d98" width="70%"></img></div>

reward model에 따라 최적화를 하는 것은 policy가 human preference와 가까워지도록 해야하지만, reward model이 comparison data의 조그만 분포만을 보기 때문에 완벽하지 않다. 학습 중에 보지 못한 요약에도 reward model이 일반화되기를 원하지만, 보상 모델을 모델이 쓸모없는 평가를 제공하기 시작하기 전까지(overfitting 전) 얼마나 최적화할 수 있는지는 명확하지 않다.

따라서, 우리는 reward model의 이전 버전에 대해 최적화된 여러 policy들을 생성하여 최적화 정도에 따라 어떻게 달라지는지 비교하였다. (labeler들에게 여러 policy들이 만든 요약문과 원본 요약과 비교하도록 하였음)   
왼쪽 그림은 여러 KL penalty 계수 $\beta$ 에서의 PPO 결과를 나타낸다. 약한 최적화에서는 모델의 성능이 좋아졌지만, 여기서 좀 더 최적화를 진행하였을 때, 성능이 떨어졌다(labeler에 판단에 따르면). 이러한 현상은 ROUGE에서도 나타난 것으로 확인된다.

한편, reward model의 크기와 데이터 크기에 따른 ablation study를 진행했다(오른쪽 그림). 결과, data size를 2배로 늘리면 약 1.1% 성능이 증가하였으며 model size를 2배 늘렸을 때 약 1.8% 성능이 증가하였다.

#### 3-4. Analyzing automatic metrics for summarization

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/eb19f908-780c-443f-ae8b-87014409eb81" width="40%"></img></div>

우리는 다양한 자동 메트릭이 인간의 선호도를 얼마나 잘 예측하는지 연구하고, 이를 reward model과 비교하였다. 구체적으로, ROUGE, 요약 길이, 포스트에서의 복사량, baseline 지도 모델의 로그 확률을 실험하였다.

결과, reward model이 다른 metric보다 성능이 좋았으며, ROUGE의 경우 우리 모델이 개선되었어도 샘플 품질을 추적하지 못하는 것을 발견했다. ROUGE는 지도학습 모델에서 labeler와 일치율이 약 57%였지만, human feedback model에서는 약 50% 이하로 떨어졌다.

- Optimization : 위 그림을 보면, ROUGE로 최적화하는 것은 일관되게 품질을 높여주지 못하는 것을 볼 수 있다.
