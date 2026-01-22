## :page_facing_up: HFT: Half Fine-Tuning for Large Language Models

### Info

* `publication`: ACL 2025
* `author`: Tingfeng Hui et al.
* `url`: https://aclanthology.org/2025.acl-long.626.pdf

- `openreview`: https://openreview.net/forum?id=WzgcreQaNV

<div align="center"><img src="https://github.com/user-attachments/assets/cd1db1db-53f8-4287-aad0-b45cb60745d3" width="70%"></img></div>

### 1. Abstract & Introduction

현재 LLM을 학습하는 방법은 large-scale의 unsupervied pre-training 이후 1번 이상의 SFT data나 human feedback data로 fine-tuning 과정을 거치는 것이 일반적이다.   
하지만, 이런 fine-tuning 과정은 catastrophic forgetting을 유발하기 때문에, pre-trained parameter를 freeze하고 LoRA와 같은 외부 모듈을 이용해 task-specific abilities를 학습하였다.    

그러나, 이렇게 구조를 바꾸는 시도는 model 배포와 continual fine-tuning을 방해한다. 이와 반대로 Full Fine-Tuning은 모든 파라미터를 업데이트한다.   
이 때, fine-tuned 와 pre-trained model 간의 elment-wise parameter 차이는 knowledge shift를 의미(task vector)한다. 최근 연구에서는 이런 task vector의 일부를 제거하거나 잘라내도 target task 성능에 미치는 영향이 미미하다는 결과가 있다.   
즉, partial new parameters만으로 새로운 능력을 학습하는데 충분하다는 뜻이며, `old parameters의 일부만으로 pre-trained model의 능력을 유지할 수 있을까?` 라는 질문이 떠오른다.

이를 검증하기 위해, LLAMA2-CHAT-7B 모델을 이용해 일부 파라미터를 reset하고 general abilities와 basic knowldege에 대해 평가해보았다. 실험 결과, LLAMA-CHAT 모델의 파라미터 절반을 LLAMA2-7B 모델의 파라미터로 reset하였을 때, 몇몇 general abilities를 유지하면서 basic knowledge 능력을 복원할 수 있었다.

이러한 관찰을 바탕으로, 본 논문에서는 Half Fine-Tuning (HFT)를 제안한다. HFT는 각 fine-tuning round마다 랜덤하게 파라미터의 반을 선택하고 freeze하며, 나머지 반은 update된다.   
여기에서, HFT는 모델 구조나 전통적인 fine-tuning 패러다임을 바꾸지 않으므로 이론적으로 full fine-tuning이 가능한 곳 어디든 적용 가능하다(SFT, DPO, CL 등).

- contributions

  -  fine-tuned parameter의 절반을 초기 상태로 reset하는 것으로 새로운 learning ability를 유지하면서 처음의 능력(pre-trained knowledge)을 복원하는 것이 가능하다는 사실을 밝혀냄

  - parameter의 절반을 freeze하고 절반은 training하는 Half Fine-Tuning (HFT)을 제안

  - 대규모 실험과 분석을 통해 HFT의 효과와 효율성을 증명함

<br></br>

### 2. Pilot Experiments

task vector의 일부분만으로 새로운 능력을 가지는 것이 가능하다는 이전 연구를 토대로 task vector의 나머지 일부는 reset하여 이전의 pre-trained model의 능력을 복원시키는 것을 시도하였다.

- Setup

  - model: LLAMA-2-CHAT-7B, LLAMA2-7B

  - original 능력과 instruction tuning으로 얻어진 능력 사이의 균형을 위해 LLAMA2-CHAT 파라미터의 50%를 LLAMA2로 reset

  - parameter matrix 카테고리에 따라 각 transformer layer의 50%을 랜덤으로 선택

    - 4개의 self-attention matrices에서 2개, 홀수개의 feed-forward layers에서 2개 혹은 1개의 행렬 선택

- Evaluation

  - 이전 연구에 따라, General Abilities(MMLU, GSM8K, BBH, TyDiQA, TruthfulQA, HumanEval)와 Basic Knowledge(NaturalQuestion, TriviaQA, HotpotQA)로 나누어 성능 평가

- Results (위의 Figure 1)

  - 파라미터의 반을 reset했을 때, general abilities에서 약간의 성능 하락이 있었지만, basic knowledge에서 눈에띄게 성능이 회복되었다.

  - 이를 통해, large-scale instruction tuning은 pre-trained LLM이 가지고 있는 basic knowledge를 방해한다는 것을 증명했다.

  - 또한, 간단한 half-reset으로 이런 잃어버린 knowledge를 일부 회복할 수 있음을 보였다.

  - 이러한 발견은 fine-tuning동안 일부 파라미터를 freeze하는 것으로 초기의 이미 숙련된 능력을 보존할 수 있게되는 가능성을 보여준다.

<br></br>

### 3. Methodology

Multiple tasks $\mathcal{T}$ 에 대해 continual learning을 진행한다. 각 task는 다음과 같은 input-ouput pairs로 구성된다: $\mathcal{D}^t = \{x^t_n, y^t_n\} ^{N^t}_{n=1}$

Training에서 하나의 모델은 모든 task에 대해 순차적으로 align되며, t번째 round에서는 특정 dataset $\mathcal{D}^t$ 만 access할 수 있다.


- LLM이 주어지면 모든 task에 대해 다음과 같은 objective에 따라 optimize된다: $\mathcal{J(\theta) = \underset{\theta}{max} \underset{t \in \{ 1, |T| \}}{\sum}  \underset{(x^t_n, y^t_n) \in D^t}{\sum} logP_{\theta^t} (y^t_n | x^t_n) }$

  - `log P`는 model output의 확률분포를 나타낸다.

#### Half Fine-Tuning 


<div align="center"><img src="https://github.com/user-attachments/assets/789cbe82-7781-4438-914f-1d3e75c1b791" width="70%"></img></div>

위 그림은 Half Fine-Tuning (HFT)의 workflow이다.

1. 먼저, 각 transformer layer를 3개의 block으로 나눈다: self-attention, feed-forward, layernorm

    - 각 block의 절반은 update되고, 나머지 절반은 frozen된다.

2. frozen & updated parameter는 각 training round마다 달라진다.

#### Why Half Fine-Tuning Works

`(Fu et al., 2022)`의 논리흐름을 빌려 optimization 관점에서 HFT가 왜 작동하는지 해석한다.

- 파라미터 $\theta$^0 로 이루어진 pre-trained model $\mathcal{M}^0$ 과 파라미터 $\theta$ 로 이루어진 fine-tuned model $\mathcal{M}$ 이 있다. 

- 이 두 모델은 서로 같은 구조를 가지고 있으며, $|| \theta - \theta^0 ||_0 \leq pdim(\theta)$ 이다.

  - p = 0.5 in HFT

- 다음, $M \in \{0, 1 \}^{m X m}$ 을 파라미터의 mask diagonal matrix로 정의한다. 여기에서, parameter가 선택되면 diagonal은 1이 된다.

- 따라서, fine-tuning 과정은 다음과 같은 식이 된다: $\theta = \theta^0 + M \Delta \theta$

  - $\Delta \theta$ : task vector 

- 이 경우, HFT는 $||M||_0 = \lfloor mp \rfloor; M_{ij} = 0, \forall i \neq j;  M_{ii} \in \{ 0, 1 \}$ 을 만족하는 constraints ${min}_{\Delta \theta, M} \mathcal{L}(\theta^0 + M \Delta \theta)$ 에 대한 optimization problem을 푸는 것이다.

  - L: loss function

  - $\lfloor \dot \rfloor$: floor function
  - m: parameter numbers

  - mask diagonal matrix로 mask되는 파라미터는 전체의 절반(mp)이며, 이렇게 선택된 파라미터에 대해서 update를 진행

- 이전 조건과 합치면, HFT의 optimization 과정은 다음과 같이 쓸 수 있다.

  - $\mathcal{O} = \underset{\theta}{min} \mathcal{L}(\theta)   s.t. || (I - M)(\theta - \theta^0) ||^2 = 0$

- 라그랑주 듀얼리티를 이용하면, $\mathcal{O} = \underset{\theta}{min} \underset{\lambda}{max} \mathcal{L}(\theta) + \lambda || (I - M)(\theta - \theta^0) ||^2$

  - $\lambda$: Lagrange multiplier

- Minimax 부등식에 따라 다음과 같이 전개된다.

  - $\underset{\theta}{min} \underset{\lambda}{max} \mathcal{L}(\theta) + \lambda || (I - M)(\theta - \theta^0) ||^2 \geq  \underset{\lambda}{max} \underset{\theta}{min} \mathcal{L}(\theta) + \lambda || (I - M)(\theta - \theta^0) ||^2 \geq \underset{\theta}{min} \mathcal{L}(\theta) + || (I - M)(\theta - \theta^0) ||^2$

- 결과적으로, HFT는 regularization term $|| (I - M)(\theta - \theta^0) ||^2$을 가지는 FFT loss function의 상한의 최적을 찾는 것이다.

  - 선택되지 않는 파라미터의 업데이트양을 최소화하도록하는 regularization

  - 이러한, regularization은 spare fine-tuned model의 stability에 기여

<br></br>

### 4. Experiments

- Datasets

  - Full Fine-Tuning, Half Fine-Tuning: TULU V2

  - human preference alignment: UltraFeedback

  - continual learning with DPO: TRACE

- Model: LLAMA-2, LLAMA-2-CHAT

#### Results

<div align="center"><img src="https://github.com/user-attachments/assets/bf11ef4a-8286-4d8b-8413-82a07585b9bf" width="70%"></img></div>


- FFT에 비해 HFT가 LLAMA2-7B에서 2.9%, LLAMA2-13B에서 2.9% 전체 성능 향상이 있었다.

- 또한, DPO setting에서 HFT가 model 학습을 방해하지 않았다.

- Half-Reset과도 비교했을 때, HFT가 좀 더 안정적인 성능을 보였음.

<div align="center"><img src="https://github.com/user-attachments/assets/b046e01d-6bd8-45fc-83d2-ebed67c141f6" width="50%"></img></div>

- basic knowledge을 얼마나 잃지 않는지를 비교하였음.

- SFT와 DPO는 상당히 성능이 떨어졌지만, HFT는 이에 비해 basic knowledge를 잘 보존하였다.

* Half-Reset 또한 basic knowledge를 잘 보존하는 것을 볼 수 있었으며, motivation을 다시 한 번 확인할 수 있었다.

#### Impact of Parameter Selection

- fine-tuned 되는 parameter의 비율을 조사해보았을 때, 약 50% 비율에서 가장 만족스러운 성능을 내었음.

<div align="center"><img src="https://github.com/user-attachments/assets/3016df81-4c67-4089-90c0-da3b7e4315a2" width="50%"></img></div>

- parameter selection strategy에 대해 비교하였음. 

  - model level: parameter matrices에서 임의로 선택

  - layer level: layer 전체를 선택

  - category-level: 논문에서 사용한 방법

- 결과, category-level이 가장 좋은 성능을 냈으며, fine-grained 방법이 update와 non-update parameter간의 상호작용을 최대화하는 것으로 예상됨

#### Revisit the Embedding and LM_head Layers.

<div align="center"><img src="https://github.com/user-attachments/assets/41403dd4-5385-4933-b812-de54e0ef03ae" width="70%"></img></div>

- HFT는 embedding, lm_head layers를 업데이트 함. 여기에서는, 이 두 layer를 freeze시켜 어떤 영향이 있는지 확인함

- 결과적으로, 해당 layers를 freeze시켰을 때 성능이 떨어졌다.
