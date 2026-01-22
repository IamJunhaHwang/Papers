## :page_facing_up: LIMA: Less Is More for Alignment

### Info

* `publication`: NeurIPS 2023
* `author`: Chunting Zhou, Pengfei Liu, et al.
* `url`: https://openreview.net/pdf?id=KBMOKmX2he


### 1. Abstract & Introduction

LLM은 크게 2가지 과정을 거친다: (1) unsupervised pretraining, (2) large-scale instruction tuning & reinforcement learning.   
여기에서 instruction tuning에는 대규모의 human annotated data가 필요하다. 하지만, 본 논문에서는 strong LM이 주어졌을 때, 1,000개의 품질 좋은 training examples만을 이용해 fine-tuning하는 것으로 좋은 성능을 달성할 수 있음을 증명했다.

여기에서, 모델이 user와 interact하기 위한 style과 format을 배우고 이를 드러내기 위한 지식과 능력을 이미 pre-training에서 가지고 있다면, alignment는 간단한 과정이 될 수 있다고 가정한다.   
이 가정을 테스트하기 위해, 품질 좋은 1000개의 examples를 선별하였다(750은 public, 250은 직접 만듦). 이를 이용해 65B  LLaMA model을 fine-tuning하였다(LIMA).

LIMA와 SOTA를 비교하기 위해 300개의 challenging test prompts를 만들었고, LIMA가 RLHF-trained DaVinci-003과 65B의 Alpaca보다 우수했다.   
또한, human test에서 GPT-4, Claude, Bard에 대해 LIMA의 대답이 43%, 46%, 58% 선호되었다.   
Ablation test에서는 prompt 다양성 증가 없이 데이터 양을 늘렸을 때는 성능이 안좋아졌으며, data quality를 최적화했을 때 성능 향상이 있었다. 또한, multi-turn dialogue examples가 없음에도 LIMA는 multi-turn dialogue가 가능했고, 이 능력은 30개의 hand-crafted dialogue chains를 training data에 더해줬을 때 성능이 크게 증가했다.

**위 findings는 pretraining의 강력함과, instructing tuning과 reinforcement learning에 대해 pretraining이 상대적으로 중요함을 의미한다.**

<br></br>

### 2. Alignment Data

- `Superficial Alignment Hypothesis`: 모델의 지식과 능력은 전체 pretraining 동안에 거의 학습되며, alignment는 유저와 상호작용할 때 어떤 format을 사용해야 하는지 모델에게 가르침.
 
  - 이 가설이 맞고 alignment가 주로 style을 학습한다면, 적은 example로도 pre-trained LM을 충분히 튜닝할 수 있을 것

- 이를 위해, 1000개의 prompt & responses를 모았으며, 여기에서 output은 서로 스타일적으로 일치하고 input(prompt)는 다양하다. (또한, 300개의 test set과 50개의 development set 모았음)

#### 2-1. Community Questions & Answers

- data source websites: Stack Exchange, wikiHow, Pushshift Reddit Dataset


- manually authored example: 데이터를 좀 더 다양화 하기 위해 직접 prompt 만듦

  - 2가지 그룹 A, B로 나누어 각각 250개의 prompt를 만듦(주제는 각자의 관심사)

    - 그룹 A: 200 prompts(training), 50 prompts(development set)

    - 그룹 B: 230 prompts(test; 문제가 있을만한 prompt filtering함)

  - 200개 training prompt에 대해 high-quality answers를 만들었음.

    - 최대한 동일한 tone을 가지도록 노력하였으며, 모든 answer는 question에 대한 acknowledgement를 먼저 하고 시작함.

    - 또한, training prompt에 13개의 악성 prompt를 포함시켰으며, 대답으로는 이를 부분적으로나 완전히 거절하도록하고 어째서 따르면 안되는지에 대해 설명하도록 작성. (test set에도 이런 30 prompts 있음)

<br></br>

### 3. Training LIMA

다음과 같은 과정으로 LIMA 학습

1. LLaMa 65B로, 1,000 example alignment training set을 이용해 fine-tuning 진행

2. 각 speaker (user and assistant)를 구분하기 위해, special token인 EOT(End of Turn)를 각 speaker 발화 뒤에 삽입

    - 이는 생성을 멈추는 EOS 토큰과 같은 역할을 하지만 pretraining 과정에서의 EOS와의 충돌을 피함

- hyperparameter setting

  - 15 epochs with AdamW

  - warmup X, learning rate = 1e-5 with linear decaying to 1e-6

  - batch size: 32

  - max token length = 2048

<br></br>

### 4. Human Evaluation

#### 4-1. Experiment Setup

LIMA를 다른 모델과 비교하기 위해, 각 test prompt에 대해 하나의 response를 생성하였으며, 이에 대해 crowd workers에게 LIMA와 baselines 중 어떤 것을 선호하는지 묻는 식으로 평가하였다.   
위 실험을 반복하였으며, human 대신 GPT-4로도 대체해서 평가하였으며 비슷한 agreement level을 보였다.

* baselines: Alpaca 65B, DaVinci003, Bard, Claude, GPT-4

* Generation: nucleus sampling with p=0.9, temperature = 0.7, repetition penalty: 1.2

- methodology: 각 평가자에게 하나의 prmopt와 다른 모델에서 생성된 2개의 possible responses를 보여준다. 이후, 어떤 response가 더 좋은지, 2개 대답 모두 좋지 않은지 고르게 하였다.

- inter-annotator agreement

  - 두 annotators가 일치하면 1점, 한쪽이 tie를 고르면 0.5점, 다른 경우엔 0점인 식으로 계산

  - 50개의 annotation examples에 대해 author, crowd, GPT-4를 비교하였을 때, human-human, human-GPT 간의 일치도가 약 80%였다. (human-GPT 간의 일치도가 human-human과 비슷한 것으로 보아 GPT도 평가자를 대체할 수 있음)

<div align="center"><img src="https://github.com/user-attachments/assets/bd3335eb-25af-4beb-aa51-75bcb9a0c758" width="60%"></img></div>


#### 4-2. Results

- Figure 1은 human preference를, Figure 2는 GPT-4의 preference를 나타낸다.
  - human과 GPT-4는 서로 비슷한 trends를 띄고 있음

- Alpaca 65B는 52배 많은 데이터로 학습되었음에도 LIMA보다 덜 선호되었으며, 이는 RLHF를 거친 DaVinci003도 같았다.

- Bard의 경우 LIMA보다 42% 선호되는 대답을 생성하였지만, 그럼에도 58%는 LIMA가 선호된다.

- Claude와 GPT-4와 비교했을 때, LIMA보다 선호되는 대답을 생성했다. 그래도 LIMA가 선호될 때가 무시하지 못할만큼 있으며 GPT-4 preference를 보았을 때, GPT-4보다 LIMA가 선호될 때가 19%있었다.

#### 4-3. Analysis

이전에 보여줬던 결과들은 high-tuned products이기 때문에 매우 높은 기준을 가지고 있기 때문에, 50개의 random examples로 직접 absolute assessment를 진행했다.

- label: Fail, Pass, Excellent

- 실험 결과 50% LIMA 응답이 Excellent, 38% Pass, 12% Fail이었다.

- Out of Distribution: 50개 중 43개가 training example에서 보였던 format과 비슷하기 때문에 추가로 13개의 out-of-distribution examples 분석

  - 45% Excellent, 35% Pass, 20% Fail

  - 작은 sample임에도, LIMA는 training에서 보지 못했던 데이터에도 비슷한 성능을 보였으며, 일반화 능력이 좋음을 의미

- Safety: training set에서 적은 수(30) safety examples의 효과 분석

  - test set에서 30개의 민감한 prompts에 대해 LIMA에게 응답하도록 하였으며, 80%에 대해 safe하게 응답함.

  - 몇몇 case에서는 task 수행을 완전히 거부

  - 악의적인 의도가 직접적으로 표현되지 않은 식으로 내포되어있을 때에는, unsafe response를 제공하는 경향이 있었음

<br></br>

<div align="center"><img src="https://github.com/user-attachments/assets/f9f24ed6-88d5-4943-b5a9-aeae203ad9a8" width="60%"></img></div>

### 5. Why is Less More? Ablations on Data Diversity, Quality, and Quantity

training data diversity, quality, quantity의 효과를 알아보기 위한 ablation 실험 진행

- Experiment Setup

  - 7B LLaMA model fine-tuning, hyperparameters는 LIMA와 동일

  - test set prompt에 대해 5개 response를 샘플링하여 ChatGPT에게 helpfulness를 1~6점으로 점수매기게 함.

- Diversity

  - prompt diversity 효과를 테스트하기 위해, quality-filtered Stack Exchange data와 wikiHow data를 비교했다.

  - 각 소스에서 2000개의 training examples를 뽑았으며, Figure 5는 더 다양한 Stack Exchange data가 더 좋은 성능을 내는 것을 보였다.

- Quality

  - response quality 효과를 테스트하기 위해, Stack Exchange에서 filtering과정을 없이 2000개를 뽑고, filtered dataset에 학습한 모델과 비교했다.

  - 결과, 0.5점의 성능 차이를 보였다. (filtered가 우수)

- Quantity

  - 보통 알려진 사실은 학습 데이터 양을 늘리면 성능이 늘어나는 것이지만, alignment의 양을 늘려도 response quality가 좋아지지 않았다.

  - 이는, 데이터의 양 하나만으로는 alignment의 scaling law에 해당하지 않는다는 것을 의미한다.

<br></br>

### 6. Multi-Turn Dialogue

1000개의 single-turn interaction에 fine-tuning된 모델이 multi-turn dialogue도 잘 할 수 있을까? 이를 테스트하기 위해 10개의 live conversation에 대해 테스트하여 Fail, Pass, Excellent로 평가했다.

- 다른 training 없이, LIMA는 multi-turn에서도 자연스럽게 동작하였고, 10개 중 6개에서 3개의 interactions 내에서 prompt를 따라가는 것을 실패했다.

- 대화 능력을 향상시키기 위해, 30 multi-turn dialogue를 모아 LLaMa로부터 LIMA를 새로 fine-tuning 하였다. (1000 + 30 examples)

  - 똑같은 prompt로 10개의 live conversation을 만들었을 때, excellent response가 45.2%에서 76.1%로 크게 향상하였다.

  - failure rate도 `15fails per 42 turns --> 1 fail per 46` 으로 줄었다.

  - 또한, quality를 비교하였을 때 새로 fine-tune한 모델이 더 우수했다. (10개중 7개가 더 우수, 3개는 tie)

- 이렇게 30개의 example만으로 능력이 좋아지며 zero-shot model도 잘 동작한다는 것은 `pre-training에서 이러한 능력을 이미 학습하였으며, 제한된 supervision으로 이를 불러올 수 있다`는 가설을 뒷받침한다.
