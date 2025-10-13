## :page_facing_up: BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data


### Info

* `publication`: ACL 2025 Long
* `author`: Wenkai Li, Jiarui Liu et al.
* `url`: https://aclanthology.org/2025.acl-long.999.pdf

<div align="center"><img src="https://github.com/user-attachments/assets/bcde56f8-8146-4cda-85c5-3dea58bce597" width="70%"></img></div>


### 1. Abstract & Introduction

human personality를 simulate하고 text generation에서 이에 대한 영향은 중요한 문제이지만, 이제까지는 `prompting` 에 의존해왔다. 이러한 prompting에 사용된 description은 보통 personality test에서 가져왔기 때문에 validity concern이 제기되었으며, description(ex. You are the life of the party)들은 LLM에게 맞지 않는다(LLM은 party에 참가하지 못함).   
또한, personality trait으로 레이블링된 대규모의 human-generated dataset의 부족으로 training-based approaches가 제한되었었다.

본 논문에서는, test에서의 실제 사람의 personality expressions를 바탕으로 대규모 대화 데이터셋 BIG5-CHAT을 만들어 LLM에게 realistic human personality trait을 inducing한다.   

전체 outline은 위 그림과 같다.

1. product-of-experts text generation(DExperts)를 이용해 2개의 primary data sources(PsychGenerator, SODA)를 합친다.

    - PsychGenerator: Big5 annotated 850K Facebook posts
    - SODA: rich dataset of diverse social interactions

2. 이렇게 만들어진 BIG5-CHAT을 활용해, real human data를 바탕으로한 training-based methods(SFT, DPO)를 traditional prompting과 비교한다.

또한, 사람의 경우 personality traits가 reasoning abilities와 연관이 되어 있다는 연구가 있어, LLM의 personality inducing이 reasoning performance에 주는 영향도 확인해보았다.   
실험을 통해 높은 conscientiousness & agreeableness를 가진 모델은 reasoning tasks에서 일관적으로 좋은 성능을 보였으며, 낮은 extraversion & neuroticism은 일반적으로 좋은 성능을 보였다.

- Contributions

  - 다양한 personality expressions의 100,000 dialogues를 포함한 대규모 데이터 셋인 BIG5-CHAT을 소개

  - SFT & DPO 와 prompting을 비교하는 것으로 LLM personality에 대한 양적 평가 수행

  - personality trait이 social reasoning과 general reasoning 성능에 미치는 영향을 조사했다.

<br></br>

### 2. Methodology

large-scale personality-grounded dataset의 부족을 해결하기 위해, controllable text generation model을 domain-specific personality-annotated dataset과 합친다.   
이를 위해, DExperts framework와 PsychGenerator dataset을 이용해 BIG5-CHAT이라는 여러 대화 시나리오에서 다양한 personality expressions을 담고 있는 dataset을 만든다.  이 방법을 **PSYCHSTEER** 라고 명명

#### 2-1. DExperts Framework

DExperts는 expert generator를 이용해 decoding time에 model output을 steering한다(여기에서는 expert generator가 대화의 질은 유지하면서 Big Five personality trait을 내비치도록 train됨).   

1. DExperts에서 `M`을 pre-trained base LM, $M^{expert}$ 를 원하는 personality를 text에 내비치도록 fine-tuned된 expert generator 라고 하자.   

2. 각 time step t에서, 주어진 prompt와 이전 token sequence $x_{<t}$ 에 대해 base model M은 logits $z^{base}_t \in \mathcal{R}^{|V|}$ 를 계산한다.

    - V: vocabulary

3. expert generator $M^{expert}$ 는 loigits $z^{expert}_t$ 를 똑같은 방식으로 계산한다.
4. expert generator의 영향을 통합하기 위해 base model의 logits를 아래와 같이 조절한다.

    - $z^{combined}_{t} = z^{base}_t + \gamma z^{expert}_t$ ; $\gamma \in [0, +\inf)$

5. combined logits는 probability distribution으로 바뀌고 softmax function을 통해 next token이 sampling됨

#### 2-2. Expert Generator Model Based on Social Media Posts

expert generator model 학습을 위해 `LLaMA-3-8B-Instruct`를 PsychGenerator dataset으로 SFT를 진행했다.   
해당 데이터는 846,304 Facebook posts로 이루어져 있으며 저자의 Big Five personality score가 레이블링되어 있다.

총 5개의 expert generators를 만들었으며, 각각은 personality traits 중 하나를 반영해 text를 생성한다. 또한, 각 personality trait은 floating-point label -> binary label(high/low)로 변환해 사용했다.

expert generators는 Alpaca 포맷을 사용해 fine-tuned 하였으며, text completions에서 base model이 첫 5개 단어를 만들게해서 personality trait을 반영하면서도 자연스러움을 잃지 않게 하였다.

<br></br>

### 3. BIG5-CHAT Dataset

다양한 social interactions에서 Big Five personality traits를 포착하기 위해 설계된 large-scale human-grounded dialogue responses dataset인 **BIG5-CHAT** 을 제안한다.

- Dataset Construction

  1. 데이터 구축에는 다양한 범위의 realistic social scenarios를 제공하는 SODA(Social DiAlogues) dataset을 이용한다(SODA는 GPT-3.5로 생성된 social commonsense narratives가 풍부한 대화 데이터).

  2. 이 dialogues에 personality trait을 반영하기 위해 DExperts framework를 적용한다.

  3. 우선, SODA에서 랜덤하게 10,000개의 샘플을 뽑는다. 여기에서, SODA는 Speaker X, Y로 구성된 interaction으로 되어 있는데, 위에서의 PSYCHSTEER framework로 personality trait을 조정한 새로운 발화를 만든다.

      - 예를 들어, original SODA dialogue (Speaker X)에 대한 대답 (Speaker Y)를 PSYCHSTEER framework를 통해 새롭게 personality를 반영해 만듦

      - 각 trait당 20,000개의 dialogue 생성(trait별로 high/low)

<div align="center"><img src="https://github.com/user-attachments/assets/a55dca20-e430-4534-b76a-8c97f0c1211c" width="70%"></img></div>


- Evaluating Personality-Steering of the Data Generator

  - 생성한 데이터셋의 품질과 personality trait 반영 여부를 평가하기 위해 평가 모델로 RoBERTa-large classifier를 PsychGenerator dataset을 이용해 학습시켰다.

  - 해당 모델은 PsychGenerator test set에서 93.8%의 정확도를 달성하였다.

  - 이를 이용해 제안한 프레임워크의 expert generator로 생성한 데이터셋과 GPT-4o를 이용해 프롬프트+ 첫 5개 word가 주어지고 personality를 반영해 이후를 생성하게 하였을 때의 결과를 비교했다.

    - 비교결과, 우리의 데이터 셋이 더 높은 정확도 (80.4%)를 보였다.

<br></br>

### 4. Experiments

#### 4-1. Experimental setup

- Prompting & training strategies

  - LLM에게 personality를 주입하기 위해 2가지 prompting baseline 구현

    - instruction-based prompting: 직접적으로 특정 Big Five traits를 model에게 지시

    - demonstration-based prompting: BIG5-CHAT에서 랜덤으로 뽑은 10개의 특정 personality에 대한 in-context examples 제공
  
  - 위 2가지 방식을 이용한 SFT와 DPO를 수행하여 비교함. (train에는 LoRA 사용)

    - DPO의 경우 negative response는 똑같은 personality지만 반대 level의 것으로 사용

- model: LLaMA-3-8B-Instruct, LLaMA-3-70B-Instruct

- evaluation

  - personality trait evaluation: BFI test, IPIP-NEO

    - temperature를 0.6으로 사용해 5번 반복

  - reasoning capabilities: SocialIQA, GSM8K, MathQA, TruthfulQA, CommonsenseQA, PIQA, MMLU, GPQA


#### 4-2. Personality Trait Assessment Results

<div align="center"><img src="https://github.com/user-attachments/assets/9a0a547d-e659-4042-9503-a77e1572ae1f" width="70%"></img></div>

위 표는 direct inference, 다양한 alignment baseline들을 사용한 모델의 BFI와 IPIP-NEO test 결과를 나타낸다(direct는 personality-related prompt 없이 직접적으로 model inference).

- direct inference에 비해 prompting과 training method가 personality questionnaires에서 trait을 더 잘 반영했다.

- training-based methods(SFT, DPO)는 prompting-based 보다 좀 더 두드러지게 personality를 유도해냈다.

  - SFT와 DPO의 큰 차이는 없었음

  - promt-based보다 낮은 level로의 유도를 잘했음

- demonstration-based prompting은 training과 같은 데이터를 사용했지만, 유도를 training보다 못한 것으로 보아 training이 좀 더 효과적임을 알 수 있음

- 또, LLaMA-3-8B-Instruct의 demonstration-based prompting 결과를 제외했는데, 이는 모델의 instruction-following 성능이 상당히 낮아졌기 때문임 (parameter size와 instruction-following capabilities을 원인으로 추측)

- 또한, 학습한 모델의 psycholinguistic richness를 측정했는데, DPO가더 효과적으로 이를 포착함

- human data에 보이는 intra-trait correlations를 얼마나 효과적으로 모사하는지 prompting과 training method를 비교하였으며, training model(이 중, SFT)이 더 정확하게 trait correlations를 포착함


#### 4-3. Reasoning Evaluation Results

<div align="center"><img src="https://github.com/user-attachments/assets/6758e521-3bc0-473d-9c9b-4631a6900270" width="70%"></img></div>

위 표는 LLaMA-3-70B-Instruct에 대해 trianing methods와 baselines를 이용해 reasoning 평가를 한 결과이다.

- 전체적으로, SFT가 일관되게 좋거나 DPO와 일치하는 성능을 보였다. 이는 BIG5-CHAT에 대해 training하는 것이 QA 능력을 해치지 않음을 의미한다. (특정 personality에서는 social, mathematical, commonsense reasoning 능력이 향상됨)

- 높은 conscientiousness와 agreeableness를 가진 모델은 일반적으로 낮은 level의 해당 trait을 가진 모델보다 좋은 성능을 보였으며, 반대로 낮은 extraversion과 neuroticism이 우수했다.

  - 이는 전체 벤치마크 전반적으로 관찰되었다. 이는 특정 personality trait level이 reasoning task에 도움이 되는 것을 의미

  - 위 현상은 human에서도 보고된 바가 있음

    - 하지만, human과 다르게 high openness는 사람한테는 인지능력에 좋았지만, 모델의 경우 수학을 빼고는 그리 큰 향상은 있지 않았다. 이러한 차이는, LLM에서의 openness 영향이 domain-specific하거나 다소 국한되어 있는 것으로 추측됨
