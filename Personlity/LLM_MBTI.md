## :page_facing_up: The Better Angels of Machine Personality: How Personality Relates to LLM Safety


### Info

* `publication`: arxiv 2024
* `author`: Jie Zhang et al.
* `url`: https://arxiv.org/pdf/2407.12344

* `github`: https://github.com/AI45Lab/Persafety

### 1. Abstract & Introduction

심리학에서는 personality와 safety behavior의 연관성에 대해 분석하고 있음에도, LLM의 personality가 증명되었어도 safety에 관련한 연구는 부족하다.   
따라서, 이 논문에서는 LLM의 personality와 safety 능력이 크게 관련되어 있음을 발견하였으며, LLM의 personality를 edit하는 것으로 safety를 향상시킬 수 있음을 보인다.   

- Contribution

  - LLM personality와 safety간의 관계를 처음으로 연구

  - personality를 edit하는 것으로 safety 향상 가능함을 증명

<br></br>

### 2. Personality Traits in LLMs

LLM은 personality를 가지고 있음을 여러 연구들을 통해 증명되었음. 여러 personality model들이 LLM의 personality 측정을 위해 사용되었는데, 그 중 널리 사용되는 MBTI를 본 논문에서도 사용함.

#### 2-1. Optimal Selection of Factors Affecting MBTI Assessment


multiple-choice questions에서 LLM의 output이 option order에 영향을 받는다는 연구가 있다. LLM의 personality assessment에서 option order에 대한 영향을 줄이기 위해 <option, label, instruction, language>에 대한 영향을 분석하였다.

- option label: number or alphabet

  - alphabets (e.g., A. Agree B. Disagree) and numbers (e.g., 1. Agree 2. Disagree)

- Instructions

  - (1) samples that answer contains option label and corresponding description (i.e., Question: Artificial intelligence cannot have emotions. A. Agree, B. Disagree. Your answer: B. Disagree)

  - (2) answer contains only option label without descriptions (i.e., Question: Artificial intelligence cannot have emotions. A. Agree, B. Disagree. Your answer: B).

- Language: Chinese or English questionnaire

또한, option order를 랜덤하게 섞은 2가지에 대해 kappa coefficient를 계산하는 방식으로 위 3가지 factor에 대한 영향을 보았다. option order를 섞는 방식은 exchanging description (i.e., changing A. Agree, B. Disagree to A. Disagree, B. Agree)

<img src="https://github.com/user-attachments/assets/34f79876-e789-4f7e-8ebd-88c05292da26" width="75%"></img>


위 Table 1을 바탕으로 앞으로의 실험에 대해 `number, detailed description, chinese version`을 선택했다.


#### 2-2. Reliability of MBTI through Multiple-time Assessments

Table 1에서 3가지 optimal한 factor를 얻었음에도 kappa coefficient가 낮았으며, 이는 stable한 결과를 얻기가 힘든 것을 의미한다. MBTI 결과는 여러 번 검사를 진행해야 신뢰할 수 있다는 연구를 바탕으로 option을 랜덤하게 섞어 MBTI 검사를 100번 검사하는 것으로 얼마나 검사를 진행해야 Kappa coefficient를 바탕으로 신뢰성을 보장할 수 있는지 분석했다.

Figure 2-(a)와 같이, 각 모델들에 대해 신뢰성을 보장하기 위해 검사를 진행해야되는 횟수가 달랐으며, 모든 모델에 대해 30번을 진행하면 신뢰성이 보장됨을 확인 후, 이후 실험에서도 30번을 진행하였다.

보장된 신뢰성을 바탕으로 Figure 3-(b)에 각 모델의 MBTI 결과를 그려보았다. 각 모델들은 각기다른 MBTI를 가지고 있었다.

<br></br>

### 3. The Relationship between LLMs’ Personality Traits and Safety Capabilities

#### 3-1. LLMs with Different Personality Traits Have Different Safety Capabilities

심리학에서는 personality와 safety 능력 간의 상관관계를 찾아내었으며, 이러한 관계가 LLM에서도 존재하는지 탐구하였다.   
이를 위해 각기 다른 MBTI를 가지는 16개의 base model을 3가지 general, 3가지 safety 능력에 대해 평가하였다.

- Models:  Machine Mindset의 모델을 이용(특정 MBTI trait을 담아내기 위해 fine-tuning과 DPO를 거침)

  - 16개의 chinese model(mindset-ch)와 16개의 english model(mindset-en; llama2-7b)

- evaluation dataset: General(ARC, MMLU, MathQA), Safety(ToxiGen, StereoSet, ConfAIde)


<img src="https://github.com/user-attachments/assets/fc341220-d3ae-45c1-8c72-6c44942d62fc" width="70%"></img>


- 각기 다른 personality를 가지는 LLM들은 general 능력은 비슷했지만, safety 능력에서 차이를 보였다.

  - E-I 차원에서, Introversion trait이 privacy 능력이 더 좋았으며, fairness와 toxicity는 떨어졌다.

  - N-S 차원에서, Sensing trait이 privacy와 fairness performance가 더 좋았고, toxicity는 떨어졌다.

  - F-T 차원에서, Feeling trait의 toxicity 능력이 좋았다. mindset-zh에서는 privacy와 fairness가 떨어졌지만, mindset-en에서는 privacy와 fairness가 높았다.

  - J-P 차원에서, Perceiving trait이 fairness 능력이 더 좋았다. mindset-zh에서는 privacy가 높았지만, mindset-en에서는 낮았다.

#### 3.2  Safety Alignment Changes Personality Traits

LLM의 alignment tuning과 safety는 깊은 관련이 있다. 이에 따라, 이 파트에서는 11가지 open-source LLM을 이용해 safety alignment가 LLM의 personality에 어떻게 영향을 주는지 조사하였다.

<img src="https://github.com/user-attachments/assets/80411d4f-354a-4e1c-9579-d3470e7b290f" width="70%"></img>

- LLMs statistically show a tendency towards certain personality types: **ENFJ**

  - Figure 5를 보면, 대부분의 base & aligned LLM이 Extraversion, iNtuition, Feeling, Judging traits를 보였다.

  - 이러한 personality 경향의 일치는 모든 human characteristics가 반영된 extensive training data에서 온 것으로 보인다.

- Alignment generally makes LLMs exhibit more Extraversion, Sensing, Judging traits

  - Alignment는 E-I, N-S, J-P 차원에 영향을 주었다.

- The personality changes through alignment techniques are consistent with some psychological findings on humans.

  - 심리학에서 extraverts는 좀 더 긍정적이고 safety concern에 proactive하며, Judging individuals는 성실하며, unsafe behavior에 부정적이다. Sensing individuals는 세부 사항에 더 주의를 기울이고 규칙을 준수한다.

<br></br>

### 4. Enhancing LLMs’ Safety Capabilities from Personality Perspective

여기에서는, personality trait을 edit하는 것으로 LLM의 safety를 높이는 것에 대해 다룬다.

#### 4-1. Controllably Editing LLMs’ Personality Traits with Steering Vector Technique

Steering Vector를 이용하여 모델의 MBTI를 조정하였다. 

- 주어진 dataset $\mathcal{D} = {(x_i, y_i)}_{i=1}^{|\mathcal{D}|}$

  - $x_i$ : 특정 MBTI dimension 관련 sentence

  - $y_i$ = {0, 1}; 해당하는 binary label(e.g., 1: E, 0: I)

- label 1을 가진 sentence를 ${\mathcal{X}^+}$, label 0을 가진 sentence를 ${\mathcal{X}^-}$ 라고 하자.

- 모든 sentence를 LLM에 넣어 activation set $A_l(\mathcal{X}^+)$ 과 $A_l(\mathcal{X}^-)$ 를 얻는다.

  - $A_l$ 은 LLM의 l번째 layer의 activation이다.

- 다음으로, 각 activation의 centroids를 계산하여 이들의 차를 계산해 steering vector를 얻는다.

  - $v_l = \bar{A_l(\mathcal{X}^+)} - \bar{A_l(\mathcal{X}^-)}$

- 마지막으로, LLM generation 단계에서 이 steering vector를 해당하는 l번째 layer의 representation에 더한다.

  - $h_{l'} = h_l + \alpha v_l$

    - $h_l$ : l번째 representation
    - $\alpha$ : 간섭 강도를 정하는 hyperparameter

#### 4-2. Controllably Editing LLMs’ Personality Traits Enhances LLMs’ Safety Capabilities

<img src="https://github.com/user-attachments/assets/9d192716-0e52-41dd-b5fa-a7b9d3bdb680" width="70%"></img>


- Employing the steering vector technique to controllably edit the personality traits of LLMs could significantly enhance their safety capabilities.

  - Figure 7을 통해, steering vector가 LLM의 특정 ersonality를 다른 dimension은 상대적으로 덜 영향을 받으며 조정할 수 있음을 보였다. (i.e., ISTJ->ISTP, ESFJ->ESTJ, ENFP->ESFP)

  - 위 3가지 case를 통해, toxicity performance는 줄어들었지만, fairness와 privacy performance를 높였다.

<img src="https://github.com/user-attachments/assets/40e1e467-03ba-4ca6-b605-57664a5c2342" width="70%"></img>

- Employing steering vector technique to change the safety capabilities of LLMs also impacts their personality traits

  - 반대로, LLM의 safety 능력을 바꾸면 personality trait에 영향을 주는지 알아보았다.

  - Table 2는 safety 능력을 steering vector로 조정한 결과이고, personality에 영향을 미쳤다.

    - ESFJ모델에서 privacy 능력을 조정하는 것은 ESTJ demension을 높였다.
