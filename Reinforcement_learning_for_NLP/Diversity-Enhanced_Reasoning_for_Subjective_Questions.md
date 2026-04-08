## :page_with_curl: Diversity-Enhanced Reasoning for Subjective Questions

### Info

* `publication`: ICLR 2026

* `author`: Wang et al.
* `url`: https://openreview.net/forum?id=1Bf0tToGT1 


### Summary

LLM이 single reasoning chain 내에서 다양한 `Role` 관점으로 reasoning 하여, diversity를 높이는 방법인 MultiRole-R1 제안


### Motivations

- 현재의 RLVR은 math와 같은 objective task 에서 뛰어난 성과를 보였지만, subjective reasoning 에서는 성능이 떨어졌으며 최근 `diversity`의 중요성이 대두되고 있음.

- LLM이 single reasoning chain 내에서 다양한 `Role` 관점으로 reasoning할 수 있도록 synthetic data를 만들어 SFT하고, GRPO를 적용하여 diversity를 높이자.



### Contributions

- subjective reasoning tasks에 diversity-enhanced training 도입 (MultiRole-R1)

- 제안한 모델을 In-Domain / Out-Of-Domain 평가를 진행하고 성능이 향상되는 것을 보였으며, diversity가 성능과 연관이 깊음을 밝힘


<div align="center"><img src="https://github.com/user-attachments/assets/cc2f0f94-cf7d-44a5-a905-48c2d18ce3da" width="70%"></img></div>

### Core Idea (novelty)

다양한 관점에 대해 모델이 생각하여 여러 답변을 하게 하자.

###  & Methodology

#### Stage 1: Multi-Role Reasoning Paths Synthesis & Finetuning

- Role 선택

  - model에게 few-shot prompts로 conflicting viewpoints를 가지는 n개의 context-relevant roles 찾게 함. 

  - 찾아낸 role $\mathcal{R} = \{ \mathcal{R}_1, \mathcal{R}_2, ..., \mathcal{R}_n \}$ 과 LLM `M`, 질문 `Q`가 주어지면, 다음과 같은 selection probabilities를 통해 가장 높은 3개를 뽑음. (Pilot analysis에서 3개의 role을 사용하는 것이 가장 효율이 좋았음)

    - $$P(R_i \mid Q) = \mathrm{softmax} \left( E\big[ M(R_i \mid Q) \big] + \alpha \, \mathbb{E}_{R_j} \big[ 1 - \mathrm{sim}(R_i, R_j) \big] \right)$$

- Self-Consistency Filtering

  - 각 Role에 대해 k개의 reasoning paths를 구하고, 각 Role에 대한 일관성을 보장하기 위해 answer에 대해 majority voting을 함.

- Multi-Role SFT

  - Training data: position bias를 피하기 위해 random combinations를 만든다 (순서를 바꿔가며 concat한 것을 모두 train data로)

    - $$D_{\text{train}} = \bigcup_{\pi \in \Pi} \left\{ \left( Q \oplus \hat{T}_{R_i} \oplus \hat{T}_{R_j} \oplus \hat{T}_{R_k} \right) \,\middle|\, \pi \right\}$$
 
  - merging strategies

    - divergent merging: role마다 다른 answer를 기대하는 tasks 경우, 마지막 prediction은 다양한 viewpoints를 weighted aggregation

    - convergent merging: role이 달라도 똑같은 answer를 만들어야되는 tasks 경우, consensus는 majority voting

#### Stage 2:  DIVERSITY ENHANCED REINFORCEMENT LEARNING

- 2가지 타입의 reward를 이용한 GRPO

  - R_acc: role-based reasoning answer의 정답 일치 여부

  - R_div: input text로 부터 shaping signal 계산

    - diversity는 8개의 complementary diversity signals의 가중합

    - lexical, token entropy, sentence length, sentence pattern, adjacent sentence, Yule’s K, distinct N-gram and function word diversity

- 최종 shaped reward: $$R = \delta R_{\text{acc}} + (1 - \delta) R_{\text{div}}$$


### Results

<div align="center"><img src="https://github.com/user-attachments/assets/551066c0-3862-4052-83df-59d36f3d5e73" width="70%"></img></div>


- MultiRole-R1이 baseline보다 뛰어난 성능을 보였다. (OOD와 ID task에서 성능이 둘 다 높아짐)

- DPO보다 GRPO(on-policy RL)가 다양성 향상에 좀 더 기여했다. (GRPO가 +19.73%)

- SFT로 7.5%의 정확도 향상, GRPO로 3.1%의 향상을 얻었다.

- 이전 test-time scaling과 다르게, diversity에 대한 SFT+GRPO가 response length가 줄었음에도 좋은 정확도를 보였다.

  - 또한, 각 Task에 대해 `Acc-Div`, `Acc-Len` 의 상관 관계를 분석했을 때, diversity가 정확도와 상관 관계가 더 높았음 (Table 2)

<div align="center"><img src="https://github.com/user-attachments/assets/7d4a6a9e-094f-4b67-9c8d-2ad237274221" width="30%"></img></div>

- Pass@5에서 Diversity Reward를 주는 것이 정확도에 도움이 되었음.


