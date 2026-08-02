## DECEPTIONBENCH: A Comprehensive Benchmark for AI Deception Behaviors in Real-world Scenarios

### Info

* `publication`: Neurips 2025
* `author`: Huang et al.
* `url`: https://openreview.net/pdf?id=x1lSRe3hcO
  
* Motivation: 최근 연구들에서는, LLM이 deception behaviors 를 보일 수 있음을 보였고, 기존의 benchmarks는 한정적인 scenarios 에 집중하였거나, psychological experiments 에 집중하였음. --> 체계적으로 LLM의 deception behaviors 를 보기 위해 evaluation framework 를 만들어, 다음의 questions 를 다룸.

  - 다른 여러 도메인에서, deceptive tendencies 가 어떻게 나타나는가?
  - 어떤 intrinsic behavioral patterns (uncovering the thinking logic) 이  deceptive responses 를 야기하는가?
  -  extrinsic contextual factors가 deception behaviors 에 얼마나 영향을 끼치는가?
  
* Problem: LLM의 deception behavior 평가
  
* Method: 5가지 도메인 선정(Economy, Healthcare, Education, Social Interaction, and Entertainment)해 150 carefully designed scenarios 생성 -> intrinsic patterns 영향 을 보기 위해 Egoism (self-preservation)  and Sycophancy (user-appeasing) 로 시나리오를 1차로 나누고, extrinsic contextual 영향을 보기 위해 Incentivization and Coercion 과 feedback loops (Multi-turn) 시나리오로 2차로 나눔. --> 각 시나리오를 LLM에게 user input 으로 넣어 thought & response 를 뽑게 하여, 각각에 대해 honest/decept 인지 GPT-4o judge 로 평가

<img width="1735" height="349" alt="image" src="https://github.com/user-attachments/assets/a523217f-6932-48b3-bc88-791af37182e1" />


- Evaluation: LLM-as-Judge (GPT-4o)

$$DR_x = \frac{\sum_{i=1}^{N} I(l_{x,i} = \text{deceptive})}{N}, \quad x \in \{\text{thought}, \text{response}\}$$

where $N$ is the number of valid outputs, $l_{x,i}$ is the label for the $i$-th output’s thought or response, and $I(\cdot)$ is an indicator function ($1$ if deceptive, $0$ otherwise).

- findings

  - Entertainment and Social Interaction 도메인에서 deceptive behavior 가 더 많이 발생함.

  - Claude series 에서 가장 적은 deceptive behavior 가 관찰됨.

  - Effect of Intrinsic Factor:  Self perspective (Egoism)이 Other perspective (Sycophancy) 에 비해 deception 의 경향이 더 높았음.

  - Effect of Extrinsic Factors:  coercive pressure 가 incentivization 보다 deception behavior 를 더 높였으며, Feedback loops 시나리오에서 제일 deception behavior 가 높았음.
