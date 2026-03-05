## :page_with_curl: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

### Info

* `publication`: Nature 2025

* `author`: Guo et al.
* `url`: https://www.nature.com/articles/s41586-025-09422-z

  - arxiv(2024.04): https://arxiv.org/pdf/2501.12948

arxiv꺼가 더 보기 편해서 arxiv로 리뷰하였음.

### Summary

본 논문에서는 기존의 LLM 학습 방식에 사용된 RL과 CoT의 한계를 지적하고, human-annotated data(human reasoning pathway)에서 벗어난 모델 학습 프레임워크를 제안한다.

이렇게 만든 모델을 `DeepSeek-R1` 이라 이름지음.



### Motivations

- 현재 LLM을 학습하는 RL 방식은 human-annotated reasoning 경로에 의존하여, scalability 문제와 cognitive biases 문제가 있음.

  - 모델은 사람이 아니므로, 사람처럼 reasoning하는 것이 오히려 모델 성능의 상한선을 가지게 됨.

- 이러한 한계를 극복하기 위해, `GRPO`를 적용하고 RL을 하기 전의 SFT phase를 생략.

  - hypothesis: human-defined reasoning patterns may limit model exploration, whereas unrestricted RL training can better incentivize the emergence of novel reasoning capabilities in LLMs

  - 하지만, 이 과정에서 readability나 language mixing 문제가 생겨 추가 과정을 거쳐 final model을 생성하였음.


### Contributions

- human-annotated data를 기반으로 한 RL 방식을 지적하고, 이 한계를 이겨내기 위한 모델 학습 프레임워크 제시

- 각 학습 단계에 따른 모델 성능 비교



### Core Idea (novelty) & Methodology

> **모델은 사람이 아니므로 human-like reasoning 경로를 따라가게 하는 것이 오히려 모델의 성능을 제한한다.**

<div align="center"><img src="https://github.com/user-attachments/assets/2a590724-dcaf-4292-a9ee-592d98206e24" width="60%"></img></div>


- **Stage 1** DeepSeek-R1-Zero: `DeepSeek-V3-Base`모델을 시작으로, SFT를 제외하고 GRPO만 진행

  - reward hacking을 막기 위해 Reasoning Prompt로 rule-base reward RL 진행: 1) Accuracy rewards(response가 math, code 등에서 정답인지), 2) Format rewards(reasoning process를 `<think>` 토큰 사이에, 정답을 `<answer>` 토큰 사이에 적었는지)
  
    - $Reward_{rule} = Reward_{acc} + Reward_{format}$

  - response 길이가 늘어났지만 reasoning quality는 좋지 않았음. (readability, language mixing 문제)

- **Stage 2** DeepSeek-R1-Dev-2: 

  - `DeepSeek-R1-Zero` 를 이용해 cold-start Long CoT data 생성해 SFT 진행 후(DeepSeek-R1-Dev-1), Reasoning Prompt로 Rule-based Reward + Language Consistency Reward 적용해 RL 진행
  
    - 안정적인 Long reasoning을 위해

- **Stage 3** DeepSeek-R1:

  - `DeepSeek-V3` 로 부터 Non-Reasoning data를, `DeepSeek-R1-Dev-2` 로 부터 Reasoning data를 Rejection Sampling해 만든 데이터로 SFT 진행(DeepSeek-R1-Dev-3)

  - human preference에 대해 align시키기 위한 RL 진행 (Rule-based + Model-based Reward)

  - $Reward = Reward_{rule} + Reward_{reward_model} + Reward_{format} + Reward_{language}$


### Results

- `SFT`를 제외하고 Rule-based reward를 이용한 RL만을 진행했을 때, 모델이 점점 response length를 길게 만들었으며, `aha moment`를 만들어 reasoning 패턴과 self-evolution process를 보여주었음.  --> thinking time이 길어짐. 하지만 quality는 불안정

- 모델 학습 프레임워크의 각 stage별 모델에 대한 성능을 볼 때, RL 뿐만 아니라 Reasoning을 위한 SFT 또한 중요한 것을 알 수 있음. 이러한 SFT가 기반이 되었을 때, RL을 통해 큰 성능 향상을 보였음.



### GRPO vs PPO


<div align="center"><img src="https://github.com/user-attachments/assets/748ae3fc-3e54-4e73-a727-17bc22b82df4" width="60%"></img></div>


1. PPO에서 adavantage는 value model을 사용하여 `GAE`를 통해 구하지만, GRPO에서는 old policy model의 responses를 이용해 구함.

2. KL divergence를 적용하는게 다름: GRPO는 loss에 직접적으로 적용하지만, PPO의 경우 per-token KL penalty를 적용함.

PPO보다 GRPO가 long response generation에 유리함.
