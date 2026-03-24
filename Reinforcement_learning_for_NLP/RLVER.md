## :page_with_curl: RLVER: REINFORCEMENT LEARNING WITH VERIFIABLE EMOTION REWARDS FOR EMPATHETIC AGENTS

### Info

* `publication`: ICLR 2026

* `author`: Wang et al.
* `url`: https://openreview.net/forum?id=P7wBg0vPTh   

### Summary

본 논문은 감정과 같은 도메인에서 RLVR의 verifiable reward 설계가 어려운 문제를 지적하며, 이를 해결하기 위해 human emotion을 모방하는 agent를 reward에 사용하였다.



### Motivations

- human-like emotional response와 reasoning을 모방하는 Sentient Agent를 Verifiable reward로 사용해 RL training과정에 집어넣자.



### Contributions

- RLVR에서 emotion을 위한 verifiable reward 설계 방법 제안

- 제안한 방법에 효과적으로 공감능력을 높이면서도, 기존의 능력(mathematics, code generation)을 유지시킴을 실험으로 보임

<div align="center"><img src="https://github.com/user-attachments/assets/be13a9fe-0b9a-4a63-86e3-09d23974dc5b" width="60%"></img></div>

### Core Idea (novelty) & Methodology

- 사람과 같은 emotional response를 mimicking하는 LLM agent인 SAGE framework를 사용한다.

  - initial query가 input으로 들어오면, 해당 input에 대해 emotion score (0 \~ 100)를 평가하고, 이와 함께 persona, conversational goal을 고려해 reply를 생성한다.

  - 이 reply에 대해 학습 모델이 다시 input을 생성하고 agent가 emotion score를 평가하는 방식을 T번까지 반복하거나 emotion score가 0보다 작아지면 종료된다.

  - T번의 턴을 지나서 나온 마지막 emotion score $e_T$ 에 대해 normalize한 값을 reward로 사용한다. $r_\phi(h_T) = \frac{e_T}{100}$

    - $h_T = \{ x_0, y_0, ..., y_T, x_T \}$

- RL 학습에는 PPO와 GRPO를 이용한다.

  
 

### Results

<div align="center"><img src="https://github.com/user-attachments/assets/9ef785d2-2ab5-4fb6-b326-c4e389714220" width="60%"></img></div>

- RLVER-trained model이 baseline(Qwen2.5-7B-Instruct) 보다 훨씬 좋은 성능을 보였으며, `Thinking step`을 넣는 것이 높은 성능을 보였다.

- PPO는 성능에 대한 높은 천장을 찍었고, GRPO는 training stability가 좋았다.


- RLVER 학습은 general capabilities의 영향은 거의 없었다.

  - 학습 전인 Qwen2.5-7B에 비교했을 때, 성능이 오르거나 변화가 거의 없었음.
