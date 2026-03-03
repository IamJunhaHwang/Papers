## :page_with_curl: Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)

### Info

* `publication`: Neurips2025 - Datasets and Benchmarks Trak

  - **Neurips2025 Best Paper**

* `author`: Jiang et al.
* `url`: https://openreview.net/pdf?id=saDOrrnNTz



### Summary

본 논문에서는 서로 다른 LLM이 다른 데이터에서 학습되었음에도 글자그대로 같은 response를 생성하는 점(mode collapse)을 지적하고, LLM의 diversity와 creativity를 평가하기 위한 benchmark를 제안한다.



### Motivations

- 현재의 LLM들은 generation을 다양하게, 조금 더 창의적으로 말하는 것에 어려움을 보이고 있다. 또한, 이러한 output diversity를 평가하는 방법(벤치마크)이 부족하다.



### Contributions

- 모델의 diversity 평가를 위한 real-world open-ended queries로 구성된 INFINITY-CHAT 와 이에 대한 taxonomy 제안

- 동일한 LM, 심지어 다른 LM들이 동일한 response를 내뱉는 현상을 발견 --> **Artificial Hivemind**



### Core Idea (novelty) & Methodology

- real-world open-ended queries로 구성된 INFINITY-CHAT을 만들어, inter-, intra- model evaluation 진행

- human preference labeled high-quality data에 대해 모델의 평가 관찰  --> 모델에게 high-quality response pair를 주었을 때, 모두 high-quality로 구분하는가? 아니면 그 중 하나를 높게 평가하는가? 


### Results

- Intra-model repetition: model의 decoding 방식을 high-stochasticity으로 바꾸어도 generation 간의 similarity가 높게 나온 것을 관찰 (min-p sampling 사용; top-p = 1.0, min-p = 0.1, temperature = 2.0)

- inter-model homogeneity: 서로 다른 LLM에서도 높은 similarity를 보였음. (글자 그대로 완전 겹치는 부분 존재)

- human-preference labeld data(ex. <query, response1, response2>)에서 사람조차도 호불호가 나뉠 때, LM judge나 Reward model이 human rating과 weaker alignment함을 찾음.

<img width="856" height="810" alt="image" src="https://github.com/user-attachments/assets/7b387f40-b9d9-4809-91c9-a655813ab96d" />

<img width="854" height="574" alt="image" src="https://github.com/user-attachments/assets/ddc681a1-b63a-43c4-bd0d-f4e51c51ebe3" />
