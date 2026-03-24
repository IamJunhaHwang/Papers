## :page_with_curl: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

### Info

* `publication`: arxiv 2024

* `author`: Shao et al.
* `url`: https://arxiv.org/pdf/2402.03300         

### Summary

본 논문은 Open Math LRM이 필요성을 언급하며, `Deepseek-Math` 모델 개발과 이를 위한 데이터 수집 과정을 제안한다.   
추가로, 학습 과정에서의 RL로 `Group Relative Policy Optimization (GRPO)` 를 제안하고, 다른 여러 방법들(PPO, DPO, Rejection Sampling Fine-Tuning)을 이해하기 위한 unified paradigm을 제공한다.

`Deepseek-Math-7B` closed source model과 open model에 비해 좋은 성능을 보였으며, math reasoning을 위한 RL과정에서 language understanding과 reasoning 성능도 올라갔다.



### Motivations

- mathematical reasoning이 중요함에도, SOTA 모델들은 모두 closed-source 였음. (open-source model의 부재)



### Contributions

- Open math LRM인 `Deepseek-Math` 모델의 학습 방법과 이를 위한 데이터 수집 방법 제안

- GRPO 제안

- 여러 방법들에 대한 unified paradigm, extensive experiments 결과 제공(online vs offline training, etc)



### Core Idea (novelty) & Methodology

- Math reasoning을 위한 새로운 High-Quality dataset 제안

  - high-quality math web-text corpus인 `OpenWebMath`를 seed로 이용해 FastText 모델을 학습시키고, Commom Crawl로부터 math-related web pages 추출 --> deduplication --> Top-ranking score(by FastText) samples preserved --> domain명에 math가 들어간 것이나 1차 때 주로 수집된 도메인들 collect(ex. mathoverflow.net) 해서 seed에 추가 후 반복

- PPO의 value model을 없애고, KL penalty를 loss에 직접적으로 넣는 GRPO 방식 제안

  - $J_{\mathrm{GRPO}}(\theta)=\mathbb{E}[{q\sim P(Q),\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(O\mid q)}]$
  - $\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\!\left(\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}\hat{A}_{i,t},\;\mathrm{clip}\!\left(\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})},1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right)-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$

    - $\varepsilon , \beta$ : hyper-parameters
    - $\hat{A}$ : Advantage

- 이에 따라, adavantage 계산도 바뀜: old policy model로부터 group of outputs를 샘플링하여 각 output에 대해 reward를 계산하고, nomalization

  - Outcome Supervision 방식: 각 output 끝에 대해 reward 생성: $\hat{A}_{i,t} = \widetilde{r_i} = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}$

  - Process Supervision 방식: 각 output 내의 reasoning step 당 reward 생성: $R=\{\{r_{\mathrm{index}(1)_1},\cdots,r_{\mathrm{index}(K_1)_1}\},\cdots,\{r_{\mathrm{index}(1)_G},\cdots,r_{\mathrm{index}(K_G)_G}\}\}$ ,where `𝑖𝑛𝑑𝑒𝑥(𝑗)` is the end token index of the 𝑗-th step, and `𝐾𝑖` is the total number of steps in the 𝑖-th output.
  
    - $\hat{A}_{i,t} =  \sum_{\mathrm{index}(j) \ge t} \widetilde{r_i}^{\mathrm{index}(j)}$

    - $\widetilde{r_i}^{\mathrm{index}(j)} = \frac{r_{\mathrm{index}(j)_i} - \mathrm{mean}(R)}{\mathrm{std}(R)}$

  

  
 

### Results

- DeepSeekMath Corpus는 다른 오픈 소스 데이터보다 많은 크기의 high quality 데이터를 포함함을 보임. (각 corpus에 대해 1.3B 짜리 DeepSeek LLMs 모델로 학습시켜 benchmark에서 성능 비교)

- CoT format을 이용해 SFT Data를 구성하는 것으로 reasoning 성능을 올릴 수 있었으며, GRPO의 효과 또한 확인

<div align="center"><img src="https://github.com/user-attachments/assets/7533d6e6-a81a-426e-ad70-50985211b61e" width="40%"></img></div>

- Code Training이 Mathematical Reasoning에 도움이 되는 것을 확인

<div align="center"><img src="https://github.com/user-attachments/assets/f7efaa06-eb39-4e18-aa3b-9ef15fe44632" width="55%"></img></div>

- arxiv Paper data는 mathematical reasoning에 효과적이지 않음. (성능 향상이 거의 없거나 떨어짐)

- Online Rejection Sampling Fine-Tuning (Online RFT)가 일반 RFT보다 좋은 성능을 보였으며, GRPO+ process supervision이 가장 좋았다.

  - 초기 단계에서는 Online RFT와 RFT가 비슷하더라도 후반에는 엄청난 차이를 보임. (diversity 증가?)

  - Online RFT는 GRPO와 다르게 incorrect response를 penalize하지 못하고 모든 정답에 대한 response만 균일하게 강화함. (GRPO는 reward에 따라 penalization 규모가 다름)

<div align="center"><img src="https://github.com/user-attachments/assets/e434279a-89dc-44c6-81fb-7ae15bcf88a0" width="55%"></img></div>


#### Why RL Works?


<div align="center"><img src="https://github.com/user-attachments/assets/e62c35ad-a5ff-40f7-8f04-32f62f2f6e0f" width="55%"></img></div>



RL은 Maj@K 성능은 높였지만, Pass@K는 그러지 못했음. --> 근본적인 능력을 향상시켰다기 보단 TopK로 얻어지는 정답을 boosting (알고 있는 지식 안에서 정답을 확실하게 찾도록 함)

#### How to Achieve More Effective RL?

- Data Source: instruction tuning data의 question을 기반으로 RL을 진행했지만, Out-Of-Distribution questions를 advanced sampling 전략을 함께 사용하면 exploration 효과가 올라갈 것

- Algorithms: data 내의 노이즈 때문에 reward signal을 fully trust하기 힘듦 --> Weak-To-Strong alignment methods가 근본적인 변화를 이끌 것 (Burns et al., 2023)

- Reward Function: 

  1. Reward model의 일반화 성능을 어떻게 향상시킬 것인가? OOD question 과 advanced decoding outputs를 다루기 위해 효과적으로 일반화 되어야 함

  2. Reward model의 reward 불확실성을 어떻게 반영할 것인가? 이러한 불확실성을 수치화할 수 있다면? (Weak-To-Strong alignment)

  3. High-Quality process reward model을 어떻게 만들 것인가?
