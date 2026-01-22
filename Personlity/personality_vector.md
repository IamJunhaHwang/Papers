## :page_facing_up: Personality Vector: Modulating Personality of Large Language Models by Model Merging

### Info

* `publication`: EMNLP 2025 -  Long

* `author`: Sun et al.
* `url`: https://aclanthology.org/2025.emnlp-main.1253.pdf
* `github`: https://github.com/RSS-researcher/Personality_vector



### 1. Abstract & Introduction

최근 personalized AI의 수요가 늘어나면서, personality를 기반으로 LLM behavior 조정에 대한 연구가 진행되고 있다.   
Big 5와 같은 personality trait을 기반으로 model output을 aligning하는 것이 promising direction으로 떠올랐으며, 이전 연구들에서는 prompting, fine-tuning, activation intervention과 같은 방식으로 LLM의 personality induction을 진행하였다.

하지만, 이러한 연구들은 특정 personality type으로 모델을 조정할 수 있을 뿐, `continuous control` 하지는 못했다(모든 개개인은 특정 type과 intensity가 존재해 unique한 personality를 구성하므로 이는 한계로 볼 수 있음).   
따라서, 이 논문에서는 `model merging`을 기반으로 additional training 없이 llm을 원하는 personality trait으로 조정하는 방법을 제안한다.

`task vectors` 논문에서 영감을 받아, **personality vector** 를 계산(각 big5 trait으로 fine-tune한 model에서 pre-training model을 subtract)한 후 이 vector를 모델에 merging하는 것으로 원하는 personality를 모델에게 induce한다.

- 이 논문에서는 4가지 질문을 풀기 위한 실험을 진행했다:

  - merging하는 동안의 personality vector의 규모를 조정하는 것으로 personality trait intensity를 조정할 수 있는지

  - 모든 big 5 personality vector를 한번에 single model로 merging하는 것으로 multiple trait inducing이 가능한지

  - base model에서 personality vector를 subtract하는 것으로 반대 trait을 induce할 수 있는지

  - personality vector를 다른 domain(Role-playing agents, Korean Language Model, Vision LM)에 적용시킬 수 있는지

<br></br>

### 2. Method


#### 2-1. Personality vector

Big 5 traits를 기반으로 각 personality에 대해 pre-trained model $\theta_{pre} \in \mathcal{R}^d$ 를 fine-tune한다.   

각 personality conditon $p \in P = \{OPN_{high}, OPN_{low}, ..., NEU_{high}, NEU_{low}  \}$ 에 대해 fine-tuned model은 $\theta_p \in \mathcal{R}^d$ 파라미터를 가지고 있다.

personality vector는 다음과 같이 정의된다: $\phi_p = \theta_p - \theta_{pre}$

이렇게 얻어낸 personality vector를 target model $\theta \in \mathcal{R}^d$ 에 merge하는 것으로 원하는 personality trait를 모델이 반영할 수 있는지 실험한다.


#### 2-2. Model Merging

- Task Arithmetic: task vector를 더하고 뺌으로써 capabilities를 주입하거나 무효화

  - $\theta' = \theta_{base} + \alpha \phi$
  
    - $\alpha$ : optional scaling coefficient

- TIES-Merging
 
  - 여러 vector를 합칠 때, parameter interference를 막기위해 
  
    - minor parameter update를 0으로 만듦
    - task vector간의 방향을 맞춤 (같은 방향으로의 변화만 있게 만듦)

- DaRE: task vector를 희소화(sparsify)

  - `Bernoulli(p)`를 이용해 random mask 생성
  
  - 이를 이용해 sparsified vector 생성: $\tau^k = ((1-m_k) \cdot \tau_k) / (1-p)$


<br></br>

### 3. Experimental setting

- 아래 research questions를 다루기 위한 실험을 보일 것임

  - RQ1: model의 personality intensity는 personality vector scaling으로 조정할 수 있는가?

  - RQ2: multiple personality traits가 model에 merging될 수 있는가?

  - RQ3: personality vector를 빼는 것으로 반대 trait을 유도할 수 있는가?
  - RQ4: 다른 domains에 fine-tune된 모델에 personality trait을 주입할 수 있는가?

#### 3-1. Data, models, baselines

- Data: Big5-Chat dataset (big five를 기반으로 구축된 대화 데이터; 10개의 personality category로 되어 있으며 각각 10k example로, 총 100k)
  - 총 10개의 fine-tuned 모델 만듦

- models: Llama-3.1-8B-INSTRUCT, Qwen2.5-7B-INSTRUCT

  - 각 모델을 SFT를 이용해 fine-tune 모델 생성 (personality trait 부여에는 DPO보다 SFT가 좋다는 연구 기반)

  - 각 fine-tuned model에서 base model weights를 빼는 것으로 10개의 personality vector 얻음

  - 모든 실험은 temperature 0.6 을 이용해 5번 반복했음.

- baselines:

  - `Prompt`: prompt-based personality conditioning; 원하는 방향(high, low)에 대해 5개의 adjectives를 랜덤으로 선택한 후 intensity modifiers(very, a bit, or none)과 함께 사용. (Jiang et al., 2023)의 Personality Prompting(P2) 사용

  - `NPTI`: Neuron-level Personality Trait Intervention; 특정 trait과 관련된 neuron activations 조정

    - scaling gamma $\gamma \in [0.1, 2.0]$ 을 사용해 관련 뉴런을 증폭시키거나 억누름

#### 3-2. Evaluation: 2가지 task

- Big Five Inventory (BFI): 5가지 major personality traits를 평가하기 위한 44개 질문으로 구성

  - self-report 방식에서 LLM이 어려움을 겪기 때문에 interview-style format(Wang et al., 2024)을 적용
   
    - 모델이 자연스러운 대화에 대답하고, GPT API를 이용해 대답을 5-point scale로 평가

    - GPT-based 평가를 검증하기 위해 human judgements와 비교
   

- Linguistic feature

  - personality는 언어 사용에 눈에 띄게 반영되므로, personality 표현을 평가하기 위해 모델의 대답에서 linguistic feature를 분석

  - 각 모델은 다음 프롬프트를 instruction으로 받음: _"Tell me about yourself in 300 words"_

  - LIWC-22를 이용해 분석한 후, trait-specific linguistic feature를 만듦

<br></br>

### 4. Main Experiments

- personality scaling: $θ^α_p =θ_{base} + \alpha \phi_p$

  - trait intensity 조절 실험, α= [0.1, 2.0]

  - 10개의 personality vector에 대해 20개의 α에 따른 모델 생성 => 200개 모델

- Multi-Personality Composition: $θ^α_{multi} = θ_{base} + \sum_{p∈P} α \phi_p$

  - α의 합이 2.0이 넘어가면 instruction-following 성능이 떨어지는 것을 관찰하여, α= [0.1, 0.4]

  - 각 α에 대해 32개의 multi-personality model 생성 --> 128개 모델

  - parameter interference 확인을 위해 <TIES-Merging, task arithmetic with DaRE, and
TIES-Merging with DaRE> 적용

- Personality Negation: $θ^α_p =θ_{base} + \alpha \phi_p$ 에서 α = -1 로 설정

- Transferability : 다른 domain 모델로의 transfer

  - 예를 들어, 다음과 같이 fine-tuned model을 만든 후, base model에서 subtract해서 vector를 얻은 후: $\phi_{chl} = θ_{chl} − θ_{base}$

  - 이를 personality vector와 함께 base model에 mergning: $θ_{p,chl} = θ_{base} + α \phi_{chl} + βphi_{p}$

    - VLM의 경우 VLM 내의 `llm` 부분에만 적용

<br></br>

### 5. Results

BFI score를 기준으로 fine-tuned model들이 base model과 비교해서 눈에 띄는 personality 차이를 보였음.

아래의 모든 분석들은 `Llama3.1-8B-Instruct`를 기반으로 진행됨. (Qwen2.5는 appendix 참고)

#### 5-1.  RQ1: Scaling-Based Control of Personality Intensity


<div align="center"><img src="https://github.com/user-attachments/assets/0f93df74-a3f3-457e-81a3-5a46a93ce9d6" width="80%"></img></div>

위 그림은 scaling coefficients를 바꾸어 가며 base model에 personality vector를 merge한 결과이다.

- baseline에 비해 personality vector가 조금 더 세밀하게 trait을 바꿀 수 있었다. 

- 또한, coefficients가 1.0보다 클 때 fine-tuned model을 넘어가는 trait을 보일 수 있었다. 이는 personality vector 를 이용하면 interpolation과 extrapolation을 통해 연속적으로 peronality 조정이 가능함을 의미한다.

- linguistic feature에서도 차이가 나타났다: coefficient가 1.0인 low agreeableness와 2.0일 때 tone과 expression이 바뀌는 것을 확인했다.

##### 5-2. RQ2: Multi-Trait Composition

multiple personality trait merging이 가능한지 실험했으며, 평균 trait-score correlation이 0.58이 나왔다(single-vector의 경우 0.9).

<div align="center"><img src="https://github.com/user-attachments/assets/dc32db84-370a-4859-af2e-7661a176b439" width="40%"></img></div>

이렇게 조정되는 정도가 줄어든 이유를 알기 위해 personality vector간의 cosine similarity를 계산해보았다. 그 결과, 0.3 이상의 높은 유사도를 가지는 것을 확인했으며, 이는 parameter redundancy를 나타낸다.

이를 완화하기 위해, DaRE를 적용하였으며, 이를 적용했을 때 average correlation을 상당히 높여주었다.

#### 5-3. RQ3: Reversal via Vector Subtraction

<div align="center"><img src="https://github.com/user-attachments/assets/2b2ddaea-bc00-4d99-af75-6925b17f3ab0" width="40%"></img></div>

personality vector를 base model에서 뺐을 때의 결과이다.

high-trait vector를 뺐을 때, BFI score가 줄었으며, low-trait vector의 경우 BFI score를 늘렸다.

하지만, 이렇게 vector를 뺄 때, generate response에서 disclaimers("As an AI, I do not have feelings") 뱉는 현상을 관찰했다. 이는, 위와 같은 vector를 빼는 것이 personality trait 뿐만 아니라 general ability까지 영향을 주는 것을 의미한다.

위의 코사인 유사도에서도 vector들이 넓은 범위의 latent subspace를 공유하고 있었음으로, 여기에는 target trait 뿐만 아닌, common conversational structures affective expressions등을 포함하고 있을 것이다. 

#### 5-4.  RQ4: Cross-Domain Transferability of Personality Vectors

personality vector를 role-playing agents에 적용할 수 있는지 보았으며, 위 그림은 그 결과이다. personality vector를 character-specific model에 merging하는 것으로 personality trait을 control할 수 있었다.

또한, 다른 domain 실험(cross-lingual transfer, cross-modal transfer to VLM)에서도 동일하게 trait을 잘 control하였다.

#### 5-5.  Personality Steering Analysis

<div align="center"><img src="https://github.com/user-attachments/assets/0e96ca28-8a12-47c1-9bb9-4a24174d019f" width="80%"></img></div>

personality vector가 personality trait을 어떻게 바꾸는지 알아보기 위해 layer 별로 hidden representations가 어떻게 바뀌는지 분석하였다. (base model과 fine-tuned model 간의 코사인 유사도)

결과, 코사인 유사도는 점점 낮아지다가 layer가 깊어질수록 좀더 크고 다양하게 유사도 떨어지는 것을 볼 수 있다. 이는 layer가 깊어질수록 trait-specific feature들이 늘어나는 것을 의미한다. (이전 연구들에서도 보인 현상과 일치함: LLM은 마지막 layer일수록 좀더 abstract하고 subjective concepts를 담고있음)
