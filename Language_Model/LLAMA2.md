## :page_facing_up: Llama 2: Open Foundation and Fine-Tuned Chat Models

### Info

* `preprint`

* `author`: Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al **\[Meta\]**
* `url`: [paper link](https://arxiv.org/abs/2307.09288)

- `github` : [LLAMA 2](https://github.com/facebookresearch/llama)

- `github` : [LLAMA 2 - CHAT](https://github.com/facebookresearch/llama-recipes/)


### 1\. Abstract & Introduction

기존의 Open-Sourced LLM들은 Closed-Source LLM(ChatGPT, BARD 등)보다 낮은 성능을 보였으며 Closed LLM들을 재현하기에는 엄청난 비용이 들기 때문에 쉽지 않다.

따라서, 이 논문에서는 7B ~ 70B의 파라미터를 가지는 fine-tuned LLM인 LLAMA2, LLAMA-2-CHAT을 공개한다. 추가로, 개발 중에 보였던 것들(emergence of tool usage and temporal organization of knowledge 등)을 공유한다.

- `LLAMA 2` : LLAMA 1의 업데이트 버전으로 새로운 공공 데이터로 훈련하였으며 이는 40%정도 늘어난 양이다. 또한, context length를 2배로 늘렸고 `grouped-query attention (GQA)`을 적용했다.

- `LLAMA 2 - CHAT` : LLAMA 2를 dialogue use cases로 fine-tune한 버전

<br></br>

### 2. Pre-Training

- Pre-Training Data

  - 개인 정보 누출에 대한 위험이 있는 사이트에 대한 데이터를 지우려 노력함.

  - 2 trillion tokens로 훈련하였으며 사실에 대한 정보들을 담은 데이터(factual sources)를 up-sampling 하였음. **-> hallucination 완화, 지식 증가**

- Training Details

  - `LLAMA 1`의 구조, 토크나이저, 훈련 설정값들을 적용함.

  - 달라진 점은 GQA를 적용하고 context length를 늘린 것

    - 7B, 13B 모델의 경우 GQA 적용 X


<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/2fcbb78e-0606-48fa-be4b-04bbfda57f6f" width="80%"></img></div>

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/765a3599-2298-46a8-af90-8b268045501f" width="80%"></img></div>



- LLAMA 2 Pretrained Model Evaluation **[Table 3]**

  - Code, Commonsense Reasoning, World Knowledge, Reading Comprehension, Math, Popular Aggregated Benchmarks 에 대해 평가함.

  - LLAMA 2 가 LLAMA 1를 능가했으며 LLAMA 2 7B & 34B는 비슷한 크기의 MPT 모델보다 코드 벤치마크를 제외하면 성능이 좋았다. 또한, LLAMA 2 7B & 34B 는 비슷한 크기의 Falcon 모델보다 성능이 좋았다.

- Closed-source Model과도 비교해보았다. **[Table 4]**

  - MMLU와 GSM8K에서 LLAMA 2 70B는 GPT-3.5와 가까운 성능을 보였지만 코드 벤치마크에서는 차이가 컸다. 또, PALM과 비교했을 때는 비슷하거나 더 좋았지만 GPT-4와 PALM2-L과는 큰 차이가 있었다. 

    - World Knowledge : TriviaQA, Natural Questions

    - Math : GSM8K

    - Code : HumanEval

    - Popular Aggregated Benchmark : MMLU, Big Bench Hard

<br></br>

### 3. Fine-Tuning

이 파트에서는 `LLAMA 2 - CHAT`의 fine-tuning에 대해서 공유한다. (supervised fine-tuning 
,initial and iterative reward modeling, RLHF)

#### 3-1. Supervised Fine-Tuning (SFT)

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/53d703b8-88e0-4cd9-9bd9-01a40f1a99a5" width="80%"></img></div>

  - Third-party SFT 데이터에는 다양성과 품질이 부족하기 때문에 위의 Table 5와 같은 고품질의 SFT 데이터를 직접 수집하였다.
  
  - Third-party SFT보다 적지만 품질이 좋은 SFT 데이터를 이용하면 성능이 좋아지는 것을 확인했고 이는 이전 연구(Zhou et al. 2023)에서 보여 것과 비슷하다.

  - 우리는 이러한 SFT 데이터를 대략 수 만개 정도 사용하면 높은 품질의 결과를 달성하기에 충분하다는 것을 알아냈고 총 27,550개를 수집한 후 이 이상 수집하지 않았다.

#### 3-2. Reinforcement Learning with Human Feedback (RLHF)

`RLHF`는 fine-tuned LM을 사람의 선호와 지시를 따르는 행동을 하도록 모델을 더욱 align 시키는 훈련 과정이다. 우리는 사람이 직접 두 모델의 output 중 더 선호하는 것을 선택하게 하는 것으로 human preferences 데이터를 수집한 후 `reward model` 훈련에 사용하였다.

- Human Preference Data Collection

  - Binary comparision protocol 사용

  - annotator들이 prompt를 작성하고 두 가지 model responses 중 주어진 기준을 기반으로 하나를 선택하게 한다.

  - 다양성을 극대화하기 위해, 두 가지 responses들은 다른 2개의 model variants(temperature 변수가 다름)에서 뽑아낸 것이다.
  
  - 또, response 선택과 더불어서 해당 선택에 대한 degree를 레이블링하도록 했다. (선택한 답변이 다른 것에 비해 얼마나 좋은지; significantly better, better, slightly better, unsure)


- Reward Modeling

  - `Reward Model`은 model response와 해당 prompt(이전 턴의 context 포함)를 input으로 받아, model generation의 품질을 나타내는 점수를 output으로 준다. 이러한 점수를 보상으로 활용해 RLHF동안 LLAMA 2 -CHAT을 사람의 선호에 맞게 조정과 helpfulness와 safety를 향상시키도록 최적화할 수 있다.

  - helpfulness와 safety는 종종 상충되는 것을 다른 연구에서 찾아냈으므로 2개의 다른 reward model을 만들어 각각 helpfulness, safety에 최적화되게 한다.

    - `helpfulness`는 요청 정보를 전달하고 유저의 요청을 만족시키는 것이고, `safety`는 해당 답변이 안전하지 않은지 여부이다.

    - `ex. giving detailed instructions on making a bomb`에 대한 대답은 helpfulness 측면에서 높을 수 있지만 이렇게 되면 unsafe하게 된다.

  - 보상 모델을 사전 훈련된 chat model 체크포인트를 사용해, 두 모델 모두 사전 훈련에서 습득한 지식을 활용할 수 있게 한다. 이렇게 하면 두 모델 간의 정보 불일치를 막을 수 있다. (hallucination 방지)

  - 모델 구조와 하이퍼 파라미터는 Pretrained LM과 같으며, NTP를 위한 classification head를 점수 출력을 위한 regression head로 바꿨다.

  - Reward Model Results : 우리의 reward model이 가장 성능이 좋았음.

- Training Objectives

  - 사람이 선호하는 대답에 대해 반대의 것보다 더 높은 점수가 나오도록 한다.

  - <img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/111ac510-7080-4acb-9ac6-2e9ea7ac76ec" width="50%"></img>

    - $r_{\theta}(x, y)$ : prompt x와 completion y의 점수값

    - $y_c$, $y_r$ : preferred response, rejected response

    - $m(r)$ : margin; preference rating의 이산함수

      - 선호하는 정도도 이전에 데이터 수집시 레이블링 했었음


<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/f26ae805-0d41-49ed-9fe9-ac5a32ce3f7e" width="80%"></img></div>


- Iterative Fine-Tuning

  - 두 가지 메인 알고리즘으로 RLHF fine-tuning을 진행했다.

    - Proximal Policy Optimization (PPO) : standard RLHF

    - Rejection Sampling fine-tuning : model로 부터 K개의 output을 샘플링한 후 best reward 점수를 가지는 후보를 선택하여 gradient update를 위해 사용한다. 각 프롬프트에 대해 가장 높은 reward 점수를 가지는 샘플을 새로운 gold standard로 삼아, fine-tuning을 진행한다.

      - PPO는 하나의 샘플을 사용하지만 Rejection Sampling은 K개의 샘플을 사용함.

      - RLHF-V4 까지는 Rejection Sampling FT만 사용했고, 그 이후에는 두 가지를 섞어 사용했다. (다시 샘플링하기 전에 Rejection Sampling 위에 PPO 사용)

<br></br>

### 3.3 System Message for Multi-Turn Consistency

RLHF 모델에서 초기 지시 사항에 대해 몇 턴 이후에 잊어버리는 현상이 있는데, 이를 해결하기 위해 `Ghost Attention (GAtt)` 를 제안한다.

- GAtt method

  - 대화에서 모든 유저 메시지에 지시 사항을 삽입한다. 그 후, RLHF 모델을 이용해 해당 input에 대한 output을 샘플링한다.

  - 이렇게 만들어진 데이터셋에서, 첫 번째 턴에 있는 지시 사항을 제외하고는 모두 삭제한다.

  - 하지만, 위와 같이 만들게 되면 훈련 시간에서 메시지 간의 mismatch를 야기하므로 이전 턴들의 대한 모든 토큰의 loss를 0으로 설정한다.
  

### 3.4 Human Evaluation 

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/4eaa2538-2d08-46aa-979a-1a984b42e213" width="80%"></img></div>

사람 평가자에게 helpfulness 와 safety 에 대해 평가하도록 하였다.

`LLAMA 2 - CHAT`이 single turn, multi turn prompt 모두에서 open-source 모델을 능가하였다. 가장 큰 LLAMA 2 - CHAT 모델이 ChatGPT와 비등했다.
