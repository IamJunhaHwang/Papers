## :page_facing_up: SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling

### Info

* `publication`:  NAACL 2024 - Industry Track

* `author`: Dahyun Kim et al.
* `url`: https://arxiv.org/pdf/2312.15166

### 1\. Abstract & Introduction

LLM의 등장으로 NLP 분야가 크게 바뀌었으며, performance scaling law에 따라 점점 더 큰 모델을 학습하고 있다.   
이를 효율적으로 하기 위해, 최근 mixture of experts(MoE) 가 제안되었지만 training과 inference framework에 변화가 필요하여 applicability에 문제가 존재한다.

따라서, 본 논문에서는 base model의 layer 수를 늘리고 continual pre-training을 진행하는 `depth up-scaling(DUS)` 를 제안한다. 이 방법은 MoE같이 추가적인 module이나 dynamism이 필요하지 않으며 HuggingFace와 같은 LLM framework에서 쉽게 사용할 수 있다. 또한, 모든 transformer 구조에 적용이 가능하다.

DUS를 사용하여 만든 모델인 SOLAR 10.7B를 공개하였으며 이 모델은 Llama2와 Mistral 7B를 능가하였다.

- Contribution

  - 효율적이고 효과적으로 small LLM으로부터 모델을 scaling하는 방법인 depth up-scaling(DUS)을 제안

  - DUS를 이용해 만든 모델인 SOLAR 10.7B 모델을 공개

  - 추가로, 복잡한 지시를 지켜야만하는 task를 위해 fine-tuning된 SOLAR 10.7B-Instruct 개발

<br></br>

### 2. Depth Up-Scaling

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/1c4b1164-8a3e-4f10-a865-14a82c068cd2" width="80%"></img></div>

효율적으로 LLM의 크기를 늘리기 위해, base model의 pre-trained weights를 사용하며, 이후 continual pre-training 과정을 거친다.

- base model : 32-layer Llama 2 (어떤 n-layer transformer 구조든 가능함)

  - 해당 모델을 `mistral 7B`의 pretrained weights로 초기화한다.

- Depthwise scaling : `n`개의 layer를 가지는 base model과 target layer `s` 를 선택했다고 하자.

  1. `n`개의 layer를 가지는 base model을 복사한다. [base model이 2개가 됨]

  2. original model의 마지막 `m`개의 layer를 삭제하고, 복제된 model의 처음 `m`개의 layer를 삭제한다. [각각 n - m 개의 layer가 남게됨]

  3. 두 모델을 concatenate함. [ s = 2*(n-m) ]

      - 논문의 세팅은 다음과 같음 : `n=32, s=48, m=8`

- Continued pretraining : depthwise scaled model은 처음에 성능이 base model보다 떨어지게 되므로 `continued pretraining` 과정을 적용했다. 실험적으로 이를 통해 빠르게 성능이 회복하는 것을 관찰했다.

  - depthwise scaling에서 `m`개의 layer를 제거하는 것이 아니라 단순히 모델을 복사하여 그대로 붙여 크기를 늘릴 수도 있을 것이다. 하지만, 이와 같은 방식은 모델을 붙이는 접합선부분에서의 maximum layer distance를 유발하게 되고 continued pretraining으로 빠르게 성능을 회복하기 어려울 수 있다.

  - 따라서, `m`개의 layer를 제거하는 것이 빠르게 성능을 회복하는 것에 기여한다.
  
<br></br>

### 3. Training Details 

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/f4cfec65-b487-4817-b58b-92990080ab61" width="80%"></img></div>


continued pretraining 과정을 포함한 DUS가 끝난 후, SOLAR 10.7B를 2단계로 fine-tuning 하였다. 1) instruction tuning 2) alignment tuning

- Instruction Tuning : QA format에서의 instruction을 따르도록 학습함.

  - open-source dataset을 사용하였으며, 수학에 대한 능력을 높이기 위해 math QA dataset 또한 합쳐서 사용하였다.

  - dataset을 만들 때, seed math data를 수집한 후, GSM8K와 같은 benchmark와의 중복을 피하기 위해 MetaMath의 경우와 같이 seed math data의 QA를 rephrase하였다. (이를 `Synth. Math-Instruct`라 명명함)

- Alignment tuning : sDPO를 사용해 instruction-tuned model을 사람이나 strong AI와 더 가깝도록 further fine-tuned 함.

  - instruction tuning 단계와 같이 alignment tuning에도 math-focused dataset을 합쳐서 사용하였다.

  - rephrase한 수학에 대한 능력을 높일 수 있었으므로(Result section에서 설명), rephrased QA가 original 보다 좋다고 판단하여 rephrased question을 prompt로 rephrased answer를 chosen response, original answer를 rejected response로 사용하여 DPO tuple을 구성하였다. {prompt, chosen, rejected}

  - 위와 같이 만든 데이터셋을 `Synth. Math-Alignment` 라고 명명함

<br></br>

### 4. Results

#### 4-1. Experimental Details

- Training datasets : instruction datasets를 Alpaca-style chat template로 바꾸었으며, OpenOrca와 같은 데이터 셋의 경우 benchmark 데이터셋과의 중복을 피하기 위해 이를 filtering하였다. Alignment dataset의 경우 [Zephyr](https://arxiv.org/abs/2310.16944) 와 같이 전처리하였다.

- Evaluation : HuggingFace Open LLM Leaderboard의 6가지 benchmark datasets를 사용해 평가하였다.

  - ARC, HellaSWAG, MMLU, TruthfulQA, Winogrande, GSM8K

#### 4-2. Main Results

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/e0b06f68-131b-4a65-8151-454893545c4e" width="80%"></img></div>

위 표는 SOLAR 10.7B와 SOLAR 10.7B-Instruct 모델을 다른 top-performing 모델들과 비교한 것이다.   
SOLAR 10.7B는 비슷한 크기의 다른 LLM보다 좋은 성능을 보였으며, SOLAR 10.7B-Instruct 모델은 evaluation dataset의 평균으로 보았을 때 SOTA를 달성하였다.

이는 DUS가 효과적으로 base LLM을 스케일 업 시킬 수 있음을 의미한다.

#### 4-3. Ablation Studies

Instruction tuning과 Alignment tuning 단계 모두에서의 Ablation studies를 진행하였다.

##### Instruction Tuning

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/5bef9540-fa2d-446c-8efd-d71431b97407" width="80%"></img></div>

- Ablation on the training datasets : 다른 training datasets를 사용하는 ablation studies를 진행하였다.

  - Alpaca-GPT4 데이터만 사용한 `SFT v1`보다 여기에 OpenOrca 데이터를 추가한 `SFT v2`가 GSM8K 데이터 셋을 제외하고는 안좋은 성능을 보였다.

  - 또한, Synth. Math-Instruct 데이터가 도움이 되는지 확인하기 위해 `SFT v1`과 `SFT v2` 데이터에 추가하여 `SFT v3`와 `SFT v4`를 구성하였다.  두 모델 모두 GSM8K benchmark에서 큰 성능 향상을 보였으며, OpenOrca 데이터를 사용하지 않은 `SFTv4`가 전체적으로 점수가 더 좋았다. 

  - OpenOrca 데이터를 사용한 모델과 사용하지 않은 모델을 단순히 가중치에 평균을 취해 merging한 `SFT v3 + v4`을 구성하였는데, GSM8K에서 가장 좋은 성능을 보이면서 다른 benchmark에서도 높은 성능을 유지하였다.

    - 따라서, 각 task에서 좋은 성능을 보이는 모델들을 merging하는 것은 일반적으로 좋은 성능을 보이는 모델을 만드는 유망한 방법이 될 수 있음을 의미한다.

##### Alignment Tuning

alignment tuning을 위해 sDPO를 이용했기 때문에 다양한 ablation studies가 있다 : 1) 다른 training datasets를 사용, 2) sDPO 초기화를 위해 다른 SFT base model 사용, 3) final alignment-tuned 모델을 얻기 위한 방법으로 다른 merging 전략 사용

<img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/f7703373-fb5c-4111-b11c-67412d3a2af7" width="80%"></img>

- Ablation on the training datasets : 다른 training datasets를 사용

  - `SFT v3` 모델을 DPO의 base model로 사용하였다.

  - `DPO v1`는 Ultrafeedback Clean 데이터만 사용한 것이고, `DPO v2`는 이와 함께 Synth. Math-Alignment를 추가한 것이다. `DPO v1`의 경우 GSM8K를 제외하고 대체로 좋은 성능을 보였지만, GSM8K에서 base model보다 성능이 떨어졌으며 `DPO v2`에서는 GSM8K의 성능이 base model보다 낮지만 `DPO v1`보다 좋아진 것을 확인했다.

  - 그리고 `DPO v1` 와 `DPO v2`을 단순히 가중치에 평균을 취해 merging한 `DPO v1 + DPO v2`를 구성해보았는데, `DPO v2`보다 낮은 성능을 보였다.

    - 이는 SFT때와 다르게 `DPO v2`가 `DPO v1`에서 좀 더 strict하게 향상시킨 버전이기 때문이다. (이 부분은 SFT와 똑같은 방식인 것 같은데, 이유가 합당하진 않은 듯 함)

<img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/08b0beab-8ea0-4ed3-ac1b-7798439d4b6e" width="80%"></img>

- Ablation on the SFT base models : `DPO v2`의 데이터를 이용하였으며, 다른 SFT base model을 사용한 ablation study를 진행

  - base model로 `SFT v3` 와 `SFT v3+v4`를 이용하였는데, 실험 결과 만들어진 두 모델에서의 성능 차이는 크지 않았다.

  - 이는 SFT model의 성능 차이가 alignment-tuned model까지 이어지는 것은 아니라고 볼 수 있다.
