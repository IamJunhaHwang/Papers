## :page_facing_up: Gradient-Based Language Model Red Teaming

### Info

* `publication`: EACL 2024 - Long paper
* `author`: Nevan Wichers et al.
* `url`: https://aclanthology.org/2024.eacl-long.175/

### 1\. Abstract & Introduction

Generative LM들이 여러 task에서 SOTA를 달성했지만, 혐오 발언이나 해로운 정보를 주는 등의 답변을 내놓을 수도 있다. Red Teaming은 사람이 악의적으로 unsafe한 답변을 하게 만든 prompt를 찾아내는 것을 목표로하며 이러한 prompt를 red teaming prompts라 부른.

Red teaming은 노동 집약적이므로 규모와 다양성이 제한될 수 있다. 따라서 이를 위한 자동화 기술이 필요하게 되었고 본 논문에서는 LM이 unsafe한 답변을 생성하게 하는 다양한 프롬프트를 자동으로 만들어주는 **Gradient-Based Red Teaming (GBRT)** 를 제안한다.   
GBRT에는 학습 가능한 프롬프트가 LM(언어 모델)에 입력으로 주어지며, 이 LM이 red teaming의 대상이되며 프롬프트에 대한 응답이 생성된다. 그런 다음, classifier가 응답의 safety를 측정하고, safety score를 최소화하기 위해 역전파를 이용해 프롬프트를 업데이트한다.

우리의 방법은 safety classifier로부터 gradient 접근할 수 있는 이점이 있으며, 이 gradient 정보를 이용하는 것이 safety score만을 이용하는 것보다 더 이득이 있다는 것을 증명했다. 

Automatic red teaming은 보통 사람이 유해한 답변을 하도록 하는 프롬프트를 입력하기 때문에 realistic red teaming prompts를 만들어야 한다. 이를 위해 (1) 사전 훈련된 모델의 logits와의 차이를 패널티로 삼는 realism loss를 추가하고 (2) 학습 가능한 프롬프트를 학습하는 대신 별도의 LM을 fine-tune하여 red teaming prompt를 생성하게하는 실험을 했다.

<br></br>

### 2. Gradient-Based Red Teaming (GBRT)

- notation

  - $X$ : prompt probabilities (몇몇 token probabilities의 concatenation)

  - $p_{LM}$ : 모델이 가지고 있는 토큰의 확률 분포

  - $y$ : autoregressive 방법으로 LM이 생성한 응답

  - $p_{safe}$ : safety classifier
    - output response `y`에만 단독으로 적용할 수도 있고, input prompt와 output response의 concatenation `(X, y)` 에 적용할 수도 있음.
    - response가 안전한지를 확률로 반환해준다(이를 loss에 바로 사용함; 이를 줄이도록).
    - 여기에서 prompt를 업데이트하기 위해 gradient를 역전파한다(LM과 classifier는 frozen시킴).

Autoregressive sampling은 카테고리 분포로부터 뽑아내기 때문에 미분이 불가능하므로, `Gumbel softmax trick` 을 사용해 미분 가능하도록 근사시킨다. 각 decoding step에서 Gumbel softmax 분포를 사용한 model output logits로부터 샘플링한다. 이후, 이 결과 값을 입력으로 하여 다음 decoding step을 진행한다. 

prompt token들의 `learnable` 카테고리 분포에서부터 샘플링하기 위해, 여기에서도 Gumbel softmax trick을 사용한다(prompt distribution $X$ 에서 샘플링 후 나온 결과를 모델에 입력). 이 때, prompt distribution은 uniform 분포로 초기화되며 training 동안 업데이트된다.

- Gumbel softmax trick : 확률을 input으로 받아, vocab 안의 각각의 항목에 대한 weights를 output으로 줌(vocab에 대한 softmax).

  - 보통 하나의 토큰으로 확률 값이 집중된다.

  - Gumbel softmax의 output을 soft prompt라 부른다(vocab의 각 항목들의 weight로 표현되기 때문에).

    - soft prompt : $\tilde X = G(X)$
    - soft response : $\tilde y = G(p_{LM}(\tilde X)) = G(p_{LM}(G(X)))$
  
    - $P_{LM}$ 은 response logits를 출력하는 LM decoding이며, soft prompt를 각 임베딩 항목에 대해 가중치를 주기 위해 LM에 넣는다.

##### Architecture

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/155389ac-7c0e-4602-b60a-7922e19a21e0" width="70%"></img></div>



- GBRT의 구조는 위 그림과 같다. training procedure는 soft prompt 확률 `x`에 대해 다음의 loss function을 최소화 시키는 것이다

  - $L = p_{safe}(\tilde x, \tilde y)$ 

    - $p_{safe}$ : model response가 안전한지를 확률로 출력하는 safety classifier 

    - classifier는  각 토큰에 대한 가중치와 함께 soft model response 또한 받는다.

- GBRT-ResponseOnly

  - safety classifier는 prompt를 context로 활용하여 대답의 안전성을 판단하는데, prompt 자체가 안전하지 않아 대답이 safe해도 `unsafe`라고 분류하는 경우가 있다. 이를 완화하기 위해 다음의 loss를 최적화하도록 하는 GBRT-ResponseOnly를 제안한다.

    - $L = p_{safe}(\tilde y)$

    - prompt를 context로 사용하지 않음

##### LM realism loss

좀 더 실용적인 prompt들을 찾아내기 위해, prompt distribution과 pre-trained LM간의 divergence penalize하는 추가적인 realism loss regularization term을 도입한다. LM은 각각의 이전 프롬프트 토큰에 대해 다음에 올 확률이 제일 높은 프롬프트 토큰을 예측한다.   

- 여기에 다음과 같은 loss term 사용 : $-\sigma(X) * X'$

  - $\sigma$ : softmax, `X` : prompt token probabilities, `X'` : 이전 프롬프트 토큰들이 주어졌을 때 예측한 프롬프트 토큰의 log probabilities

##### Model-based prompts

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/4d74df93-c2a4-44ef-b6c2-f780346d6801" width="60%"></img></div>

프롬프트를 만들기 위해 GBRT로 fine-tune한 모델을 사용해 실험해보았다(제안한 방법을 이용하면 일종의 cycle이 가능하니 이를 실험한 것인듯).

여기에서도 위와 같은 loss term과 setup을 사용했다. 그리고 prompt probabilities를 직접 업데이트하는 대신에 prompt model의 가중치를 업데이트했다. 또한, L2 정규화를 사용해 model weights가 pretrained model과 많이 멀어지지 않도록 하였다.

<br></br>

### 3. Experiment Setup

 GBRT를 위해 LM, safety classifier로 `LaMDA model(2B)` 을 사용했다. 비교를 위해 각 method당 200개의 red teaming prompts를 얻어냈다. 훈련 시간 동안 6개의 input prompt tokens를 사용하며 모델은 4개의 response tokens를 그리디 디코딩한다. 또한, 제안한 방법의 경우 다른 random seeds에서 200번 실행시켰다. 

`GBRT-Finetune`의 경우 model을 한 번만 훈련 시킨후 200개의 각기 다른 prompt를 만들게 하였다. 이후, 가장 좋은 결과를 내는 하나를 선택해 사용했다.

각 방법들을 평가하기 위해, prompts를 LaMDA 모델에 넣어 response를 생성하게 하였으며, response는 15 토큰들로 이루어졌다. 

- 논문에서 새롭게 제시하였으며 실험에 사용한 방법들

  - GBRT : safety classifier가 prompt와 response를 둘 다 취함.

  - GBRT-RealismLoss : GBRT method에 realism loss 또한 optimize됨.

  - GBRT-ResponseOnly : safety classifier가 response만을 취함.

  - GBRT-Finetune : GBRT method에서 LM이 prompt를 생성하기 위해 fine-tuned 되었음.

- Baselines : RL Red teaming, Bot Adversarial Dialogue dataset (BAD)

  - RL Red teaming : unsafe response를 주기 위해 RL을 사용해 훈련

  - BAD : 미국의 English-speaking annotators가 toxic response를 주기 위해 진행한 multi turn dialogue 모음
    - 해당 데이터 셋 내의 대화들에서 첫 번째 턴만을 프롬프트로 사용

- metrics

  - safety classifier : 크라우드 워커로부터 측정된 safety 점수로 훈련된 8B LaMDA model, unsafe score가 `0.9` 를 넘어가면 unsafe한 문장으로 판단

    - prompt를 포함했을 때와 포함하지 않았을 때를 나누어 평가

  - $E{f_{toxic}(y)}$ : Perspective API를 이용해 toxicity 측정, 0.5를 기준으로 함

  - self BLEU : 각 프롬프트의 BLEU 점수의 평균, 낮을수록 diverse한 prompt임을 의미

<br></br>

### 4. Results & Analysis

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/28b60db9-031d-4c07-ad23-c4f58f07a068" width="70%"></img></div>

위의 표는 각 방법들이 safety classifier를 trigger시키는 것에 성공한 비율을 나타낸다.

GBRT와 GBRT-RealismLoss 방법은 $f_{unsafe}(x, y)$ 에서 red teaming prompts를 잘 찾아냈다고 나오는데, 이는 해당 평가 classifier가 prompt와 response 모두를 받는 것으로 training되어서 잘 찾아낸 것으로 보인다. 이와 같이 GBRT-ResponseOnly 방법 또한 $f_{unsafe}(y)$ 에서 잘 찾아냈다고 나온다.

`GBRT-RealismLoss`가 red teaming prompts를 가장 잘 찾아내었으며, vanilla GBRT와 RL Red Team 은 그러지 못했다. 또, BAD dataset은 model을 성공적으로 trigger시키지 못했다. 

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/d5d6c1ce-e5d1-4b88-a7c0-ffb31b14775d" width="70%"></img></div>

DPO를 이용해 safe response만을 출력하도록한 model을 만든 후, 제안한 방법을 적용해 unsafe response를 주는 prompt를 찾아낼 수 있는지 실험하여 그 결과를 `Table 4`에 기록하였다. GBRT 방법이 unsafe response를 출력하는 몇 개의 prompt를 찾아낼 수 있었다.   
또한, prompt와 response length에 따른 변화를 보기 위한 실험을 `Table 5`에 기록하였다. prompt와 response의 길이가 길어질수록 red teaming prompts를 더 잘 찾아냈다.
