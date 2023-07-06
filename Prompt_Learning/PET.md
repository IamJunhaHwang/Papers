## :page_facing_up: Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference


### Info

* `conf`: EACL 2021

* `author`: Timo Schick, Hinrich Schutze

* `url`: https://arxiv.org/pdf/2001.07676v3.pdf

* `github`: https://github.com/timoschick/pet


### 1. Abstract & Introduction

몇몇 NLP task 들은 task의 설명과 함께 사전 학습 언어 모델을 제공함으로써 비지도 방식으로 task를 수행할 수 있었다. [GPT]   

이 방식은 zero-shot 시나리오에서만 시도되었으며 이는 지도 방식보다 성능이 떨어진다. 따라서, 본 논문에서는 지도 방식과 비지도 방식을 조합할 수 있음을 보인다. (few-shot setting의 지도학습) 이 방법은 `Pattern Exploiting Training (PET)`라고 하며 input examples를 cloze-style(빈칸 맞추기)로 바꾸어 언어 모델이 task를 잘 이해하도록 돕는 것이다. 이렇게 바꾼 문장들은 unlabeled examples에 soft labels를 할당하는 것에 사용되며 앞선 과정을 모두 마친 결과 training set에 표준 지도 학습을 적용한다.

몇몇 task와 언어에서, PET는 낮은 리소스 환경에서 지도 학습 및 준지도 학습 결과를 큰 차이로 능가했다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/6665c9c5-c5b2-40b8-8eb0-b21f5d411700" width="40%"></img></div>

PET는 3가지 step으로 구성된다.

1. 각 패턴에 대해 서로 다른 PLM이 작은 training set(labeled set) $\mathcal{T}$ 에 대해 fine-tune 된다.

2. 모든 모델의 앙상블이 large unlabeld set $\mathcal{D}$ 를 soft label로 레이블링 한다.

3. 일반적인 classifier가 이 soft-labeled dataset으로 학습된다.

<br></br>

### 2. Pattern-Exploiting Training

- 기호 정의

  - `M`: Masked Language model
  - `V`: Vocabulary
  - $\mathcal{L}$ : label set for our target classification task
  - $x = (s_1, ... , s_k), s_i \in V^*$ : input sequence
  - `P`: Pattern; x를 입력으로 받아 하나의 mask token을 포함하는 phrase나 sentence를 만드는 함수
  - $v$ : $\mathcal{L}$ -> V : verbalizer; 각 label을 M의 Vocab으로부터의 단어로 매핑하는 함수
  - $(P, v)$ : pattern-verbalizer pair(PVP) 
  
- `PVP (P, v)`를 사용하면 다음과 같이 task A를 풀 수 있다.

  - 입력 x에 P를 적용해 input representation `P(x)`를 얻고 M을 이용해 label y를 결정한다. 여기서, `v(y)` 는 mask를 가장 잘 대체할 수 있는 값이다.

  - 예를 들어, 두 문장(a, b)가 모순인지($y_0$), 모순이 아닌지($y_1$)를 알아내는 task를 가정하자. 그러면 $y_0$ -> "YES", $y_1$ -> "No"로 매핑하는 verbalizer `v`와 패턴 `P(a,b) = a? ____, b.` 를 선택할 것이다.
    
    -  x = (Mia likes pie, Mia hates pie) 라는 input pair는 아래와 같이 바뀌게 되고,
    -  P(x) = `Mia likes pie? ___, Mia hates pie.`, 이는 label을 부여하는 것에서 마스킹된 위치에 Yes와 No 중에서 가장 그럴듯한 선택을 맞추는 것으로 바뀌는 것이 된다.

#### 2.1 PVP Training and Inference

`p = (P, v)` 인 PVP가 있고 small training set $\mathcal{T}$ 와 large unlabeled examples $\mathcal{D}$ 에 접근한다 가정하자. 각 시퀀스 $z \in V^*$는 하나의 mask token $w \in V$ 를 포함한다.

$M(w|z)$ 를 LM이 마스크된 위치의 w에 할당하는 unnormalized score로 정의하자. 

- 입력 x가 주어지면, label $l \in \mathcal{L}$ 을 위한 score를 다음과 같이 나타낼 수 있다.

  - $s_p(l|x) = M(v(l)|P(x))$
  - `v(l)`은 $\mathcal{L}$ -> V 로 매칭하는 verbalizer임에 유의

- 그리고 softmax를 사용해 label간의 확률 분포를 얻을 수 있다.

  - $q_p(l|x) = \frac{e^{s_p(l|x)}}{\sum_{l' \in \mathcal{L}}e^{s_p(l'|x)}}$

- $q_p(l|x)$ 와 training example $(x,l)$ 의 true(one-hot) distribution 과의 cross-entropy(모든 $(x, l) \in \mathcal{T}$를 더한 )를 `p`를 위한 M의 fine-tuning loss로 사용한다.

#### 2-2. Auxiliary Language Modeling

위의 작업이 적은 양의 example만 사용하기 때문에 catastrophic forgetting이 일어날 수 있다. 또, PVP에 대해 fine-tune된 PLM은 본질적으로 language model이므로 language modeling task를 보조 task로써 사용해서 forgetting을 막고자한다. 전체 Loss는 아래와 같다.

$L = (1 - \alpha) \cdot L_{CE} + \alpha \cdot L_{MLM}$

$L_{MLM}$ 이 $L_{CE}$ 보다 훨씬 크기때문에 $\alpha$ 값을 작게($10^{-4}$) 설정하는 것이 좋은 결과를 보였다.

여기에서 language modeling task는 unlabeled dataset $\mathcal{D}$ 에서 뽑은 문장(x)을 그대로 학습하는 것이 아닌 `P(x)`로 학습 한다. (패턴 내의 masked position에 대한 예측을 요구하지 않고 형식만 사용)

#### 2-3. Combining PVPs

그렇다면, 어떤 PVP가 잘 동작하는지 알 수 있을까? 이를 위해 우리는 knowledge distillation과 비슷한 전략을 취한다.   
먼저, 주어진 task A에 대해 직관적으로 타당한 PVP들의 집합 $\mathcal{P}$ 를 정의한다. 이 PVP들을 아래와 같이 사용한다.

1. 각 PVP에 대해 서로 다른 LM을 fine-tune한다. ($\mathcal{T}$ 가 작기 때문에 PVP가 많더라도 여기에 드는 cost는 작다)

2. 위에서 만든 fine-tuned model들을 앙상블($\mathcal{M}$)하여 $\mathcal{D}$ 의 example들을 레이블링한다. 각 example $x \in \mathcal{D}$ 에 대한 unnormalized class score를 결합한다. (아래와 같이 표현 가능)

- $s_{\mathcal{M}}(l|x) = \frac{1}{Z} \sum_{p \in \mathcal{P}}w(p) \cdot s_p(l|x)$

  - $Z = \sum_{p \in \mathcal{P}}w(p)$ 이고 $w(p)$ 는 PVP의 가중치이다.

  - 저자들은 이 가중치를 1로 두는 방법(단순 평균)과 훈련 전에 training set으로 해당 p를 사용했을 때의 accuracy로 설정하는 방법을 제안

- 위의 $s_{\mathcal{M}}(l|x)$ 에 softmax를 취하여 확률 분포를 얻어 unlabeled data를 soft-labeled data로 바꾸고 이를 모아 새로운 training set $\mathcal{T_C}$ 를 만든다.

3. 만들어진 $\mathcal{T_C}$ 를 이용해 PLM with standard sequence classification head를 fine-tune 한다.

#### 2-4. Iterative PET(iPET)

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/ae848958-3179-4b88-9e72-3aededd17e6a" width="60%"></img></div>

패턴마다 성능이 다르기 때문에 잘못 레이블링된 example들이 있을 수 있다. 이를 위한 보상작용을 주기 위해 iterative PET(iPET)를 제안한다.

핵심 아이디어는 데이터 양을 늘려가며 모델의 여러 세대를 학습하는 것이다.

- 아래와 같이 진행된다. (위 그림 참조)

  - (a) 훈련된 PET model들의 random subset을 사용해 $\mathcal{D}$ 에서 선택된 example들을 레이블링하는 것으로 original dataset $\mathcal{T}$ 를 확장한다.
    - model의 random subset은 다음과 같이 정해진다. -> $\lambda \cdot (n-1)models$, $\lambda$ 는 하이퍼파라미터
    - 간단하게 모델의 몇 %를 사용할지 정하는 

  - (b) 그 후, 이렇게 만들어진 데이터 셋으로 새로운 세대의 PET 모델을 학습한다.
  - (c) 위와 같은 작업을 반복한다.

<br></br>

### 3. Experiments

- `Dataset`: Yelp, AG'sNews, Yahoo Questions, MNLI

- `Model`: RoBERTa-large for LM, XLM-R for x-stance
  - PET가 얼마나 잘 동작하는지 조사하기 위해 x-stance 사용
  
#### 3-1. Patterns  
 
pattern 과 verbalizer를 설명 (`||`으로 문장 구별)

- YELP : 1~5 스케일의 리뷰

- AG's NEWS: World(1), Sports(2), Business(3), Science/Tech(4) 분류

- Yahoo: 1~10 class

- MNLI: 두 문장이 모순되는지, 한 문장이 다른 문장을 암시하는지, 둘 다 아닌지 예측
  - 2개의 verbalizer로 진행

- x-stance: 사회적 이슈에 관한 질문이 주어지고, 답변이 찬성인지 반대인지 분류

  - multilingual data로, verbalizer를 English, French 두 개를 정의

<br></br>

### 4. Results

#### 4-1. English Datasets

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/183eb046-6216-4eaf-aae8-532095cd9b7b" width="60%"></img></div>

전체적으로 iPET가 성능이 좋았으며, example 수를 늘릴 수록 성능향상은 낮아졌다.

#### 4-2. Comparison with SOTA

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/42c856ad-fc09-480a-b6e2-40a51bfa27b3" width="30%"></img></div>

Data augmentation에 의존하는 semi-supervised learning method 2개와 비교 [모두 back translation 방식]

PET와 iPET가 모든 task에서 더 좋은 성능을 보였으며 이는 PVP형식에서 오는 인간의 지식(도움)이 있기 때문임

#### 4-3. X-stance

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/51132a0f-c69a-4507-b9c3-0bbd56ecc2b3" width="30%"></img></div>

PET가 영어 말고 다른 언어에도 잘 작동을 하는지(1), 중간 크기의 훈련 셋에서도 성능 향상을 보이는지(2)를 조사하기 위해 x-stance로 평가함.

여기에서 우리는 프랑스어, 독일어에 대한 unlabeled data가 없었기 때문에 labeled data를 그대로 가져와 사용했다. 또한, 이탈리아어 데이터가 없기 때문에, 독일어와 프랑스어 모델의 zero-shot 성능 평균을 사용했다.

<br></br>

### 5. Analysis

#### 5-1. Combinig PVPs

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/537dbbb2-1b60-4ad9-b46b-164f6cc58cd5" width="30%"></img></div>

몇몇 PVP가 안 좋게 동작하는 것에 대처할 수 있는지 조사하였다. example 수를 10개로 주었으며, 가장 좋은 성능을 보인 PVP와 그 반대 PVP, 일반 PET, iPET에 대해 비교하였다.

PET는 worst PVP에 대해 대응할 수 있으며 성능 또한 높였다. Distillation 또한 성능 향상을 가져왔다.

#### 5-2. Iterative PET

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/0b06158c-0d6b-472e-b9b5-bcd5c0667642" width="30%"></img></div>

iPET가 정말 모델의 성능을 향상시켜주는지 확인해보았다. 위 그림은 zero-shot setting에서 각 단계의 모델의 평균 성능을 나타낸 것이다. 각 iteration은 정말 앙상블 성능을 높여주었다.

#### 5-3. In-Domain Pretraining
<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/98f5ec37-7676-4c53-b8fa-d6ecd14ae139" width="30%"></img></div>

PET가 추가적인 unlabeled data를 활용하기 때문에 supervised 모델보다 더 높은 성능을 가지는 것일 수도 있다. 이를 테스트하기 위해 in-domain data로 RoBERTa를 further pretraining을 진행해서 비교해보았다.

위 결과를 보아, PET의 성공은 단순히 추가적인 unlabeled data 때문이 아니다.
 
<br></br>
