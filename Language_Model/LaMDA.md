## :page_facing_up: LaMDA: Language Models for Dialog Applications

### Info

* `publication`: arXiv preprint 2022

* `author`: Romal Thoppilan et al.
* `url`: https://arxiv.org/pdf/2201.08239.pdf

### 1\. Abstract & Introduction

본 논문에서는 `Language Models for Dialog Applications (LaMDA)`를 소개한다. LaMDA는 대화에 특화된 Transformer 기반 LM으로 137B까지의 파라미터를 가지며 1.56T 단어의 public dialog 와 web text로 사전 학습되었다.

- Motivation : `Adiwardana et al.,2020`에서 dialog model 또한 모델의 크기가 늘어나면 성능이 좋아짐을 보였음. --> 모델의 크기와 대화 퀄리티는 강한 상관관계가 있음.

LaMDA는 하나의 모델로 여러 task를 수행한다 : 가능한 response를 만든 후 safty를 위해 필터링하고, 외부 knowledge source에 기반해 높은 질의 response를 찾기 위 re-rank한다.


모델의 크기만을 늘리는 것은 quality를 향상시키지만 safety와 factual groundnig에서 낮은 성능을 보였다. 우리는 annotated data로 fine-tuning하는 것과 모델이 외부 지식을 참고하도록 하는 것으로 이러한 2가지 측면에서의 성능을 향상시킬 수 있음을 증명했다.

<br></br>

### 2. LaMDA pre-training

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/47ef39b0-876a-4e79-a6d7-a5830fb549e5" width="50%"></img></div>

LaMDA는 text corpus에서 `next token prediction`을 수행하는 것으로 사전 학습 된다. 대화 데이터만 사용해 만든 이전 대화형 모델과 다르게 LaMDA는 2.97B의 문서, 1.12B의 대화, 13.39B의 대화내 발화로 사전 학습 데이터 셋을 구성한다. 따라서, fine-tuning 전에는 일반적인 LM이다.

LaMDA는 decoder-only Transformer 구조이며, T5의 relative attention과 Raffel et al.의 gated-GELU 활성화 함수를 차용했다. 1024개의 TPU-v3 chips로 57.7days를 학습했으며 배치당 256K의 토큰을 가지고 있다.

fine-tuning 전의 모델을 `"PT"`라고 부를 것이며 PT는 디코딩을 위해 Meena에서 사용한 `sample-and-rank`전략을 사용했고 top-k sampling(k=40, no temperature)을 사용해 16개의 독립적인 candidate response를 샘플링한다. 최종 출력은 가장 높은 점수를 받은 candidate이다(점수는 candidate의 log-likelihood와 length를 기반).

<br></br>

### 3. Metrics

#### 3-1. Foundation metrics: Quality, Safety and Groundedness

논문에서 제시하는 전체 quality score는 Sensibleness, Specificity, Interestingness (SSI)의 평균이다.

* Sensibleness : 모델의 응답이 말이 되는지, 이전에 말한 것과 반대되지는 않는지 여부 측정

- Specificity : 모델의 응답이 주어진 context에 특정한 것인지 여부 측정

  - 예를 들어, `I love Eurovision`이라 했을 때, `"Me too"`라는 대답의 경우 0점이고 `"Me too. I love Eurovision songs"`라는 대답의 경우 1점

- Interestingness : 주의를 끌거나 호기심을 부르거나 insightful한지 여부 측정

  - crowdworkers에게 0/1로 태깅하도록 함.

  - Sensibleness, Specificity 만으로는 모델의 질을 판단하기 충분하지 않음. (좀 더 만족스러운 답변인지를 체크하기 위한 metric)

- safty

  - 대화형 모델이 높은 SSI 점수를 얻을 수 있지만 user에게 안전하지 않을 수 있음. 따라서, unsafe model output을 측정하기 위해 새로운 safy metric을 고안함. (Google의 AI Principle에서 유래됨)

- Groundedness

  - 현재 언어 모델은 그럴듯하지만 잘못된 대답을 생성하는 경향이 있기 때문에, 우리는 LaMDA가 가능하면 알려진 소스와 연관될 수 있는 응답을 생성하고, 원하는 경우 교차 확인을 가능하도록 한다.

  - `"Groundedness"`는 외부 세계에 대한 주장을 포함하는 응답 중 외부 소스에서 가져올 수 있는 비율로 정의한다.

  - `"Informativeness"`는 알려진 소스로부터의 외부 세계에 대한 정보를 담고 있는 응답의 비율로 정의한다.

    - `"That’s a great idea"`와 같이 외부 세계 정보를 전달하지 않는 응답은 groundedness에 영향을 미치지 않지만 Informativeness에는 영향을 미친다. 그러나 `"Rafael Nadal은 2020년 롤랑 가로스 우승자입니다"`와 같은 응답은 grounded response의 예이다.
    
  - `"Citation accuracy"`는 URL을 인용하는 모델 응답의 비율로 정의한다.

#### 3-2. Role-specific metrics: Helpfulness and Role consistency

기본 metric(quality, safety, groundedness)은 일반적으로 대화 에이전트에게 중요한 속성들을 측정하지만 에이전트의 설계 목적(예. 동물에 대한 정보 교육)에 종속되지 않는다. 따라서, 에이전트가 특정 역할을 수행하는 대화 어플리케이션에서 Helpfulness 와 Role consistency를 측정한다.

- Helpfulness : 모델의 응답은 독립적인 정보 검색 시스템을 기반으로 정확한 정보를 포함하고 사용자가 도움이 되었다고 판단하는 경우 도움이 되었다고 표시된다.

- Role consistency : 목적 역할을 수행하는 것처럼 보이는 에이전트의 경우 role consistent로 표시된다.

  - 이전 응답에 대해 일관적인 답변을 하는지 측정하는 것은 sensibleness metric이다.

  - Role consistency는 대화에서 외부 에이전트 역할 정의의 consistency를 의미

<br></br>

### 4. LaMDA fine-tuning

#### 4-1. Discriminative and generative fine-tuning for Quality (SSI) and Safety

PT 모델에 여러 fine-tuning을 진행해 LaMDA 모델을 만들었다. 이 fine-tuning에는 주어진 context에 대해 응답을 생성하는 `generative task`와 context에서 응답의 질과 안정성을 평가하는 `discriminative task`의 혼합이 포함된다. 이는 단일 모델이 generator와 discriminator 모두로써 동작할 수 있게 한다.

- LaMDA가 decoder기반 생성LM이기 때문에, 모든 fine-tuning 예시는 토큰 시퀀스로 표현된다. 

  - Generative fine-tuning : `"<context> <sentinel> <response>"`, loss는 response 부분에만 적용됨.

    - ex: “What’s up? RESPONSE not much.”, What's up==<context>, RESPONSE==<sentinel>, not much==<response>

  - Discriminative fine-tuning : `"<context> <sentinel> <response> <attribute-name> <rating>"`, loss는 rating 부분에만 적용됨.

    - ex: “What’s up? RESPONSE not much. SENSIBLE 1”

discriminator를 평가하는 것은 P("<desired-rating>" | "<context><sentinel><response><attritube-name>")를 계산해야 하는데, 모델이 response를 생성하면서 `"<context><sentinel><response>"`까지 진행되므로 추가적인 토큰인 `"<attribute-name><desired-rating>"`를 계산하면된다.

LaMDA fine-tuning은 생성된 candidate response의 SSI와 safety rating을 예측하기 위해 진행된다. 그 후, generation동안 safety 예측 값이 threshold 밑으로 떨어지면 이 response를 필터링한다. 나머지 response는 quality에 대해 rank된다. ranking하는 동안 `sensibleness`에 3배 높은 가중치를 부여하고 가장 높은 점수를 가진 response를 선택한다(i.e., 3 *
P(sensible) + P(specific) + P(interesting). 

LaMDA SSI & safety discriminator는 사전 학습 데이터셋에서 샘플링된 2.5M turns의 대화 데이터를 필터링하고 점수를 매기기위해 사용되며, 결과적으로 800K의 turn이 만들어진다. 이에 대해 LaMDA모델을 fine-tune한다.

이 기술을 사용하면 LaMDA의 safety와 quality가 아래와 같이 상당히 좋아진다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/83c49b3f-ebed-4344-9280-68a533541476" width="50%"></img></div>

#### 4-2. Fine-tuning to learn to call an external information retrieval system

LaMDA와 같은 LM은 그럴듯하지만 거짓인 출력들을 만들어내는 경향이 있다. 한 가지 가능한 해결법은 학습 데이터가 늘어나면 모델이 기억을 좀 더 잘한다는 가정에 따라 model size를 늘리는 것이다. 또한, “How old is Rafael Nadal?” 또는 “What time is it in California?”와 같은 시간이 지나면서 변하는 사실에 대해 대답해야하는 `temporal generalization problem`이 있다.

최근 연구들은 dynamic or incremental architecture를 통해 이러한 이슈를 완화했지만, user는 사람이 가지는 지식 속의 어떤 것이든 대화하길 바랄 것이므로 이를 위해 모델 capacity와 학습 데이터를 충분하게 얻기는 쉽지 않다. 따라서, 우리는 외부 knowledge resource나 도구들을 참조하는 것을 학습하게 하는 fine-tuning을 제시한다.

- The toolset (TS) : 정보 검색 시스템, 계산기, 번역기를 포함하는 toolset을 만듦

  - TS는 단일 string을 입력으로 받아 하나 이상의 string의 list를 출력한다. (ex. "135+7721"을 입력으로 받아 ["7856"]을 출력)

  - TS는 input string을 모든 tool에 시도하고 output들을 붙여서 final output list를 만든다. 여기서, 계산기->번역기->정보 검색 순으로 진행되며 파싱할 수 없는 경우 빈 리스트를 반환한다.

- Dialog collection : `(generative data)`로 태깅된 40K의 dialog turns와 ranking task (discriminative data)의 입력으로 사용된 `correct` 나 `incorrect`로 태깅된 9K의 LaMDA가 생성한 dialog turns의 모음
  
  - information-seeking interactions에 집중해 crowdworkers간의 사람대사람 대화를 수집했고 그들의 진술이 외부 소스에서 지원될 수 있는지 평가했다.

  - 알고리즘에서 사용되는 파인튜닝의 교육 데이터를 수집하기 위해 정적 및 대화형 방법을 모두 사용한다. 다른 하위 작업과의 주된 차이점은 Crowdworkers가 모델의 출력에 반응하는 것이 아니라 모델이 모방할 수 있도록 수정하는 방식으로 개입한다. 대화형 경우, Crowdworker는 LaMDA와 대화를 수행하며 정적인 경우에는 이전 대화 기록을 차례로 읽는다. Crowdworker는 각 문장이 외부 지식을 참조해야 할 주장을 포함하는지 여부를 판단한다.

##### Fine-Tuning : LaMDA를 2가지 task에 대해 fine-tune한다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/affb2911-c538-4075-b34b-916f11909434" width="50%"></img></div>

- 첫 번째 task는 현재까지의 다중 턴 대화 컨텍스트와 base 모델이 생성한 응답을 취한다. 그 다음, special string("TS"는 Tool Set를 의미)을 생성하여 다음에 나올 텍스트가 Tool Set에 전송되어야 하는 쿼리(e.g., “How old is Rafael Nadal?”)임을 나타낸다: context + base → “TS, Rafael Nadal’s age”

- 두 번째 task는 tool이 반환한 정보와 대화문을 취한다(e.g., “He is 31 years old right now” + “Rafael Nadal / Age / 35”).

  - 이후, grounded version을 예측한다 : context + base + query + snippet → “User, He is 35 years old right now” 

  - 그렇지 않으면, 추가 research query를 출력한다(context + base + query + snippet → “TS, Rafael Nadal’s favorite song”).

  - inference time에, 모델의 출력은 첫 번째로 생성된 문자열이 'TS'인지 'User' 인지에 따라 정보 검색 시스템 또는 유저에게 전달된다.

##### 모델의 크기만을 키우는 것은 모델의 quality, groundedness metric 성능을 올려주었지만, safety는 성능이 크게 올라가지 않았다. 하지만 Fine-Tuning을 진행했을 때는 모든 metric의 성능 향상을 관찰하였다.
