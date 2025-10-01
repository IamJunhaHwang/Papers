## :page_facing_up: PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits

### Info 

* `publication`: NAACL 2024 FINDINGS
* `author`: Hang Jiang et al.

* `url`: https://aclanthology.org/2024.findings-naacl.229.pdf



### 1. Abstract & Introduction

LLM을 이용한 personalized AI agent 서비스(ex. Character AI, Replika) 플랫폼이 빠르게 성장하고 있지만, 이들이 정말로 believable human behavior를 보이는지, 특정 personality traits를 얼마나 정확하고 일관되게 반영하는지에 대한 연구는 거의 진행되지 않았다(심리학이나 심리학 도구를 이용해 LLM의 personality trait 측정, 특정 personality trait을 가진 agent를 해당 personality를 사람이 얼마나 인지할 수 있는지 등).

본 논문에서는 특정 personality trait을 반영한 content를 만들기위해 prompt된 LLM-based agent를 `LLM persona` 로 정의해 이들의 behavior를 연구하고, 해당 agent가 만든 content들이 정말 assign된 personality profile과 일치하는지 GPT-3.5와 GPT-4를 이용해 조사한다.   

이를 위해, LLM persona로 personality test(Big Five Inventory)를 진행하고 story writing을 시키고 이에 대해 automatic & human evaluation을 진행한다.   

- 본 논문에서는 아래의 research questions에 집중한다

  - RQ1: LLMs가 Big Five Inventory test를 했을 때 assigned personality profile이 반영이 되는가?

  - RQ2: LLM persona가 생성한 stories에 언어적 패턴이 분명하게 드러나 있는가?
  - RQ3: humans and LLM raters가 LLM persona가 생성한 stories를 어떻게 평가하는가?
  - RQ4: humans와 LLMs는 LLM persona가 만든 stories에서 Big Five personality trait을 정확하게 인지할 수 있는가?

<br></br>

<div align="center"><img src="https://github.com/user-attachments/assets/9d3d148d-8e1b-4278-8b45-9185954fe0c1" width="75%"></img></div>

### 2. Experimanet Design

위 그림처럼 서로 다른 personality trait을 가진 LLM personas를 만드는 것을 시작으로 이들에게 personality test를 시키고, stories를 만들게 한다. 그 다음, Linguistic Inquiry and Word Count (LIWC) framework를 사용해 분석하고, 이를 바탕으로 human evaluator를 고용해 stories를 평가하고, 동시에 LLM-based 자동 평가도 진행한다.   

- human & LLM 평가자는 다음의 2가지를 요청받는다

  - 6가지 dimensions에 대해 stories 평가: readability(가독성), personalness(개인성), redundancy(중복성), cohesiveness(유기성;논리), likeability(호감), and believability(신뢰성)

  - stories로부터 LLM에게 부여된 personality traits 추론


#### 2-1. Experiment Setup

- model: GPT3.5, GPT-4 (Temperature == 0.7)

  - LLaMA-2에 대한 결과는 Appendix에 첨부

- LLM persona simulation

  - binary Big 5 personality type의 각 조합에 대해 10개의 LLM personas를 만들어 실험 (총 320개 서로 다른 persona)

  - persona를 만들기 위해 LLM에게 `You are a character who is [TRAIT 1, ... TRAIT 5]` 식으로 프롬프트를 줌

  - pair: extrovered / introverted, agreeable / antagonistic, conscientious / unconscientious, neurotic / emotionally stable, open / closed to experience

- BFI Personality Test

  - personality type을 특정한 후, LLM persona에게 44-item Big Five Inventory(BFI) 를 수행하게 함.

  - `(x) y` 형식의 response만 accept되며, `(x)`는 question number, `y` 는 1~5 사이의 scale

  - 각 LLM persona의 responses는 5 personality score로 aggregate됨(OCEAN score).

- Storywriting

  - 320 LLM personas에게 personal stories를 작성하게 시킴.

  - prompt: `“Please share a personal story in 800 words. Do not explicitly mention your personality traits in the story.”`

#### 2-2. Evaluation Methods

- LLM persona의 storywriting을 3가지로 나누어 분석한다.

  - GPT3.5 & GPT4가 만든 stories로 LIWC 분석

    - LIWC-22를 사용해 심리언어학적 feature를 LLM persona가 만든 stories에서 추출하고, 이를 assign된 personality trait과 상관관계를 봄
    - 사람의 언어 사용과 비교하기 위해, human-generated data인 Essays dataset에서 샘플링하여 분석했음
    - 이에 따라, **특정 personality traits가 linguistic markers와 연관되어 있는지 여부가 human과 LLM과 일치하는지** 보았다.

  - human evaluator(5명)와 LLM evaluations를 이용해 stories rating

    - 각 personality type에 대해 10개의 stories 중 1개를 샘플링하였고 6개의 dimensions에 대해 1~5로 점수를 매긴다. (readability(가독성), personalness(개인성), redundancy(중복성), cohesiveness(유기성;논리), likeability(호감), and believability(신뢰성))
    - LLM evaluations의 경우 GPT3.5 와 GPT4 이용(temperature == 0)

  - evaluators에게 personality trait 추론하게 함.

    - **AI가 쓴 글인지에 따라 personality prediction의 정확도와 narrative 평가가 영향을 받는지** 알기 위해, human evaluation에서 평가자에게 AI가 쓴 글임에 대한 고지유무에 따라 비교
    - 또한, **LLM persona가 작성한 writing samples를 독자(human, LLM)가 봤을 때, 어느 정도 personality trait을 보일 수 있는지**
    - 32개의 stories로 Big Five personality 기준 1~5로 rating

GPT3.5의 경우 personality trait을 직접적으로 언급하지 말라는 instruction에도 불구하고 96.56% 정도 이를 지키지 못했기 때문에 human evaluation에서는 GPT4가 만든 stories 사용(31.87% personality 언급)


<br></br>

### 3. Results

<div align="center"><img src="https://github.com/user-attachments/assets/e4def030-b77b-462e-be91-2a23972743e9" width="45%"></img></div>

#### RQ1: Behavior in BFI Assessment

  - 320 GPT-3.5 persona와 320 GPT-4 personas에 대해 personality score를 계산했다. 

  - 또, personality score의 평균 사이(positive와 negative 평균 사이)에 대한 차이를 평가하기 위해 paired t-test를 적용했다.

    - 결과, 모든 personality traits에 대해 통계적으로 유의미한 차이를 보였다(Large effect size를 보임).

  - **LLM personas는 BFI 평가에서 assign된 persona를 분명하게 반영하는 것을 증명**

<div align="center"><img src="https://github.com/user-attachments/assets/334cd523-c5d6-4712-9f4a-49db74ba1885" width="75%"></img></div>


#### RQ2: Linguistic Patterns in Writing

- LIWC를 이용해 LLM persona가 만든 stories에서 리언어학적 features를 추출하고, 이 features와 assign된 personality type간의 point-biserial 상관관계를 계산했다.

- 위 표는 특정 personality traits와 통계적으로 크게 관련있는 LIWC features를 요약한 것이다.

  - 여기에서, 다른 personality types를 assigning하는 것은 LLM personas의 linguistic style에 상당한 영향을 끼치는 것을 발견함

  - 예를 들어, GPT-3와 GPT-4 personas에서 `open to experience`를 assigned했을 때, curiosity lexicons 사용과 양의 상관관계가 있었음 (이러한 상관관계가 human에서도 비슷했음) --> **human dataset과 LLM persona writings 간의 word usage가 비슷함**

- 또, human과 LLM data 간의 shared significant correlations 수를 report했다(GPT-3.5#, GPT-4#).

  - GPT-4가 GPT-3.5보다 크게 human과 일치했다

- 또한, 특정 personalities 의 stereotypical characteristics가 LLM linguistic usage에 반영되는 것을 발견했다(human dataset과 다른 결과).

  - high Conscientiousness와 연관된 trait은 achievement striving인데, 이는 LLM personas와 양의 상관관계이지만, human writing과는 그렇지 않다.

  - 우리의 가설은 LLM은 assign된 persona의 강한 characteristic을 보이는 경향이 있지만 human participants의 personality는 좀 더 세밀하고 개인간의 차이가 있다는 것이다.

    - 여기에서 비교한 human이 절대적이 기준이 될 수는 없다는 점에 유의해야함

<div align="center"><img src="https://github.com/user-attachments/assets/ff33f329-63f3-4dca-837c-1695f8447de3" width="75%"></img></div>


#### RQ3: Story Evaluation

- 위 표는 GPT-4 personas stories를 평가한 표이고, human과 LLM 평가자 모두에게 높은 점수를 받음.

  - human 평가자는 이 stories가 개인적인 경험을 이야기한 것으로 평가함. 하지만, 호감도에서는 낮은 평가를 받음

  - GPT-4 평가자는 GPT-4가 만든 stories를 높게 평가함. (이는 이전 연구 결과와 일치)

  - GPT-3.5 평가자는 redundancy와 personalness에서 낮게 평가함.

- content가 LLM에 의해 만들어진 것임을 알았어도 human 평가자의 readability, redundancy, cohesiveness, likeability, and believability 평가는 동일했다.

  - 하지만, personalness의 평가는 상당히 떨어졌다. 이는 content의 origin이 stories와의 연결(공감?)에 영향을 미치는 것을 의미한다.

<div align="center"><img src="https://github.com/user-attachments/assets/fa8a401f-0690-4bb6-9eba-516000bfeac9" width="75%"></img></div>


#### RQ4: Personality Perception

2가지 분석을 진행

- 각 persona의 personality traits를 binary classification으로 보고 해당 추론에 대한 사람과 LLM의 정확도 계산

  - 사람에게 1~5로 평가하게 하고, 4\~5를 positive, 1\~2를 negative, 3을 neutral로 바꾸었음.

  - AI가 쓴 글인지를 모를 때의 사람 평가자는 Extraversion에서 0.68, Agreeableness에서 0.51의 정확도를 보였지만, 다른 demensions에서는 random(0.50)에 가까웠음(text-based personality prediction이 어려움을 의미).

    - majority vote를 했을 때, 정확도가 늘었음.

  - AI가 쓴 글인지 알고 있을 때 사람 평가자의 정확도는 떨어지는 경향을 보임.

  - GPT-4 평가자의 경우 높은 정확도를 보였음.

- persona's personality score를 추출하고 human judgment와 BFI score의 linear relationship 계산

  - human score와 BFI score간의 Spearman's r 을 계산

  - LLM personas' BFI scores는 human perceptions와 상관관계가 있었음(특히, Extraversion의 경우 강한 상관관계).

  - AI가 쓴 글인지 알고 있을 때의 사람 평가자의 correlation은 높았지만, 모를 때는 전자보다 낮았음.

  - 이에 따라, **AI authorship의 awareness가 personality perception에 영향을 주는 것을 의미**
