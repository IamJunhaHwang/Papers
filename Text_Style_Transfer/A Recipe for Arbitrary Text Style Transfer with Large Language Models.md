## :page_with_curl: A Recipe for Arbitrary Text Style Transfer with Large Language Models

### Info

* url: https://aclanthology.org/2022.acl-short.94/
* conf: ACL 2022 (short paper)
* video: https://aclanthology.org/2022.acl-short.94.mp4

### 1. Abstract + Introduction

TST 태스크를 위해 labeling 되어 있는 데이터를 필요로 했고, label-free 문제를 제기한 bleeding-edge approaches 또한 지시한 타겟 스타일의 예시 문장을 필요로 했음.

본 논문에서는 타겟 스타일에 대한 어떤 예문도 없이 large-LM이 임의의 스타일로 TST를 가능하도록 하는 prompting method를 제안함. 
 ==> called `augmented zero-shot learning`

- Contribution

  1. label, train 없이 large-LM으로 직관적이게 TST 수행하는 방법 제시
 
  2. human evaluation에서 표준과 비표 TST 모두 좋은 성능을 보임.
  
  3. 본 논문에서 제시한 방법을 구현한 텍스트 수정 UI의 사용자가 이야기하는 현실에서 필요한 Style Transfer를 분석

<br></br>

### 2. Augmented zero-shot prompting

특정 style로의 transfer를 위해 해당 style의 예문을 prompting하는 대신에 다양한 sentence rewriting 명령 예시를 prompting한다.

[Reynolds and McDonell (2021)](https://arxiv.org/abs/2102.07350)의 방법에서 생각해 냄.

zero shot prompt의 유연성은 보존하면서 특정 템플릿의 출력을 모델이 만들어내게 함.

프롬프트에 주는 예문의 형식을 일정하게 유지하면서 같은 포맷에서 원하는 sentence transform을 삽입함.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/210164944-622f281b-7fd7-464d-afb6-0c44bd9eedd4.png" width="40%"></img></div>

<br></br>

### 3. Experimental Setup

#### 3-1. Style Transfer Task

- 비표준 TST task: contribution 3번에서 가장 빈도 높은 Style Transfer task로 정함.

  - source sentence: Reddit Writing Prompts validation set에서 50개를 랜덤하게 추출. [하나의 스타일로 이미 확실하게 보여지거나 문법이 이상하거나 일관성이 없는 것은 제외함]

  - human evaluation 사용. [모든 스타일에 대한 분류기가 없기 때문에]

- 표준 TST task: sentiment & formality

  - 사용 데이터-> sentiment:Yelp polarity dataset,   formality: Grammarly's Yahoo Answers Formality Corpus

  - 이전의 방법들과 성능 비교 가능

#### 3-2. Model

- 사용 모델: LaMDA from Google, LLM-Dialog, GPT-3

  - LLM-Dialog: 대화 형식으로된 고품질 데이터로 fine-tune 된 LaMDA model, `top-k=40`으로 디코딩 됨.
  - GPT-3: `p=0.6`으로 설정한 nucleus sampling(Top-p sampling)을 적용해 디코딩 됨.

### 4. Results

#### 4-1. 비표준 TST

평가자들에게  `<input sentence, target style, output sentence>` 튜플을 평가해달라고 함.

평가자들간의 불일치를 줄이고 우리의 지시를 확실하게 하기 위해, 테스트 평가를 진행함. [결과에 미포함하는 10개 데이터로]

50개 문장에 대해 3개의 베이스라인과 제시한 방법의 결과를 비교.

각 튜플은 3명의 평가자들에게 평가 되었음. [TST 평가 표준에 대해]

- TST 평가 표준 [Mir et al., 2019](https://arxiv.org/abs/1904.02295)

  - Transfer strength: 타겟 스타일과 일치하는 정도 [1~100]

  - Semantic preservation: 스타일을 제외하고 입력의 내용이 보존되었는지 [1~100]

  - Fluency: 얼마나 자연스러운지 [유창한지] [Yes/No]

Style Transform에 필요하기 때문에 생성된 정보에 대해서 penalty를 가하지 않음. [몇몇 평가에서는 input보다 더 많은 정보를 포함하면 안된다고 엄격하게 평가함]

**dialog-LLM 사용, 3가지 방법 비교: zero-shot, paraphrase(augmented zero-shot으로 target style로 paraphrased 사용한 것), human**

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/210164952-b5496d98-bf78-437a-8a7b-f848f5e8631f.png" width="40%"></img></div>

- `paraphrase`가 사람만큼 좋은 성능을 보임.
- input sentence는 66자 였고 paraphrase는 107자, 사람은 82자를 출력 문장 평균 길이로 만듦.

#### 4-2. 표준 TST

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/210164954-9f2caed4-937f-4d0c-bc9e-3a6c2da44272.png" width="40%"></img></div>

전의 유명한 두 방법(Unsup MT, Dual RL)을 추가로 사용해 비교. [이전과 같이 사람이 평가함]

- 제시한 방법이 전의 두 방법과 더불어 사람과 비슷하게 성능을 냄.

- 아래는 자동으로 평가한 것. [사람X]

  - Transfer strength: huggingFace Transformer로 평가

  - Semantic similarity: BLEU-score로 평가

  - Fluency: GPT-2를 이용해 PPL로 평가

  <img src="https://user-images.githubusercontent.com/46083287/210164958-412534bf-32e7-4dba-975e-01302f229639.png" width="40%"></img>

- 자동 평가에 대한 4가지 키포인트

  - 논문에서 제시한 방법이 높은 정확도와 낮은 PPL을 보였고 BLEU-score는 낮지만, 이는 새로운 단어(정보)를 생성했기 때문으로 보임
  - 다른 large-LM에도 제시한 방법을 적용할 수 있음. [GPT-3로 보임]
  - GPT-3는 size가 클수록 TST를 더 잘 풀었음.
  - large-LM과 large-LM-dialog에 제시한 방법을 적용하면 상당히 뛰어난 성능을 보임. 

<br></br>

### 5. Limitations and Failure Modes

Unparsable answers: 자동으로 원하는 답을 생성하지 않음. ex) 이 커피 맛있다. => 맛있다를 맛없다로 바꾸어야 합니다. [원하는 문장은 이 커피 맛없다.]

Hallucinations: 새로운 정보들을 생성함. [이는 소설쓰기 같은 곳에는 장점이 될 수 있지만 요약에서는 단점이 됨]

Inherent style trends: 특정 스타일에 강점을 보임. [제시한 방법의 생성 결과가 일반적으로 특정 스타일의 경향을 보임을 암시]

Less reliable than trained methods: train data로 fine-tune하거나 pretrain한 TST는 train data와 같은 문장을 생성함. [제시한 방법이 BLEU-score가 낮았음]

Large LM safety concerns: 제시한 방법으로 model이 어디까지 잘 생성가능한지 어디부터 이상한 답을 하는지 확인할 수 있음.

### 6. Conclusions

augmented zero-shot learning을 소개하고 좋은 성능을 보임을 증명.

가능한 Text Transfer Style을 넓히고 레이블링된 데이터가 필요하다는 한계를 넘어설 수 있을 것임.

또한, large-LM을 task에 특화된 예제 없이 프롬프팅하는 전략이 다른 NLP task에 적용될 수 있을 것임.
