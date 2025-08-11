## :page_facing_up: You Are What You Talk About: Inducing Evaluative Topics for Personality Analysis

### Info

* `conf`: EMNLP2022-FINDINGS

* `author`: Josip Jukić, Iva Vukojević, Jan Snajder
* `url`: https://aclanthology.org/2022.findings-emnlp.294/

- `github` : https://github.com/josipjukic/quasi-snowballing

### 1. Abstract & Introduction

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/231948577-e4103037-80fd-4aeb-8c2f-74117f061595.png" width="45%"></img></div>

어떤 개체나 개념에 대한 태도나 입장을 표현하는 것은 그 사람의 행동 및 성격(personality)을 내포하고 있다.

최근, Evaluative language(Hunston. 2010.) data에 좀 더 접근이 쉬워지고 있지만 성격과 evaluative language에 사이의 관계에 대한 연구는 거의 없다. (`Evaluative language` : 평가적 언어, 어떤 물체나 개념에 대한 평가나 주관을 이야기하는 것)

우리는 소셜 미디어에서의 평가적 언어가 성격 분석에 어떻게 쓰일 수 있는지를 보일 것이다. 자세하게는 사람들이 말하는 주제가 그들의 성격과 어떻게 연관이 되어있는지 탐구하는 것을 목표로 한다.

따라서, 소셜 미디어로부터 evaluative text를 필터링하기 위한 방법, topic model의 적용, `evaluative topics`에 대한 생각을 소개한다. 그 후, `evaluative profile`을 만들기 위해 `evaluative topics`를 개개인(author)과 연결한다.

이렇게 만들어진 evaluative profile을 성격 점수로 레이블링된 레딧 댓글들에 적용하고 evaluative topic과 Big 5 personality facets 사이의 관계를 탐구한다. [더 해석가능한 facet-level 분석(trait을 세분화한 것)을 목적으로]

마지막으로, 우리가 제시한 접근법이 이전 연구(심리학 등)와 일관된 상관관계를 보이는지 검증한다.

- **Contribution**

  1. evaluative filtering과 topic model을 기반으로한 evaluative author profiling 방법

  2. Reddit Comments에 evaluative profiling 적용

  3. 위의 profile과 Big-5 personality가 얼마나 관련 있는지 연구

<br></br>

### 2. Methodology

- 3가지 단계로 처리함.

  1. `evaluative filtering`을 통해 evaluative text를 추출

  2. `filtered dataset`으로 부터 topic을 생성

  3. 해당하는 evaluative profile을 위해 위의 2번에서 만든 evaluative topic을 사용

- `Dataset`: PANDORA [아래는 전처리 과정임]

  - Big-5에 대한 코멘트만 가져옴.

  - English 코멘트만 뽑음. (50% 이상이 알파벳이 아닌 문장 제거)

  - 5 단어 미만 문장 제거

  - `spaCy의 en_core_web_lg`를 이용해 댓글을 문장으로 잘랐음. (이 과정을 거친 후 3 단어 미만의 문장 제거) 

#### 2-1. Evaluative filtering

evaluative patterns(ex. 의견을 표현하는 구절)을 포함하는 문장을 찾고 의견과 입장의 어휘를 이미 만들어져있는 감정 분석 툴을 함께 사용해 evaluative language의 다른 측면을 발견했다. [아래는 이에 대한 설명]

sentiment-laden(감정이 풍부한) 문장을 추출하기 위해 우리는 `VADER`을 사용했다. (positive, negative VADER score를 합쳐서 전체 감정 점수를 결정) 또한, `Hu and Liu (2004)`가 고안한 opinion lexicon과 `Pavalanathan et al. (2017)`의 lexicon of stance markers를 사용했다.

정규 표현식을 이용해 데이터 셋에서 evaluative pattern을 포함하는 문장을 모았다. [ IMO (in my opinion)과 같은 줄임말도 포함시킴] 이 때, opinion, stance score가 각각 50%를 넘어야하며 sentiment의 경우 상한을 설정했다. (높은 sentiment 점수를 가진 문장은 보통 target이 암묵적으로 있는 감정 표현 문장이었음)

위의 과정은 엄격한 패턴 매칭을 통해 이루어지기 때문에 높은 precision에 집중한다. Recall을 향상시키기 위해 `quasi-snowballing(QSB)`를 적용한다. QSB는 간단한 paraphrase mining 기법으로 평가 표현의 seed set으로 시작해 반복적인 과정을 통해 비슷한 문장들을 찾아 늘려가는 것이다. 여기에는 문장의 contextualized representation이 필요한데 이를 위해 `sentence transformer`를 사용했다.

처음 필터링한 문장을 seed set으로 사용하며 비슷한 문장을 찾기 위해 임계값 $t_{sim}$ 의 코사인 유사도를 사용한다. 반복되는 과정에서 문장 set들이 늘어나기 때문에, 수렴을 용이하게 하기 위해 임계 값을 각 step마다 증가시켰다. 이 과정은 더 이상 후보들이 없으면 멈추게 된다.

모든 과정을 마치고 evaluative marker가 있는 `310K`의 문장을 얻었다.

#### 2-2. Evaluative topics

evaluative topic를 얻기 위해 위에서 구한 문장들에 토픽 모델링을 적용했다. (전통적인 확률 모델, 신경망 베이스 모델, 짧은 텍스트를 위한 모델 [데이터로 사용한 reddit comment가 짧은 경향이 있으므로])

우리는 topic에 대한 정의로 용어의 고정된 vocab에 대한 분포(Blei and Lafferty, 2009)를 채택했으며 `evaluative topic`으로 evaluative language로 파생된 target-specific opinion에서 만들어진 topic을 채택했다.

- 사용 모델

  - LDA: 전통 토픽 모델로, 강한 베이스라인이 됨. 짧은 텍스트의 경우 성능이 안좋기 때문에 문서별로 묶었음. (각 author별로 subreddit 단위로 묶음)

  - BTM(Biterm Topic Model): 짧은 문장을 위해 디자인됨.

  - ABAE(Attention-Based Aspect Extraction): AutoEncoder로 제안된 신경망 기반 구조

    - 단어 임베딩을 사용해 유사한 맥락에서 나타나는 단어들을 그룹화함

    - 어텐션을 사용해 관련 없는 단어와의 관련성을 줄임

    - 훈련 시, Word2Vec을 PANDORA 데이터로 새로 훈련했고 reconstruction loss function을 코사인 유사도와 일치되게 바꾸었음.

  - CTM(Combined Topic Model): VAE와 sentence transformer 임베딩을 섞은 것.

- 평가: 토픽 모델은 토픽 일관성으로 많이 평가하는데 validation gap 문제가 있음.

  - 따라서, 전체 데이터 셋에 대해 `token co-occurrence`를 측정함.

  - 자세하게는 NPMI를 사용하며 평가 다양성을 위해 IRBO를 사용함.

#### 2-2. Evaluative author profiles

각 유저별로 토픽의 출현빈도 관점에서 표현한다. 유저의 텍스트는 다른 타겟간의 sentiment weighted 혼합 토픽으로서 분포된다. 유저 문장(0~1 값)의 전체에 대해 평균 토픽 분포를 계산한다.

- 특정 문서에 대한 토픽 분포는 다음과 같이 공식화 된다. => $d = [c_1c_2...c_K]^T$ : 특정 문서 d는 k개의 토픽에 대한 혼합 요소를 가진다.

- 유저의 evaluative sentence 값을 모아 각 토픽에 대한 n번째 유저 요소를 계산하고 이를 붙여서 벡터 $u^{(n)}$ 으로 만든다. => $u^{(n)} = \frac{1}{N_n}\overset{N_n}{\underset{i=1}{\sum}}d^{(n, i)}$

  - $N-n$ : n번째 유저 문서의 수
  - $d^{(n,i)}$ : i번째 문서

- sentiment-enhanced representation $v^{(n)}$ 을 얻기 위해 감정집중정보를 포함한다. => $v^{(n)} = \frac{1}{N_n}\overset{N_n}{\underset{i=1}{\sum}}s^{(n, i)d^{(n, i)}$

  - $s^{(n, i)$ : n번째 유저의 i번째 문서에 대한 sentiment intensity (==VADER 점수의 합)

<br></br>

### 3. Experiments

evaluative language와 personality의 연관을 조사하기 위해 아래 두 가지 단계로 진행된다.

1. PANDORA 데이터로부터 evaluative author profile 만들기 (CTM 모델로 evaluative topic을 20개 유도했음)

2. Big 5 facets과 evaluative topic 간의 상관 관계 연구 수행. (NEO PIR 채택)

첫 번째로 특정 topic 과 facet 간의 각각의 상관 관계를 분석 하고 두 번째로 topic 전체와 facet 전체 간의 공통 연관성을 탐구하기 위해 canonical correlation analysis 사용

#### 3-1. Pairwise correlations

evaluative author profiles 과 Big Five facets 간의 partial pairwise correlation 계산 (성별과 같은 confounder를 조절함)

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/231948614-683070b5-65b7-49a6-a4e6-f9eedeffee60.png" width="70%"></img></div>

#### 3-2. Canonical correlation analysis(CCA)

CCA의 목표는 evaluative profile과 facet간의 선형 결합을 찾는 것. (+ 새롭게 만들어진 canonical 변수 간의 상관 관계를 최대화하는 목적)

CCA를 4가지 데이터에 대해 적용해보았음. (unfiltered text, evaluative text, non-evaluative text, evaluative text with sentiment intensity)

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/231948633-420c5037-d7e4-471c-8a31-854087914c26.png" width="40%"></img></div>

- 결과 아래 두 가지를 보였다.

  - evaluative prefiltering이 Big 5 personality에 더 잘 맞았음 (unfiltered, non-eval은 상관 관계가 낮음)

  - 감정 정보가 상관 관계를 증폭시킴

<br></br>

### 4. Conclusion

- text author와 evaluative language를 담은 토픽간의 연결을 통해 evaluative language와 personality 간의 관계를 연구함.

- Big 5로 레이블링된 reddit 댓글 데이터에 evaluative profiling을 적용하였으며 canonical correlation anlysis를 사용해 동일한 trait을 갖는 facet들이 canonical space에서 더 강한 연관을 가짐을 보여줌.

- 또, evaluative 표현이 personality 분석을 위한 정보를 더 많이 갖고 있음을 증명함.
