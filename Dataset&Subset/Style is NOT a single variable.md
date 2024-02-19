## :page_facing_up: Style is NOT a single variable: Case Studies for Cross-Style Language Understanding

### Info

* `conf`: ACL 2021
* `author`: Dongyeop Kang, Eduard Hovy
* `url`: https://arxiv.org/pdf/1911.03663.pdf

* `github` : https://github.com/dykang/xslue

### 1. Abstract & Introduction

모든 text들은 어떠한 스타일로 쓰여진다. 이 스타일은 다른 스타일적 요소(감정, 은유, formality marker 등)들의 복잡한 조합과 함께 변함(상호작용)으로 만들어진다. 이러한 함께 변하는 조합들에 대한 연구는 stylistic language(== Cross-style language understanding) 분야에 빛을 비추었다. 하지만 이전 연구들은 한정된 영역에서의 스타일 상호 의존성을 다루었다. 따라서, 본 논문이 첫 번째로 포괄적인 cross0stylistic language를 다루는 연구이다. (서로 다른 스타일이 어떻게 상호 작용하는지, 어떻게 체계적으로 text를 만드는지에 집중)

본 논문에서는 이미 존재하는 데이터셋과 `sentence-level cross-style language undersanding`과 평가를 위한 새로운 데이터 셋을 모은 것을 조합한 벤치마크 말뭉치인 `XSLUE`를 제공한다.

벤치마크는 4가지 이론적 그룹(비유적, 개인의, 정서적, 대인관계)을 따르는 15가지 다른 스타일로 이루어진다. (+ 유효한 검증을 위해 똑같은 텍스트에 15개 스타일로 어노테이션을 진행한 데이터셋을 만듦)

- **Contribution**

  - 15가지의 서로 다른 스타일과 23개의 sentence-level classification task를 종합함.

  - 유효한 검증을 위해 똑같은 텍스트에 15개 스타일로 어노테이션을 진행한 cross-style dataset을 모음(만듦)

  - classification, correlation, generation에서의 cross-style 변형을 연구함.

    - multiple style로 학습한 분류기가 하나의 스타일로 학습한 분류기보다 더 나은 성능을 보임.

    - 사람이 작성한 글에서 특정 스타일들이 서로 강한 의존성을 지님을 밝힘. (correlation)

    - conditional stylistic generator를 만들어 더 나은 스타일 분류기가 더 나은 스타일 생성이 가능함을 보였음. (몇 개의 상충적 스타일의 조합(무례함-긍정 감정)은 말이 안되는 문장을 만들어 냈음.)

- cross-style analysis의 예시: 감정과 인구통계학적정보(성별, 나이 등)의 상관관계, 텍스트에서 은유와 감정의 상호작용 등

<br></br>

### 2. Method - XSLUE

#### 2-1. Style selection and groupings

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/233261886-90e9b7b3-ca43-4385-97ad-4fd2bed4768b.png" width="70%"></img></div>

포괄적인 스타일 연구를 위해 다른 스타일을 가지는 여러 데이터 셋들을 모아야한다. 따라서, 15개의 널리 쓰이는 스타일의 레이블링된 데이터 셋들을 선택했다. (위의 그림처럼 스타일을 4가지 그룹으로 나눔, 각각의 그룹은 대화에 있어서 각각의 목적을 가지고 있음)

이런 그룹핑이 추후 case study에서 이들의 의존성을 탐지하는 basic framework로 쓰일 것이다.

#### 2-2. Individual style dataset

1K 이하의 작은 크기의 데이터셋은 포함하지 않았다. (large model training에 적합하지 않다고 봄)
또, 데이터 셋에 다른 타입(문서-수준 분류)들이 존재하지만 하나의 문장을 분류하도록 제한했다.

데이터 셋은 `0.9:0.05:0.05`로 나누었으며 레이블링이 positive만 존재하는 경우 negative sampling을 진행했다. [Khodak et al. (2017)]

#### 2-3. Cross-style diagnostic set

각각의 데이터 셋들은 도메인, 레이블 분포, 데이터 사이즈가 모두 다르므로, 각각의 데이터에 해당하는 테스트 셋을 multiple style을 함께 사용해 평가하는 것은 적절하지 않다. 따라서, 우리는 `cross-set`이라는 추가적인 진단 데이터 셋을 만들었다. (같은 텍스트에 15개 스타일의 레이블을 사람을 모아 만들었음)

- 아래의 두 가지 다른 소스에서 총 500개 텍스트 모았다

  1. 15개의 style dataset에서 랜덤하게 선택하였다. (각 stlye dataset에서의 샘플링을 밸런스하게 수행)
  
  2. pre-train한 스타일 분류기를 사용해 style 예측 점수들 사이에서 높은 차이를 보이는 트윗을 랜덤하게 선택


5명의 어노테이터에 의해 레이블링 되었으며 각 스타일의 마지막 레이블은 다수결에 의해 결정되었다. personal style에 `Don't know` 선택을, 이진으로 분류되는 스타일에는 `Neutral` 선택을 추가하였다.

<br></br>

### 3. Experiments

#### 3-1. Cross-Style Classification

각각을 모델링하는 것이 아니라, 어떻게 멀티 스타일들을 함께 모델링하는지 연구하였다.

- 모델: `single` 과 `cross` 모델을 비교함. (single: 스타일 각각만을 사용해 훈련, cross: 모든 데이터 셋을 섞은 것으로 훈련)

  - single model로 다양한 baseline model을 사용함. (훈련 데이터에서 가장 많이 등장한 레이블을 선택하는 모델, biLSTM with GloVe, BERT, RoBERTa, T5)

  - cross 모델로는, 스타일 사이에서 공유하는 내부 표현과 cross-tyle 패턴을 학습하는 인코더-디코더 모델을 제안함.

    - 다른 스타일의 입력("STYLE: formality TEXT: would you please")을 인코딩하고 출력 레이블("formal")을 디코딩함.

    - 사전학습된 T5를 사용했으며 XSLUE 데이터로 Fine-Tuning하였다.

  - 또한, single model을 multi-task setup으로 XSLUE 데이터를 이용해 훈련하였다. (낮은 성능 보임)

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/233262001-b83fd3b0-0108-40f6-a6dc-845781d3e365.png" width="30%"></img></div>

- Tasks: 개별 데이터 셋의 테스트 셋으로 분류기를 평가한 `individual-set evaluation` 과 cross-set 으로 평가한 `cross-set evaluation` 으로 나뉜다.

  - 데이터의 불균형인 점을 고려해, F1으로 평가하였으며 회귀 task의 경우 Person-Spearman 상관관계로 평가했다. 또한, 다중 레이블(Multi-label)의 경우 모든 점수에 각 레이블의 macro-average를 취했다.
  
<div align="center"><img src="https://user-images.githubusercontent.com/46083287/233262042-20d6db78-c7b0-45af-86f6-18a43f4b0242.png" width="70%"></img></div>

- Results

  - 평균적으로, individual-set에서 biLSTM보다 fine-tune한 트랜스포머 모델들의 성능이 높았다.

  - 우리가 제안한 모델이 single 모델보다 성능이 더 좋았으며 이는 multiple style을 함께 학습하는 이점을 보여줬다고 볼 수 있다. (특히, cross model이 personal style에서 상당한 성능 향상을 보임.)

  - cross-set에서 전체적인 성능이 떨어졌다. 이는 cross-style 변형의 올바른 평가를 위해서는 제안한 diagnostic set이 왜 중요한지 보여준다.

  - cross-set에서도 cross 모델이 single 모델보다 더 좋은 성능을 보였다.
   
#### 3-2. Style Dependencies

분류기로의 silver prediction을 이용해 사람이 쓴 글과 스타일이 어떻게 연관되어 있는지 경험적으로 찾았다.

- setup

  - 다른 도메인에 비해 스타일이 다양한 트윗을 target domain으로 설정하고 100만개의 트윗을 랜덤하게 추출했다.

  - 이전의 fine-tune한 cross-style classifier로 각 트윗에 대해 53개의 스타일 속성을 예측했다. (감정 스타일에 대한 긍/부정)

  - 스타일 속성간의 피어슨 상관관계를 나타내었다. (p-value < 0.05인 값만 보여주고 나머지는 지웠음)

- Results

  - 두 명의 어노테이터가 평가한 스타일 의존성과 비슷한 양상을 보였음. (합리적인 상관관계 정확도(accuracy)]

  - 또한, ` Ward hierarchical clustering`를 사용해 스타일의 경험적 grouping을 제공한다. (논문의 figure 참고)

#### 3-3. Cross-Style Generation

input text, target style을 각각 $x$, $s$라고 하자. PLM과 같은 생성기 $P(x)$ 와 위에서 사용했던 스타일 분류기 $P(s | x)$ 를 합칠 것이다. 이를 위해, `plug-and-play language model(Dathathri et al., 2019)`를 사용해 fine-tune하지 않은 사전 학습된 GPT와 XSLUE로 훈련된 스타일 분류기를 합쳤다.

모델이 만드는 출력에 대해 평가함. (모델이 만드는 문장에 대한 예시가 논문에 있음)

- 평가

  - XSLUE 훈련 셋에서 랜덤하게 추출한 20개의 빈번한 프롬프트가 주어지면, 각 프롬프트마다 4 개의 스타일에 대한 각각의 이진 레이블을 위해 10개의 연속된 텍스트를 생성함. (= 20 * 10 * 2 * 4)

  - automatic, human-measure 둘 다 사용해 평가함. (automatic: fine-tuned 분류기로 F1-score, human: Likert scale)

- 결과

  - XSLUE의 각 test set과 비교했을 때 생성자의 출력이 20.5% 낮은 점수를 보였다.

  - 사람 평가에서 negative label이 적은 스타일적 타당성 점수를 받았다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/233262163-a8f99e4a-3ad0-4047-9868-c31bccca5278.png" width="40%"></img></div>

- Better classification, better generation

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/233262184-c3725c02-146c-4b83-902e-e3106711de2f.png" width="40%"></img></div>

분류기의 성능과 생성의 품질간의 관계를 조사하기 위해 분류기 훈련을 제한하여 실험하였다. -> $P_{C \%}(s|x)$

결과, 스타일에 대해 더 많이 이해할수록(분류기 성능이 높을수록) 더 높은 스타일 생성이 가능했다.

- Contradictive styles in generation

이전 생성에서는 하나의 스타일에서만 실험했지만, 여러 스타일(본 실험에서는 2개; 감정, 정중함)로 문장을 생성해보려 한다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/233262222-bb4d4bad-3d58-444a-95a8-8dfcea48a227.png" width="40%"></img></div>

(positive, impolite)나 (negative, polite)가 서로 반대되기 때문에 낮은 stylistic appropriateness 점수를 받았다.
