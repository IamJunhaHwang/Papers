## :page_facing_up: Multi-Domain Targeted Sentiment Analysis

### Info

* `conf`: NAACL2022

* `author`: Orith Toledo-Ronen, Matan Orbach, Yoav Katz, Noam Slonim

* `url`: https://aclanthology.org/2022.naacl-main.198/

* `video`: https://aclanthology.org/2022.naacl-main.198.mp4

### 1. Abstract & Introduction

`Targeted Sentiment Analysis(TSA)`는 소비자의 리뷰로부터 인사이트를 만들어내기 위한 주요 task이다. TSA는 텍스트에서 감정을 수반하는 용어를 탐지하고 어떤 감정인지 분류하는 것을 목표로 한다.   
Ex) `방이 시끄러웠어요. 그렇지만 음식은 맛있었어요`: `방:부정`과 `음식:긍정`

이런 소비자의 리뷰들은 많은 도메인에 걸쳐 존재하기 때문에, 다양한 도메인에서 잘 동작하는 multi-domain model이 필요하다. (도메인이 훈련 데이터에 없었더라도 잘 예측해야 함)

도메인마다 모델을 만들어 TSA를 서비스하는 것은 불가능하다. (GPU, memory 등의 수요가 커지므로) 따라서, 우리는 훈련 중 해당하는 도메인을 봤든 보지 않았든 좋은 성능을 내는 `single multi-domain model`을 목표로 한다. 

단적으로는 여러 도메인이 포함된 large dataset을 학습시키면 될 일이지만, 데이터가 다양하지 않으며 이를 새로 만드는 것도 어려운 일이다. 그러므로, `self-training`[(Chapelle et al., 2009)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4787647)을 통해 적은 레이블을 가진 훈련 데이터를 늘리는 것을 기반으로한 multi-domain TSA 모델을 제안한다.

- main contribution

  1) 최초의 multi-domain TSA

  2) 다양한 평가를 통해 `1`의 실현가능성을 증명

  3) TSA 데이터 셋을 새로 공개

<br></br>

### 2. Method

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/226094701-4401a1fc-a10b-42de-95a5-31473346adb5.png" width="70%"></img></div>

우리의 self-training 접근법은 `large multi-domain corpus`에서 생성된 `weak-label`들과 주어진 TSA 훈련 데이터 셋을 늘리는 것이다.

순서는 위의 그림과 같이 진행된다. 

1) 주어진 훈련 데이터 셋으로 `initial TSA model training`을 진행

2) `large unlabeld corpus`로부터 훈련된 모델로 TSA 예측을 한다.

3) 예측 중 몇 개를 선별하고 `weak-label`로서 훈련 데이터셋에 추가한다.    

#### 2-1. TSA Engine

우리는 TSA를 `sequence tagging problem`으로 보았고 모델은 input의 각 토큰마다 label(Pos,Neg,nOne)을 예측하게 된다.   
Ex) "Here is a nice electric car" => O O O O P P

위와 같이 연달아 같은 label이 나오는 경우 하나의 word-piece가 되도록 합쳤다.   

#### 2-2. Unlabeld Data Set

`YELP`리뷰 데이터를 훈련을 위한 `weak-label dataset`을 만들기 위해 사용했다. 

`2M`개의 문장을 추출했고 `not useful` 평가되었거나 특정 상업(식당, 숙소 등) 카테고리가 없는 리뷰는 제외했다. 문장에 `domain`을 할당했는데 이는 리뷰가 등록되어 있는 카테고리와 `YELP`에 존재하는 카테고리 중 할당된다. (즉, 여기에 없는 카테고리는 제외되는 것)

다음의 문서-레벨 필터링을 진행함.

1) 10~50 개의 단어가 있는 문장만 추출

2) 적어도 한 개의 sentiment를 담은 단어가 존재해야 함. (필터링을 위해 Opinion Lexicon 사용)

3) 각 리뷰의 도메인을 할당함. (60%: 식당, 40%: 그 외 다양한 도메인 이었음)

<img src="https://user-images.githubusercontent.com/46083287/226094709-7b9aaffb-5443-4cff-9112-70442dc7976e.png" width="40%"></img>   

#### 2.3 Generating Weak Labels

`Figure 1`을 보듯이, TSA labeld data에 모델(LD 모델)이 학습하기 시작한다.

LD 모델은 unsupervised data(YELP)에서 TSA target 영역(span)과 감성을 예측하는데 사용된다.

각 예측들은 `confidence score S`에 따라 다음과 같이 선택된다.

1) **targets**: `S > 0.9`인 target(tokens)이 있고 다른 target은 `S <= 0.5`라면 이 문장을 선택하며 높은 점수를 가진 target만 `weak label`로 추가하고 나머지 0.5 이하인 것은 label을 무시한다.(label 안붙임)

2) **non-targets**: 예측이 없는 문장이거나 모든 target이 `S <= 0.5`라면 이 문장을 선택하고 label을 무시한다. 이 파트의 양을 제안하기 위해 이 파트는 data의 10%만 랜덤으로 골라서 진행된다.

3) **domain balancing**: 도메인당 문장의 양은 각 파트별로 `20K`로 제한된다. (weak-label을 준 파트, label 안 준 파트)   

위 과정을 통해 선택된 문장들은 `labeled data`에 추가되고 TSA 모델이 fine-tune 된다. 그리고 이 과정을 3번 반복한다. (총 280K 문장이 새로 생성됨)

<br></br>

### 3. Empirical Evaluation

#### 3-1. 평가 데이터

- **YASO**: 다양한 소스의 유저 리뷰로 이루어진 TSA 데이터 셋

  - 다양한 리뷰가 존재하지만 일반적인 도메인에 편향되어 있기 때문에 `per-domain 평가`가 필요하다.

  - 이를 위해, YASO 리뷰 데이터 셋에서 각 리뷰들을 domain에 관해 labeling 했다.

  - 평가를 위해서만 쓰임.

- **MAMS**: 적어도 두 개의 다른 감정을 나타내는 target이 있는 식당 리뷰 데이터 셋  

  - `sentiment`: positive, negative, neutral(이건 제거함)

  - 500 문장이 test set으로 쓰임

- **SE**: 속성 기반 감정 분석을 위한 식당과 노트북 리뷰 데이터 셋

  - 문장 수 => `train`: 6072, `test`: 1600

  - 도메인의 균형이 맞도록 나누었음.

  - MAMS 처럼 neutral과 mixed sentiment label은 제거했음.

#### 3-2. Language Models

- `BERT-B`: BERT-Base uncased with 110M parameters

- `BERT-MLM`: YELP 데이터 셋을 이용해 domain-specific pre-training **with MLM** 을 진행한 BERT-B 모델

  - `각 문장에서 단어의 15%`, `각 문장에서 감정이 드러난 단어의 30%`를 랜덤하게 마스킹

- `BERT-PT`: YELP의 식당 리뷰와 QA 데이터로 domain-specific pre-training **with MLM & NSP**를 진행한 BERT-B 모델

- `SENTIX`: sentiment-aware language model (YELP, Amazon 리뷰 데이터 사용해 사전학습)

  - Transformer Encoder만을 사용하며 감정이 드러난 단어, 이모티콘, 일반 단어를 랜덤으로 마스킹하는 MLM task 사용.

#### 3-3. Experimental setting

Training Setting|Contents
|:---:|:---:|
|Loss| Cross-Entropy|
|Learning-rate|3e-5|
|Epsilon|1e-8|
|batch-size|32 x 2 GPU|
|# of epoch|15|

- **평가**

  - 다른 random seed로 10개의 모델 훈련

  - 각각 per-domain performance를 측정했고 마지막 결과의 평균을 계산함

  - 각 데이터셋의 전체 성능을 포함하기 위해 macro-average됨.

  - 평가는 precision, recall, F1으로 나타냈음.

<br></br>

### 4. Results

#### 4-1. In-Domain Results

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/226094716-14ec61ea-3a1f-4ca9-9b05-22fd8720c00c.png" width="70%"></img></div>


널리 사용되는 SE(Sentiment Evaluation) 평가 데이터로 in-domain 성능을 나타냈음.

WL(weal-label)이 모든 모델의 성능을 높였음. => self-training이 in-domain에서 효과가 있었음.

**[내 생각]: 이는 `in-domain`이라고 할지라도, training data에서 학습하지 못했거나 약하게 학습한 부분을 unlabeld data를 이용해 찾아서 추가 학습하는 과정을 통해 성능 향상이 되는 것을 의미하는 것.**

#### 4-2. Multi-Domain Results

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/226094719-25562427-b24c-4013-80ee-7b75ce836cf1.png" width="70%"></img></div>

논문에서 제안한 self-training을 통한 WL이 성능 향상을 이끌었으며 이는 Precision의 증가에 기반했다.

또한, stronger base model인 `SENTIX`, `BERT-PT`에서도 성능의 향상을 보였다.

`Table 4`는 이전의 `cross-domain TSA` 연구와 우리의 접근법을 비교한 것이다. `Gong-UDA`가 `Gong-BASE`보다 성능을 높였지만 우리 것보다는 성능이 낮았다.

<br></br>

### 5. Analysis

#### 5-1. Impact of the Initial LD Model

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/226094726-940c4259-3e38-4cb5-b488-f91e24c2f6ae.png" width="70%"></img></div>

우리가 제안한 TSA 모델은 처음에 생성하는 LD Model이 핵심이다. (self-training의 질을 결정하므로)

이와 관련해 `self-training process`에 LD Model이 얼마나 영향을 주는지 LD Model 훈련 데이터에 제한을 두는 방식으로 분석했다.

예상한대로, single domain이나 데이터의 반만 훈련하는 것은 성능을 낮췄다.

그럼에도, `self-training`은 전체적으로 성능을 높여 주었다.

#### 5-2. Diversifying the Training Set

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/226094733-b9dc87f6-4f85-49d0-aa05-08d59cbe8dc8.png" width="70%"></img></div>

`self-training`의 대체할 방법은 TSA training set이 넓은 도메인을 포함할 수 있게 하는 것이다. 따라서, 이를 비교하기 위해 어노테이터를 모아 레이블링된 다양한 도메인의 리뷰 데이터(952 sentence)를 만들었다.

우리의 방법과 비교한 결과, 우리의 방법이 성능이 더 좋거나 비슷했다.

**[내 생각]: 그런데 이 비교는 의미가 없는 것 같다. 어노테이터를 고용해 추가로 만든 데이터가 얼마 없는 것에 비해 `self-training`은 280K의 데이터로 진행되었기 때문이다.**
