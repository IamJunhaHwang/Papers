## Language Models are Unsupervised Multitask Learners

### Info

* `conf`: Preprint 2019
* `author`: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever

* `url`: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

* `github`: https://github.com/openai/gpt-2

### 1. Abstract & Introduction

NLP 태스크들은 해당 태스크를 위한 특정 데이터 셋에 지도 학습을 적용하는 방법으로 이루어졌다. 하지만 이 논문에서 우리는 이러한 특정 데이터 셋 없이 몇 백만 개의 웹페이지 데이터 셋인 WebText로 학습해 앞서 말한 태스크들을 배울 수 있다는 것을 증명한다. (좀 더 일반적인 시스템을 만들고자 함) 언어 모델의 크기는 zero-shot task transfer에 필수적이며 이를 증가시킬수록 선형적으로 성능이 증가한다.

기존의 ML들은 특정 task에 대한 지도 학습을 통한 접근법이 우세했으며 여러 연구가 이와 같은 접근법이 다양성 측면에서 모델의 불규칙한 움직임이 있음을 보였다. 이런 일반화의 부족은 단일 태스크 학습으로부터 기인한다고 의심하며 robust system을 위해서는 다양하고 넓은 범위의 도메인과 태스크로 훈련되고 테스트되어야 한다.

해당 논문에서는 언어모델이 파라미터나 모델 구조 변경 없이 zero-shot 세팅으로 다운스트림 태스크들을 수행할 수 있다는 것을 증명한다.

<br></br>

### 2. Approach

우리의 접근법의 핵심은 `Language Modeling` 이다. Language Modeling은 가변 길이의 심볼 시퀀스의 집합으로부터 비지도 분포 추정으로 구성된다.

언어는 자연적으로 순차적 순서를 가지고 있기 때문에, 주로 조건부 확률의 곱으로써 심볼간의 결합 확률로 공식화 된다. (아래와 같음)

$p(x) = p = \overset{n}{\underset{i=1}{\prod}} p(s_n | s_1, ... , s_{n-1})$

general system은 다른 많은 태스크를 수행할 수 있어야 하므로, input 뿐만 아니라 수행할 task에 대해서도 조건화되어야 한다.

$p(output | input, task)$

여기서 태스크 조건은 자주 모델 구조 레벨이나 알고리즘 레벨에서 구현된다. 하지만, `McCann et al. (2018)`에서 예처럼 언어는 특정 태스크, 입력, 출력을 모두 심볼 시퀀스로 제공한다. [ex. (translate to french, english text, french text)]

Language Modeling은 위와 같은 명시적 지도 없이 McCann 의 태스크를 배울 수 있다. `Next-Token Prediction` 측면에서 볼 때, 비지도 목적함수가 곧 지도 목적함수가 되기 때문에 비지도 목적함수의 전체 최적 값은 지도 목적 함수의 전체 최적 값이 된다. (next token 자체가 레이블이 된다)

dialog data가 매력적이지만, 제약을 우려해 많은 양을 가지고 있는 인터넷 데이터를 사용하기로 하였다. 우리의 추측은 충분한 크기를 가지는 언어 모델은 다음 단어를 더 잘 예측하기 위해 여러 태스크들에서의 정답을 추론하고 수행하는 것을 배우기 시작할 것이다.

<br></br>

#### 2-1. Training Dataset

많은 이전 연구들이 단일 도메인 텍스트로 학습했지만 우리는 가능한 크고 다양한 데이터셋으로 학습하고자한다. 따라서, 다양한 도메인과 거의 무한에 가까운 텍스트들인 `Common Crawl`와 같은 웹 스크랩이 유망한 소스인데 여기에는 데이터 질에 대한 이슈가 있다. (거의 대부분이 의미가 없는 문장들로 이루어짐)

그래서 우리는 문서의 질에 집중한 웹 스크랩 데이터를 새로 만들었다. 사람 손으로 필터링하였고 Reddit에서 최소 3개의 karma를 받은 모든 아웃바운드 링크를 스크랩하였다. 결과로 `WebText`는 45M개의 링크를 포함한다. HTML response로부터 텍스트를 추출하기 위해 Dragnet과 Newspaper content extractors를 이용했다. 2017년 12월 이후에 만들어진 링크를 포함하지 않았으며 중복 제거, 휴리스틱 정제 과정을 거쳐 8M의 문서 (40GB의 텍스트)를 얻었다.

<br></br>

#### 2-2. Input Representation

범용 언어 모델은 어떤 문자열이라도 그 확률을 계산하고 생성할 수 있어야 한다. 현재의 대규모 언어 모델은 전처리 과정을 포함하기 때문에 모델이 생성 가능한 문자열의 공간을 제한한다. 유니코드 문자열을 UTF-8 바이트 시퀀스로 처리하면 이런 문제를 피할  수 있는 반면 현재 byte-level 언어 모델은 word-level 언어 모델보다 큰 데이터 셋에서 낮은 성능을 보인다.

BPE가 돌파구가 될 수 있는데 BPE는 자주 등장하는 단어와 자주 등장하지 않는 문자의 중간을 모델링한다. BPE는 이름과 다르게 바이트 시퀀스가 아닌 유니코드 포인트에서 동작한다. 이는 전체 유니코드 심볼 공간을 필요로 하게 된다. (130,000 base vocab)

우리는 BPE가 흔한 단어의 많은 다른 표현들을 포함하고 있는 것을 관찰했다. (ex. d`og.` `dog!` `dog?`)  
이를 방지하기 위해 우리는 BPE가 문자 범주를 넘어가는 병합을 막았다.

이러한 입력 표현은 byte-level 접근법의 일반성과 word-level 언어 모델의 경험적 이득을 결합할 수 있게 한다. 이 접근법이 어떤 유니코드 문자열이든 확률을 부여할 수 있기 때문에 우리의 언어 모델이 전처리, 토크화 등과 관련 없이 평가할 수 있게 만들어 준다.

<br></br>

#### 2-3. Model

GPT-1과 같은 구조를 가지되 몇 가지 변경점이 있다. `Layer normalization`이 각 sub-block의 입력으로 옮겨졌고 마지막 self-attention block후에 추가 `Layer normalization`을 거친다. 모델 깊이에 따른 residual path 누적의 초기화가 변경되었다. (residual layer의 가중치를 residual layer 수인 N을 이용해 $\frac{1}{\sqrt N}$ 로 스케일링)

vocab 크기를 50257로 확장했으며 text max_length를 1024로 늘렸다. 그리고 batch-size로 512를 사용했다.

<br></br>

### 3. Experiments

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/c80dc9ee-d10a-49ef-8067-5acc93fc9f9a" width="40%"></img></div>

위와 같이 4개의 LM을 훈련하고 성능을 관찰함. 가장 작은 모델은 이전의 GPT-1과 크기 같으며 2번째로 작은 모델은 BERT와 같은 크기이다. 가장 큰 모델은 GPT-2는 GPT보다 몇 배 이상 많은 파라미터를 가지고 있다. 각 모델의 학습률은 WebText의 valid set(전체의 5%)에 대해 가장 좋은 성능을 내도록 수동으로 조절되었고 underfit 되어있다.

#### 3-1. Language Modeling

zero-shot task transfer을 향한 첫 걸음으로 WebText LM이 zero-shot domain transfer에서 어떻게 작동하는지에 관심을 기울였다. (language modeling으로만 훈련하였을 때, 여러 NLP task에 대해 어떻게 작동하는지)

우리 모델은 byte level로 작동하므로 어떤 언어 모델 벤치마크라도 평가할 수 있다. 각 데이터 셋에 대한 결과는 예측한 token별로 negative log probability를 구한 것의 평균을 지수 함수로 표현한 것이나 스케일링한 것으로 보고한다. (PPL)

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/866f636e-c59b-4030-ab4a-a88c88794429" width="70%"></img></div>

WebText LM이 8개 중 7개에서 SOTA를 달성했다. 

#### 3-2.  LAMBADA

LAMBDA 데이터셋은 텍스트에서 장거리 의존성 모델링의 능력을 평가한다. [최소 50 토큰을 필요로하는 문장의 마지막 단어를 예측하는 태스크]

GPT-2는 기존 연구에 비해 다음과 같이 성능 향상을 이뤘다.

- PPL: 99.8 -> 8.6

- Accuracy: 19% -> 52.66%

GPT-2의 오류를 조사해보았는데 대부분의 예측이 유효한 문장의 연속이었지만 마지막 단어는 그렇지 않은 경우였다.

#### 3-3. Reading Comprehension

CoQA는 질문자-답변자 사이의 자연어 대화 쌍으로 이루어져있으며 7가지 서로 다른 도메인으로 구성된 QA 데이터 셋이다.

문서, 관련된 대화 기록, 최종 토큰 A에 따른 GPT-2의 Greedy decoding: 55 F1-score 달성

4개 중 3개의 베이스라인 성능을 능가하였음. [127K의 질문-정답 쌍으로 훈련되지 않았음에도]

<br></br>

### 4. Generalization vs Memorization

CV 분야의 최신 연구는 흔한 이미지 데이터셋에서 train 데이터가 test 데이터와 겹치는 것이 있어 성능이 높게 잡히는 문제가 있음을 보였는데, 이는 우리가 사용한 WebText에서도 발생할 가능성이 크다. 따라서, test 데이터가 얼마나 train 데이터와 겹치는지 분석하는 것은 중요하다.

위를 위해 WebText의 train 데이터셋 토큰의 8-gram을 포함하는 Bloom filter를 만들어 테스트했다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/11925fc7-1249-41cb-ae74-406f67ea22e6" width="70%"></img></div>

보통의 LM 데이터셋들의 test 데이터는 1~6% WebText의 train 데이터와 겹쳤으며 각자의 train 데이터와 겹쳤다.
