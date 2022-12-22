## :page_with_curl: Text based personality prediction from multiple social media data sources using pre‑trained language model and model averaging


### Info

- Journal: Journal of Big Data
- url: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00459-1
* year: 2021 

* 추가로 데이터 구할 수 있나 확인하기

### Abstract + Introduction

SNS의 등장으로 텍스트 데이터가 특히 급증하였고, 사람들이 SNS를 통한 활동을 함에 따라 이를 이용해 개인의 개성(성격)을 특징화(characterize)할 수 있게 됨. 실제로 온라인(SNS)에서의 행동과 개인 성격과 강한 상관관계가 있음이 일찍이 연구로 증명됨.

personlaity prediction이란 디지털 컨텐츠로부터 특징을 뽑아내 personality model에 mapping하는 것. [여기서 personality model은 딥러닝 모델이 아닌 사람의 성격 특성 모델로 잘 알려지고 이미 검증된 `big five personality`]

이를 위한 이전 연구(RNN, LSTM)에는 아래의 한계가 있었음

1. train 시간 오래 걸리며 순차적 입력에 의존

2. 단어에서 semantic한 의미를 찾아내는 능력이 부족함

#### 논문에서 주장하는 Contribution

1. 위 문제들을 해결하기 위해, 해당 논문에서는 SNS에서의 특징 추출 방법으로 다중 언어 모델을 조합한 것을 이용하고 시스템은 모델의 평균에 기반해 예측을 함. (+ 추가적인 NLP Features 이용; TF-IGM, NRC)

2. 하나의 SNS 데이터만 사용한 이전 연구들과 다르게 두 개의 SNS 데이터(facebook, twitter)사용.

3. 각 성격 특징마다 예측 모델을 만들었으며 특징 추출 방법을 조합한 양방향 context feature 사용. 

4. 논문에서 제시한 모델들이 이전 연구보다 좋은 성능을 냈음.

<br></br>

### Methodology

- **Data**
  1. Facebook의 `MyPersonality` [250 user 참여]
  2. 메뉴얼대로 수집된 Bahasa Indonesia(인도네시아 공용어) Twitter 데이터 + 추가 수집(Twitter API 이용) [심리학 전문가가 labeling]
  3. 위 데이터들은 70:15:15로 각각 Train:Valid:Test

    <img src="https://user-images.githubusercontent.com/46083287/209074589-7641d6ea-f20f-4b38-b11f-53d8bf313a90.png" width="60%"></img>

- Entire Flow Chart

  <img src="https://user-images.githubusercontent.com/46083287/209074776-f4bf44c2-7bf1-43ee-b4e8-16a7022603da.png" width="80%"></img>

  위와 같이 크게 Initiation, Model Development, Model Evaluation 스테이지로 나뉨.

- **initiation Stage**

  - 트위터 데이터 추가 수집
  - 페이스북 데이터는 오픈 소스임(당시)
  - 모든 데이터셋은 `Big Five personality traits modeling approach`에 기반
  - 각 데이터셋의 언어가 다르므로 각각에게 맞는 전처리 작업을 거침. [그 결과는 특성 추출과 선택(Token)으로 나올 것임] 

- **Preprocessing**

  - 모든 데이터셋은 특성 추출 전에 전처리를 거침.
  - 트위터 데이터의 경우 영어로 바꾸어 주는 과정이 추가되는 것을 제외하고는 두 데이터셋 모두 똑같은 전처리 과정을 거침.
  
  - [URL, 심볼, 특수문자 제거] -> (트위터 데이터의 경우 영어로 번역) -> [축약형을 원형으로 복원] -> [모두 소문자로 변환] -> [접사와 불용어 제거] -> [어간 추출]

  - 접사와 불용어 제거, 어간 추출은 `NLTK`라이브러리 이용
  - 트위터 데이터를 영어로 번역하는 과정은 `Google Translate API`이용

- **Feature Extraction**

  - 본 논문에서는 특징 추출을 사전 학습 모델을 이용한 특징 추출[토큰 임베딩]과 통계학적 특징 추출[TF-IDF]의 두 가지 유형으로 나눔.

  - 통계학적 특징 추출로 이전 연구에서 사용된 TF-IDF대신 `TF-IGM` 사용
    - 큰 값을 가지는 값이 분류 모델을 만들 때의 특징으로 사용됨. [특정 클래스의 중요한 뜻을 담고 있는 것으로 판단]
    
    - `TF`: 문서 d에 등장한 단어 t의 개수 / 문서 d의 총 단어 개수
    - `IGM`: 1 + λ(문서 d에 등장한 단어 t의 개수 / 특정 클래스에 등장한 단어 t 개수) [λ는 조절 계수]

      <img src="https://user-images.githubusercontent.com/46083287/209075135-65d4aa02-1f58-4292-addf-f69adfa7ad6f.png" width="70%"></img>

  - 사전 학습 모델의 경우 아래 사진과 같음. [특별히 추가된 과정 없이 흔하게 사전 학습 모델의 input 생성 과정]
    - 각 모델에 설정된 `max_length = 512`; 각 모델 논문에 언급된 대로

    <img src="https://user-images.githubusercontent.com/46083287/209074972-b5519392-426e-4bfd-b4a1-1d0931dbc178.png" width="70%"></img>

  - 추가로 `Semantic analysis`와 `NRC emotion lexicon` 특징 사용.

    <img src="https://user-images.githubusercontent.com/46083287/209075230-c19ad702-67b4-4940-be7c-44d7df5555fa.png" width="70%"></img>

- **Model Prediction**

  - 미리 정의한 모델 feature와 통계학적 text feature를 조합한 멀티 모델 딥러닝 구조를 제시함.
  
    - 멀티 모델이기 때문에 나온 결과를 `averaging`함.
    
    - 각 모델의 `softmax probability`를 조합해 계산함. [아래와 같이; k= N of class]
    - 평가 지표로 `F1-score`, `Accuracy` 사용

      <img src="https://user-images.githubusercontent.com/46083287/209075275-a61f35bb-f326-4efb-9969-53cecd27f5c7.png" width="40%"></img>

  - big five personality model에 맞게 5개의 분류기를 만듦 (위의 구조를 가진 모델을 5개 만드는 것)

    <img src="https://user-images.githubusercontent.com/46083287/209075288-568040e4-14c7-4fcc-a335-369d94ce46df.png" width="70%"></img>

<br></br>

### Experiment & Result

- 이전 연구와 비교하기 위해 아래 세 가지로 나누어 실험을 진행 함.
  1. 사전 학습 모델만 사용(RoBERTa, BERT, XLNet
  2. 위의 모델 + 통계학적 feature(TF-IGM, NRC 등)
  3. 본 논문에서 제안한 모델(2번에 대해 3가지 모델[!번에 해당하는]을 만들어 앙상블)

- 여러 batchsize-learning-rate를 조절해보며 실험. (GridSearch) [각 시나리오: 논문 Table 6]
  - 10-fold Validation 사용.
  - 통계학적 feature를 함께 사용한 것이 좋은 성능을 보임. (사용하지 않은 것에 비해)

  <img src="https://user-images.githubusercontent.com/46083287/209075355-c6b7d9f5-f507-44ee-a4ac-860d63f75c0e.png" width="70%"></img>

  <img src="https://user-images.githubusercontent.com/46083287/209075375-c7cbb4cb-1765-4c53-9f6a-929d2f672e0a.png" width="70%"></img>

### Conclusion

- 본 논문에서 제안한 모델(BERT, RoBERTa, XLNet과 각각에 통계학적 feature를 적용하고 각 모델의 평균을 취한 것)이 가장 좋은 성능을 보임

- 통계학적 feature들(TF-IGM, 감성 분석, NRC lexicon database)이 개성 예측 시스템에 상당히 큰 기여를 함.

- 후속 연구들로 생각해볼 수 있는 것: 데이터 셋 늘리기, 다른 사전 학습 모델(ALBERT, DistilBERT, BigBird BERT) 사용해보기
