## :page_facing_up: Convolutional Neural Networks for Sentence Classification

### *Info*

* `authors`: Yoon Kim

* `Conference`: EMNLP 2014
* `url`: https://aclanthology.org/D14-1181/
* `code\[non-official\]`: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb

  \+) https://github.com/graykode/nlp-tutorial

- `code[official]`: https://github.com/yoonkim/CNN_sentence

- `Guide`: https://chriskhanhtran.github.io/posts/cnn-sentence-classification/#2-data-preparation

<br></br>

### *Abstrct*

* 문장 분류 task에서 **pretrained word vector**에 CNN을 학습시키는 실험을 보여줄 것임.

- `task-specific`과 `static vectors`를 사용하기 위한 간단한 구조의 변환을 제시함.

- 본 논문의 CNN model이 7개의 task 중 4개(sentiment, question classification, etc)에서 좋은 성능을 보임.

<br></br>

### *Introduction*

- CV에서 CNN은 주목할만한 성능을 보였다. 이러한 CNN은 CV를 위해 발명되었지만, NLP에도 효과를 보이고 있다.

- **본 논문에서는 비지도 학습으로 얻어진 단어 벡터(Word2Vec)을 통해 하나의 `Conv Layer`를 가지는 CNN을 훈련시킬 것이다.**

  - 단어 벡터는 그대로 두고, 모델의 다른 파라미터만 학습시킬 것임.

  - `fine-tuning`을 통한 `task-specific` 단어 벡터를 학습하는 것으로 성능을 더 향상 시켰음.

- `pretrained & task-specific`한 벡터를 사용하기 위해 다수의 채널을 가지게 하는 작은 변경을 소개한다.

<br></br>

### *Model*

<div align="center"><img src="[/uploads/342fe6e5ae682ea9144d134685520b4d/image.png](https://user-images.githubusercontent.com/46083287/209088719-6342b0ee-f7c0-4d70-ad08-46b8e487201f.png)" width="70%"></img></div>

<br></br>

- `Collobert et al. (2011)`의 모델을 조금 변형한 모델

- *문장의 표현*

  <img src="https://user-images.githubusercontent.com/46083287/209089161-79db5b52-25ab-472d-99e7-2a63adec3335.png" width="40%"></img>

  - `x_i`: k차원 단어 벡터 (i = i번째 단어, 실수 값을 가짐)
  - `n`: 문장의 길이 (필요하다면 pad 진행)
  
  - `⊕`: concatenation 연산자

- *Convolution operation*

  - _filter_ **w** ∈ R^{hk} (h: window, k: k-demension)

  - h개의 단어를 보고 새로운 feature 생성. (window) ==> 이렇게 생성된 새로운 feature `c_i`는 아래와 같다.

    <img src="https://user-images.githubusercontent.com/46083287/209089259-b00c213c-d770-4d12-b3de-f0dc6a4312a8.png" width="30%"></img>

      - `b`: 편향 (실수 값)
      - _f_: non-linear function (`tanh`)
  
  - 위와 같은 `filter`는 가능한 모든 단어의 window에 적용되며, `feature map`을 생성함.

    - 모델은 다양한 윈도우 크기를 가진 다수의 필터를 사용한다. (그림에서도 여러 개의 필터 사용)

    - 다채널의 경우, 각 필터는 모든 채널에 적용되며, `c_i`에 더해져서 들어감. (ex. c_1 = channel 1의 1번째 윈도우 feature + channel 2의 1번째 윈도우 feature)

<br></br>

- Max-overtime pooling을 사용. ==> *feature map 중에서 가장 큰 값(max) 사용*

- 그 후, `fully connected softmax Layer`를 지난다. (with dropout)
  - label의 확률 분포를 출력

<br></br>

- *Regularization*

  - 위에서 말햇듯이 `dropout`을 적용했음. [가중치의 크기 (ℓ2​  norm)을 패널티로 사용]

  - 끝에서 두번째 layer인 z가 아래와 같이 주어졌을 때(`c_hat`은 `c`에서 `max-over-time pooling`을 적용 후)

    <img src="https://user-images.githubusercontent.com/46083287/209089399-78e196d7-0547-4fea-a6a5-edca2b9507ca.png" width="20%"></img>

  - 아래와 같이 (4) 대신 (5)의 식을 사용한다.

    <img src="https://user-images.githubusercontent.com/46083287/209089416-2b6a4a80-4d74-496d-a14a-1b73a792cbec.png" width="60%"></img>

    - `∘`: 원소별 곱셈 연산자
    - **r**: 1일 확률이 `p`인 베르누이 랜덤 변수의 `masking` 벡터 (마스킹이 안된 유닛을 통해서만 기울기가 역전파됨.)

  - test 시에는, 학습된 벡터가 `p`에 의해 조정된다. --> `w_hat = pw`
    - `w_hat`은 보지 않은 문장의 score로 사용된다. 

  - 추가적으로, 가중치 벡터의 `ℓ2​  norm`을 제한한다. (`ℓ2​  norm`의 값이 `constraint`를 넘으면 해당 `constraint`값으로 고정)

<br></br>

### *Experiment*

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209088929-77f783a0-960d-44b0-aed0-caccdfa1e733.png" width="40%"></img></div>

<br></br>

- _Datasets_

  - **MR**: 한 문장의 영화 리뷰들 (긍정/부정 분류 탐지)

  - **SST-1**: Stanford Sentiment Treebank

    - `MR`의 확장판, train/dev/test 셋이 나뉘어 있으며 클래스가 <매우 긍정, 긍정, 중립, 부정, 매우 부정>으로 세부적으로 나뉨.

  - **SST-2**: `SST-1`과 같지만, 클래스가 <긍정/부정> 으로만 나뉨.

  - **Subj**: 문장이 주관적(Subjective)인지 객관적인지를 구분하는 `task`의 데이터셋

  - **TREC**: Text REtrieval Conference Dataset, 질문을 6가지 타입으로 분류하는 task

  - **CR**: 다양한 상품에 대한 소비자 리뷰 데이터 (긍정/부정 탐지)

  - **MPQA**: MPQA 데이터셋에서 의견이 긍정인지 부정인지 분류하는 `subtask`
    - Multi-Perspective Question Answering
    - 535개의 뉴스 기사 데이터

<br></br>

- _하이퍼 파라미터 & 학습_ ==> SST-2의 dev set에서 grid search를 통해 아래 값들을 찾음.

  - `ReLU` 사용
  
  - 하이퍼 파라미터
  
    - `filter window`: 3, 4, 5 
    - `feature maps`: 100개
    - `dropout rate`: 0.5
    - `ℓ2​ constraint`: 3
    - `mini-batch size`: 50

  - `dev set`에서의 조기 종료 외에는 특별히 데이터셋별로 조정을 하지 않음.
  
  - `dev set`을 주어주지 않는 데이터셋의 경우 `dev set`을 무작위로 훈련 셋의 10%를 뽑아 만듦.
  - 훈련은 `SGD`를 통해 진행되며 update rule은 `Adadelta`.

<br></br>

- _Pre-trained Word Vectors_

  - 공식 Word2Vec model [(Mikolov et al. 2013)](https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)

  - 벡터 차원: 300

  - 사전 학습되지 않은 단어(OOV)의 경우, 랜덤하게 초기화됨.

<br></br>

- _Model Variation_

  - **CNN-rand**: 모든 단어가 랜덤하게 초기화되며 훈련을 거치면서 갱신됨. `[BASELINE]`

  - **CNN-static**: `word2vec`으로 사전 학습된 벡터로 단어 표현. 훈련을 거치면서 단어 벡터는 바뀌지 않고 CNN의 파라미터만 갱신.

  - **CNN-non-static**: 위와 같지만 사전 학습된 벡터가 각 `task`마다 fine-tune 됨.

  - **CNN-multichannel**: 두 단어 벡터 집합을 가진 모델.
  
    - 각 벡터 집합(from word2vec)은 `channel`로 다루어지고 각 필터는 두 채널에 모두 적용됨. 

    - 하지만 기울기의 역전파는 하나의 채널에만 적용됨. ==> 즉, 하나의 단어 벡터는 갱신하고 하나는 그대로 둠.

  - 모델들의 효과를 구분하기 위해 다른 무작위성은 제거함. (CV-fold, word2vec의 [UNK] 벡터 값, CNN 파라미터 초기화 값을 모든 데이터셋에 똑같은 값이 적용되게 함)

<br></br>

### *Result & Discussion*

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209089009-2c21ddfd-e0e0-405f-9192-b375e56318f8.png" width="60%"></img></div>

<br></br>

- baseline은 좋은 성능을 보이지 않았지만, `word2vec`의 단어 벡터만을 사용해 CNN의 파라미터만 갱신한 모델(CNN-static)이 놀라운 성능을 보임. (단어 벡터는 갱신 X)

  - 이는 사전 학습된 벡단어 벡터가 범용으로 사용될 수 있는 `feature extractor` 역할을 한다고 볼 수 있음.

- 또한, 각 태스크에 맞게 사전 학습된 벡터를 `fine-tune`을 진행하면 좀 더 좋은 성능을 보였음.

<br></br>

- _Multi-channel vs Single-channel Models_

  - `Multi-channel`이 오버피팅을 방지하길 바랬고 작은 데이터셋에서는 실제로 성능이 더 좋았지만,

  - 전체 결과로는 그렇지 않은 것도 있었다.

  - `fine-tuning`작업을 시켜준 정규화의 경우 더 좋은 결과를 보임.

    - 예를 들어, 벡터가 바뀌지 않는 부분의 채널을 쓰지 말고, 하나의 채널로 두고, 몇몇 차원은 갱신이 되고 몇몇은 안되도록

<br></br>

- _Static vs Non-static Representations_

  - `CNN-non-static`처럼 `multichannel model`도 fine-tune하여 task에 특화되게 만들 수 있음.

  - 아래와 같이 task에 맞게 비슷한 단어가 바뀌는 것을 볼 수 있음.

  - 사전 학습된 벡터에 없는 토큰의 경우(OOV) `fine-tune`을 통해 의미 있는 표현으로 배울 수 있음.

    - `!`는 감탄사와, `,`는 접속사와 연관이 있음을 배움.

<br></br>

- _Further Observations_

  - [A Convolutional Neural Network for Modelling Sentences](https://aclanthology.org/P14-1062/)의 논문은 본 논문과 같은 모델 구조이지만 본 논문의 모델에 비해 낮은 성능을 기록했다. (본 논문의 모델이 더 많은 필터 너비와 feature map을 가지기 때문)

  - `Dropout`은 좋은 정규화 수단임. (2~4% 성능을 높임, 큰 네트워크에서도 문제없이 작동)

  - `word2vec`에 없는 단어(OOV)를 랜덤으로 초기화할 때, 사전 학습 모델과 같은 분산을 같도록 하는 `a`를 선택해 _U_[-a, a]에서 각 차원을 샘플링하여 조금의 성능 향상을 얻음.

  - `Collobert et al. 2011`를 Wikipedia 데이터 셋으로 훈련한 단어 벡터보다 `Word2Vec`이 우수했음. (Word2Vec의 구조 덕분인지 데이터셋 덕분인지는 명확하지 않음)

  - `Adadelta`는 `Adagrad`와 비슷한 결과를 주지만 epoch이 적게 듦.
    
<br></br>

### *Conclusion*

- `word2vec`으로 표현한 단어 벡터에 CNN을 적용하는 것은 엄청난 성능을 보임. (작은 하이퍼파라미터 튜닝에도 불구하고)

- 사전학습된 단어 벡터가 딥러닝을 이용한 `NLP`에서 중요한 역할을 한다는 증거를 보여줌.
