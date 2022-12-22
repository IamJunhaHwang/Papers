## :page_facing_up: Character-level Convolutional Networks for Text Classification

### *Info*

* Author: Xiang Zhang, Junbo Zhao, Yann LeCun
* Conf: NIPS 2015 (accepted)
* URL: https://papers.nips.cc/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html
* Code[비공식]: https://github.com/ahmedbesbes/character-based-cnn
* Code[공식]: https://github.com/zhangxiangxiao/Crepe

  - 해당 논문의 전 버전에서 사용된 코드. 해당 논문에서도 이 코드를 사용했다고 한다.
  
  - 원래 `technical report(?)`였는데 몇 가지 실험결과를 추가하고, introduction을 다시 쓴 것이 이 논문이라고 한다.

  - `lua`로 작성되었다.

<br></br>

### *Abstract*

- `text classification` task에 `character-level convolutional networks(ConvNets)`을 적용한 실험을 통한 결과를 얻음.

* `bag of words`, `n-grams`, 앞 두 개의 `TF-IDF 변형`, `word-based ConvNets`과 `RNN`을 비교.

<br></br>

### *Introduction*

- 본 논문에서는 문자 단위로 텍스트를 다루며 이를 `1D ConvNet`에 적용시켜 볼 것.

  - 또, `ConvNet`이 텍스트를 이해하는 능력을 보여주는 전형적인 예시를 들기 위한 방법으로 `classification task`만 사용할 것.

  - `ConvNet`을 사용하기 위해 현재(2015)까지 보았을 때 큰 데이터셋이 필요하므로 이를 만들었고 다양한 여타 모델과의 비교를 보여줄 것.

- `ConvNet`은 이전 연구에서 언어의 의미론, 구문론적 사전 지식 없이 바로 단어 임베딩에 적용할 수 있음을 보여주었다. 따라서, 이 접근법은 기존 모델과 견줄 만하다고 증명됨.

- 이 논문은 `문자 단위`로 `ConvNet`을 적용한 최초의 시도임.

  - 해당 논문에서 이전 연구에서의 결론과 같이 큰 데이터셋을 사용한 `deep ConvNet`에서는 사전지식이 필요 없음을 보일 것.

  - 오직 문자로만 작동하기 때문에 비정상적인 문자 조합도 학습하게 될 것이라는 장점이 있음. (이모티콘이나 오타 등)

<br></br>

## *방법*

### *Key Modules*

- Main Component: 간단한 `1D Convolution module`

  - Function

    - discrete input function: `g(x)∈[1,l]→R`

    - kernel function: `f(x)∈[1,k]→R`

    - Convolution: `h(y)∈[1,⌊(l−k)/d⌋+1] `

      - `c = k - d + 1` : offset 상수

  - 기존의 `CNN`과 같이 `kernel` 함수의 집합(가중치)으로 파라미터화 된다.

  - `output`은 `input`과 `kernel`의 Convolution.

<br></br>

- key module: **max-pooling**

  - 모델이 더 깊게 훈련될 수 있게 함. ==> Vision에서 사용되는 `max-pooling`의 1D 버전이다.

  - discrete input function: `g(x)∈[1,l]→R`

  - max-pooling function: `h(y)∈[1,⌊(l−k)/d⌋+1] `

    - `c = k - d + 1` : offset 상수

  - 풀링 계층의 이해를 돕기 위한 그림

    - **이를 통해 특성 맵의 크기를 줄일 수 있음.**
    - `ConvNet`을 6 layers 이상 훈련 시킬 수 있게 함.

<br></br>

- `non-linearity`: ReLUs 사용. ==> `h(x) = max{0, x}`

- 학습 방법: `SGD` with minibatch of size `128`

- 모멘텀 상수는 0.9로 하였고 학습률은 0.01 시작해서 10에폭 중 3에폭마다 절반으로 줄이는 방식을 사용

- 각 에폭에는 여러 class에서 두루두루 샘플링된 랜덤 훈련 샘플들의 고정된 크기가 있는데, 이는 데이터셋마다 다르므로 뒤에서 설명한다.

<br></br>

### *Character quantization(문자 인코딩)*

- `input`으로 인코딩된 문자의 시퀀스를 받는다. ==> 정해진 m개의 `alphabet(기호)`을 one-hot encoding 한 형태.

  - 문자의 시퀀스는 고정된 길이(l_0)의 `m`차원 벡터로 변환됨.

  - 최대 길이인 `l_0`를 넘어가면 무시되며, 공백 문자를 포함해 `alphabet`에 없는 문자는 0벡터로 변환됨.

  - 문자의 양자화(인코딩) 순서는 반대로 된다. (최근에 읽은 문자가 항상 output의 시작 쪽에 위치하게 하기 위해)

- `alphabet`을 총 70개의 문자로 정의한다. ==> 26개의 영어 문자, 10개의 숫자, 그리고 33개의 특수문자, 줄 바꿈 문자

<br></br>

### *Model Design*

- 2개의 `ConvNets`을 디자인 함. (하나는 `feature`가 큰 것, 하나는 작은 것)

  - 둘 다, 9 Layers를 가진다. (6: Convolution layer, 3: Fully-Connected Layer)

  <img src="https://user-images.githubusercontent.com/46083287/209087175-2ace631a-4b68-48a3-a141-42b66d90d5fd.png" width="70%"></img>

- `input`은 70개의 `feature`를 가지며, `length`는 1014이다.

  - 위의 문자 인코딩 방법에 의해 70개의 feature를 가짐.

- 3개의 Fully-Connected Layer 사이에 2개의 dropout 모듈을 넣었다. (dropout 확률은 `0.5`)

- 아래의 `Table1`은 Convolutional layer의 설정이고, `Table2`는 Fully-Connected Layer의 설정이다.

  <img src="https://user-images.githubusercontent.com/46083287/209087225-42f9cb98-84da-479b-a559-b0f6af97bbfa.png" width="70%"></img>

  <img src="[/uploads/694bbdde7bb0a32341989b6973493bcc/image.png](https://user-images.githubusercontent.com/46083287/209087268-eb9dc9ee-589e-4f96-83e0-5cd4c198f9c4.png)" width="70%"></img>

- 가중치는 `가우시안 분포`로 초기화 했음. (분포의 `평균`과 `분산`은 아래와 같다.)
  
  - large model: (0, 0.02)
  - small model: (0, 0.05)

- 어떤 문제이냐에 따라, input length가 달라질 수 있다.

  - model design에 따라 `frame length`를 구할 수 있으므로 Fully- Connected Layer의 input dimension 설정에 문제 없음.

<br></br>

### *Data Augmentation using Thesaurus(유의어 사전)*

- 텍스트의 경우 `signal transformation`이 불가능 함. (문자의 정확한 순서가 그 의미와 문법을 담고 있으므로(

  - 따라서, 단어나 구를 유의어로 교체하여 데이터를 늘렸다.

- 교체할 단어의 수를 정하기 위해, 주어진 text로부터 교체가능한 단어를 추출하고 랜덤하게 `r%`를 선택할 것.

  - `r`은 **기하 분포(등비 분포)**에 의해 결정된다. (parameter `p`를 이용. `P[r] ∼ p^r`)

  - 주어진 단어에 의해 선택될 유의어의 index `s`는 또 다른 기하 분포(등비 분포)에 의해 결정된다. (`P[s] ∼ p^s`)
  
    - 선택된 유의어의 확률은 가장 자주 보이는 의미에서 떨어질수록 작아진다.

<br></br>

## *실험*

### *비교 모델*

- Traditional Methods

  - 직접 만든 feature extractor를 사용하였으며, 다항 Logistic Regression을 사용.

  - `Bag-of-words` & its `TF-IDF`

    - 훈련 subset에서 가장 많이 등장한 50,000개의 단어로 만듦.
    - `feature`는 가장 큰 feature value로 정규화 됨.

  - `Bag-of-<n-grams>` & its `TF-IDF`

    - 각 데이터셋마다 훈련 subset에서 가장 많이 등장한 n-gram 500,000개로 만듦. (5-gram까지)
    - 위와 같은 방법으로 feature 정규화.

  - `Bag-of-means` on word embedding

    - 각 데이터셋의 훈련 subset으로부터 학습된 word2vec에 `k-menas`를 적용한 모델. (임베딩 차원: 300)
      - 군집화된 단어의 대표로서 학습된 평균(k-means)을 사용. (`K`:5000)

    - 각 단어가 훈련 subset에서 5번 넘게 나오도록 함.
    - 위와 같은 방법으로 feature 정규화.

<br></br>

- Deep Learning Methods

  - Word-based ConvNets

    - pretrained word2vec embedding과 lookup table을 사용하는 경우 둘 다 비교.
      - 임베딩 차원: 300

    - 공정한 비교를 위해, 본 논문의 Convnet과 똑같은 size 사용. (layer 수, output size)

    - `data augmentation` 또한 진행함.

  - LSTM

    - 임베딩 차원이 300인 `pretrained word2vec`을 사용한 `word-based LSTM`.

    - 모든 LSTM cells 출력의 평균을 취하도록 구성되며 다항 Logistic regression을 사용한다.

    - `gradient clipping` 사용 (기울기 노름을 `5`로 제한)

<br></br>

### *Large-scale Datasets & Results*

- 공개되어 있는 text 분류를 위한 데이터 셋이 너무 작기 때문에 실험을 위해 만듦.

  <img src="https://user-images.githubusercontent.com/46083287/209087393-d7ce0d9e-4c37-440a-b30e-1944792b4ce6.png" width="70%"></img>

  <img src="https://user-images.githubusercontent.com/46083287/209087406-f56671e3-375e-401f-8e34-d412650fa1db.png" width="70%"></img>

  - `Sogou` 데이터 셋은 `중국어`이며, 중국어 유의어 사전이 없기에 data augmentation을 진행하지 못함.

  - best result: :large_blue_circle: , worse result: :red_circle: 

<br></br>

### *Discussion*

  <img src="https://user-images.githubusercontent.com/46083287/209087421-b94f59d8-3e65-4cc2-acf0-e4dc3be0041f.png" width="70%"></img>

- 위는 비교 모델과 본 논문의 `Char-level ConvNet`과의 상대 오차를 나타낸 것.
  - 두 모델의 차를 구한 후, 비교 모델의 에러로 나눈 값.
  
  - **즉, 그래프가 양의 방향이면 본 논문의 모델이 성능이 더 좋은 것.**

- `Char-level ConvNet`은 단어의 필요 없이 text 분류가 가능하므로, 언어 데이터를 다른 데이터와 같이 신호로 취급할 수 있다는 강점이 있음.

- 큰 데이터 셋이 좀 더 좋은 성능을 보임.

- `user-generated data`에 좋은 성능을 보임. ==> 현실 시나리오에 적용 가능

  - `Figure 3-c, 3-d, 3-e`의 Yahoo A., Amazon F., Amazon P. 참고

- 소문자와 대문자를 구분하는 것이 더 성능에 악영향을 미침.

- 감성 분석(Yelp & Amazon reviews)과 주제 분류 (다른 데이터들)에 대해 성능이 차이가 없었음.

- `word2vec`을 단순한 분포로 사용하는 것은 text 분류에서 좋지 않음.

- 모든 종류의 데이터 셋에 뛰어난 머신 러닝 모델은 없었음. (특정 부분에서 좋은 성능을 내는 방법을 결정하는 데 도움을 줄 것임)

<br></br>

## *결론*

- text 분류를 위한 `Character-level convolutional network`의 실험적인 결과를 얻어냄.

- 큰 데이터셋을 사용해 여러 모델들과 비교하였고, 본 논문에서 제시한 `Char-level Convnet`이 효율적인 방법임을 보임.

- 또한, 데이터셋 크기, text와 alphabet의 선택에 따른 본 논문의 모델의 성능 비교를 함.

- 구조화된 출력이 필요할 때, `Language Processing`의 범주에서 본 논문의 모델이 적용되길 바람.
