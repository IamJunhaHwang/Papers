## :page_facing_up: PANDORA Talks: Personality and Demographics on Reddit

### Info

  * `conf`: NAACL2021
    
  * `author`: Matej Gjurkovic, Mladen Karan, Iva Vukojevic, Mihaela Bosnjak, Jan Snajder

  - `url`: https://aclanthology.org/2021.socialnlp-1.12.pdf

  - `data`: https://psy.takelab.fer.hr/datasets/all/pandora/

### 1. Abstract & Introduction

`Personality`와 `demographic`은 사회언어학의 중요한 요소이지만 이에 따라 레이블링된 데이터가 부족하다. 설령 데이터가 존재한다해도 적은 수의 유저가 만들은 데이터이기 때문에 편향이 존재한다. (특정 주제만 다룬다거나 길이가 적다던가)

따라서, 우리는 `MBTI`, `Big-5 factor`, `Enneagram` 으로 레이블링한 레딧 데이터셋(PANDORA)을 소개한다. (+ 인구통계학적 정보도 있음)

우리가 제시한 데이터가 연구에 도움에 된다는 것을 보여주기 위해 세 가지 실험을 시행한다.

- Big-5 예측 model을 만들기 위해 `MBTI`, `Enneagram `레이블을  사용하는 방법

- 심리-인구통계학 프로파일을 채우는 것이 성별 예측에서 편향을 찾아내는 것에 어떻게 도움을 주는지

- 사회 과학에서 우리가 제시한 데이터셋이 도움이 되는지 

### 2. PANDORA dataset

레딧은 많은 토픽을 다루며 익명성이 보장되므로 사회언어학 연구에 적합하다. (데이터 양도 많음)

`10K 유저 & 17M 코멘트`

- MBTI & Enneagram Labels

  - 유저들이 `flair`에 기재해놓는 것을 가져옴. (flair는 프로필에 나의 정보를 써놓는 것과 비슷함)

  - 몇몇은 comment에서 가져옴.

- Big 5 Label

  - 이 경우에는 `flair`에 기재해놓지 않기 때문에 유저의 코멘트에서 이를 언급한 것을 찾아낸다. (테스트 글에 달린 댓글들이 주류)

  - `text-based test`의 경우 버렸다. (`16personality` 처럼 공신력 없는 테스트)

  - 찾아낸 코멘트 상의 점수들이 잘 추출되었는지 직접 확인했다.

  - 찾은 코멘트 중에서는 어떤 테스트를 보았는지 기재하지 않은 것도 있었는데 이 부분은 classifier를 하나 만들어서 어떤 테스트인지 예측하게 만들었다. (F1: 81.4% 성능)

- Demographic Labels (ex. age, gender, location, etc)

  - 이것 또한 `flair`에 기재된 것을 가져옴.

  - 대부분이 영어권이었음.

<br></br>

### 3. Experiments

- 풍부한 MBTI/Enneagram 레이블이 Big5를 예측하는 데 주는 영향을 주는지 실험

  - MBTI/Enneagram 레이블을 Big5 레이블로 바꾸는 도메인 특화 태스크임.

  - 코멘트에서 MBTI 예측 => 나온 결과를 feature로 이용해 Big5 예측

  - MBTI가 Big5 예측에 도움이 되었음.


- Gender Classification Bias

  - Reddit 데이터로 훈련된 간단한 성별 분류 모델이 편향되어 있는지 본다.

  - PANDORA 데이터에서 `남성:8.1%`, `여성:14.4%`를 잘못 분류했음. ==> 저자들은 편향이 존재한다고 봤음. (Z-검정 통해)

  - `T/F`, `P/J`를 가지는 여성이 남성보다 잘못 분류되기 쉬웠. (남성은 반대)

  - 더 외향적인 남성이 잘못 분류되기 쉬웠음.
