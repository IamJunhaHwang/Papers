Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks
========================================================================================

- 저자: Binxuan Huang, Yanglan Ou and Kathleen M. Carley
- https://gitlab.com/bytecell/dailab/-/issues/60#note_1256789806

Introduction
--------------

- aspect-level sentiment classification(속성 기반 감성 분류)는 문장에서 특정 속성의 감성을 분류하는 것을 목표로 하는 task임(예. "음식은 맛있지만 직원이 불친절했어요."라는 문장에서 음식은 '긍정'이고 직원은 '부정'인 것으로 분류함)
- 이전의 LSTM 기반 모델들은 개별 텍스트에 초점을 두었지만 여기서는 속성과 텍스트를 동시에 모델링함
- 이 논문의 모델은 target 표현과 text 표현의 상호작용은 attention-over-attention을 통해서 이루어짐
- text에서 가장 중요한 부분에 집중하는 것이 필요한데 이를 위해 AOA를 선택하였음

Related work
-------------------

- Sentiment Classification
  - 감성 분류는 text의 감성 극성을 감지하는 것임
  - 지도 학습 머신 러닝 방식이 대부분이고 최근에는 신경만 기반의 접근이 좋은 결과를 나타냈음

- Aspect Level Sentiment Classification
  - 문장의 한 측면의 감성을 구별하는 감성 분류의 한 종류
  - 규칙 기반 모델로 해결을 하다가 신경망을 이용하는 방식이 도입되었음
  - 기존의 모델에 사용되는 pooling operation은 문장과 목표 사이의 단어 쌍 간의 상호 작용을 무시하지만 이 논문의 모델은 이 부분에서 우수성을 보였음

Method
--------------

- Problem Definition
  - 문장 $s=[w_1, w_2, ..., w_i, ..., w_j, ..., w_n]$과 aspect target $t=[w_i, w_{i+1}, ..., w_{i+m-1}]$이 주어졌을 때, 문장에서 aspect target의 감성을 분류하는 것이 목표임
  - ![image](https://user-images.githubusercontent.com/49019184/221735296-6728ac10-115e-4456-a571-050bb806fb62.png)
  - 신경망 모델의 구조, word embedding, Bi-LSTM, AOA module, final prediction으로 구성됨

- Word Embedding
  - n개의 문장과 m개의 타겟이 주어지면 각 단어를 낮은 차원의 실수 벡터로 매핑시킴
  - 결과로 각 문장과 aspect구에 대한 단어 벡터 집합 두 개를 얻음

- Bi-LSTM
  - Bi-LSTM 신경망이 문장과 타겟에 있는 단어의 숨겨진 의미를 학습함
  - LSTM을 사용하여 기울기에 관련된 문제를 피할 수 있고 장기 의존성을 학습하기 좋음

- Attention-over-Attention
  - 타겟 표현벡터와 문장 표현벡터를 행렬곱을 함 $I = h_s\codt h_t^T$
  - column-wise softmax와 row-wise softmax를 이용해 target-to-sentence attention $\alpha$와 sentence-to-target attention $\beta$를 얻음
  - $\alpha_{ij} = \frac{exp(I_{ij})}{\sum_i{exp(I_{ij})}}, \beta_{ij} = \frac{exp(I_{ij})}{\sum_j{exp(I_{ij})}}$
  - target-level attention을 구함 $\overline{\beta_j} = \frac{1}{n} \sum_i {\beta_{ij}}$
  - 최종 sentence-level attention $\gamma$를 각 target-to-sentence attention인 $\alpha$에 가중치를 더하여 계산함 $\gamma = \alpha \cdot \overline{\beta^T}$

- Final Classification
  - sentence attention을 이용한 문장 hidden semantic states의 가중 합으로 최종 문장 표현을 계산함 $r = h_s^T \cdot \gamma$
  - 최종 문장 표현을 최종 분류 feature로 하고 $r$을 목표 클래스$C$로 투영하는 선형 계층에 입력함 $x = W_l \cdot r + b_l$
  - 그 후에 감성의 확률을 계산하기 위해 softmax 계층을 사용함 $P(y=c) = \frac{exp(x_c}{\sum_{i\in C}exp(x_i)}$
  - 마지막으로 예측된 aspect 타겟의 감성은 확률이 가장 높은 것으로 label됨
  - $L_2$ 정규화를 통해 cross-entropy loss를 줄이는 모델 학습함 $loss = -\sum_i{\sum_{c\in C}{I(y_i=c)\cdot\log{P(y_i=c))+\lambda\Vert\theta\Vert ^2}}}$
  - 추가로 과적합을 피하기 위해 dropout을 적용했고 Adam 업데이트 규칙과 SGD를 사용하여 loss function을 최소화하였음

Experiments
-------------

- Dataset
    |Dataset|Positive|Neutral|Negative|
    |-------|--------|-------|--------|
    |Laptop-Train|994|464|870|
    |Laptop-Test|341|169|128|
    |Restaurant-Train|2164|637|807|
    |Restaurant-Test|728|196|196|

  - 두 개의 도메인별 데이터셋에 대해 실험하였음

- Hyperparameters Setting
  - 학습 데이터의 20%를 하이퍼파라미터 튜닝을 하는 검증 데이터로 사용하였음   
  
  |Hyperparmeter|detail|
  |-------------|------|
  |$L_2$ 정규화 계수|$10^{-4}$|
  |dropout keep rate|0.2|
  |word embedding dimension|300-dimensional Glove vectors|
  |dimension of LSTM hidden states|150|
  |initial learning rate|0.01|
  |batch size|25|
  
  - 3 epoch 동안 학습 loss가 낮아지지 않으면 learning rate를 반으로 줄임

- Model Comparisons
  - accuracy로 성능 측정을 하고 Majority, LSTM, TD-LSTM, AT-LSTM, ATAE-LSTM, IAN과 비교했음

  |Methods|Restaurant|Laptop|
  |-------|----------|------|
  |Majority|0.535|0.650|
  |LSTM|0.743|0.665|
  |TD-LSTM|0.756|0.681|
  |AT-LSTM|0.762|0.689|
  |ATAE-LSTM|0.772|0.687|
  |IAN|0.786|0.721|
  |AOA-LSTM|**0.812**|**0.745**|

  - 10번 반복 실험하여 최고 점수와 평균, 표준편차를 계산했음
  - 꽤 큰 차이로 성능이 더 좋았음

- Case Study
  - ![image](https://user-images.githubusercontent.com/49019184/221735321-a77411a1-e370-4ecc-91c8-bcc5c0aa3f8c.png)
  - 5가지 예시에 대해 aspect 감성을 보기 위해 최종 문장 attention vectors $\gamma$를 시각화하였음

- Error Analysis
  - 직접적으로 드러나지 않는 비구성적 감성에 대해서는 찾지 못하는 오류가 있었음
  - 관용구로 표현되는 감성을 찾지 못하는 오류가 있었음
  - 복잡한 감성 표현을 모델이 오해하는 오류가 있었음

Conclusion
------------

- 속성 감성 분류를 위한 신경망 모델 제시, 중요한 부분에 대해 학습을 시키기 위해 AOA모듈 이용
- 기존 methods와 비교 실험을 하였고 case stduy를 통해 모델이 target을 효과적으로 학습했음을 확인했음 
- 오류 분석에서 복잡한 감성 표현에 대한 오류가 있었는데 이것은 문장의 문법을 구조화하는 방법으로 해결할 수 있을 것으로 예상
