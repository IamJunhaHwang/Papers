Fine-tune BERT with Sparse Self-Attention Mechanism
===========================================================   

- https://gitlab.com/bytecell/dailab/-/issues/74#note_1306438802
- 저자: Baiyun Cui, Yingming Li, Ming Chen, and Zhongfei Zhang

Introduction
-------------

- ![image](https://user-images.githubusercontent.com/49019184/233263703-fe6ebe56-ce5a-48f7-8e62-dcf01e724c7a.png)
- 각 모델의 attention 행렬을 시각화한 그래프
- 사전학습을 하지 않은 경우보다 한 경우가 더 구조적인 attention 분포를 나타냄
- 하지만 다른 미세조정 모델은 (b)에서 (d)처럼 더 희소해지도록 하지 않음
- 소프트맥스 함수를 sparse transformation으로 대체해서 희소성을 늘리는 메커니즘과 모델을 제안
- 높은 attention 가중치를 가지면 가장 중요한 관계로 만들고 의미가 없는 관계는 정확히 0으로 만듦
- 제안한 메커니즘은 다양한 목적의 신경망 모델에 적용하기 쉬움
- 실험은 감성분석, 질의응답, 자연언어 추론 세 목적에 대해 7개의 공개된 데이터셋으로 진행했음

Fine-tuning with Sparse Self-Attention
-----------------------------------------  

- Self-attention mechanism
  - 출력 representation $y$는 입력 sequence $x$와 관계된 요소의 가중합으로 결정됨
  - $y_i = \sum_{j=1}^L \alpha_{ij}(x_j W_v)$
  - $e_{ij} = \frac{(x_i W_q)(x_j W_k)^T}{\sqrt{d}}, \alpha_{ij} = \rho(e_{ij})$
  - $\rho$는 확률 매핑 함수임, 보통은 softmax 함수를 사용함
  - 하지만 softmax는 더 attention을 하지 않음

- Sparse Self-Attention Mechanism
  - softmax 대체로 sparsegen-lin을 사용했음(희소성을 조절할 수 있음)
  - 확률 매핑 함수 $\rho(e_{ij}) = p_{ij} = \text{max}\{0, \frac{e_{ij} - \tau(\text{e}_i)}{1-\lambda}\}$
  - $\lambda$를 통해 희소성을 조절할 수 있음
  - 이 메커니즘을 통해 의미없는 관계는 0 확률로 만들고 가장 중요한 관계는 더 강조되도록 할 수 있음

- Sparse Self-Attention Fine-tuning model
  - N개의 sparse self-attention layer로 구성된 모델
  - $h^{-n} = LN(h^{n-1}+SSAM(h^{n-1})),  h^n = LN(h^{-n}+FFN(h{^-n})), h^0 = embed(x)$
  - 토큰 임베딩의 합과 위치 임베딩을 입력으로 받음, LN은 layer normalization을 의미함

- Relationships with existing methods
  - 이전의 sparse formulation은 대부분 분류 계층이나 attention기반 인코더-디코더 구조에 적용되었음
  - 하지만 본 연구에서는 self-attention기반 transformer encoder에 적용함

Experiments
-------------

- Datasets
![image](https://user-images.githubusercontent.com/49019184/233263724-f47eefaa-0232-4090-bffe-bbef28312e8b.png)

  - Sentiment Analysis(SA)
    - 평가 지표: Accuracy
    - 데이터셋: SST-1, SST-2, SemEval, Sentube-A, Sentube-B
  - Question Answering(QA)
    - 평가 지표: Exact Match, F1
    - 데이터셋: SQuAD v1.1
  - Natural Language Inference(NLI)
    - 평가 지표: Accuracy
    - 데이터셋: SciTail

- Implementation details
  - 사전 학습된 BERT_BASE 모델 사용
  - $\lambda$ 데이터셋에 따라 다르게 사용했음
  - optimizer: Adam, learning rate: 2e-5, batch size: 16

- Baselines
  - 제안한 모델 SSAF와 base 모델 BERT_BASE를 모든 task에 대해 비교했음
  - 감성 분석에 대해서는 Ave, LSTM, BiLSTM, CNN, SSAN과도 비교했음
  - Np: 사전학습 없이 BERT_BASE모델 사용, SSAF(Np): 사전학습 없이 SSAF모델 사용

- Results
  - ![image](https://user-images.githubusercontent.com/49019184/233263743-2d458a70-bd08-461a-9750-2a4b9413c895.png)
  - 감성분석 결과
  - ![image](https://user-images.githubusercontent.com/49019184/233263756-e0f5f741-a0f3-4a99-ab28-365c563ead93.png)
  - 질의응답 결과
  - 모든 곳에서 SSAF의 결과가 더 좋고 Np와 SSAF(Np)의 결과 비교를 통해 self-attention model에 희소성의 효과를 증명했음
  - ![image](https://user-images.githubusercontent.com/49019184/233263763-3a7567e4-0155-44a6-a91e-530cbaf176e7.png)
  - $\lambda$에 대한 분석: -8 ~ 0 범위로 실험을 진행하였음, -7 ~ 3 사이의 값이 적당한 결과를 이끌었음
  - ![image](https://user-images.githubusercontent.com/49019184/233263774-354da99b-6289-4cd3-848e-676550536d28.png)
  - attention 행렬을 시각화해서 분석한 결과 $\lambda$가 너무 크면 행렬의 희소성이 너무 커져서 다른 단어 사이의 관계를 무시함 반대로 너무 작으면 불필요한 관계가 늘어남

Conclusion
------------

- self-attention 메커니즘에 희소성을 추가하는 Sparse Self-Attention Fine-tuning model을 제시함
- 감성 분석, 질의응답, 자연언어 추론에 대해 총 7개의 데이터셋으로 실험을 진행하였음
- 실험 결과에 나타난 높은 성능으로 제안한 모델의 효과를 입증했음
