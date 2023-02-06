Fine-grained Sentiment Classification using BERT
===================================================

- Manish Munikar, Sushil Shakya and Aakash Shrestha
- https://gitlab.com/bytecell/dailab/-/issues/60#note_1256954692

Introduction
-------------------

- Fine-grained sentiment classification은 텍스트를 5단계의 감성으로 분류하는 지도학습 머신러닝임
- ![image](https://user-images.githubusercontent.com/49019184/216865669-8cb698d4-1f93-4155-99fa-6a94f6dcdc00.png)
- Motivation
  - Stanford Sentiment Treebank(SST) 데이터셋 중 BERT를 활용한 SST-2에 대한 연구는 있었지만 SST-5에 대한 연구는 없었음
  - BERT는 빠르고 강력하고 downstream task에 적용도 쉽기에 좋은 결과를 기대할 수 있음

Related Work
--------------------

- IMDb 영화 리뷰 데이터셋과 같은 대규모 공개 데이터셋이 있어 대부분의 감성 분류 연구는 이진 분류로 진행되었음
- 감성 분류를 위해서 먼저 텍스트를 임베딩하여 벡터로 전환하는 것이 필요함
- 여러 단어 벡터를 합쳐서 하나의 문서 벡터로 바꿔야함
- context-free가 아닌 contextual 임베딩을 하여 context-sensitive feature를 추출함

Dataset
-----------

- Stanford Sentiment Treebank는 fine-grained sentiment classification task 데이터셋 중 가장 인기 있는 공개 데이터셋임
- 각 문장은 Stanfor constituency parser에 의해 개별 단어가 리프 노드고 문장 전체를 루트 노드로 하는 트리 구조로 분석됨
- ![image](https://user-images.githubusercontent.com/49019184/216865695-e87ec0fa-8e5d-489e-891f-3becc7a64bed.png)
- label: 0 ~ 4 (very negative, negative, neutral, positive, very positive) 

Methodology
---------------

- BERT
  - ![image](https://user-images.githubusercontent.com/49019184/216865703-20f1298e-d6f2-46c1-8602-4244f0a2d0c7.png)
  - unlabeled text에서 양방향 representation을 학습하도록 설계된 임베딩 계층
  - 입력 문장을 한번에 처리할 수 있음
  - 입력 형식: [CLS]토큰 시작, [SEP]토큰 끝
  - 출력 임베딩인 [CLS]토큰은 전체 문장을 분류할 수 있는 임베딩임

- Preprocessing
  - canonicalization: 숫자, 기호, 특수문자를 제거하고 소문자로 변환
  - tokenization: WordPiece 토크나이저 사용, 접두사, 어근, 접미사를 분해함
  - special token addition: 문장 구분을 위한 [CLS]토큰과 [SEP]토큰을 문장 앞뒤에 추가함

- Proposed architecture
  - ![image](https://user-images.githubusercontent.com/49019184/216865720-be823833-412c-4e7b-8b3f-5285832e1240.png)
  - BERT 이후에 dropout regularization과 softmax classifier를 추가함

Experiments and results
-----------------------------

- Comparison models
  - 성능 비교를 위해 Word embeddings, Recursive networks, Recurrent networks, Convolutional networks도 실험을 진행했음

- Evalauation metric
  - 데이터셋에는 감성 5단계가 골고루 분포되어 있으므로 accuracy를 사용함
  - $\text{accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}} \in [0, 1]$

- Results
  - ![image](https://user-images.githubusercontent.com/49019184/216865735-03b587dd-2b4b-4c89-8be7-f4bd4d17efe2.png)
  - BERT를 사용한 모델이 단순한 구조임에도 좋은 성능을 보였음

Conclusion
--------------

- 사전학습된 BERT에 미세 조정을 통하여 세분화된 감성 분류를 하였음
- 단순한 구조이지만 다른 복잡한 구조의 모델보다 좋은 성능을 보였음
- 결과를 통해 BERT와 같은 언어모델의 NLP에서 전이학습 능력을 입증하였음
