What Does BERT Look At? An Analysis of BERT's Attention
============================================================

- https://gitlab.com/bytecell/dailab/-/issues/74#note_1320120656
- 저자: Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D.Manning

Introduction
---------------

- 사전학습 언어모델로 지도학습 finetuning을 하면 높은 정확도를 달성하지만 왜 그런지는 완전히 알 수 없음
- 이를 밝히기 위해 출력이나 내부 벡터 representation으로 시험을했지만 본 논문에서는 attention map을 통해 연구함
- 같은 계층의 attention head는 비슷한 경향이 있음을 발견함
- attention head의 syntactic 능력을 더 얻어내기 위해 attention-based probing classifier를 제안함

Background: Transformers and BERT
-------------

- BERT의 각 인코더에는 여러 attention head가 포함되어 있고 이들은 각각 attention 가중치를 계산함
- 본 연구에서는 인코더 attention head를 12개씩 갖는 12계층이 있는 BERT "base" 크기의 모델을 사용했음

Surface-Level Patterns in Attention
---------------------------------------------
- ![image](https://user-images.githubusercontent.com/49019184/233264355-8e8b2397-5530-4ea4-9bf1-27fc0e92e24e.png)
- surface-level pattern을 보여주는 attention head의 예시
- 실험 Setup
  - BERT-base 모델에서 attention map 추출
  - 기본 사전학습된 BERT의 설정을 따르고 paragraph단위로 [SEP]토큰을 추가함
- Relative Position
  - ![image](https://user-images.githubusercontent.com/49019184/233264371-d642c678-c969-4908-8698-64d2c60bc94d.png)
  - 대부분의 attention head는 현재 토큰에는 attention 가중치가 작고 이전이나 다음 토큰의 attention 가중치가 큼(특히 BERT의 앞쪽 인코더에서 그런 경향을 확인하였음)
- Attending to Separator Tokens
  - ![image](https://user-images.githubusercontent.com/49019184/233264391-232c5069-4139-4912-b1fb-5565aebae3bd.png)
  - 6-10번째 인코더 계층에서 [SEP] 토큰에 초점이 맞춰져 있음
  - 이는 uncased BERT 모델에서도 비슷하게 발생하는데 시스템적인 이유가 있을 것임
  - 5번째 계층 이후로 [SEP]에 대한 attention이 거의 변하지 않는데 이를 통해 [SEP] 토큰이 attention head에 영향을 끼치지 않음을 확인하였음
- Focused vs Broad Attention
  - attention head가 넓게 퍼진 여러 단어에 초점을 두는지 소수의 단어에만 초점을 두는지 확인해봄
  - ![image](https://user-images.githubusercontent.com/49019184/233264406-2afa2ddd-d58f-46c5-b322-f668fd019358.png)
  - 각 attention head의 엔트로피를 계산하고 평균을 표시했음, 앞쪽 계층의 attention이 넓게 퍼져있음
  - [CLS] 토큰에 담긴 정보로만 엔트로피 계산을 했을 때 대체로 평균과 비슷했지만 마지막 계층에서는 엔트로피가 높게 나왔는데 이는 BERT가 다음 문장 예측 task를 위해 사전학습되었으므로 입력 전체를 넓게 봐야 한다고 하면 타당한 결과임

Probing Individual Attention Heads
-----------------------------------------
- ![image](https://user-images.githubusercontent.com/49019184/233264427-5ae63525-13d4-48b7-8a9f-038307ef15cf.png)
- 각 attention head가 무엇을 학습하는지를 확인하고 평가하기 위해 dependency parsing이 레이블링된 데이터셋을 사용했음
- Method
  - 단어 단위를 평가하고 싶기에 토큰 단위의 attention map을 단어 단위로 전환했음
  - `to`의 attention 가중치는 토큰의 가중치를 더했고 `from`의 attention 가중치는 토큰의 가중치의 평균을 사용했음
  - 이는 각 단어의 attention의 합이 1이 되는 것을 유지하기 위함임
  - 가장 많은 attention을 받는 단어를 모델의 예측값으로 함
- Dependency Syntax
  - 실험 Setup
    - Stanford Dependencies로 레이블링된 Wall Street Journal 일부를 사용함
    - 단어들 사이의 attention 방향을 양방향 모두에 대해 평가함
  - Results
    - ![image](https://user-images.githubusercontent.com/49019184/233264445-743c4a40-bb16-4423-9ab9-a24bdf73f6a8.png)
    - 전반적으로 다 좋은 정확도를 보이는 단독 attention head는 없음
    - 특정 head는 특정 의존 관계에 대해 높은 정확도를 보여줌
    - 위의 attention head의 attention map 그림을 보면 사람이 생각하는 의존 관계나 BERT가 만든 의존 관계가 유사한 것을 볼 수 있음
- Coreference Resolution
  - 실험 Setup
    - CoNLL-2012 데이터셋 사용, antecedent selection accuracy로 평가
    - antecedent를 고르는 방법을 세 가지 다른 베이스라인과 비교함
      - 가장 가까운 다른 mention 선택
      - 현재 mention과 같은 head word인 것 중 가장 가까운 다른 mention 선택
      - 다음 우선순위를 만족하는 mention 선택(단어 전체 일치, head word 일치, 숫자/성별/사람 일치, 다른 mention)
  - Results
    - ![image](https://user-images.githubusercontent.com/49019184/233264462-215dfac6-d41c-4b5c-a5b2-a3f6b798c55b.png)
    - BERT의 attention head가 좋은 성능을 보여줬음, 거의 rule-based system(우선순위 기준 mention 선택)와 비슷함

Probing Attention Head Combinations
------------------------------------------

- 모델의 전반적 지식은 여러 attention head에 퍼져있음
- 이 전반적 지식을 측정하기 위해 attention-based probing 분류기 그룹을 제안하고 그들을 의존 parsing에 적용함
- Attention-Only Probe.
  - ![image](https://user-images.githubusercontent.com/49019184/233264483-7ff1d64f-2387-4d59-a59f-6235462a63a2.png)
  - attention 가중치의 간단한 선형 조합을 학습함
  - attention의 양방향을 고려하여 candidate head와 dependent 서로의 attention을 모두 포함함
- Attention-and-Words Probe.
  - 특정 head가 특정 syntactic 관계와 연결된다는 것으로 probing classifeir가 입력 단어에서 정보를 가져오는 과정에서 이득이 있을 것임
  - 예) dependent `the`, candidate head `cat`이 있을 때, probing classifer는 대부분의 가중치를 determiner relation에서 뛰어난 head 8-11에서 할당함
  - ![image](https://user-images.githubusercontent.com/49019184/233264504-db891ca0-3c16-4700-ab1d-fcf3e860af6a.png)
  - 이는 attention-and-words probing classifier에서 단어의 head간의 확률임
- Results.
  - Penn Treebank dev set으로 평가함
  - 베이스라인 세 가지
    - head가 항상 dependent의 오른쪽에 있게 예측하는 right-branching baseline
    - dependent와 candidate head의 GloVe 임베딩을 입력으로 받는 간단한 one-hidden-layer network
    - BERT의 랜덤 초기화된 word/positional 임베딩을 사용하는 attention map인 attention-and-words probe
  - ![image](https://user-images.githubusercontent.com/49019184/233264514-077065e0-32fd-4dd1-8623-236132410417.png)
  - Attn+GloVe의 성능이 제일 좋음, BERT의 attention map이 영어 syntax 표현을 잘한다는 것을 의미함

Clustering Attention Heads
----------------------------------

- 모든 attention head 사이의 거리를 계산하여 같은 층에 있는 attention head는 어떤 지에 대해 조사함
- attention distribution 사이의 Jensen-Shannon Divergence를 이용하여 두 head사이의 거리를 계산
- $\sum_{token\in data}{JS(H_i(token),H_j(token))}$
- 이 식으로 계산한 거리를 다차원 스케일링을 통해 시각화함
- ![image](https://user-images.githubusercontent.com/49019184/233264528-79c0652f-d22a-457d-97df-7bb362ecb3b9.png)
- 같은 군집으로 묶인 head를 살펴보면 같은 계층에 있는 경우가 자주 있음

Conclusion
--------------

- BERT 모델의 attention 메커니즘을 이해하는 분석 method 제안
- hidden state뿐만 아니라 attention map에서도 언어적 지식이 담겨 있음
- attention map을 분석하는 것이 모델 분석 기술과 언어 학습 신경망 이해를 보완해줌
