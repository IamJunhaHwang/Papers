Improving Language Understanding by Generative Pre-Training
==================================================================

- 저자: Alec Radford, Karthik Narashimhan, Tim Salimans, Ilya Sutskever
- https://gitlab.com/bytecell/dailab/-/issues/60#note_1227064275   

Introduction
--------------------

- raw text(unlabeled data)에서 얻은 언어 정보는 labeled data의 대안이 될 수 있으며 지도 학습이 가능한 경우라도 이를 활용하여 더 좋은 성능을 기대할 수 있음
- unlabeled data의 word-level 정보 활용은 어떤 optimization objective가 transfer에 유용한 text representation을 학습하는데 가장 효과적인지 불명확하고 target task로 학습된 representation을 transfer하는 가장 효과적인 방법에 대한 합의점이 없음
- unlabeled data에서 language modeling objective를 통해 파라미터를 학습하고(unsupervised pre-training) supervised objective를 사용하여 파라미터를 target task에 적용함(supervised fine-tuning)
- Transformer 사용, 텍스트의 장기 의존성을 처리하기 위한 더 구조화된 메모리를 제공하여 다양한 작업에 강력한 transfer 성능을 보여줌
- 네 가지 언어 이해 작업에 대해 평가함 - 자연어 추론, 질의응답, 의미론적 유사도, 텍스트 분류

Related Work
-----------------

- Semi-supervised learning for NLP
  - unlabeled corpus에서 pre-train한 word-embedding을 사용하는 것이 다양한 작업에서 성능을 향상시킴
  - 주로 word-level 정보를 전송하지만 본 연구는 더 높은 수준의 의미론을 목표로함
- Unsupervised pre-training
  - 비지도 pre-training은 지도학습 목표를 수정하는 대신 좋은 초기 지점을 찾는 것을 목표로 하는 준지도 학습임
  - 상당한 양의 새로운 파라미터가 포함되지만 본 연구는 최소한의 변경을 필요로 함
- Auxiliary training objectives
  - 보조적으로 비지도 학습 objective를 추가하는 것도 준지도 학습이라고 볼 수 있음
  - 비지도 pre-training이 target task에 해당하는 언어적 특성을 학습했음을 본 실험과 전 연구에서 보였음

Framework
----------------

![image](https://user-images.githubusercontent.com/49019184/211460227-9e01a51c-1a1c-4a02-86d6-866d4492be32.png)
- high-capacity language model을 학습하는 단계와 fine-tuning 단계를 거침
- Unsupervised pre-training
  - 토큰$v$의 비지도 말뭉치가 주어지면 아래의 likelihood를 최대화하기 위한 standard language modeling objective를 사용함
  - $L_1(v) = \displaystyle\sum_i \log{P(u_i | u_{i-k},..., u_{i-1};\Theta)}$
  - $k$는 context window의 크기, $\Theta$는 모델의 파라미터(SGD로 학습됨)
  - Transformer의 변형인 multi-layer Transformer decoder를 사용함
  - multi-headed self-attenttion operation을 적용
  - $h_0 = UW_e + W_p,  h_l = \text{transformer}\underbar{ }\underbar{ } \text{block} (h_{l-1} \forall i\in [1, n], P(u) = \text{softmax}(h_n W_e^T)$
  - $U$는 토큰의 context 벡터, $n$은 계층 수, $W_e$는 토큰 임베딩 행렬, $W_p$는 위치 임베딩 행렬
- Supervised fine-tuning
  - 위의 학습으로 얻은 파라미터를 supervised target task에 적용함
  - $P(y|x^1,...,x^m) = \text{softmax}(h_l^m W_y)$
  - labeled dataset $C$에서 입력을 받고 그 입력이 최종 transformer block을 지나고 파라미터 $W_y$가 있는 출력 계층도 지났을 때의 label $y$의 확률
  - $L_2 (C) = \displaystyle\sum_{(x,y)} \log{P(y|x^1,...,x^m)}$
  - 위에서 설명한 최종 transformer block 다음의 출력 계층
  - 이전 연구에서 보조적인 목표로서 언어모델을 fine-tuning에 추가하는 것이 지도학습 모델의 생성 성능을 향상시키고 수렴을 가속화하는 것이 증명되어 있으므로 해당 방법을 이용함
  - $L_3(C) = L_2(C) + \lambda * L_1(C)$
  - 가중치 $\lambda$를 찾는 것이 목표
  - fine-tuning 과정에서 학습이 필요한 파라미터는 $W_y$와 다음에 나올 delimiter embedding token이 끝임
- Task-specific input transformations
  - 사전 학습된 모델이 처리할 수 있는 순서로 입력이 주어지는 traversal-style 접근법을 사용함
  - 약간의 변형만으로 여러 task에 적용할 수 있음

Experiments
-------------------

- Setup
  - Unsupervised pre-training
    - BooksCorpus 데이터셋 사용(7000 이상의 출판되지 않은 책)
  - Model specifications
    - 큰 틀은 original transformer를 따름
    - self-attention head(768 dimensional states, 12 attention heads)가 있는 decoder-only transformer 12층을 학습함
    - position-wise feed-forward network에는 3072차원의 내부 상태가 사용됨
    - Adam optimization 사용(max learning rate 2.5e-4)
    - epoch 100, batch size 64
    - bytepair 인코딩 사용, 활성 함수 GELU 사용
    - 전처리 fifty library 사용, punctuation과 whitespace 표준화, 토크나이저 spaCy 사용
  - Fine-tuning details
    - 비지도 사전학습에 사용한 파라미터 재사용
    - learning rate 6.25e-5, batchsize 32, dropout 0.1, $\lambda$ 0.5
    - 3 epoch만해도 학습이 충분했음

- Supervised fine-tuning 
    - ![image](https://user-images.githubusercontent.com/49019184/211460254-1b1921a1-b503-456e-a4c6-a773e8e61349.png)
    - task별 사용 데이터셋 
    - Natural Language Inference
      - SNLI, NMLI, QNLI, SciTail, RTE 평가 데이터셋 사용
      - ![image](https://user-images.githubusercontent.com/49019184/211460261-ad2c32a4-31fa-40db-9ec0-33939ea2e494.png)
      - 이전 연구와 비교했을 때 높은 성능을 보임
    - Question answering and commonesense reasoning, Semantic Similarity
      - QA&CR에서 RACE 평가 데이터셋 사용, Story Cloze Test로 평가
      - ![image](https://user-images.githubusercontent.com/49019184/211460277-d498e277-ce1b-4366-be90-1e7bb1c76872.png)
      - 이전 연구와 비교했을 때 높은 성능을 보임
  
Analysis
-------------

- Impact of numver of layers transferred
  - 비지도 사전학습에서 지도 목표 태스크 사이의 계층의 변수를 transfer하는 영향을 관찰하였음
  - ![image](https://user-images.githubusercontent.com/49019184/211460296-0db72d05-deb8-4395-be08-4afdb60c976a.png)
  - 임베딩을 전송한 결과에서 성능 향상을 발견함
  - 이는 사전 학습된 모델이 목표 태스크에 유용하다는 것을 나타냄
- Zero-shot Behaviors
  - 생성된 모델이 언어 모델 cpability 향상을 위해 많은 task를 학습하고 LSTM과 비교했을 때 구조화된 attentional memory가 transformer의 transfer에 더 도움이 된다고 가정함
  - ![image](https://user-images.githubusercontent.com/49019184/211460304-08610fff-ee41-40cd-87ba-f384d516b42b.png)
  - generative pre-training 과정에서 이러한 휴리스틱 솔루션의 효과
  - 학습을 반복하는 동안 성능이 안정적이고 지속적으로 향상된 것으로 보아 generative pre-training이 다양한 태스크에 도움이 주는 것을 확인할 수 있음
- Ablation studies
  - ![image](https://user-images.githubusercontent.com/49019184/211460321-21ad4d9d-c6b7-46cf-9e08-7c98101b2976.png)
  - fine-tuning 과정에서 auxiliary LM objective의 성능과 pre-training의 성능을 실험한 결과임
  
Conclusion
----------------

- generative pre-training과 discriminative fine-tuning이 다양한 task에 적용되어 좋은 성능을 보였음
- 긴 연속 텍스트를 가진 다양한 말뭉치로 사전 학습을 하여 장기 의존성과 다양한 지식을 잘 처리할 수 있었음
