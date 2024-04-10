#### 논문 - Pretrained Language Models for Text Generation: A Survey

- 저자: Junyi Li, Tianyi Tang, Wayne Xin Zhao, Ji-Rong Wen
- [논문 링크](https://arxiv.org/pdf/2105.10311.pdf)
------------------------------------------------
- **서론**
  - 우리는 텍스트 생성을 위한 PLM 에서 달성된 주요 발전에 대한 개요 제시
  - RNN, CNN, attention mechanism에 기반한 텍스트 생성 작업을 해결하기 위한 다양한 연구가 제안됨
    - 입력에서 출력으로의 semantic mapping의 end-to-end 학습이 가능
  - 그러나 주요 문제는 대규모 데이터 세트의 가용성에 있음
    - 대부분의 supervised 텍스트 생성의 데이터 세트는 매우 작음
    - 그러나 심층 신경망은 학습해야할 매개변수가 많으므로 작은 데이터 세트를 사용하면 overfit될 가능성이 매우 높음
  - PLM(Pretrained Language Model)은 일반적으로 다운스트림 task에 유용하며 새로운 모델을 처음부터 훈련하는 것을 피할 수 있음
  - Computational power의 증가와 tranformer 아키텍쳐의 등장에 따라 PLM이 더 발전했으며 BERT 및 GPT와 같은 많은 task에서 뛰어난 성능을 달성함
  - 연구자들은 PLM을 기반으로 텍스트 생성 작업을 해결하기 위한 다양한 방법을 제안
  - Zaib et al. [2020] and Guan et al. [2020]는 dailogue 시스템 및 요약과 같은 일부 텍스트 생성 subtask에9 대한 연구 종합을 제공
  - Qiu et al. [2020]는 전체 NLP 도메인에 대한 두 세대의 PLM을 요약하고 다양한 확장 및 접근법 제시
  - 텍스트 생성 연구자에게 관련 연구에 대한 포인터를 제공하는 것을 목표로 함
---------------------------------------------------
- **본론 및 결론**
  - Task and Typical Applications
    - 텍스트 생성 작업을 공식적으로 정의
    - 텍스트 생성의 핵심은 ![image](https://user-images.githubusercontent.com/49019292/223050188-5ce7d4b3-0927-4c06-9302-ba9d9a30c410.png)의 일련의 이산 토큰을 생성하는 것
      - 각 y<sub>j</sub>는 단어 vocab V에 추출됨
      - 대부분의 경우 텍스트 생성은 속성, 텍스트 및 구조화된 데이터와 같은 입력데이터에 따라 결정되며 X로 표시됨
      - 밑의 식 1은 텍스트 생성 작업을 보여줌   
      ![image](https://user-images.githubusercontent.com/49019292/223050294-1c9e68b6-af1d-4ec2-aa2e-d2173c54ebc0.png)   
        - X의 정보는 텍스트 생성 과정을 안내하고 생성된 텍스트의 모드를 제어하는 역할을 함
        - X의 가장 일반적인 형태는 텍스트 시퀀스이며, 기계 번역, 요약 및 대화 시스템과 같은 여러 응용 프로그램이 있음
        - 아래 표 1은 주요 텍스트 생성에 대한 공식을 제시    
        ![image](https://user-images.githubusercontent.com/49019292/223050404-f68ed836-5d3a-4496-ac52-f10c460056fd.png)      
  - Standard Architectures for Text Generation
    - PLM은 unlabeled large-scale text data로 pretraining하고 다운스트림 task에서 fine-tuning될 수 있음
      - Transformer를 기반으로 함
      - 텍스트 생성 작업의 경우 일부 PLM은 인코더-디코더 framework를 따르는 표준 트랜스포머 아키텍쳐를 활용하거나 디코더 전용 트랜스포머를 적용함
    - Encoder-decoder Transformer
      - 표준 트랜스포머는 두 개의 트랜스포머 블록으로 구성된 인코더-디코더 아키텍쳐 사용
      - 인코더는 입력 시퀀스를 받고 디코더는 출력 시퀀스를 생성하는 것을 목표로 함
      - MASS [Song et al., 2019], T5 [Raffel et al., 2020], and BART [Lewis et al., 2020] 등 
    - Decoder-only Transformer
      - GPT [Radford et al., 2019; Brown et al., 2020] and CTRL [Keskar et al., 2019]는 단일 트랜스포머 디코더 블록 사용
      - 각 토큰은 이전 토큰에만 적용 가능한 단방향 self-attention masking을 적용
    - Raffel et al. [2020]은 명시적 encoder-decoder attention을 추가하는 것이 좋다는 결론을 내림
    - 택스트 생성 작업의 핵심은 입력에서 출력으로의 semantic mapping을 학습하는 것
  - Modeling Different Data Types from Input
    - Unstructured input
      - 대부분의 연구는 비정형 텍스트 입력을 모델링하는데 중점을 둠
      - Liu and Lapata [2019] and Zheng and Lapata [2019]는 텍스트의 의미를 대부분 보존하면서 저차원 벡터로 응축하기 위한 텍스트 인코더로 PLM을 사용했음
    - structured input
      - 구조화된 데이터(예-그래프, 표)는 기상 보고서 생성과 같은 실제 응용 프로그램에서 텍스트 생성을 위한 중요한 종류의 입력임
      - 그러나 label이 지정된 구조적 데이터를 대량으로 수집하기 힘듬
      - Chen et al. [2020b] and Gong et al. [2020]은 few-shot 설정에서 데이터-텍스트 생성을 위한 PLM 통합을 연구하였음
      - Gong et al. [2020]  구조 정보를 복구하기 위한 보조 재구성 작업을 제안하여 구조 정보 모델링 용량을 향상
    - Multimedia Input
      - 이미지 캡션 및 음성 인식과 같은 입력 멀티 미디어 데이터(예-이미지, 비디오 및 음성)로 가져가려는 몇가지 시도가 존재함
        - VideoBERT 등
  - Satisfying Special Properties for Output Text
    - Relevance
      - linguistic literatures [Li et al., 2021c]에 따르면 관련성은 출력 텍스트의 주제가 입력 텍스트와 높은 관련성을 갖는 것을 말함
        - 대표적인 예는 대화 시스템의 작업으로, 생성된 응답이 입력 대화 이력과 관련이 존재해야 함 
      - TransferTransfo [Wolf et al., 2019]와 DialoGPT [Zhang et al., 2020]는 기존 RNN 기반 모델보다 더 관련성 있고 맥락에 맞는 응답을 생성할 수 있었음 
    - Faithfulness
      - 생성된 텍스트의 내용이 입력 텍스트의 사실과 모순되지 않아야 한다는 것을 의미함
        - 텍스트 요약 작업이 대표적인 예임
    - Order-preservation
      - Order-preservation은 입력 및 출력 텍스트 모두에서 의미 단위(단어, 구문 등)의 순서가 일관됨을 나타냄
        - 가장 대표적인 얘는 기계 번역 작업 
        -  Lin et al. [2020]은 보편적인 다국어 기계 번역 모델을 사전 교육하는 접근법인 mRASP 제안
    - 전체 정리 표   
      ![image](https://user-images.githubusercontent.com/49019292/223050503-71ed65b3-575e-4d19-947c-0e73ce8a9583.png)      
  - Fine-tuning Strategies for Text Generation
    - Data view
      - 새로운 도메인에서 텍스트 생성 작업에 PLM을 적용할 때, 새로운 도메인의 특성에 적합하고 효과적인 미세 조정 전략을 설계하는 방법은 중요한 고려사항임
      - Few-shot Learning, Domain Transfer 등이 있음
    - Task view
      - 새로운 도메인의 특성 외에도, PLM을 미세 조정할 때 특정 생성 작업에서 언어 일관성 및 텍스트 충실도와 같은 몇 가지 특별한 문제를 고려하는 것도 의미가 있음
      - Enhancing Coherence, Preserving Fidelity 등의 방법이 있음
    - Model view
      - 이 부분에서는 모델을 고려하여 몇 가지 미세 조정 방법도 있음
  - 이 논문은 텍스트 생성을 위해 사전 훈련된 언어 모델에서 달성된 최근의 발전에 대한 개요를 제시
    - 우리는 또한 텍스트 생성을 위한 몇 가지 유용한 미세 조정 전략에 대해 논의했음
-----------------------------
- **고찰**
  - 오랜만에(? 읽는 기술 논문이 아닌 survey 논문!
  - 텍스트 생성을 위한 PLM에 대한 이야기가 주로 있긴 했지만 중간에 비전 관련된 모델도 같이 나와서 처음보는 모델이 몇 개 보이기도 했었다.
  - 매우 세분화해서 설명하고 있다는 생각이 들었다. 
  - input에 관해서도 정의하고 가는게 색다르게 느껴졌다.
    - Unstructured Input, structured input, mutilmedia Input 등.. 근데 텍스트 생성이 초점인 것 같은데 multimedia input은 왜 설명한거지....
  - Output text의 주요 특성에 대해서도 3가지를 설명했는데, 3가지로 나눈것은 무엇을 보고 참고한걸까?
  - 미세 조정 전략도 마찬가지로 3가지로 나누었는데(data, task, model), 내가 나눠도 이렇게 나누었을 것 같아서 비교적 이해하기 좋았던 것 같다.