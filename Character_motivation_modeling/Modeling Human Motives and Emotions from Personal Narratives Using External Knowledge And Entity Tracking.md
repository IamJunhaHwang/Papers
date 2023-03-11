#### 논문 - Modeling Human Motives and Emotions from Personal Narratives Using External Knowledge And Entity Tracking
- 저자: Prashanth Vijayaraghavan, Deb Roy (MIT Media Lab)
- WWW 2021
- [논문 링크](https://dl.acm.org/doi/pdf/10.1145/3442381.3449997)
------------------------------------------------------------------
- **목적**
  - 학자들은 Narrative는 metalizing(정신화) 과정을 환기 시켜 사회적 인식에 영향을 미친다고 주장
    - mentalizing은 다른 사람의 생각, 신념, 태도, 감정, 동기 등의 정신 상태 종류의 추론을 설명하는데 사용됨
    - 즉, Narratives를 이해하는 것은 인간을 이해하는 열쇠라고 할 수 있음
  - Narratives와 mentalizing 사이의 관계의 특정 측면을 발견하는 것이 목표
  - Narratives에서 인간의 동기와 감정을 모델링하는 계산 접근법을 개발하는 것이 목표
  - character-specific contexts 모델링과 free-text responses에 대한 사전 교육을 통해 이 새로운 리소스에 대한 벤치마크 결과를 제공
  - Character의 mental state를 저장하고 설명하기 위한 Nemo라는 트랜스포터 기반 인코더-디코더 아키텍쳐 구현
  - Motivation 및 Emotion text 표현식의 weak-annotations을 포함하는 개인 Narratives 및 Social Commonsense Knowledge 데이터를 수집함
    - 개인적인 이야기와 사회적 상식에서 나오는 감정들은 웹에서 오는 지식을 의미함
    - 아래 그림 1은 레딧의 개인적인 이야기 샘플을 보여줌   
    ![image](https://user-images.githubusercontent.com/49019292/223055811-fc50d6f6-2f40-4675-bfa7-d72b9cad59a1.png)   
    - Narrator인 "I"가 병원에 급히 갔다라는 문장은 그의 아버지를 돌보기 위해서, 그의 아버지를 걱정하기 위해서 라는 감정 정보가 들어가 있다고 볼 수 있음 
----------------------------------------------------------
- **방법**
  - 데이터 수집 파이프라인   
    ![image](https://user-images.githubusercontent.com/49019292/223055834-5c2b7d13-af83-4c9c-8fbc-5deae33962a0.png)      
    - 개인 Narrative 말뭉치는 전체 이야기 맥락을 고려한 Motive와 감정 추출을 포착하고 자함
    - Personal Narratives Corpus  
      - 일상적인 상호작용, 삶의 경험, 관계, 코믹한 상황과 관련된 레딧의 게시물을 수집하여 Personal Narratives Corpus를 구성함 
        - 439,408개의 게시물, 평균 길이는 12.08개의 문장으로 이루어짐
      - 아래 그림은 길이와 관련된 데이터 분포를 보여줌   
        ![image](https://user-images.githubusercontent.com/49019292/223055866-16ed09cf-0103-4aa0-8295-403e004aef93.png)      
      - Motives와 관련된 데이터 세트를 만들기 위해 의도 또는 목적과 관련된 특정 표현을 찾음
        - 인간의 Motives와 Emotions는 여러 가지 방법으로 언어적으로 표현될 수 있음
      - Narratives에서 등장인물의 감정을 추출하기 위해 유사한 전략을 채택함
        - emotion-directed, FrameNet의 어휘 단위, 감정 어휘 목록에서 추출한 400개의 키워드를 식별함
        - 문장이 부정적이거나 감정 키워드가 술어의 일부가 아니며 첫번째 인수가 명사도 대명사도 아니면 문장을 삭제함
        - spaCy의 rule-based matching tool를 사용하여 텍스트에서 특정 패턴을 캡쳐함으로써 이를 달성함
        - 아래는 데이터 통계 표   
          ![image](https://user-images.githubusercontent.com/49019292/223055906-382c15a3-c243-4f0e-b60b-c12e636eb992.png)      
        - 아래는 샘플 추출   
          ![image](https://user-images.githubusercontent.com/49019292/223055935-c02886ba-6633-4033-a5fe-02d6b429e53f.png)     
      - 3명의 non-author annotators는 검증을 위해 300개의 인스턴스의 랜덤 샘플에 레이블을 지정함
        - 서술적 맥락이 주어질 때 등장인물이 표현하거나 암시적으로 느끼는 올바른 의도나 감정 설명을 선택하도록 함
        - 89%와 93%의 경우에서 추출된 의도(Fleiss' λ = 0.87)와 감정(Fleiss' λ = 0.90) 텍스트에 동의한다는 것을 발견
    - Social Commonsense Knowledge
      - 암묵적인 mental state를 얻기 위해 Atomic 및 ConceptNet과 같은 기존 소스에서 얻은 사회적 상식 지식 활용 및 서술 말뭉치에서 이벤트에 대한 더 많은 지식 강화를 위한 웹마이닝 
      - 우리는 사회적 역할(예: 학생, 어머니, 남자친구 등)이 행동 뒤에 숨겨진 동기와 감정에 대한 추가 정보를 제공한다고 가정
  - Nemo: Our propsed model
    - 전반적인 목표는 mental state의 캐릭터별 임베딩, 특히 motives와 emotions를 학습하고 사회적 역할 정보, 선행 narratives context 및 mental state 인코딩과 함께 외부 지식을 통합하여 텍스트 셜명을 생성하는 것
    - 이를 위해 트랜스포머 기반 인코더-디코더 아키텍쳐 소개
    - 아래 그림 4는 Nemo 모델의 개요를 제공함   
      ![image](https://user-images.githubusercontent.com/49019292/223055970-a649068d-d606-4830-94c3-415f5670b694.png)      
    - 모델 주요 구성 요소
      - 지식-강화 모듈(Kem), 스토리 엔티티 인코더, 엔티티 기반 메모리 모듈, Intent-Emotion 설명 생성기로 구성됨 
    - Story Entity Encoder
      - 아래 그림 5는 모델 아키텍쳐에 대한 자세한 내용을 보여줌   
        ![image](https://user-images.githubusercontent.com/49019292/223056004-ff3a8550-385d-4cae-9f64-9b8602de5cb4.png)      
      - 인코딩 전략인 StoryEntEnc는 다음과 같이 정의됨   
        ![image](https://user-images.githubusercontent.com/49019292/223056065-b8261447-01f8-4d29-b768-45fa70cf1bcb.png)      
        - 여기서 e<sub>j</sub>는 ∈ ε는 고려 중인 엔티티
        - StoryEntEnc는 N개의 동일한 레이어 스택으로 구성 
        - 현재 문장과 함께 캐릭터 정보를 연결하여 이야기 문장의 실체 또는 문자 인식 표현 생성
        - 이야기 컨텍스트를 인코더에 통합하는 추가 컨텍스트 주의 하위 계층을 도입
        - 특정 정신 상태 속성과 관련된 지식 표현을 융합   
          ![image](https://user-images.githubusercontent.com/49019292/223056113-f38f862b-c823-495b-aa0b-4763a15365c8.png)      
      - Context-Attention & Gating
        - 이전 문장에서 스토리 컨텍스트 정보를 계산하기 위한 표준 트랜스포머 인코더 계층 구현    
        - 각 문장 시작과 끝에 [CLS]와 [SEP] 토큰을 각각 삽입
        - 새로운 하위 계층이 추가되므로 하위 계층의 정보가 현재 문장 표현에 미치는 통제되지 않은 영향을 방지하기 위해 게이팅 메커니즘 도입   
          ![image](https://user-images.githubusercontent.com/49019292/223056161-1b2273a2-9854-405f-8385-3affb3418484.png)      
      - Intent-Emotion Explanation Generator
        - 인간의 인지 행동에 의해 동기 부여 되어 시퀀스 생성 프레임워크에 대한 심의 과정을 탐구 
        - 1단계 디코딩 출력은 엔티티 기반 메모리 모듈로 부터 획득된 엔티티의 정신 상태 컨텍스트와 함께 second pass decoder에 공금됨
        - 2단계 디코딩 절차는 다음과 같이 표현됨   
          ![image](https://user-images.githubusercontent.com/49019292/223056193-70d25129-d61f-475c-9a12-58c22deda1c8.png)      
        - First-pass Decoding
          - 디코딩 절차는 다음과 같이 표현됨   
            ![image](https://user-images.githubusercontent.com/49019292/223056232-c11956dd-8cc6-47e4-a532-d30dae27238f.png)      
        - Second-Pass Decoder
          - 엔티티별 외부 메모리에 저장된 엔티티의 이전 정신 상태 임베딩을 첫번째 패스 디코더 출력과 함께 사용하여 현재 엔티티 상태를 맥락화함   
            ![image](https://user-images.githubusercontent.com/49019292/223056279-5d84948e-5324-42b0-9367-7244b28d28d3.png)      
    - Knowledge-Enrichment Module
      - 지식 강화 모듈은 의미론적 기억과 유사하게 볼 수 있음
      - 일반적으로 시맨틱 메모리는 사건, 사실, 개념과 관련된 일반 지식의 장기 저장고를 의미
      - 핵심 아이디어는 스토리 문장을 프래그매틱스 인식 임베딩으로 인코딩하는 것
      - 입력은 정신 상태 소석과 스토리 문장의 연결임
      -  EventBert를 통해 제공되어 속성별 상황별 소셜 이벤트 임베딩, R(t)m을 생성
    - Entity-based Memory Module
      - 엔티티 기반 메모리 모듈은 특정 이야기에서 등장인물의 정신 상태를 이상적으로 저장하는 에피소드형 메모리로 볼 수 있음 
      - Memory Attention
        - 우리의 디코더는 각 정신 상태 속성에 대해 엔티티의 이전 정신 상태 표현에 대해 Multi head attention 메커니즘을 적용
  - Training & Hyperparameters
    - 데이터는 7:1:2 = 훈련:검증:테스트세트로 분할
    - 첫번째 단계에서 엔티티 또는 문자 정보가 입력 텍스트와 연결되는 모든 사회 상식 지식 데이터를 사용하여 모델을 사전 훈련함
    - 두번째는 이전 서술적 맥락과 함께 현재 이야기 문장을 모델링하는 것을 포함 
    - 그리드 서치를 사용하여 하이퍼 파라미터 조정
    - 과적합 방지를 위한 0.2  비율의 드롭아웃 사용
    - GloVe 벡터와 ELMo 기반의 상황별 임베딩으로 실험
    - Adam을 learning rate = 0.0002, batch size = 8 설정
    - 추론에서는 빔 서치 활용
----------------------------------------
- **실험 및 결과**
  - 실험
    - 데이터세트, baseline, model variants, modes, metrics와 같은 다양한 평가 설정에 대해 설명
    - 다음과 같은 임의의 질문을 연구하기 위해 실험 설계
      - RQ1: 우리 모델은 설명 생성 작업의 다른 baseline과 비교하여 얼마나 잘 수행되는지, 각 구성 요소가 전체 성능에 얼마나 영향을 미치는지
      - RQ2: 모델 표현을 사용하여 레이블이 지정된 동기 및 감정 반응 범주를 기반으로 상태 분류를 수행할 수 있는지
      - RQ3: 학습된 정신 상태 표현이 다운스트림 작업으로의 전달 능력을 나타내는지
    - Explanation Generation Task (RQ1)
      - 표 2는 설명 생성 작업 평가에 사용되는 데이터 세트 요약   
         ![image](https://user-images.githubusercontent.com/49019292/223056312-2fd8b21e-8c2b-4fcc-90fb-a61cde9959a6.png)       
      - LSTM, REM, NPN, GPT와 성능을 비교함
      - personal narratives corpus에 대해 평가함으로써 Kem, Emm, Eie-Dec의 세가지 모델 구성 요소에 영향을 평가함 
      - 단어 중복 기반 메트릭을 피하고 임베딩 평균 및 벡터 극한값과 같은 임베딩 기반 메트릭을 계산함
      - 문장 수준 벡터 사이의 코사인 유사성을 계산
      - 미세조정된 GPT 모델에 비해 ~9% 및 ~12%의 평균 향상을 달성한다는 것을 관찰
      - 그림 7은 GPT와 함께 여러 패스에서 제안된 모델에 의해 생성된 샘플 동기를 제공           
        ![image](https://user-images.githubusercontent.com/49019292/223056341-63c912f4-2a20-4741-a43a-e8f6bb679bf2.png)      
    - State Classification Task (RQ2)
      - Story Commonsense 데이터 세트는 ROC Stories 교육 세트에서 선택한 15,000개의 단편에 걸쳐 동기와 감정에 대한 30만 개 이상의 낮은 수준의 주석으로 구성
      - Zero-shot, Supervised, Low-resource regimes 과 같은 설정으로 실험 수행
      - 결과    
        ![image](https://user-images.githubusercontent.com/49019292/223056385-f3fd3ec5-d02f-45b6-b806-b138f6fae2b7.png)     
      - 그림 8의 결과는 설명과 함께 미세조정된 모델 변형이 설명 미세조정 없이 해당 모델보다 더 적은 도메인 내 레이블링된 데이터로 더 빠르게 학습한다는 것을 시사함   
        ![image](https://user-images.githubusercontent.com/49019292/223056411-de2471f6-a44f-48dd-9b8a-8e3442cfc388.png)   
      -  컨텍스트와 소스 텍스트 사이에 attention 맵을 그려 텍스트에 대한 설득력 있는 설명을 생성하는 데 컨텍스트의 효과를 조사함   
        ![image](https://user-images.githubusercontent.com/49019292/223056452-8c32a104-6eec-4e4d-a893-1dc3b59b3397.png)     
    - Application: Empathetic Dialogue Generation (RQ3)
      - Nemo는 이야기에서 동기와 감정 상태를 추론하기 때문에, 우리는 학습된 임베딩이 이 대화 생성 작업에서 성능 향상으로 이어질 수 있다고 제안함
      - 우리의 Ensem-SCS+ 모델이 학습된 표현을 사용하는 것의 영향을 정량화하여 자동화된 메트릭에서 상당한 개선 보여줌   
        ![image](https://user-images.githubusercontent.com/49019292/223056482-4f517195-350f-40da-bb33-e9dae1d23883.png)      
------------------------------------------------------
- **고찰**
  - 매우 어렵다..
  - 컨셉은 금방 이해했는데, 확실히 임베딩 방법이라던지 기술적인 부분 or 수식은 이해가 가지 않는다...
    - 아마 30% 정도 이해한 것 같은데 ㅠㅠ
  - 그래도 인간의 Motives나 Emotions을 추출한 정보나 방법에 대해서는 흥미로웠다.
    - 기준없이 아무 문장에서 추출한 것은 아니고, 다 근거가 있었다.
  - 잘만들어진 모델에도 한계점은 있었음!   
    ![image](https://user-images.githubusercontent.com/49019292/223056517-52328bba-9731-476f-b0ec-20288995f425.png)      
    - 위의 그림이 논문에서 제시하는 Nemo의 한계를 보여주는 샘플임
    - 논문에선 입력을 작은 단위로 쪼개서 하면 더 잘 생성할 수 있을 것이라는 해결책을 제시했음
    - 이는 번역기도 문장을 짧게 번역하면 더 번역이 잘 되는데, 이러한 점이 생각났음  
    - 그렇다면 문장이 난해 해질수록 성능이 좋지 않다는 의미가 되겠군..
  - 보통 이런 논문들은 방법만 제시하기보단 방법과 학습시킨 모델까지 한번에 제시하는구나 싶었다.
    - 가령 이 논문에서도 모티브랑 이모션 데이터를 사용하는 방법만 제시했다기 보단, 이를 적용하여 만든 사전 학습 모델까지 제안했다는 점
  - 흥미롭긴 했지만 너무 어려워서 나중에 다시 한번 읽어보고 싶다..ㅎㅎ
