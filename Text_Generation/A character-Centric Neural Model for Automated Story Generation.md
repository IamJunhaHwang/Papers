#### 논문 - A character-Centric Neural Model for Automated Story Generation

-  저자: Danyang Liu, Juntao Li, Meng-Hsuan Yu, Ziming Huang, Gongshen Liu, Dongyan Zhao, Rui Yan
- Accepted by AAAI 2020
- [논문 링크](https://ojs.aaai.org/index.php/AAAI/article/view/5536)
---------------------------------------------------------
- **목적**
  - 자동화된 스토리 생성은 일관된 캐릭터와 플롯으로 구성된 이해할 수 있는 스토리를 자동으로 생성하는 것을 목표로 하는 작업
  - 최근에는 무수한 신경 모델이 수작업 도메인 지식 없이 스토리 생성 작업을 모델링하도록 설계되어 다양한 시나리오에 더 유용함
    - variational autoencoder, GAN, convolutional sequence to sequence model과 같은 신경망을 기반으로 함
  - 신경망 기반 이야기 생성의 경우, 이전 프레임워크에서는 주로 이 문제를 standard long document 생성 task으로 처리하고 long-range 종속성 문제를 해결하기 위해 multi-stage 생성 프로세스로 분해할 것을 제안함
    - 그러나 이러한 모델은 훈련 이야기에서 구문 및 어휘 정보를 잘 포착할 수 있지만, 캐릭터, 플롯 등은 고려하지 못할 수 있음
  - 앞서 언급한 문제를 해결하기 위해, 우리는 스토리 장르의 관점에서 스토리를 생성하기 위해 조사
  - deep neural generation networks를 캐릭터 신뢰성 향상에 효과적인 것으로 확인된 캐릭터 모델링과 결합하려고 시도
  -  dialogue systems(Li et al. 2016)에서 이러한 전략이 neural response generation에서 화자 일관성을 향상 시킬 수 있음을 검증하고자 함
  - 본 논문에서는 distributed embedding에서 캐릭터를 인코딩하는 character-centric neural story telling model을 제안
  - 아래 그림 1 처럼 모델에 성격, 상황, 행동 요소를 포함함   
  ![image](https://user-images.githubusercontent.com/49019292/223050985-3024663e-2b9f-4c9d-ab99-c89ac336103c.png)      
  - 아래 표 1은 generation process의 예를 보여줌   
  ![image](https://user-images.githubusercontent.com/49019292/223051086-d1e04942-9039-4820-aaab-dd3afb32a22a.png)      
-------------------------------------------------------------
- **방법**
  - character-centric neural story-telling model을 자세히 소개
    - 이전의 생성된 문장은 다음 문장을 생성하기 위한 입력으로 사용됨
      - INPUT, title T = {t<sub>1</sub>, t<sub>2</sub>, ... , t<sub>m</sub>}와 훈련 단계에서 학습된 문자 임베딩 C = {c<sub>1</sub>, c<sub>2</sub>, ... , c<sub>p</sub>}가 이야기 생성을 위한 입력으로 모델에 제공됨
        - 여기서 ti는 i번째 단어, m은 제목의 길이, ci는 이야기의 i번째 문자 임베딩을 나타냄
      - OUTPUT, 스토리 Y = {y<sub>1</sub>, y<sub>2</sub>, ... , y<sub>n</sub>}이 모델의 결과로 생성됨
    - 아래 그림 2에서 보는 바와 같이 모델은 스토리 생성을 2단계로 분해함
      - 현재 컨텍스트에 대한 캐릭터 반응 결정 액션 예측, 완전한 문장을 구성하는 문장 생성   
      ![image](https://user-images.githubusercontent.com/49019292/223051140-2045967f-a422-4d02-a6a5-8b5a3f092714.png)      
  - Character Embedding
    - 우리의 모델은 각각의 개별 문자를 임베딩으로 나타냄
    - 각 캐릭터에 대해 Related verbs(agent verbs, patient verbs), Attributes(형용사 및 형용사 수식어)와 같은 언어적 특징을 추출 
    - 이는 문자 임베딩을 초기화하는 데 사용됨   
    ![image](https://user-images.githubusercontent.com/49019292/223051183-776926b0-6c3b-4e4e-83a6-12bb2d9dc965.png)      
    - 캐릭터 임베딩은 캐릭터의 행동 결정에 영향을 미치는 캐릭터의 속성(예: 성격, 직업, 감정, 성별, 나이)을 인코딩 함
  - Action Predictor
    - 우리 모델은 등장인물이 이야기의 중심임
    - 이야기 전개는 현재 상황에 근거하여 건전하고 믿을 수 있는 일련의 행동으로 묘사될 수 있음
    - 컨텍스트 임베딩은 현재 환경 정보와 스토리 세계의 등장인물을 둘러싼 상황을 나타냄(LSTM에 의해 계산됨)   
      ![image](https://user-images.githubusercontent.com/49019292/223051227-889a4312-8fb7-4f44-926b-d9cfd3f7d261.png)      
    - 캐릭터 임베딩은 캐릭터가 상황에 어떻게 반응할 것인지에 대한 정보를 전달
    - 동작에 대한 확률은 문자 임베딩 C와 현재 상황 Si의 연결을 기반으로 MLP에 의해 계산   
    ![image](https://user-images.githubusercontent.com/49019292/223051261-6a1d66d8-67f8-4f03-be9c-12f6249ee370.png)   
    - time step i에서 해당 문자 C를 가진 동사 Vi가 주어지면 훈련 데이터의 negative log probability을 최소화하도록 훈련됨   
    ![image](https://user-images.githubusercontent.com/49019292/223051435-aec4fb3e-6735-40a6-a57d-62b10a792ea2.png)      
  - Sentence Generator 
    - 우리는 수행할 동작 Vi, 문자 임베딩 C 및 time step i의 현재 상황 Si를 기반으로 문장을 생성하는 조건부 생성 문제로 공식화 함
    - LSTM 문장 생성기에서 seq2seq 구조를 사용함
    - 인코더   
      ![image](https://user-images.githubusercontent.com/49019292/223051475-ab6d4a08-988e-4288-a12a-a2648fc56f41.png)      
    - 디코더   
      ![image](https://user-images.githubusercontent.com/49019292/223051529-000e3d82-6197-4c1f-8ae8-c8ec495313c8.png)     
    - 손실함수   
      ![image](https://user-images.githubusercontent.com/49019292/223051574-845b144d-448a-498d-bf83-8d90999e0777.png)      
    - 우리 모델은 모든 time step에서 캐릭터 임베딩을 언급하며, 이는 모델에게 캐릭터의 속성(예: 나이, 성별, 성격)에 맞는 적절한 행동을 선택할 수 있는 능력을 제공함
 -----------------------------------------------------
- **실험 및 결과**
  - Dataset
    - 우리는 위키피디아에서 추출한 영화 줄거리 요약의 말뭉치에 대한 실험을 수행
      - 인물들에 대한 설명과 함께 영화 속 사건들에 대한 간결한 요약을 포함
      - 42,306개의 이야기가 존재
      - 토큰화를 위해 Moses 디코더 도구를 사용하고 모든 단어를 소문자로 변환하였음
      - vocab 크기는 50,000개, 2.06% 단어가 <unk>기호로 대체됨
      -  훈련, 검증, 테스트를 위해 말뭉치를 34,306/4,000/4,000 스토리로 무작위로 분할함
    - 우리 모델은 이야기 제목을 기반으로 이야기를 생성하도록 설계되었기 때문에 우리는 영화 ID를 해당 영화 이름으로 대체
    - Stanford CoreNLP library를 사용하여 구조와 암시적 정보를 추출
    - 전치사구 제거, 접속사가 있는 문장구를 나눔
  - Baselines
    - Conditional Language Model(C-LM), Seq2Seq with Attention(Vanilla-Seq2Seq), Incremental Seq2Seq with Attention(Incre-Seq2Seq), Plan-and-write, Event Representation, Hierarchical Convolution Sequence Model (Hierarchical) 방법들을 사용
  - Experimental Settings
    - 문장 생성기의 경우 인코더, 디코더 모두 512차원의 hidden state를 가짐
    - 문자 임베딩은 단어 임베딩 크기와 동일한 512로 설정됨
    - learning rate α = 0.001인 Adam 사용
  - Evaluation
    - BLEU, Perplexity(복잡성), Human Evaluation 방법 사용 
      - Huma Evaluation은 5명의 인간 평가자를 고용하여 생성된 이야기에 점수를 매김
        - 점수가 높을수록 성능이 높음
        - 서로 다른 모델에서 생성된 100개의 스토리를 무작위로 선택하여 평가자에게 배포함
        - 세가지 측면의 평균을 최종 점수로 사용
        - 아래 표 3은 평가 기준을 나타냄   
        ![image](https://user-images.githubusercontent.com/49019292/223051630-bea45758-2e7c-40f4-a512-1d5fc10dc0f2.png)   
  - 실험 결과
    - 아래 표 4는 제안된 모델 및 baseline 모델의 성능을 보여줌   
    ![image](https://user-images.githubusercontent.com/49019292/223051658-b1184893-967f-463a-9615-717d4de7f721.png)   
    - 아래 그림 5는 Ablation 연구를 수행하고 캐릭터 임베딩이 생성된 스토리에 어떤 영향을 미치는지 조사하기 위해 모델 구현   
    ![image](https://user-images.githubusercontent.com/49019292/223051691-ab800bee-06f7-4a9d-b088-14d0811648bb.png)   
    - 아래 그림 6은 영화 줄거리 말뭉치에서 제안된 캐릭터 중심 모델에 의해 생성된 예시 이야기를 보여줌   
    ![image](https://user-images.githubusercontent.com/49019292/223051718-5b41744f-d23e-4dac-a6f2-2677576c8787.png)   
--------------------------------------------------
- **고찰**
  - 확실히 수식을 제안하는 논문이 큰 contribution을 가지는 것 같다.
  - 캐릭터 신뢰성, 이야기 일관성을 향상하는데 가장 큰 비중을 두고 실험을 수행한 것 같다.
  - 그리고 실제로 그럴듯한 결과를 형성해낸 것이 실험이 성공적이었다고 말할 수 있을 듯
    - 그래도 말이 안되는 이야기를 생성하기도 했던 것 같다.   
    ![image](https://user-images.githubusercontent.com/49019292/223051743-a1171e88-7c79-441e-a4d3-e3b4234e2759.png)      
      - 위에 요약에는 싣지 않았던 예시인데, 캐릭터 중심이라서 그런지 캐릭터에 대한 설명만 나열된다.
      - 근데 이게 유기적으로 연결되어 있지 않고, 갱스터의 특성이나 아예 다른 특성을 단순 나열하고 있는 것 같이 보인다...
        - 머가 문제일깡
  - 낯선 상황에서 캐릭터가 자신의 특성과 일치하는 행동을 수행할 수 있도록 학습할 수 있다는 점이 가장 신기했다.
    - 논문 속 예시는 경찰이라는 직업을 가진 인물이 총을 쏘려는 사람을 만나면 `체포`라는 행동을 수행할 것을 예측할 수 있다고 나와있다.
    - 수식이 결과로 나타나는 순간이 아닐까 싶다 ㅎㅎ
  - 꽤 재밌게 읽은 것 같다.
