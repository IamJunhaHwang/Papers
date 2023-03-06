#### 논문 - Generating Expository Dialogue from Monologue: Motivation, Corpus and Preliminary Rules
- 저자: Paul Piwek, Svetlana Stoyanchev
- NAACL 2010
- [논문 링크](https://aclanthology.org/N10-1048.pdf)
------------------------------------------------------------------
- **목적**
  - 본 논문은 expository dialogue을 생성하는 NLG task를 소개
    - expository dialogue는 두 허구의 등장인물 사이의 대화 내용임
    - 텍스트, 오디오 또는 필름으로 표시할 수 있음
    - expository dialogue에서 등장인물들의 기여는 서로 맞물려야 함
  - 최근에 많은 경험적 연구들은 특정 목적에서 expository dialogue가 독백보다 유리하다는 것을 보여줌
    - 설득에도 더 효과적임
  - 아래와 같은 대화를 생성하기 위한 시스템을 설명함( (1) -> (2)의 대화로 )
    ![image](https://user-images.githubusercontent.com/49019292/223058159-3192745d-eb4f-4382-9985-3c6f2e734ddc.png)   
      ![image](https://user-images.githubusercontent.com/49019292/223058184-3c11bd32-27c6-4869-a1aa-b7f39b29aac2.png)   
    ![image](https://user-images.githubusercontent.com/49019292/223058203-6729bb3c-c302-4c6d-99d6-c334590205cd.png)
  - 전문적으로 작성된 대화에서 규칙을 자동으로 배우는 것이 최선의 전략이라고 주장함
  - 최종적으로 Monologue-to-Dialogue (M2D) 생성 규칙의 반자동 구성 작업을 제안
-------------------------------------------------
- **방법**
  - The CODA Corpus
    - Monologue에서 Dialogue 텍스트로의 매핑을 학습하기 위한 parallel corpus가 필요함
    - CODA 프로젝트에서 동일한 정보를 표현하는 monologue(수동 작성)과 일치하는 전문적으로 작성된 대화로 구성됨
    - 궁극적인 목표는 호평을 받은 작가들이 쓴 것과 유사한 대화를 만드는 것
      - 전문적으로 작성된 대화로 시작하여 그에 상응하는 monologue을 만들어냈음
      - 전문적인 작가를 고용하여 monologue를 바탕으로 대화를 만드는 것 보다 효평을 받은 작가들의 실제 대화를 활용하는 것이 더 실현 가능성이 높았음
    - dialogue와 monologue에 모두 주석을 달았음
      - 대화 행위가 있는 dialogue와 담론 관계가 있는 monologue
    - 표 1은 대화 단편, 정렬된 독백 및 대화 행동 주석의 예시   
      ![image](https://user-images.githubusercontent.com/49019292/223058224-f7ebf53c-9b8e-4fec-8371-40123c327687.png)   
    - 그림 1은 독백의 담론 구조를 도시화한 것      
      ![image](https://user-images.githubusercontent.com/49019292/223058247-42a9eb14-24ae-49b9-8882-ab7feb2ba889.png)   
    - 표 2는 전문가와 비전문가 사이의 대화 행위의 분포를 보여줌   
      ![image](https://user-images.githubusercontent.com/49019292/223058270-814b7975-bdcf-41cb-94a6-d0ecb35b9c11.png)   
      - 두 대화의 가장 빈번한 대화 행동은 설명이고 여기서 등장인물은 정보를 제시함
    - 스타일의 차이는 M2D 매핑 규칙이 작성자 또는 스타일에 따라 다르다는 것을 나타냄
      - 두 명의 다른 저자(Twain과 Gurevich)로부터 얻은 M2D 규칙을 동일한 텍스트(예시 - 아스피린 예제)에 적용함으로써 두 개의 다른 대화를 생성할 수 있었음
      - 이를 통해 자동으로 생성된 대화의 프레젠테이션 스타일을 다양화 할 수 있음
----------------------------------------
- **실험 및 결과**
  - Rules
    - 본 논문에서는 자동으로 monologue와 dialogue의 병렬 말뭉치에서 정렬된 담화 관계와 대화 행동에서 M2D 규칙을 도출함
    - 표 3은 표 1의 병렬 dialogue-monologue 단편에서 생성된 세 가지 규칙을 보여줌   
      ![image](https://user-images.githubusercontent.com/49019292/223058290-ec260ea5-9bb6-424b-a835-c07f2082c0b5.png)   
      - 첫 번째 규칙인 R1은 독백(i-iv)의 완전한 담화 구조를 기반으로 하는 반면 R2와 R3는 일부만 기반으로 함
      - 대화를 생성하기 위해 일치하는 M2D 규칙을 적용함
      - 그럼 2와 같이 관계 노드를 제거하여 monologue의 담화 구조를 단순화 할 수 있음
    - 그림 2 (2)의 단순화된 구조는 표 3의 규칙 R2와 일치함 
    - R2를 적용함으로써 표 4의 대화를 생성함   
      ![image](https://user-images.githubusercontent.com/49019292/223058303-0b26487a-22eb-4e19-b777-183b0b5d1cb2.png)  
      - 전문가는 절 a와 b로 구성된 복잡한 질문을 하고 일반인은 동일한 절 집합에서 생성된 설명으로 대답함
      - 그런 다음 전문가는 c와 d에서 생성된 상반된 설명을 제공함
----------------------------------------
- **고찰**
  - 논문이 짧아서 그런지 기술 구현 부분은 많진 않았지만, 아이디어는 충분히 엿볼 수 있었다.
  - 처음에 독백을 통해서 대화를 생성한다는 것을 보고 독백에서 어떤 내용을 뽑아서 대화로 만드는 걸까 궁금해서 살펴보았다. 
  - 독백에는 보통 인물의 감정이나 자신이 생각하는 내용이 나올 것이고 이를 활용하나? 싶었다.
    - 논문에서는 단순히 독백에 드러난 지식 정보 혹은 무언가를 설명하는 정보를 가지고 그럴듯한 대화를 생성하는 것이었다.
    - 감정이나 생각을 사용하지 않는 것이 약간 아쉬웠지만 나름 이 시대엔 획기적인 논문이었겠구나 싶었다. 
  - 여기서 사용한 데이터셋은 직접 구축한 것 같다.
    - 검색해도 별로 안나오고... 논문 읽어보면 독백 부분은 수동을 작성했다는 것을 보면 그런 듯하다.
    - 확실히 딱 맞는 데이터셋을 찾기 힘들었을 듯 싶다.   
    - 대화 규칙을 자동으로 찾으려면 질 좋은 데이터셋(task에 적합한)이 필요하긴 할 듯..
