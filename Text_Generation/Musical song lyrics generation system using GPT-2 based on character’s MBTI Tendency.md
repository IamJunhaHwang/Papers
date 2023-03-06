#### 논문 - Musical song lyrics generation system using GPT-2 based on character’s MBTI Tendency

- 저자: 전이슬, 박준석, 문미경
- 한국차세대컴퓨팅학회 논문지 2022
- [논문 링크](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002858320)
-----------------
- **목적**
  - 기존의 가사를 생성하는 연구들은 압운(Rhyme)을 고려하여 얼마나 가사다운 문장을 생성하는지, 주어진 음표에 갯수에 맞게 가사를 생성할 수 있는지에 대해 집중하였음
  - 이에 본 논문에서는 뮤지컬 극 중 캐릭터의 MBTI (Myers-Briggs Type Indicator, 마이어스-브릭스 유형지표) 성향에 맞는 GPT-2 기반 뮤지컬 가사 생성 시스템을 제안
    - 캐릭터 성향 분석을 위해  (내향,외향)/(감각,직관)/(사고,느낌)/(판단,지각)의 네 가지 범주를 지정하여 16가지 성격 유형으로 분류하는 MBTI를 활용
      -  MBTI란 스위스 정신과 의사 Carl G. Jung의 심리 유형 이론을 바탕으로 Isabel Briffs Myers와 Katharine Briggs에 의해 개발된 성격 유형 평가 도구
    - MBTI 성향과 유사한 키워드 리스트를 활용하여 노래 플레이리스트를 찾아 학습 데이터를 자동으로 생성 
  - KoGPT-2 모델을 학습시키고 극 중 캐릭터 성향과 분위기에 맞는 뮤지컬 가사를 생성
----------------------------------------------------------- 
- **방법**
  - 본 논문은 아래 그림 1과 같이 크게 두 가지 단계로 구성된 방안을 제안   
    ![image](https://user-images.githubusercontent.com/49019292/223052687-fd1b6b89-78f2-4e71-a58f-c48467eef521.png)      
    - 첫번째, 캐릭터 성향과 유사한 가사 학습 데이터를 구축하는 단계(MBTI 활용)
    - 두번째, 학습 데이터를 전처리하여 KoGPT-2 모델의 학습 데이터로 사용하여 뮤지컬 가사 생성
  - 캐릭터의 성향에 어울리는 학습 데이터를 구축하는 것이 중요
    - 인물 소개 및 대본을 통해 표현되는 캐릭터를 분석하고 MBTI를 활용하여 캐릭터 성향 분류 
    - MBTI 성향은 등장인물에 대해 알고 싶은 사람들이 모여서 활동하는 커뮤니티(Personality Database) 사이트의 약 4000명의 사람들의 투표 데이터 기반으로 파악
  - MBTI 성향과 관련된 키워드는 아래 표 1로 분류   
    ![image](https://user-images.githubusercontent.com/49019292/223052720-9b95c1c8-6d8d-4ce3-b183-7ff1b8bdb3f4.png)      
    - 국내 음원 사이트 '벅스'와 '멜론'에서 MBTI별 플레이리스트의 태그를 기반으로 검색 키워드 추출
    - 추출된 키워드로 '멜론' 음원 사이트에서 플레이리스트를 검색하여 크롤링하여 가사 학습 데이터 구축
    - 한국어 가사 생성을 위해 가요 및 K-POP 키워드를 우선 순위로 두고 데이터 수집
    - 아래는 예시   
      ![image](https://user-images.githubusercontent.com/49019292/223052755-6ac6803a-54c0-4477-aa1d-92f55661722a.png)      
  - 본 논문에서는 KoGPT-2에서 진행한 사전학습을 기반으로 파인튜닝 진행
    - 아래 표 3을 참조하여 코드 수행 시 Top-K, Top-P 방식을 적용하여 파인튜닝 진행   
      ![image](https://user-images.githubusercontent.com/49019292/223052772-b442ee29-a872-46bb-99fb-c3f490706737.png)      
-----------------------------
- *실험 및 결과**
  - 실험 환경
    - 데이터는 뮤지컬 등장인물 캐릭터들의 MBTI에 맞춰 수집한 플레이리스트 8개 및 총 309곡의 가사 데이터를 수집하여 진행
    - tensorflow 환경에서 실험 진행
    - NVIDIA GeForce RTX 3090 GPU 1개가 장착된 PC 3대에서 실험 진행   
  - 결과   
    - 가상 인물(뮤지컬 배우)의 가사 생성 예시   
      ![image](https://user-images.githubusercontent.com/49019292/223052804-1261febe-f6da-4af9-b53d-d28568b4a5f9.png)      
      ![image](https://user-images.githubusercontent.com/49019292/223052890-c9a34c15-0e28-4f4a-a913-09c5837aa7da.png)         
  - 사전학습된 KoGPT-2 모델을 총 3098곡의 가사 데이터 학습을 진행
  - 가사 생성을 위한 최적의 하이퍼 파라미터
    - top_k, top_p, temperature 하이퍼 파라미터 값을 각각 휴리스틱하게 조절하여 가사 생성에 어떠한 영향을 미치는지 확인  
    - 아래 표 5는 '사랑'이라는 키워드를 주었을 때 top-k값만 변화하여 생성된 결과   
      ![image](https://user-images.githubusercontent.com/49019292/223052922-be31c2d8-17d1-4917-af92-0abde86fa9f3.png)      
      - 100개의 토큰으로 높였을 때 반복되는 문장 생성 횟수가 줄고 다양한 문장이 포함된 가사를 생성
    - Temperature scaling은 토큰 확률 분포를 변형하여 문장을 다양하게 생성하는 기법   
      ![image](https://user-images.githubusercontent.com/49019292/223052957-8d02fb0c-19d6-4d28-b4ad-ceef5913337f.png)      
    - 아래 표 7은 누적확률값을 이용하여 문장을 생성하는 top-p 인자값만 조절하여 생성된 결과   
      ![image](https://user-images.githubusercontent.com/49019292/223052979-94a06990-5de6-4eac-a48d-8c21fcf851c6.png)      
    - 본 논문에서는 허깅페이스(Hugging Face) 에서 
작성한 “텍스트 생성 방법: Transformers로 언어 생성에 다양한 디코딩 방법 사용[19]”연구에서 제시한 하이퍼파라미터 값을 기준으로 하였음
      - 아래 표 8은 최적의 하이퍼파라미터 값   
      ![image](https://user-images.githubusercontent.com/49019292/223053030-36b93744-5628-45dc-b24b-ffaf8bca9d34.png)   
    - 아래 표 9는 캐릭터 MBTI 기반 뮤지컬 가사 생성 결과   
      ![image](https://user-images.githubusercontent.com/49019292/223053057-2e1679cd-0b9a-4577-81ee-07f71aa0ec17.png)   
  - 평가
    - 표 9의 가사와 연관된 MBTI 타입을 설명하는 자료를 제시하고 각각의 MBTI에 어울리는 가사를 선택하는 평가를 진행
    - 생성된 가사는 표절 검사를 위해 카피킬러 활용
      - 스노는 0%, 버니는 12%의 표절률 결과를 얻음   
    - 평가 최종 결과   
      ![image](https://user-images.githubusercontent.com/49019292/223053084-ec1c8376-829f-4a52-bbd0-b5ef069340fe.png)   
-----------------------------------------
- **고찰**
  - 제목이랑 초록만 보고 MBTI를 어떻게 모델에 접목하는지, 방법이 궁금해서 읽었는데.......
    - MBTI 성격표를 모델에 바로 적용하는 것은 아니고, 그냥 MBTI별 태그를 찾아서 태그에 맞는 플레이리스트의 음악 가사를 활용하여 새로운 가사를 생성하는....
    - 이런 내용이었다니.... 속았다.. :cry: :cry:
  - 그래도 결과를 보면 태그에 맞게 가사 자체가 잘 생성되긴 한 것 같다.
    - 다만 뮤지컬보단 k-pop 가사 느낌이 많이 나긴 한다.
    - 근데 왜 하필 뮤지컬 가사를 생성하는 쪽으로 잡았을까? -> 아마 아직 연구된 적이 없어서 그런 듯
  - 성격별 태그를 국내 음원 사이트의 플레이리스트에 달린 태그를 사용했는데, 이게 공신력이 있는걸까?
    - 국내에서 가장 점유율이 높은 음원 사이트를 사용했다고는 하지만.. 태그 자체가 학습해서 자동으로 달린게 아닐텐데, 라는 생각을 해봤다.
  - 모델 관련 논문이라기보다는 주제 그 자체나, 데이터셋을 모으는 방법에 대한 내용이 더 초점인 것 같다.
  - GPT-2를 어떻게 활용할지도 궁금해서 끝까지 읽어보긴 했지만, 주목할만한 기여는 없는 것 같다.
    - 이런 논문도 있는거지 머...ㅎㅎㅎ 
  - 그래도 나름 재밌게 읽은..
