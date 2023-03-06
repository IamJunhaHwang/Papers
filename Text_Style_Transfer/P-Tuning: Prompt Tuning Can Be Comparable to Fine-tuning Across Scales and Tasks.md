#### 논문 - P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks

- 저자: Xiao Liu, Kaixuan Ji1, Yicheng Fu1, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, Jie Tang
- ACL 2022
- [논문 링크](https://aclanthology.org/2022.acl-short.8.pdf)
------------------------------------------------------------------
- **목적**
  - finetuning은 target task에 대한 전체 모델 매개 변수를 업데이트하므로 메모리가 많이 소모됨 
  - prompting은 사전 훈련된 모델의 모든 매개변수를 동결하고 자연어 프롬프트를 사용하여 언어 모델을 쿼리함
  - Prompt tuning은 연속 프롬프트만 튜닝하는 아이디어
  - contibution은 properly optimized prompt tuning이 다양한 모델과 NLU 작업에 걸쳐 fine-tuning과 비교할 수 있다는 것의 발견임
    - 우리의 발견은 NLU에 대한 prompt 조정의 보편성과 잠재력을 보여줌 
  - P-Tuning v2는 미세 조정에 비해 작업당 0.1~3%의 훈련 가능 매개 변수를 가지고 있어 훈련 시간 메모리 비용과 작업당 저장 비용을 크게 절감      
    ![image](https://user-images.githubusercontent.com/49019292/223054300-72bbd90f-f1f6-4321-8f66-ecadf8804d74.png)      
-------------------------------------------------------------------
- **방법**
  - P-Tuning v2 
    - Deep Prompt Tuning
      - (Lester et al., 2021) 및 (Liu et al., 2021)에서 연속 프롬프트는 입력 임베딩 시퀀스에만 삽입됨 
      ![image](https://user-images.githubusercontent.com/49019292/223054344-72c140f0-39e2-4ad5-85f3-bf4cacbd4e0c.png)      
      - 위의 한계점을 해결하기 위해 P-Tuning v2는 딥 프롬프트 튜닝 아이디어를 사용함
      -  아래 그림 2와 같이 서로 다른 계층의 프롬프트는 접두사 토큰으로 추가됨   
        ![image](https://user-images.githubusercontent.com/49019292/223054360-c2c7d107-7ba8-4a34-b84f-ac3582ab25d1.png)      
    - Optimization and Implementation
      - 최상의 성능을 달성하기 위한 최적화 및 구현
      - Prompt Length
        - 프롬프트 길이는 P-Tuning v2에서 중요한 역할
        - 단순한 분류 작업은 프롬프트가 더 짧은 것(20개 미만)을 선호하며, 하드 시퀀스 레이블링 작업은 더 긴 것(약 100개)을 선호
      - Multi-task Learning
        - 멀티태스킹 학습은 개별 작업을 미세 조정하기 전에 공유된 연속 프롬프트로 여러 작업을 공동으로 최적화함
        - 멀티태스킹은 P-Tuning v2의 경우 선택 사항이지만, 더 나은 초기화를 제공함으로써 추가적인 성능 향상을 위해 사용될 수 있음
      - Classification Head
        - language modeling head를 사용하는 것은 전체 데이터 설정에서 불필요하고 시퀀스 레이블링과 호환되지 않음
        - 대신 BERT에서와 같이 토큰 위에 randomly-initialized classification head 사용
      - 기존 프롬프트 튜닝 방식과 P-Tuning v2 비교   
        ![image](https://user-images.githubusercontent.com/49019292/223054381-7e4fb8c1-71bf-45ef-89ba-00c85208944d.png)      
------------------------------------------------------
- **실험 및 결과**
  - Experiments
    - P-튜닝 v2의 효과를 검증하기 위해 일반적으로 사용되는 다양한 사전 훈련 모델과 NLU 작업에 대해 광범위한 실험을 수행
      - 또한, 모든 실험은 few-shot setting이 아닌 fully-supervised setting에서 수행됨
    - NLU Tasks
      - SuperGLUE(Wang 외, 2019)의 데이터 세트에서 실험  
    - Pre-trained Models
      - 평가를 위해 BERT-large, RoBERTa-large, DeBERTa-xlarge, GLMxlarge/xxlarge를 포함
        - NLU 작업을 위해 설계된 양방향 모델로 300M ~ 10B까지 크기가 다양함 
    - Multitask Learning
      - multi-task setting은 각  task 유형에서 데이터 세트의 훈련 세트를 결합함
      - 연속 프롬프트를 공유하면서 각 데이터 세트에 대해 별도의 linear classifier를 사용함
  - Experiment results
    - P-tuning v2: Across Scales  
      - 아래 표 2는 모델 규모에 걸쳐 P-Tuning v2의 성능을 보여줌   
        ![image](https://user-images.githubusercontent.com/49019292/223054413-a4e43901-7982-4dfb-8ed3-236d9c4fb1bf.png)      
    - P-tuning v2: Across Tasks
      - 아래 표 3에서, 우리는 P-Tuning v2가 일반적으로 모든 작업에서 미세 조정과 유사할 수 있음을 관찰하였음   
        ![image](https://user-images.githubusercontent.com/49019292/223054445-18509fed-5ba1-45db-9334-b5f415156cad.png)      
  - Ablation Study
    - Verbalizer with LM head v.s. [CLS] label with linear head
      - 아래 표 4는 다른 하이퍼 파라미터를 유지하고 linear head를 가진 [CLS] 레이블만 LM head를 가진 언어자로 변경한 것을 비교하여 제시   
      ![image](https://user-images.githubusercontent.com/49019292/223054468-e05c19de-e2b8-40e7-8f16-8ea23d317cc5.png)      
    - Prompt depth
      -  ester et al. (2021); (Liu et al., 2021) and P-tuning v2의 주요 차이점은 multi-layer 연속 프롬프트임
      - 정확한 영향을 확인하기 위해 프롬프트를 추가할 특정 k계층이 주어지면 오름차순과 내림차순으로 선택하여 프롬프트를 추가하고 나머지 계층에 대해서는 변경되지 않은 상태로 둠
      - 아래 그림 3에서 보듯이 동일한 양의 매개변수를 사용하면 항상 오름차순보다 내림차순으로 추가하는 것이 더 낫다는 것을 알 수 있음   
        ![image](https://user-images.githubusercontent.com/49019292/223054492-c62c084c-1c4b-4cf9-baa7-8f977136d89d.png)      
----------------------------------------------
- **고찰**
  - ACL 2022 accept paper list을 찾아보다가 제목에 꽂혀서 읽어보았다. 
  - 탑티어 논문은 매우 깔끔하구나를 다시 한번 느꼈다. 
    - 군더더기 없이 구성도 깔끔하고 문체며, 서술 방식도 너무 좋다..ㅎㅎ 해석이 잘 돼..!(그나마)
    - 이 정도는 되어야 ACL에 accept 되나..?   
  - P-Tuning이 fine-tuning을 대체할 수 있다는 점을 시사하는 논문이라고 할 수 있겠다...
  - 생각보다 기술적인 파트는 적다고 느껴졌음...!
  - 논문에서 보면 기존 prompt tuning 방식보다 구조도 단순해진 것 같은데 성능은 비슷하게 가져갈 수 있다니...
  - 매개변수를 전부 업데이트 하지 않고 학습 시키는 방법은 전에도 있었던 것 같은데, 이를 결합한거구나..
    - 전에 논문에서 봤을 때도 메모리 문제 때문에 전체 업데이트를 진행하지 않았던 것 같음
  - 이 정도 실험하려면 준비 기간이 얼마나 될까..
  - 이런 논문에서 사용하는 메소드도 전부 최신에 나온 것만 사용하는 듯
    - 트렌드라는게 많이 중요한 것 같다.
  - 확실히 구조 파트는 좀 어렵고 오래걸렸다 ㅠㅜㅜ 
    - 그래도 나름(? 재밌게 읽은 것 같음!
