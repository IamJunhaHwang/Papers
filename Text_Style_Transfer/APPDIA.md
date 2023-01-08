## :page_with_curl: APPDIA: A Discourse-aware Transformer-based Style Transfer Model for Offensive Social Media Conversations

### Info

* url: https://aclanthology.org/2022.coling-1.530/
* conf: coling 2022
* github: https://github.com/sabithsn/appdia-discourse-style-transfer

### 1. Abstract + Introduction

SNS의 익명성으로 인해, 공격적인 댓글들이 증가했고 이를 막기 위한 AI-system이 요구 되었음.

이전에는 단순히 공격적 표현이 있는 댓글을 분류해 삭제했지만 이는 댓글의 다양성을 줄이게 되고 토의에 대한 내용 또한 지워버리는 문제가 있었음.    
ex) 지금 뚫린게 입이라고 그딴 식으로 지껄이는 거야? OOO때문에 OO해서 OO인건 당연하잖아  ==> 전체 삭제됨.

따라서, 텍스트의 내용은 살리되 공격적 표현만 바꾸는 `supervised style-transfer task`로 접근함.

`discourse coherence framework`를 transformer 기반 style-transfer 모델에 접목시키는 것이 문장 내용을 보존하는 것에 도움이 된다고 가설을 세움.

- Contribution

  - 2K의 데이터셋을 수집 및 공개함.

    - 데이터셋은 레딧의 공격적인 댓글과 이를 사회언어학자가 바꾼 순화된 댓글로 이루어져 있음.
    - `discource relation`또한 태깅함. [이건 자동으로 태깅된 것]

  - 사전학습된 transformer 모델에 discource relation framework를 접목시는 두 가지 방법을 제안

    - Penn Discourse Treebank를 단일 댓글에 이용
    - Rhetorical Structure Theory Discourse Treebank를 이용해 댓글과 그 응답을 분석

<br></br>

### 2. Data Collection and Annotation

#### 2-1. Data Collection Pipeline

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/211184550-e138f22b-246d-4db3-9eac-0c7b8fbe6177.png" width="80%"></img></div>

14개의 subreddit(게시판)에서 다양한 주제의 댓글을 수집([PRAW](https://praw.readthedocs.io/en/stable/)이용)

수집된 댓글은 OLID dataset으로 fine-tune된 공격성 분류를 위한 BERT 모델로 태깅됨. (해당 모델이 공격성이 없다 판단한 것은 버림)

그 후, 주기적으로 해당 댓글에 레딧 상에서 접근가능한지 주기적으로 확인한다. 해당 댓글이 삭제되었을 경우, 부모 댓글 혹은 게시글을 확인하고 이 또한 없다면 댓글을 데이터셋에서 지운다.  [게시글에 직접 달린 댓글이면 게시글이 부모에 해당되며 대댓글일 경우 상위 댓글(대댓글의 대상)이 부모가 된다]

부모 댓글 혹은 게시글이 확인된다면 이를 포함해 데이터셋에 넣는다. (부모-댓글(레딧 상에서 지워짐))

엄청 긴 댓글은 필터링하고 총 5.8K 댓글을 수집했다.

#### 2-2. Data Annotation

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/211184555-1c34ec23-a6cc-4d2e-a399-450119343bea.png" width="80%"></img></div>


3명의 사회언어학자 전문가에 의해 annotation.

`SemEval 2019 shared task (Zampieri et al., 2019b)`와 비슷하게 공격성을 정의했음. [모욕, 욕설, 혐오, 폭력성]

단어 몇 개만 수정하면 공격성이 완화 되는 것을 `Local change`, 문장 전체를 다시 써야되는 정도인 경우를 `Global change` 라고 보았음.

- annotation key points

  - 댓글이 이미 공격성이 없거나 댓글 내용을 바꾸지 않고는 공격성을 완화시킬 수 없다면 버림. (BERT에 의해 공격성 있는 것만 우선 가져왔으므로 False Positive가 있을 수 있지만 사람 어노테이터가 이를 버림)

  - `local change`로 공격성을 완화시킬 수 없을 시 `global change`를 함.

내용을 보존하는 어노테이션을 평가하기 위해 `BLEU`를 사용. (annotated text and the original간의)

`BLEU`로 중복성을 평가하므로 내용이 잘 보존되었는지 알 수 있음. **60.6 BLEU-score를 얻음**

+) BERT가 공격성이 있다고 한 문장들의 68%를 사람 어노테이터가 버렸음. 공격적으로 민감한 단어들이 포함되었을 경우 문장을 공격적이라 하기 때문인 듯 함. 이는 공격성 분류기의 한계로 볼 수 있음. ex) ""a rape victim should not be the one to blame"

### 3. Discourse-Aware Models

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/211184556-74e94f4b-e875-4278-8dd1-a49e0902663d.png" width="80%"></img></div>

PDTB and RST-DT discourse frameworks를 트랜스포머 모델에 접목시키는 접근법 제안

- PDTB Within-Comment Relations

  - PDTB relations을 얻기 위해 댓글을 먼저 파싱한다. 
    - `Lin et al. 2014b` 사용해 외재적 발화 관계 추출

    - `Kim et al.2020`의 XLNet-large를 이용해 인접 문장 쌍으로부터 내재적 발화 관계 추출

  - `PDTB-tagged Reddit corpus`가 없기 때문에 `PDTB-2 corpus`를 이용해 위 모델들을 훈련함.
    - 각각 80.61(F1-score), 57.74(accuracy)가 나옴. [Kim et al은 F1으로 평가하지 않았었음]

  - `argument pairs`의 위치와 발화 관계를 이용할 것임

- RST-DT Context-Based Relations

  - 댓글과 그 부모의 RST-DT 관계를 파악하기 위해 댓글과 부모를 연결한 후 아래 모델에 넣음.

    - `Li et al.(2018a) EDU segmenter`를 돌리고 `Wang et al. (2017)`모델에 넣어 결과를 얻음

    - `RST-DT and GUM corpus Zeldes (2017)`를 합친 것으로 훈련하고 테스트 함.

  - `RST tree`의 루트를 관계로 style-transfer 모델에 사용했음. 

- Integration with transformer model

  - RST-DT의 각 관계와 PDTB의 관계 쌍(시작, 끝)을 `special token`으로 추가함. (당연히 임베딩층을 resize함)

<br></br>

### 4. Experiments

Pretrained Transformer Models와 Discourse-aware Transformer Models를 실험함

- Pretrained Transformer Models

  - BART-base: formal data로 훈련됨.
  - T5-base: formal data로 훈련됨.
  - DialoGPT-medium: Reddit data로 훈련됨.

- Discourse-aware Transformer Models

  - DiaoloGPT가 Reddit으로 훈련되었고 본 논문에서 Discourse-aware모델을 위해 Reddit을 사용했으므로 DialoGPT에 discourse-aware approaches를 적용하기로 함.

  - i. PDTB relation을 적용하기 위해 아래와 같이 시도함.
 
    - Level 1 & Level 2 explicit PDTB relations
    - Level 1 & Level 2 implicit PDTB relations
    - combining level 2 explicit & implicit relations

  - ii. RST-DT를 적용하기 위해 top-level RST-DT classes를 이용함.
    - 데이터셋에서 하위 수준 클래스를 자주 접할 가능성이 낮기 때문에 범위를 최상위 RST-DT 클래스로만으로 제한

  - iii. 위 2방법을 합치는 것도 시도함.

    - 댓글 자체와 부모간의 `root-level RST-DT relation`, `PDTB relations(both implicit and explicit)`를 텍스트에 삽입
    
    - PDTB implicit and RST parser가 낮은 accuracy를 가지므로 threshold $α$를 도입
    
    - 주어진 관계에 대한 confidence score가 $α$보다 낮으면 해당 관계를 버림

      - $α = 0$: 모두 취함.
      - $α = µ − σ$: 평균 - 분산 [classifier score의]
      - $α = Q1$: classifier scores의 4분위를 계산한 것의 첫 번째 값

### 5. Results

- 데이터를 8:1:1로 나눔.

- 자동 평가로 BLEU, BERT-score, SafeScore를 사용.
  - SafeScore: 댓글의 공격성을 분류하는 BERT 모델로 예측했을 때, style-transfer한 댓글 중 성공적으로 공격성 완화된 댓글의 퍼센트

- Human Evaluation도 진행

#### 5-1. Automatic Evaluation

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/211184575-37457579-a089-4c77-9079-b564f1df977f.png" width="40%"></img></div>

- manual annotation and style transferred text 간의 내용이 얼마나 보존되었나 평가
- original comment and style-transferred text 간의 내용이 얼마나 보존되었나 평가

- BART나 T5가 BLEU, BERT-score가 높더라도 본 논문의 주 목적은 텍스트의 공격성을 완화하는 것임. 이런 측면에서 SafeScore를 보면 DialoGPT가 다른 모델에 비해 좋은 성능을 보임.

- **DialoGPT를 아래 모든 실험들의 Baseline으로 씀**

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/211184579-3933809c-6beb-4157-b019-69c89de183f0.png" width="70%"></img></div>

- discourse aware model이 baseline보다 좋은 성능을 보임.
  - RST-DT relation이 BLEU, BERTScore를 높이는데 도움을 주었음. (이는 생성 모델이 댓글의 맥락을 아는 것이 중요하다는 것을 시사)
  
  - PDTB Implicit 이 그 자체로 낮은 accuracy를 보였지만 성능을 높여주었음. (이는 단순히 데이터셋에 많이 등장했기 때문)

  - RST-DT와 PDRB를 합친 것이 가장 높은 성능을 보였음.

  - $α$로 낮은 confidence score를 버리는 것은 악영향을 미쳤음. (낮더라도 여전히 모델이 추론하는데 도움이 됨을 시사)


#### 5-2. Human Evaluation

- 100개의 랜덤 예시에 대해 basemodel과 best discourse-aware 모델이 만든 문장에 대해 세가지 측면에서 평가

  - i. 어떤 문장이 내용을 잘 보존했는지
  - ii. 어떤 문장이 제일 그럴듯한지(어휘)
  - iii. 전반적으로 어떤 문장이 선호되는지

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/211184583-4e63ff10-fc1c-4369-a0ab-e43719938ab0.png" width="40%"></img></div>
