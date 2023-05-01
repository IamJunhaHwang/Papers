## :page_facing_up: Efficient Pre-training of Masked Language Model via Concept-based Curriculum Masking

### Info

* `conf`: EMNLP 2022
* `author`: Mingyu Lee, Jun-Hyung Park, Junho Kim, Kang-Min Kim, SangKeun Lee

* `url`: https://aclanthology.org/2022.emnlp-main.502.pdf

- `github`: https://github.com/KoreaMGLEE/Concept-based-curriculum-masking


### 1. Abstract & Introduction

MLM pretraining은 효과적이지만 비용이 크다. 따라서, 효과적인 pretraining을 위한 방법으로 개념 기반 커리큘럼 마스킹( Concept-based Curriculum Masking; CCM)을 제안한다.

- CCM과 기존 Curriculum Learning는 2가지가 다르다. (MLM의 자연스러움을 반영하기 위해)

  - 몇몇의 다른 개념과 관련된 (단어와 구)를 쉬운 것으로 간주하고 초기 개념으로써 처음으로 마스킹되게 한다.

  - 각 토큰의 MLM 난이도(어려움)을 평가하는 언어적 난이도 기준을 도입한다.

    - model-based, linguistic-inspired 방법이 있는데 model-based는 cost가 많이 들기 때문에 후자를 선택함.

  - 지식 그래프(knowledge graph)의 검색을 통해 이전에 마스킹된 단어와 관련된 단어를 점진적으로 마스킹하는 커리큘럼을 만들었다.

  - 단어나 구 사이의 의미적, 문법적 관계를 이용하는 지식 그래프를 사용 [ConceptNet]

  - ConceptNet의 각 개념별로 관련된 개념의 수를 세고 가장 많은 빈도의 개념의 집합을 만들어 커리큘럼의 첫 단계에서 마스킹되게 한다.
  - 그 후, 이전에 마스킹한 개념과 관련된 것들을 마스킹해나간다. (반복)

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/235385044-d001b726-ded3-4fb7-85c8-0a811130eb7f.png" width="50%"></img></div>

위와 같이 기존의 언어학기반(linguistics-inspired) MLM 난이도 평가 기준에는 MLM의 자연스러움을 효과적으로 반영하지 못했다. 또한, MLM과 연관된 난이도는 주어진 sequence 자체가 아닌 sequence 내에서 어떤 토큰이 마스킹 되었느냐에 영향을 받는다. (ex. "The man is <mask> University student")

MLM은 마스킹된 토큰과 관련된 다른 contextual token(맥락과 관련된 토큰)에 기반하여 마스킹을 예측하는 것으로 볼 수 있다. 그러므로 만약 단어가 많은 다른 (단어, 구)와 관련되어 있다면 MLM을 쉽게 만들 수 있게 하는 힌트가 될 것이다.

제안한 커리큘럼의 효과를 검증하기 위해 GLUE 벤치마크에서 실험을 하였으며 그 결과로 기존 BERT training cost의 반으로도 기존과 견줄만한 성능을 달성했으며 pre-training 성능을 향상시켰다.

- Contribution

  - pre-training을 위한 커리큘럼 러닝을 조사하고 증명함.

  - 뛰어난 커리큘럼 마스킹 프레임워크 제안

  - 제안한 프레임워크가 pre-training의 MLM의 효율을 높일 수 있음을 증명함.

<br></br>

### 2. Method - Concept-based Curriculum Masking

관련된 기본 개념들을 활용하는 것으로, 점진적인 학습 스타일은 사람이 추상적인 개념을 쉽게 배울 수 있게 만든다. 이를 모방하기 위해 우리는 지식 그래프를 사용해 초기 개념과 관련된 개념을 더해나가는 여러 단계의 커리큘럼을 만들었다.  [초기 개념 선택 -> 커리큘럼 생성 -> pre-training]

우리는 지식 그래프를 다음과 같이 정의할 것이다.

$G = (N, E)$ [N: 노드의 집합, E: 엣지의 집합]

노드와 상응하는 단어나 구를 개념(concept)로 정의한다. => $concept c \in N$   

또한, $S_i$를 _i_ 번째 단계에서 마스킹되어야하는 개념의 집합으로 정의한다. 마지막 커리큘럼은 여러 단계를 포함한다. => ${S_1, S_K}$

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/235385049-caa36ab0-5e94-4222-be01-7aa5983f7923.png" width="85%"></img></div>

#### 2-1. 초기 개념 선택 (Initial Concept Selection)

CL의 핵심은 쉬운 예시를 먼저 배운다는 것이다. 우리는 많은 다른 개념들과 연관되어있을수록 쉬운 개념이라고 가정하고 지식 그래프에서 연결의 강도가 가장 높은 개념을 선택해 초기 개념의 집합으로 만든다. 

초기 개념을 선택하기 위해, 지식 그래프안의 연결 강도에 따라 각 개념의 순위를 매긴다. 그리고 pre-training 말뭉치에서 10만번 이하로 등장한 개념은 제외했다. (등장 빈도가 높은 개념이 더 영향력있기 때문; 10만번 이하의 개념은 대부분 의료, 과학 분야 용어였음)

마지막으로, 초기 개념의 숫자 $M$ 을 지정하면 상위 M개의 개념이 선택된다.

#### 2-2. 커리큘럼 생성 (Curriculum Construction)

_i_ 번째 단계에서 마스킹되어야 하는 개념의 집합은 다음 원칙에 따라 생성된다.

- $S_i$ 는 $S_{i-1}$ 를 포함해 $S_{i-1}$ 와 관련 있는 개념까지 계속 확장하여 만들어진다.

위는 새로운 관련된 개념을 모델링하기 위해 이전에 배운 개념을 단서로서 이용할 수 있게 만들어준다. 

우리는 관련된 개념을 지식 그래프의 관계를 사용해 찾는다. 직관적으로, 지식 그래프에서 두 개념이 가까이 연결되어 있을 수록 더 많이 연관되어 있다.

원칙에 기반해, 우리는 지식 그래프에서 `k-hops` 내의 이전에 배웠던 개념과 연결되어 있는 개념을 점진적으로 마스킹한다. 

=> $S_i = S_{i-1} \cup N_k(S_{i-1})$ , 여기서 $N_k(S_{i-1})$ 은 $S_{i-1}$ 의 개념으로부터 k-hops 내에 있는 개념들을 말한다.


#### 2-3. Pre-training

언어 모델 pre-training에 CCM을 도입하기 위해, pre-training 말뭉치의 토큰 시퀀스로부터 각 i번째 단계를 위 $S_i$ 를 포함하는 개념을 찾아 마스킹한다.

- Concept Search

  - 개념의 어휘(lexicon)을 컴파일한다. [pre-training 말뭉치에서 10번 이상 등장하고 5 단어 미만의 단어를 지식 그래프에서 추출한다]

  - 이후 `string match`를 통해 pre-training 말뭉치 안의 토큰 시퀀스에서 추출한 개념을 찾는다. [이 과정은 전처리 중 한 번만 실행하면 되므로 무시할 수 있는 cost임]

- Concept masking

  - 개념을 찾은 후, 커리큘럼에 따라 토큰 시퀀스를 마스킹한다.

  - 우리는 여기에서, 단일 개념으로 구성된 모든 토큰들을 동시에 마스킹하는 `Whole Concept Masking` 을 도입한다.

    - ex) `Standford` 개념을 마스킹한다면, `Stan` 과 `##ford`가 함께 마스킹됨.

    - 마스킹 비율은 BERT와 같다. (80:10:10 = mask: random: normal)

  - 식별된 개념의 수가 각 단계와 문장에 상당한 영향을 주므로, 정적 마스킹 확률은 너무 적거나 많은 마스킹을 야기한다. 그러므로 각 시퀀스의 전체 토큰의 대략 15%가 마스킹되도록 동적으로 마스킹 확률($p_d$)를 계산한다.

  - 주어진 입력에서 어떤 개념이 다른 개념을 포함하고 있어도, 모든 개념들은 독립적으로 취급된다.

    - `Stanford University` 와 `Stanford `가 하나의 시퀀스에 등장하더라도 각 개념은 독립적으로 마스킹됨.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/235385077-e890512f-85eb-49be-94ac-6f25082dda85.png" width="50%"></img></div>

<br></br>

### 3. Experiment

#### 3-1. Experimental Settings

BERT를 사용했 GLUE 벤치 마크에서 이들의 성능을 측정했다.

|BERT|Token Embed size|Hidden size|# of Layer | FFNN layer size | # of attention head|# of parameters|
|:------:|:---:|:---:|:--:|:--:|:--:|:--:|
|Small|128|256|12|1024|4|14M|
|Medium|128|384|12|1536|8|26M|
|Base|768|768|12|3072|12|110M|

- **Pre-training**

  - BERT 모델을 CCM과 MLM을 사용해 수동으로 pre-train하였다.

  - pre-trained 언어 모델의 성능이 말뭉치 크기에 크게 의존하기 때문에, 모든 모델은 HuggingFace Datasets(BooksCorpus & English Wikipedia 포함)으로 pre-train 하였다.

  - CCM에서 100k step동안 MLM을 사용해 `warmup`을 진행한 후, CCM의 각 stage별(단계별)로 100k step을 학습한다는 점이 주목할만하다.

    - 마지막 단계가 끝나면 MLM으로 다시 돌아가고 남은 step을 위한 커리큘럼을 반복한다.

  - 우리는 4단계(four-stage) 커리큘럼을 사용했고 이는 마지막 단계에서 모든 개념과 개념으로 구성되지 않은 모든 단어들을 마스킹할 수 있다. (4단계가 가장 성능이 좋았음)

  - 각 BERT 모델당 4개의 랜덤하게 초기화된 모델을 만들었으며 가장 낮은 validation MLM loss를 가진 모델을 사용했다.

|BERT|# of steps|Batch size|sequence length | maximum learning rate |optimizer|warmup steps|
|:------:|:---:|:---:|:--:|:--:|:--:|:--:|
|Small, Medium|1M|128|128|5e-4|Adam (0.9, 0.999, 0.01)|after 10K (Linear)|
|Base|1M|256|128|1e-4|Adam(0.9, 0.999, 0.01)|after 10K(Linear)|

- **Evaluation**

  - 학습한 모델을 GLUE 벤치마크로 평가하였으며 이 GLUE 벤치 마크에는 8개의 데이터 셋으로 이루어진다.

    - RTE, MNLI: textual entailment(문장 함의) task

    - QNLI: QA 함의 task
    - MRPC: paraphrase(문장 바꿔쓰기) task
    - QQP: Question paraphrase task
    - STS: textual similarity task [Spearman 상관관계]
    - SST: sentiment task
    - COLA: linguistic acceptability [Mathew 상관관계]

  - 위의 데이터 셋에 대해 따로 작성하지 않은 평가 메트릭은 Accuracy로 평가했다.

  - STS, RTE 데이터셋은 10 epochs, 나머지는 3 epochs 로 fine-tuning 했다.

  - 각 task에서 가장 좋은 learning rate를 선택했다. (5e-5, 4e-5, 3e-5, 2e-5; BERT 논문의 세팅을 따름)

  - 5번의 random restart를 했으며, 각각에서 같은 checkpoint를 사용했지만 데이터 셔플링과 분류기 초기화 방법을 다르게하였다.

<br></br>

### 4. GLUE Results

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/235385137-437da7c5-20ce-4883-a5cd-3df9aad191fc.png" width="50%"></img></div>

100k step마다 GLUE 벤치 마크의 성능을 실험하여 CCM이 모델의 수렴 속도를 가속시켰음을 증명하였다.

위의 그림에서 CCM이 모든 모델을 낮은 cost로 baseline의 성능을 달성할 수 있게 하였고 특히, base size 모델에서 MLM의 50% cost만 사용한 것이 성능이 더 좋았다. 또한, 모델이 클수록 CCM이 더 효율적인 것을 관찰하였다. (적은 크기의 모델은 개념단위 마스킹을 진행하기에 너무 어렵다고 추측, 그래도 충분한 iteration 후에는 MLM의 점수를 달설함)

`Table 2`는 각 GLUE task의 1M step 학습한 모델의 성능을 보여준다. RTE task에서의  Base 모델을 제외하고 모든 task에서 CCM을 적용한 모델의 성능이 더 좋았다. (이는 데이터셋이 pre-training 말뭉치와 비교했을 때 상당히 다른 개념을 가지고 있기 때문으로 볼 수 있다. 또한, 훈련 step을 좀 더 높여주면 baseline 성능을 능가하는 것을 관찰했다.)

<br></br>

### 5. Analysis

#### 5-1. Ablation Study

  - medium 모델로 GLUE를 이용해 Ablation Study를 진행했다.

  - CCM을 사용한 것이 가장 성능이 높았다. (CL, WCM을 적용하지 않으면 성능 안좋아짐)

#### 5-2. Effect of the Curriculum

우리의 커리큘럼 디자인을 증명하기 위해, 커리큘럼을 쓰지 않은 것과 다른 커리큘럼을 사용한 medium 모델로 비교하였다. (다른 커리큘럼들로 사용된 것들은 같은 커리큘럼 세팅을 따)

reverse 커리큘럼을 제외하고는 모든 커리큘럼이 커리큘럼을 사용하지 않은 것보다 좋은 성능을 보였다. CCM이 다른 접근법과 비교했을 때, 가장 좋았다.

#### 5-3. Difficulty of Curriculum Stages

우리의 커리큘럼이 쉬운 데이터 -> 어려운 데이터 순으로 모델 학습이 되도록하는지 조사하기 위해, `Bengio et al. (2009)`와 같이 커리큘럼의 각 단계에서 데이터의 난이도를 확인해보았다.

결과, 각 단계별로 MLM loss가 올라가는 것을 확인했으며 이는 각 단계별로 난이도가 올라간 것을 의미한다.

#### 5-4. Analysis of Initial Concept

- 초기 개념의 수: `1k`가 성능이 가장 낮았으며 `3k`가 가장 높았음. => 충분한 초기 개념이 있어야함을 의미

- 개념 선택 기준: `많이 연관된 순`, `많이 등장한 순`, `많이 등장하고 많이 연관된 순`을 비교했을 때, 우리가 사용한 `개념이 많이 등장하고 많이 연관된 순`이 가장 성능이 높았다.

  - 많이 연관된 순: `carbohydrate` 와 같은 많은 학술적 용어들이 초기 개념으로 포함되게 된다.

  - 많이 등장한 순: 불용어가 초기 개념에 포함되게 된다.
