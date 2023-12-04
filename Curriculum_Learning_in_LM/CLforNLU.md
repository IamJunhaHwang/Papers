## :page_facing_up: Curriculum Learning for Natural Language Understanding

### Info

* `conf`: ACL 2020, long paper
* `author`: Benfeng Xu, Licheng Zhang et al.
* `url`: https://aclanthology.org/2020.acl-main.542.pdf
* `github` : https://github.com/google-research/distilling-step-by-step

### 1\. Abstract & Introduction

Pre-Trained Language Model(PLM)이 두각을 드러낸 이후 PLM을 사용하는 pretrain-finetune 파라다임으로 task를 푸는 것이 자연스러운 수순이 되었다.   
fine-tune 단계에서, target task data는 각각의 example을 동등하게 취급했고 무작위로 모델에게 보여졌다. 하지만, NLU task에서는 example들의 난이도가 각각 크게 다를 수 있으며, LM은 easy-to-difficult 커리큘럼으로부터 이득을 볼 수 있다.

이러한 아이디어를 기반으로, 우리는 trainset을 Cross Review 방법을 사용해 각 example의 난이도를 구분하고 LM을 위해 커리큘럼을 만드는 Curriculum Learning (CL) approach를 제안한다.

Cross Review 방법이란, 해당 task에 적용할 모델이 잘 풀 수 있는 example을 easy example로 정의하는 것이다. 여기에서 difficulty score로 사용할 metric은 각 데이터에서 사용되는 metric을 사용한다.   
우리는 실험을 통해 제안한 CL이 fine-tuning stage에 도움이 되는 것을 보였고 많은 NLU task에 외부 데이터나 특정 모델 디자인 없이 성능 향상을 시킬 수 있음을 보였다.

+) 이전 CL 연구들은 주로 사전에 정의한 규칙들을 사용했는데, 예를 들면 (Tay et al., 2019) 는 문장의 길이를 난이도로 설정했다. 하지만 이렇게 사전에 정의한 규칙들은 task나 domain에 따라 달라지므로 일반화할 수 없다는 단점이 있다.

- **Contributions**

  - NLU task의 LM fine-tuning 에서의 CL의 효과를 증명했음

  - Difficulty Review method와 Curriculum Arrangement 알고리즘으로 구성된 새로운 CL framework를 제안함.

  - MRC, NLI task를 포함한 다양한 NLU task에서의 성능 향상을 관찰했음. (어려운 task일수록 성능 향상 높았음)

<br></br>

### 2. Preliminaries

본 논문의 제안한 방법을 실험할 모델로 BERT를 사용하였다. BERT는 output으로 각 position $i$ 에 대해 $H$ 차원의 벡터를 주는데, 다음과 같이 표기하도록 한다. $h_i^l \in \mathbb{R}^H$, l == stack of self-attention layers


다음은 각 task에 사용한 configuration과 metric을 나타낸 것이다.

- Mahine Reading Comprehension

  - MRC task는 passage **P** 가 주어지면 상응하는 질문 **Q** 에 대해 연속적인 영역 $\langle p_{start}, p_{end} \rangle$를 추출해 정답 **A** 를 추출하는 것이다. 

  - 다음과 같이 Q와 P를 concatenate하여 모델에 전달한다.  `<CLS>, Q, <SEP>, P, <SEP>`

  - i번째 토큰에 대해 이 토큰이 정답의 start 나 end일 확률은 다음과 같이 계산된다.

    - $W^T_{MRC} \in \mathbb{R}^{2 \times H}$

    <img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/3dd3bbb3-7b4e-4c46-9452-eb6131e5f222" width="30%"></img>

    <img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/b9e917ec-8e7a-4c75-94ee-d774d9e9ff83" width="30%"></img>

  - log-likelihood가 training objective로 사용되며, **F1** 을 golden metric으로 사용했다.

- Sequence Classification

  - final contextual embedding <CLS> 토큰 $h^l_0$ 을 전체 시퀀스의 pooled representation으로 사용해 어떤 label에 속하는지 분류하는 task

  - 이는 linear output layer $W_{SC} \in \mathbb{R^{K \times H}}$ 로 계산되며 다음과 같이 나타낼 수 있다.

    - $P(c | S) = softmax(h^l_0 W^T_{SC})$

    - K는 클래스의 갯수이다.

    - log-likelihood가 training objective로 사용되며, **Accuracy** 를 golden metric으로 사용했다.

- Pairwise Text Similarity

  - <CLS> 토큰이 입력 텍스트 쌍의 유사도를 나타내기 위해 사용되었다.

  - $Similarity(T_1, T_2) = h^l_0 W^T_{PTS}$

    - $W_{PTS} \in \mathbb{R}^H$

    - 다음과 같이 **MSE** 를 golden metric으로 사용했다. $MSE = (y - Similarity(T_1,T_2))^2$

<br></br>

### 3. Our CL Approach

$D$ 는 학습 데이터 셋이며 $\theta$ 는 언어 모델이다.

CL framework는 2가지 단계로 나뉜다. `Difficulty Evaluation, Curriculum Arrangement`   

#### 3-1. Difficulty Evaluation

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/b7f48390-eb07-4431-a044-99023a758cc8" width="40%"></img></div>

다양한 방법(문장 길이, 드문 언어의 사용, masking token의 갯수)으로 난이도 평가를 해왔지만, 우리는 모델 자체가 평가하게하여 모델의 시점에서 보는 난이도로 평가하여야 한다고 주장한다. 그리고 best metric은 target task의 golden metric이 되어야한다.

난이도 평가는 다음과 같은 순서로 진행된다.

1. 학습 데이터 셋 $D$ 를 같은 양을 가지는 $N$ 개의 **meda-datasets** 으로 나눈다. $\{ \widetilde{D}_i : i= 1,2, ..., N \}$

2. 그리고 각각의 meda-datasets에 대해 독립적인 모델들인 **teachers** 을 학습시킨다. $\{ \widetilde{\theta}_i : i = 1,2,..., N \}$
3. 이와 같은 작업은 다음으로 수식화할 수 있다. $\widetilde{\theta}_i = \underset{\widetilde{\theta}_i}{argmin} \underset{d_j \in \widetilde{D}_i}{\sum} L(d_j, \widetilde{D}_i)$    
  즉, N개의 meta-dataset에 대해 N개의 모델이 어떤 Loss Function에 대해 optimize되는 것
4. 어떤 meta-dataset을 $\widetilde{D}_k$ 라고 하면, 이 데이터 셋의 각 example $d_j$에 대 k번째 teacher를 제외한 모든 teacher의 inference를 진행한다.
5. 모든 inference를 마치면, target task의 metric으로 $d_j$의 점수를 계산한다.  $c_{ji} = M(\widetilde{\theta}_i(x_j), y_j)$
  이를 통해 N-1개의 teachers에서 N-1개의 점수를 얻게 된다. 위 식에서 M은 metric을 구하는 formula를 의미한다.
6. 마지막으로, 위에서 구한 N-1개의 score를 합치는 것으로 최종 점수가 정의된다.  $c_j = \underset{i \in (1,...,N), i \not = k}{\sum} c_{ji}$
7. 위와 같은 방법으로 난이도 점수 집합 $C$ 를 얻는다.

우리는 위와 같은 난이도 평가 방법을 **Cross Review** 라고 부르겠다.

#### 3-2. Curriculum Arrangement

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/da879469-484d-478c-af58-7c9634939f57" width="50%"></img></div>

이 섹션에서는 난이도 점수 $C$ 에 따른 학습 커리큘럼을 만드는 방법을 설명한다. 
우리는 커리큘럼을 multi-stage $\{ S_i : i = 1,2, ..., N \}$ 로 설정하였으며, 각 stage $S_i$ 의 example들은 다른 stage와 겹치지 않게 구성된다.

1. 먼저, 모든 example들을 난이도 점수 $C$ 에 대해 정렬하고 N개의 buckets로 나눈다: $\{ C_i : i = 1,2,..., N \}$
  이렇게되면, example들은 N개의 다른 레벨의 난이도(easiest $C_1$ to hardest $C_N$)로 분류되고 다음과 같이 분포를 가진다. $num(C_1) : num(C_2) : ... : num(C_N)$

2. discrete metric의 task는 위와 같은 분포 난이도 점수 계층(hierarchy)에 의해 자연스럽게 형성되며 데이터 셋의 고유한 난이도 분포를 직접 반영한다. 다른 task의 경우에는 수동으로 uniform하게 나눈다.

3. 각 stage $S_i$ 는 선행 버킷 $\{ C_j : j = 1, 2, ..., i \}$ 의 다음에 나타낸 부분 만큼 포함한다. $\frac{1}{N}num(C_1) : \frac{1}{N}num(C_2) : ... : \frac{1}{N}num(C_i)$ 

4. 각 stage마다 1 epoch 학습을 진행하며, $S_N$ stage에 도달하면 전체 training set $D$에 대해 학습하는 $S_{N+1}$ stage가 있다. $S_{N+1}$ stage 는 모델이 수렴할 때까지 진행한다.

우리는 위와 같은 arrangement algorithm을 **Annealing** 이라 부르겠다.

<br></br>

### 4. Experiments

#### 4-1. Datasets

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/554a661b-d0da-4e83-a0fe-83b7cda33ed8" width="70%"></img></div>

- SQuAD : Wikipedia article을 사용해 만들어진 machine reading comprehension(MRC) dataset

- NewsQA : CNN의 뉴스 기사로 수집된 더 어려운 난이도의 MRC dataset (사람의 경우 0.694 F1 점수가 나옴)

- GLUE : well-designed NLU benchmark datasets

#### 4-2. Experimental Setups

- Model : BERT-Large, BERT-Base, BERT-Base(re-implementation)

  - BERT + CL과 BERT의 다른 점은 training example의 순서뿐이다.
  
- 좀 더 안정적인 난이도 점수를 얻기 위해, binarize를 진행했다. --> Accuracy는 이미 binary 이고 F1의 경우 micro-F1을 적용했으며 MSE와 같은 연속적인 값을 가지는 metric은 그대로 두었다.

- meta-dataset의 숫자를 경험적으로 `N=10`으로 설정하였다. 좀 더 작은 데이터 사이즈를 가지는 RTE, MRPC, STS-B의 경우에는 `N = 3`으로 설정했다.

- 기타 하이퍼파라미터 세팅은 논문 참조

<br></br>

#### 4-3. MRC Results

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/315837cf-b1fa-4867-ad6f-63e7e451dfab" width="35%"></img></div>

- MRC tasks의 모든 실험에서 제안한 CL approach가 다른 baseline들을 능가했다.   

- SQuAD 2.0에서 re-implement BERT와 비교했을 때, base model과 large model의 performance gain은 각각 +1.30 EM/+1.15 F1, +0.31 EM/+0.57 F1 이다.

- NewsQA에서는, +0.02 EM/+0.47 F1, +0.10 EM/+0.30 F1 의 성능 향상을 보였다.

#### 4-4. GLUE Results

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/3631ff8a-7390-43d8-86e5-6f5b12d109f7" width="65%"></img></div>

- dev set에서의 결과를 보면, 제안한 CL method가 baseline을 능가하거나 비슷한 성능을 보였다.

  - 모델 구조와 하이퍼 파라미터 세팅이 모든 실험에서 동일하므로 성능 향상은 온전히 CL approach 덕분이라 볼 수 있다.

  - 특히, 좀 더 어려운 task인 CoLA와 RTE에서 효과적이었다. (각각 +3.3, +1.8)

  - 이는 task가 어려울 수록 모델이 이른 시간에 어려운 example을 먼저 보는 것이 학습에 좋지 않다고 할 수 있다.

<br></br>

### 5. Ablation Study

BERT-Base model로 SQuAD 2.0 task를 이용해 여러 흥미로운 주제에 대해 실험해보았다.

#### 5-1. Comparision with Heuristic CL Methods

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/fa00af35-abf2-4787-910c-d0851d89ad78" width="35%"></img></div>

- 난이도 측정 방법으로 word rarity, answer length, question length, paragraph length 를 사용해 비교해보았다.

  - word rarity의 경우 전체 question의 average word frequency를 이용했다. 
  
  - difficult example는 낮은 빈도를 가지는 단어, 긴 answer, question, paragraph로 정의했다.

- 모든 example을 각 metric으로 정렬하고 10개의 버킷으로 나누었고 제안한 방법처럼 Arrangement algorithm을 적용했다.

- 또한, Curriculum Arrangement method의 경우 어떤 샘플링도 하지 않는 Naive order를 통해 비교해보았다.

- 결과로, 제안한 방법이 다른 방법들을 능가했다.

#### 5-2. Case study : Easy VS Difficult

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/5b36456e-9970-4b6f-9c26-2eb553152693" width="45%"></img></div>

- 제안한 방법이 정말 example들의 난이도를 잘 구분해냈는지 확인하기 위해, 각 bucket에 대한 통계량을 그림으로 나타내었다.

- answer length, question length, paragraph length가 단조증가하는 것을 보아, answer, question, paragraph가 길수록 어려운 example인 경향이 있다. 이러한 결과는 직관과 일치한다.

- unanswerable example의 percentage는 40% -> 20%로 떨어졌는데 이는 단순히 분류를 수행하는 것이 정확한 정답 boundaries를 찾는 것보다 쉽기 때문인 것으로 보인다.

  - `no-answer`로 처리하는 과정(분류)이 MRC task보다 쉽기 때문이라고 저자들은 추측함

#### 5-3. On Different Settings of N

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/6c61972e-cd1c-40e2-9cc4-b5c29905e4c6" width="45%"></img></div>

- 제안한 방법의 하이퍼-파라미터인 `N`을 2~20의 값으로 바꿔보면서 성능을 관찰했음.

- 어떤 `N`을 선택하던간에 제안한 방법의 성능이 항상 좋았으며, `N`을 엄청 큰 숫자(==100)으로 설정해보았을 때는 74.10 F1으로 오히려 안좋은 결과를 보였음.

  - 이는, meta-dataset의 크기가 너무 작아지기 때문인 것으로 추측함
