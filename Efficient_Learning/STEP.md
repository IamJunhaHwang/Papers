## :page_facing_up: STEP: Staged Parameter-Efficient Pre-training for Large Language Models

### Info

* `publication`: NAACL 2025 Short
* `author`: Kazuki Yano et al.
* `url`: https://aclanthology.org/2025.naacl-short.32.pdf


### 1. Abstract & Introduction

최근 LLM이 scaling raws에 기반해 나날이 그 크기가 증가하고 있으며, 이에 따라 pre-training에 필요한 GPU 요구량도 증가하고 있다.

이러한 문제를 다루기 위해, layer addition을 통한 model growth 연구와 parameter-efficient tuning 기법을 합친 방법을 제안한다(**STaged parameter-Efficient Pre-training; STEP**).   

STEP은 vanilla pre-training에 비해 53.9%의 maximum memory requirements 감소를 달성하면서 domain-specific tasks에서의 perplexity와 performance를 유지할 수 있었다.

<br></br>

<div align="center"><img src="https://github.com/user-attachments/assets/379df646-a2bc-47e0-ac37-20d3f66351dd" width="55%"></img></div>

### 2. STEP: STaged parameter Efficient Pre-training


- STEP은 다음과 같은 단계로 이루어진다.

    1. target model size보다 작은 initial model로 vanilla pre-training 진행

    2. Growth Operator를 사용해 layer를 늘린다.

        - Growth Operator로 Interpolation 선택(존재하는 layer 사이에 새로운 layer 삽입)

    3. 1단계에서 학습한 layer에 PET(parameter-efficient adapters; LoRA)를 적용한다.

    4. 2단계에서 새롭게 추가된 layer와 adaptors를 pre-train한다(이 때, 1단계에서 학습한 layer를 freeze한다).

#### 2-2. Maximum memory requirement of STEP

pre-training 동안의 maximum memory requirement는 model state(model parameter, gradient, optimizer state)의 크기로 추정할 수 있다고 가정한다. 또한, 전형적인 Transformer model에 Adam optimizer에 mixed-precision training을 한다고 가정한다.

모델과 gradient는 16-bit floating point로, optimizer states는 32-bit floating point로 표현된다.

- 이 때, Transformer 1개의 layer에 해당하는 parameter 수를 $P_layer$, layer 개수를 `n` 이라고 하면 model state의 memory 사용량은 **bytes**로 표현되면 다음과 같다

  - $P_{trn} = n(2P_{layer} + 2P_{layer} + 12P_{layer}) = 16nP_{layer}$

    - optimizer state는 model, gradient momentum, variance로 구성됨(각 4바이트(32bit)씩).

- $n_i$ 를 i-1 stage로부터 i에서 증가된 layer 개수라 하고, $N_i = \sum^i_{k=1} n_k$ (where, $N_0 = 0$)를 i번째 stage의 총 layer 개수라 하자. 그리고 $E(P_{layer})$ 는 PET로 추가된 단일 layer 파라미터 개수이다. 이때, stage i의 maximum memory requirement는 다음과 같다.

  - $P_i^{STEP} = 16n_iP_{layer} + 2N_{i-1}P_{layer} + 16N_{i-1}E(P_{layer})$

    - $2N_{i-1}P_{layer}$ : 1 ~ i-1 stages에서 이미 train된 frozen model parameter 개수

    - $16n_iP_{layer}$ : Procedure 2에서 새롭게 추가된 model parameter + optimization states 수

    - $16N_{i-1}E(P_{layer})$ : Procedure 3에서 추가된 PET parameter 수

- `L`을 target model의 layer 개수라 하면, maximum memory requirement를 minimize하는 problem으로 정의할 수 있다

  - $\underset{\{ n_1, ..., n_K \}}{minimize} \{ \underset{i=1, ..., K}{max} P_i^{STEP} \} \ \ \   s.t. \ L = N_K$

위와 같은 minimization 문제는 $n_i$에서 모든 i가 양수이기 때문에 integer linear programming 문제이며, ILP solver나 간단한 계산(K=2와 같이 작은 K)로 solution set $\{n_i \}^K_{i=1}$ 을 쉽게 얻을 수 있다.   
일반적으로 K는 작으며, 최대 L−1이고 보통 L/4 미만을 유지하므로 계산적으로 다루기 쉽고 계산 비용은 LLM 사전 학습에 비해 무시할 수 있을 만큼 작다.

<br></br>

### 3. Experiments

- Datasets: FineWebEdu

- Model: LLaMA configuration을 따르며, <368M, 680M, 1.2B> 사이즈를 선택했다.

- Evaluation: 2개의 validation set에 대해 perplexities 계산 (FineWebEdu, Wiki-Text)

  - 또한, 여러 downstream tasks에 대해 accuracy를 평가
 


<div align="center"><img src="https://github.com/user-attachments/assets/3f2a0d40-fe11-4fd5-ac92-62cb83c451b9" width="45%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/7c9ed864-1f67-4d75-bdb2-5f729387d6bf" width="40%"></img></div>


- Configuration 

  - STEP-2stages(K=2) 와 STEP-3stages(K=3)에 집중하였음

  - 고정된 hidden size와 L개의 layer 개수가 주어지고, maximum memory requirements를 minimize할 수 있는 STEP-2stages의 n1, n2를 STEP-3stages의 n1,n2,n3를 계산

  - 위의 표는 target model size가 368M, 680M, 1.2B일 때의 계산된 layer 개수를 나타낸다.

  - Growth Layer Operator는 각 stage의 전체 training step의 75%가 끝났을 때 작동되도록 하였다.

<div align="center"><img src="https://github.com/user-attachments/assets/bbef3785-0af4-43e7-8ae4-1de924975190" width="70%"></img></div>

- Results

  - STEP이 ReLoRA, GaLore보다 좋은 성능을 보였으며 Vanilla pre-training과 비슷한 성능을 내었다.

    - maximum memory requirement를 효과적으로 줄였음(각각 42.3%, 42.2%, 53.9% reduction).

  - STEP-2stages와 STEP-3stages를 통해 stage를 늘리는 것이 성능저하 없이 memory reduction에 도움이 되었다.


<div align="center"><img src="https://github.com/user-attachments/assets/c4b03414-0dd8-47c9-a593-96eb024fae57" width="70%"></img></div>


#### 3-1. Evaluation in instruction tuning

- instruction data: Alpaca dataset

- model: vanilla, STEP-2stages, STEP-3stages

- evaluation: MT-Bench

  - 80개의 multi-turn questions에 대한 대답을 생성하고 GPT-4에게 각 response에 대해 10점 만점으로 rating 시킴

- results: STEPed model이 vanilla model보다 비슷하거나 살짝 더 좋은 성능을 보였다. 이는, STEP이 downstream task에 부정적 영향을 끼치지 않는 다는 것을 의미한다.

<br></br>

<div align="center"><img src="https://github.com/user-attachments/assets/46373e5d-1562-4154-8c22-a34bb390371d" width="40%"></img></div>

### 4. Ablation Study

- effective position for adding new layers : Procedure 2의 Interpolation-Mean 이 어디에서 진행되는 것이 효과적인지 확인

  - Upper, Intermediate, Lower, Random에 대해 실험

  - 실험 결과, upper에 추가하는 것이 더 좋은 성능을 내었음(random보다도 좋았다).

- The effect of PET parameters: PET가 성능에 정말 기여를 하는지 확인

  - Procedure 3를 skip하는 실험을 구성하였음

  - 실험 결과, PET를 제외하였을 때 vanilla pre-training 보다 안좋은 성능을 보였음.
