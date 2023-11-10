## :page_facing_up: Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes

### Info

* `conf`: Findings of ACL 2023, long paper

* `author`: Cheng-Yu Hsieh et al.
* `url`: https://arxiv.org/pdf/2305.02301.pdf
* `github` : https://github.com/google-research/distilling-step-by-step

### 1\. Abstract & Introduction

LLM을 적용하기 위해서는 메모리 비효율적이고 계산량이 많기 때문에 실제로 적용하기가 어렵다. 그렇게 때문에, 연구자들은 작은 task-specific 모델을 fine-tuning하는 방법을 취했지만, 이는 대용량 학습 데이터셋을 필요로 한다.   
따라서, 우리는 _**Distilling step-by-step**_ 을 제안한다. 이는 (a) LLM을 능가하는 작은 모델을 학습하고 (b) 적은 데이터 양만을 필요로 한다.   
이 방법은 multi-task 프레임워크에서 작은 모델을 학습시키기 위한 추가적인 supervision으로서 LLM의 rationales를 추출한다.   

우리는 4가지 NLP 벤치마크로 3가지 finding을 보였다.

1. fine-tuning과 distillation 모두와 비교했을 때, 우리의 메커니즘이 더 적은 데이터로 더 좋은 성능을 보였음.

2. few-shot prompted LLM과 비교했을 때, 우리는 훨씬 작은 모델 사이즈로도 더 좋은 성능을 보였음.
3. 우리는 모델 사이즈와 LLM을 능가하기 위해 필요한 데이터 양 모두를 줄였음.

<br></br>

### 2. Method : Distilling step-by-step

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/cf71ced9-32eb-48df-8589-eb02a2cf2481" width="70%"></img></div>

우리가 제안한 방법의 패러다임은 크게 2-step을 가진다. (위 그림과 같음)

1. LLM과 unlabeled dataset이 주어졌을 때, LLM이 label과 해당 label을 생성한 근거(Chain-of-Thought) 대한 rationales를 생성한다.

2. 위에서 생성한 rationale을 사용해, small task-specific model을 multi-task learning으로 학습한다. 여기서 task 정보를 모델에 전달하기 위해, 각 input example에 `task prefix`를 붙이고 이 prefix에 따라, 모델이 다른 output을 내뱉게 학습한다.

#### 2-1. Extracting rationales from LLMs

LLM으로부터 rationales를 끌어내기 위해 **Chain-of-Tought(CoT) prompting(Wei et al., 2022)** 을 이용한다.

먼저, CoT를 위한 prompt template $p$ 를 큐레이트한다. 각 prompt는 ($x^p, r^p, y^p$) 의 triplet으로 이루어지며, 순서대로 example input, 상응하는 label, user-provided rationale 이다. [즉, 여러 개의 프롬프트를 이용; Few-shot CoT]

unlabeled dataset $x_i \in D$ 가 주어지면, 각각의 $x_i$ 를 input으로 prompt $p$ 에 넣어, rationale $\hat{r_i}$ 과 output label $\hat{y_i}$ 를 생성한다.

#### 2-2. Training smaller models with rationales

데이터 셋 $\mathcal{D} = {(x_i, y_i)}^N_{i=1}$ 과 smaller model $f$ 이 있을 때, 이전 연구들은 아래와 같이 text와 rationale 두 개를 input으로 받아 $\hat{y}$ 를 output으로 냈었다. [Rajani et al., 2019; Wang et al., 2022]

$\mathcal{L} = \frac{1}{N} \overset{N}{\underset{{i=1}}{\sum}} \mathbb{l} (f(x_i, \hat{r}_i), \hat{y}_i)$

하지만, 위와 같은 경우 smaller model이 prediction을 만들기 전에 LLM이 rationale를 생성해야하므로 한계가 있다. (test time에 LLM이 필요하게 됨)

따라서, 우리는 위와 같은 한계를 제거하기 위해 text data $x_i$ 를 input으로 받아 rationale $\hat{r}_i$ 와 label $\hat{y}_i$ 를 output으로 하는 multi-task framework를 제안한다. Loss는 다음과 같이 설정한다.

- $\mathcal{L} = \mathcal{L_{label}} + \lambda \mathcal{L_{rationale}}$

  - $\mathcal{L_{label}} = \frac{1}{N} \overset{N}{\underset{{i=1}}{\sum}} \mathbb{l} (f(x_i), \hat{y}_i)$   

  - $\mathcal{L_{rationale}} = \frac{1}{N} \overset{N}{\underset{{i=1}}{\sum}} \mathbb{l} (f(x_i), \hat{r}_i)$

이러한 multi-task frame work를 위해, input text 앞에 `task prefix`를 추가한다. [label], [rationale]

<br></br>

### 3. Experiments

#### 3-1. Experimental Setup

- Model

  - `LLM` : 540B PaLM
 
  - `task-specific downstream model` : T5

- `CoT prompting` : Original CoT(Wei et al. 2022), 새로운 데이터 셋에서는 저자들이 새로 만듦.

- `Datasets `: e-SNLI, ANLI, CQA, SVAMP

#### 3-2. Reducing training data

Distilling step-by-step을 다음 2개의 일반적인 method와 비교했음: (1) STANDARD FINETUNING (human-labeled available), (2) STANDARD TASK DISTILLATION (unlabeled examples available)

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/f4a744df-7b0e-40d1-8537-09ea2ef457a4" width="70%"></img></div>

- STANDARD FINETUNING의 경우

  - 제안한 방법이 좀 더 적은 데이터를 사용했음에도 STANDARD FINETUNING보다 좋은 성능을 보였다.

  - 특히, e-SNLI dataset의 12.5%만 사용하는 것으로도 100% 데이터를 사용한 STANDARD FINETUNING보다 좋은 성능을 보였다.

- STANDARD TASK DISTILLATION의 경우

  - STANDARD FINETUNING과 비슷한 양상

  - 모든 데이터 셋, 다양한 크기들에서 제안한 방법이 성능이 좋았다. 

#### 3-3. Reducing model size

trainging set size를 고정시키고 T5 model의 크기를 다양하게 실험했음.

LLM에 대해서는 2가지 baseline과 비교했음 : FEW-SHOT CoT, PINTO TUNING(rationale을 input에 넣은 것)

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/052d0b92-6ed6-4d0f-8e92-b2b1e9139f8a" width="70%"></img></div>

- 제안한 방법이 다양한 모델 사이즈에서 baseline을 능가했음.

- 제안한 방법이 LLM을 능가했음

  - 제안한 방법이 모든 데이터 셋에서 Few-shot CoT와 PINTO tuning을 능가했다.

- `Figure 7`의 SVAMP 데이터셋에서 제안한 방법이 낮은 성능을 보였는데, 이는 데이터 양이 적기 때문이라고 생각된다.

  - 따라서, unlabeled example을 augment하여 기록했다. (Figure 7의 rightmost)

  - data augmentation에서도 제안한 방법이 더 성능이 좋았다.

#### 3-4. Outperforming LLMs using minimum model size and least training data

LLM의 성능을 고정시켜 놓고 이 성능을 능가하면서 모델 사이즈와 데이터 셋 크기의 최적 양을 알아내는 실험을 함.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/06e26870-faab-4a86-a7fa-efad3b44f915" width="70%"></img></div>

- 제안한 방법이 데이터 셋의 일부를 사용하고 LLM보다 작은 사이즈로 LLM의 성능을 능가하였다.

- standard finetuning 과 distillation 모두 데이터와 모델 크기가 늘어날수록 좋은 성능을 보였다.

- 특히, e-SNLI 데이터에서는 전체 데이터 셋의 0.1%를 사용하는 것만으로도 제안한 방법이 LLM 성능을 능가했다.

<br></br>

### Further ablation studies

#### Distilling step-by-step works with different sizes of decently trained LLMs.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/8c13e59d-6a2b-4334-aec4-82f595312dea" width="50%"></img></div>

540B PaLM보다 작은 LLM인 20B GPT-NEoX model로 실험해보았다.

실험 결과, 제안하는 방법이 standard finetuning보다 성능이 좋았지만 큰 폭으로 좋아지지는 않았다. 이는 GPT-NEOX가 PaLM보다 작은 모델이어서 PaLM보다 좋은 품질의 정보를 제공하지 못하기 때문으로 보인다.

#### Multi-task training is much more effective than single-task rationale and label joint prediction.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/eddf8afe-cbb7-4acc-87c6-a2bfa8b8cc7e" width="50%"></img></div>

LLM-rationales를 output supervisions로 사용하는 가능한 다른 task-specific model이 있다. 가장 흔한 방법은 아래와 같은 Loss를 사용하는 것이다.

$\mathcal{L_{single}} = \frac{1}{N} \overset{N}{\underset{{i=1}}{\sum}} \mathbb{l} (f(x_i), \[\hat{r}_i, \hat{y}_i \])$   

이와 비교해보았을 때, multi-task training이 좋은 성능을 보여주었으며 위와 같은 single-task training은 standard finetuning보다 낮은 성능을 보여주었다.   
이는 이전 연구들과 똑같은 결과이다. (rationale & label prediction을 single joint task로 취급하는 것은 모델의 성능을 해친다)
