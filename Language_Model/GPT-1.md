## :page_facing_up: Improving Language Understanding by Generative Pre-Training


### Info

  * Preprint, 2018
    
  * author: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever

  * url: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

  - also known as GPT-1

### 1. Abstract & Introduction

labeled data가 부족하기 때문에 여러 task에 맞는 모델을 구성하는 것이 어렵다. 이를 해결하기 위해 각 task에 `discriminative fine-tuning`을 할 수 있으며 unlabeld text로 학습된 `generative pre-training model`을 제안한다.

또, `task-aware input`을 사용함으로써 fine-tuning 동안 모델 구조에 작은 변화만 주도록 한다.

해당 논문에서는 `unsupervised pre-training(language modeling) + supervised fine-tuning(task-specific)` 을 하는 `semi-supervised` 접근을 한다.

모델 구조로는 `Transformer`를 사용했으며 4가지 task로 평가함. (NLI, QA, semantic similarity, text classification)

<br></br>

### 2. Framework

Unsupervised Pre-training stage 와 Supervised fine-tuning stage로 나뉨.

- **Unsupervised Pre-training**

  - unsupervised corpus의 토큰이 다음과 같을 때 $\bf{u}$ $= \{u_1, ... , u_n\}$ , 아래의 likelihood를 최대화하는 language modeling

  - $L_1(\bf{u})$ $= \sum_{i} log P (u_i | u_{i-k}, ..., u_{i-1}; \theta)$

    - `k`: context window size, 조건부 확률 P는 신경망으로 모델링됨. (SGD 사용)

    - 식 해석: $u_i$ 는 예측해야 할 토큰임. 따라서 `i-k ~ i-1` 번째의 토큰들을 보았을 때 `u_i`가 나올 확률

  - `multi-layer Transformer decoder`를 사용했다.

  - **input embedding -> multi-headed self-attention -> FFNN with softmax**

- **Supervised fine-tuning**

  - supervised target task에 적용하는 단계

  - 위의 fianl transformer block을 지나 $h_l^m$ 을 얻고 이를 linear layer에 넣어 y를 예측하게 함.

  - $P(y|x^1, ... , x^m) = softmax(h_l^mW_y)$

  - 따라서, 다음을 최대화 하면 됨. $L_2(\bf{C})$ $= \sum_{(x, y)}logP(y|x^1, ... , x^m)$ `[C: labeled dataset]`

  - 또한 보조 task로 language modeling을 포함시키는 것은 fine-tuning의 일반화와 수렴을 도왔음.

  - 최종 objective는 다음과 같음. $L_3(\bf{C})$ $= L_2(\bf{C})$ $+ \lambda * L_1(\bf{C})$ $\lambda$ : 가중치

- **Task-specific input transformations**

  - text classification과 같은 task에는 바로 fine-tune이 가능하지만 몇몇 다른 task에는 input을 바꾸어주어야 한다. [아래 그림 참조]

  - structured input을 ordered sequence로 바꿨음. [traversal-style approach]

  - 이를 통해, task 간에 큰 변화가 없도록 만듦.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/220050089-18c403a2-fe29-4bf7-bad2-a0ff4b4758d7.png" width="60%"></img></div>

<br></br>

### 3. Experiments

- Unsupervised Dataset: `BooksCorpus(다양한 장르의 7,000 unique unpublished books)`

- Supervised Dataset

  <img src="https://user-images.githubusercontent.com/46083287/220050110-72cd3156-f668-4888-a85f-70ead9748377.png" width="60%"></img>

- Model Specifications

  | Pre-training Setting |    |
  |:----------------:|:--:|
  |   Decoder Layer | 12 |
  |  attention head | 12 |
  |  attention dim  | 768 |
  |  FFNN dim       | 3072|
  |  Optimizer       | Adam |
  |  learning rate   | 2.5e-4 |
  |  learning rate scheduler  | cosine |
  |  warm_up steps   | 2000   |
  |  epochs          | 100 |
  |  batch size     | 64 |
  | max_length       | 512   |
  | dropout_rate     | 0.1   |
  | Activation function | GELU |
  | Tokenizer   | spaCy |
  | Vocab | BPE |
  | weight initialization | $N(0, 0.02)$ |

- Fine-tuning detail

  - 기본적으로 위와 같은 설정 적용

  - `classifier dropout rate`: 0.1

  - `learning rate & batch-size`: 6.25e-5 & 32

  - `epochs`: 3

  - `linear learning rate scheduler` 적용. (rate: 0.2%)

  - $\lambda$: 0.5 (auxiliary task 가중치)


### 3.1 Supervised fine-tuning

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/220050137-86397a2b-ce4b-43fe-8925-6ead0dc6a63a.png" width="60%"></img></div>


<div align="center"><img src="https://user-images.githubusercontent.com/46083287/220050153-fc4ad1c6-9085-470f-a868-f738484821f9.png" width="60%"></img></div>

- **NLI**: 5개의 데이터 셋에서 모두 SOTA 달성. (RTE 데이터 셋은 작은 데이터 셋이며 biLSTM 모델이 성능이 더 좋았음)

- **QA & commonsense reasoning**: 모든 데이터 셋에서 SOTA 달성

- **Semantic similarity**: 3개 중 2개의 데이터 셋에서 SOTA 달성

- **Classification**: SST-2를 제외하면 SOTA 달성

- 12개 중 9개의 데이터 셋에서 SOTA를 달성했음. 이는 우리가 제안한 것이 서로 다른 크기의 데이터 셋들에서도 잘 적용된 것을 보여줌.

<br></br>

### 4. Analysis

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/220050168-bc84bd0d-81f6-42d9-9e91-c3866d9a598a.png" width="70%"></img></div>

- **Impact of number of layers transferred**

  - 위의 왼쪽 그림을 보면, 각 transformer layer가 성능을 올리는 것을 알 수 있다. [Embedding Transferring]

- **Zero-shot Behaviors**

  - 왜 transformer의 LM pretraining이 효과적인지 이해하고자 했음.

  - generative model이 language modeling 능력을 향상시키기 위해 위에서 평가했던 많은 task들을 배운다고 가정함.

  - 오른쪽 그림에 supervised finetuning 없이 수행한 task들의 성능을 나타냄.

  - 이는 generative pretraining이 task에 관련 없이 넓은 범위로 학습하는 것에 도움이 되는 것을 의미함.

- **Ablation studies**

  - `auxiliary objective`가 있는 것이 큰 데이터 셋에는 도움이 되지만 작은 것에는 도움이 되지 않았다.

  - LSTM보다 Transformer 구조가 성능이 더 좋았다.

  - pretraining을 하지 않는 것은 모든 task에서 성능이 안 좋아졌다.

### 5. Conclusion

- `unsupervised pre-training`은 성능을 높여주었다.

- 어떤 모델과 데이터가 잘 동작하는지 보여주었다.
