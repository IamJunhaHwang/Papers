## :page_facing_up: Unified Language Model Pre-training for Natural Language Understanding and Generation

### Info

* `conf`: NIPS 2019
* `author`: Li Dong, Nan Yang, Wenhui Wang, Furu Wei et al. **[MircroSoft Research]**
* `url`: https://proceedings.neurips.cc/paper/2019/file/c20bb2d9a50d5ac1f713f8b34d9aac5a-Paper.pdf

* `github` : https://github.com/microsoft/unilm

### 1\. Abstract & Introduction

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/0aa772b5-6f1a-4f47-8912-2ba72e6db56b" width="65%"></img></div>

이 논문은 `NLU`와 `NLG` task 모두에 fine-tune 될 수 있는 `UNIfied pre-trained Language Model(UniLM)`을 제안한다. 이 모델은 3가지 Language Modeling task로 사전 학습되며 이는 `shared Transformer Network`와 `specific self-attention mask`를 사용한다.

실험에서 `UniLM`은 GLUE, SQuAD2.0, CoQA에서 BERT보다 좋은 성능을 보였으며 5가지 NLG 데이터셋에서는 SOTA를 달성했다.

- 3가지 Language Modeling task : `unidirectional`, `bidirectional`, `sequence-to-sequence`

- **UniLM의 장점**

  1. 통합 사전 훈련 과정은 서로 다른 유형의 언어 모델에 대해 파라미터 공유와 아키텍처를 사용하는 단일 Transformer 언어 모델을 만들어서 여러 언어 모델을 별도로 훈련하고 호스팅할 필요를 줄여준다.

  2. 파라미터 공유는 학습한 text representation을 더 일반적으로 만들어준다. (여러 LM objectives들이 함께 최적화되므)
  3. UniLM을 `sequence-to-sequence LM`으로 사용하면 NLG를 위한 자연스러운 선택이 된다.



<br></br>

### 2. Unified Language Model Pre-training

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/f4b329c5-adf9-4b6b-8e65-72804c183fcf" width="55%"></img></div>

입력 시퀀스 $x = x_1 · · · x_{|x|}$ 가 주어지면, UniLM은 contextualized vector representation은 얻는다. 이후, 위의 그림처럼 사전학습을 통해 각각의 Language modeling objectives에 대한 shared Transformer network를 최적화한다. 또한, 예측할 토큰의 문맥의 접근을 제어하기 위해 self-attention에 대해 다른 mask들을 적용한다. 즉, 토큰이 얼마나 많은 문맥에 attend해야하는지 masking을 통해 제어한다.

#### 2.1 Input Representation

`sequence`의 시작에 `[SOS]`토큰을 붙이고, 끝에는 `[EOS]`를 붙인다. 여기에서 `[EOS]`는 NLU에서의 문장의 경계를 나타내 줄 뿐만아니라 NLG에서는 모델이 decoding process를 언제 종료해야 되는지 학습하게 한다.

텍스트는 `WordPiece`로 인해 토크나이즈되며, 각 토큰은 `token embedding, position embedding, and segment embedding`를 합해 vector 표현으로 나타낸다. `UniLM`은 여러 LM task으로 학습되기 때문에 `segment embedding`이 LM identifier 역할을 한다. (다른 LM objective에 대해 다른 segment embedding 사용)

#### 2.2 Backbone Network: Multi-Layer Transformer

입력 벡터 ${x_i}^{|x|}_{i=1}$ 은 $L$ 개의 Transformer Layer를 거쳐 contextual representation으로 바뀐다. 각 Transformer Block에서 `multiple self-attention heads`가 이전 layer의 출력 벡터를 다음과 같이 통합한다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/7bc50ff2-616c-4397-9350-7e2b14c6e923" width="55%"></img></div>

mask matrix $M \in \mathcal{R}^{|x| \times d_h}$ 는 토큰 페어가 서로 attend 해야되는지를 결정한다. (아래의 pre-training objectives에 따라 다르게 사용할 것임)

#### 2.3 Pre-training Objectives

`UniLM`은 각각의 Language Modeling Objectives에 대해 다른 cloze(빈칸 맞추기) task를 사용해 사전 학습한다. 기본적으로 BERT와 비슷하게 input에 있는 토큰 중 무작위로 `[MASK]`토큰으로 바꾸고 이를 맞추도록 한다. 예측한 값과 정답과의 cross-entropy loss를 최소화하도록 학습한다.

- **Undirecctional LM** : `left-to-right`, `right-to-left` 모두 이용 

  - `left-to-right`를 예로 들면, " $x_1 x_2$ [MASK] $x_4$ " 에서 $x_1 x_2$ 와 `[MASK]` 만 컨텍스트로 이용할 수 있다. 이는 self-attention mask $M$ 을 사용해 attend 하지 않을 부분을 $- \infty$ 로 만들어 줌으로써 가능하다.

- **Bidirectional LM** : `BERT`와 같이 모든 token들이 서로 attend 한다. 따라서, $M$ 은 0 행렬이다.

  - NSP task 또한 포함되어 있음

- **Sequence-to-Sequence LM** : 첫 번째 segment만 해당 segment 안에서 자유롭게 attend할 수 있고, 두 번째 segment에서는 자신을 포함한 해당 segment의 왼쪽만 attend 할 수 있다.

  - 예를 들면, “ $\[SOS\] t_1 t_2 \[EOS\] t_3 t_4 t_5 \[EOS\]$ " 에서 $t_1, t_2$ 는 `[SOS]`와 `[EOS]`를 포함해 총 4개에 attend할 수 있고 $t_4$ 는 자신을 포함해 왼쪽의 6 토큰에 attend 할 수 있음

  - 학습하는 동안 각 segment에서 무작위로 토큰을 골라 `[MASK]`로 바꾼 후 모델이 이를 맞추도록 한다.

  - source, target 텍스트가 연속적인 입력 시퀀스로 들어가게 되므로 모델이 이 두 문장 간의 관계를 학습할 수 있도록 하는 것으로 볼 수 있으며 target 텍스트에서 토큰을 더 잘 맞추기 위해 source 텍스트를 효과적으로 인코딩하도록 학습될 것이다.

<br></br>

#### 2.4 Pre-training Setup

- `Overall Training Objective` : 각기 다른 type의 LM objective의 합

  - 하나의 training batch에서 bidirectional LM, sequence-to-sequence LM을 각각 1/3 time 씩, left-to-right, right-to-left LM을 각각 1/6씩 사용

  - batch-size: 330

- `Model` : $BERT_{LARGE}$ 의 구조를 따르며, 해당 모델의 초기 값을 사용한다. (== UniLM은 BERT의 Futher pretraining)

- `Data` : English Wikipedia & BookCorpus

- `Vocab Size & Maximum input length` : 28996, 512

- `Masking` : BERT의 MLM과 동일하지만 추가로, 마스킹 토큰의 80%는 하나의 토큰만 마스킹하지만 20%는 bigram이나 trigram을 마스킹한다.

- `Optimizer & Learning Rate` : Adam($\beta_1=0.9, \beta_2=0.999$), 3e-5 with linear warmup(over 40000 steps and decay)

- `# of steps` : 770,000

<br></br>

#### 2.5  Fine-tuning on Downstream NLU and NLG Tasks

- NLU task : BERT와 같은 bidirectional Transformer Encoder로 fine-tune

  - 일반적인 classification fine-tuning임. (classification head를 추가하는 방식)

- NLG task : pre-training과 비슷하게 `[MASK]`를 예측하는 방식

  - 이는 태스크마다 다르며 sequence-to-sequence를 예로 들면, target sequence의 몇%가 랜덤으로 마스킹되고 이를 맞추도록 학습

  - 여기에서 문장으 끝을 나타내는 `[EOS]` 또한 마스킹될 수 있는데, 모델은 문장의 생성을 끝내기 위해 `[EOS]`를 예측하는 식으로 학습됨.

<br></br>

### 3. Experiments

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/96159ba0-da78-4221-ad9b-f4d05c41fdde" width="60%"></img></div>

- **Abstractive Summarization** : input의 문장을 재사용하는 한계가 없는 generation task

  - `사용한 데이터` : CNN/DailyMail dataset(non-anonmized), Gigaword

  -  sequence-to-sequence로 fine-tuning했으며, {document, summary} 쌍을 input으로 주었다. (max_length를 넘는건 버림)

  - baseline, SOTA, UniLM을 ROUGE의 F1-score로 평가하였다.

  - Table 3를 보면 UniLM은 새로운 SOTA를 달성하였으며 Table 4에서 보듯이 UniLM은 적은 리소스로도 좋은 성능을 보였다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/5af06395-28dc-4c05-b9c7-c796ec02fadb" width="60%"></img></div>


- **Question Answering** : 주어진 passage(context)에 대한 질문에 답을하는 task

  - Extractive QA : 질문이 주어지면 passage(context)에서 정답이 되는 span을 찾는 NLU task

    - bidirectional encoder로 fine-tuning

    - `사용한 데이터` : SQuAD 2.0, CoQA

  - Generative QA : free-form으로 답변하는 NLG task

  - 위 표와 같이 UniLM이 가장 성능이 좋았음.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/1dc0fe53-48d5-4746-bbaf-139fd727df86" width="60%"></img></div>

- **GLUE**

  - BERT-large에 견줄만한 성능을 달성하였음.

<br></br>

### 4. Conclusion

- 파라미터를 공유하며 여러 LM objective를 결합해 최적화하는 `UniLM`을 제안함

- bidirectional, unidirectional, and sequenceto-sequence LM은 NLU와 NLG task에서 바로 fine-tuning 될 수 있도록 한다.

- 실험 결과 GLUE에서 BERT보다 너 나은 성능을 보였으며 5개의 NLG task에서 SOTA를 달성했다.
