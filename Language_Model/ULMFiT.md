## Universal Language Model Fine-tuning for Text Classification

### Info
  * conf: ACL2018
  * author: Jeremy Howard, Sebastian Ruder
  * url: https://arxiv.org/abs/1801.06146
  * github: https://github.com/fastai/fastai/blob/master/fastai/text/models/core.py

### 1. Abstract & Introduction

`Inductive Transfer Learning`은 CV에서 좋은 성능을 보였지만 NLP에서는 성공적이지 못했음. [task에 맞게 변환하고 처음부터 훈련하는 것이 필요]

- 이유
  - 모델의 첫 layer만 target하는 간단한 transfer를 다룸.
  
  - 다른 task의 embedding을 연결하는 것은 main task를 처음부터 다시 학습해야 했고 pretrained embeddings를 고정된 파라미터로 다룸.

  - 많은 in-domain document를 필요로 했음.

  - 효과적이게 학습하는법을 몰랐음.

LM은 작은 dataset에 overfit되고 classifier를 fine-tune할 때, catastrophic forgetting을 겪음.

inductive transfer learning을 가능하게 하며 어떤 NLP task에도 적용 가능한 `Universal Language Model Fine-Tuning(ULMFiT)`을 제안하고 LM을 fine-tuning 하기 위한 핵심 기술을 소개함. [이전 연구보다 좋은 성능을 보였음]

ULMFit은 3-layer LSTM 구조로 6개의 텍스트 분류 task에서 뛰어난 성능을 보임. (같은 하이퍼파라미터를 가지며 dropout hyperparameter를 조정한 것 외엔 추가한 것이 없음)

- **Contribution**

  - CV와 같은 transfer learning을 NLP에서 가능하게 하는 ULMFiT을 제안함.

  - fine-tuning동안 catastrophic forgetting을 피하고 이전 지식을 유지하기 위한 방법인 `discriminative fine-tuning`, `slanted triangular learning rates`, `gradual unfreezing`을 제안함.

  - 6개의 저명한 텍스트 분류 데이터셋에서 SOTA 달성
  
  - 위의 방법들이 sample-efficient transfer learning(적은 데이터로 효과적인 transfer learning)이 가능하고 extensive ablation analysis(feature를 제거해보며 비교하는 분석; 인과관계 파악 가능)을 수행함.

  - pretrained model과 코드를 공개

<br></br>

### 2. Related work

- Transfer learning in CV

  - CV에서는 task-specific[first layer]에서 general[last layer]하게 전이가 이루어지는 특징이 있다. (Yosinski et al. 2014)
  
  - 따라서 대부분의 CV는 모델의 last layer를 transfer하고 있다.
  
  - 최근(2018 기준)에는 끝쪽 몇몇개 layer을 fine-tuning하고 나머지 layer는 고정시키는 방법을 취하고 있다.

- hypercolumns

  - 다른 task들을 통해 추가적인 context를 포착하는 embeddings를 pretrain하는 것

  - 다른 레벨의 embeddings는 feature로서 사용되고 word embeddings나 중간 layer의 입력과 연결됨. [CV에서 hypercolumn으로 알려진 방법]

- Multi-task Learning

  - language modeling objective를 main task와 함께 훈련되도록 추가함. [Rei(2017), Liu et al.(2018)]

  - 매번 처음부터 훈련해야되어 비효율적이고 task-specific objective function이 필요함.

- Fine-tuning

  - Fine-tuning은 비슷한 task간의 transfer로 성공적이게 사용되었지만 관련없을 경우엔 실패함.

  - `Dai and Le (2015)`는 LM을 fine-tune했지만 dataset이 작아 overfit 되었고 좋은 성능을 위해서는 백만개의 도메인 문서가 필요했음.

  - **하지만 ULMFiT은 general-domain pretraining을 하고 fine-tune 기술로 작은 데이터로도 overfit을 피하며 SOTA 달성함**

<br></br>

### 3. Universal Language Model Fine-tuning

Language Modeling은 이상적인 source task로 볼 수 있다. (downstream task 관련한 언어의 많은 면들을 포착할 수 있음) 실제로 이미 핵심 요소로 사용되고 있다.

또, 기계 번역, 논리적 함의(entailment) task와 다르게 많은 도메인과 언어를 위한 거의 제한 없는 양의 데이터를 제공할 수 있다. [Language Modeling은 next token 예측이기 때문에 label이 필요하지 않으므로]

우리는 `Universal Language Model Fine-tuning(ULMFiT)`을 제안한다. [large general-domain corpus에 LM을 pretrain하고 target task에 후에 제안할 기법을 이용해 fine-tune한]

- 여기에서 `universal`은 아래의 경험적인 기준을 만족시키기 때문이다.

  - 다양한 문서 크기, 수, label 종류에서 잘 동작함.

  - 단일 구조와 훈련 과정을 사용함.
  - 수동으로 해주어야할 feature engineering 이나 전처리 과정이 불필요함.
  - 추가적인 in-domain 문서나 label이 필요하지 않음.

실험에 사용한 모델은 당시 SOTA였던 `AWD-LSTM`을 사용했다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/218041431-86ab21ce-4ee2-4768-b0a4-da3f10efc58a.png" width="70%"></img></div>

- **ULMFiT은 위와 같은 과정을 거친다.** 

  - `a`: LM이 각 층에서 언어의 일반적인 특징을 포착할 수 있게 general-domain corpus로 훈련됨.

  - `b`: 전체 LM이 task의 특징을 배울 수 있게 `discriminative fine-tuning(Discr)`, `slanted triangular learning rates(STLR)`을 사용해 target task data에 fine-tune 진행

  - `c`: 분류기(classifier)를 target task에 fine-tune 함.

    - `gradual unfreezing`, `Discr`, `STLR`을 사용. [low-level representation 보존하고 high-level 것을 조절]

  - 아래에 자세히 설명


### 3.1 General-domain LM pretraining

언어의 일반적인 특징을 잡아내기 위해서 충분히 corpus가 커야함. ==> `Wikitext-103` 사용.

- Wikitext-103

  - 28595개의 전처리된 Wikipedia article과 103M의 단어로 이루어져있다.

Pretraining은 target task가 작은 데이터셋을 가지더라도 일반화할 수 있게 함. 이 과정은 시간이 많이 걸리므로 한 번만 하면 충분.

### 3.2 Target task LM fine-tuning

pretraining에 사용한 데이터가 아무리 다양하더라도 target task의 데이터가 다른 분포에서 올 가능성이 있으므로 target task 데이터에 LM을 fine-tune 해야 한다. [pretrain을 한 번 진행했기 때문에 target task의 특징에 적응하면 되므로 수렴이 빠르고 작은 데이터에서도 잘 동작하게 만듦]

LM을 Fine-Tuning하기 위한 기법들을 아래에 제안함.

- **Discriminative fine-tuning**

  - [Yosinski et al., 2014](https://arxiv.org/pdf/1411.1792.pdf)에서 각 layer마다 다른 type의 정보를 담고 있다고 했으므로 layer마다 다른 범위로 fine-tune 되어야 함.

  - 따라서, 각 layer마다 다른 learning rates를 적용하는 `discriminative fine-tuning`을 제안함.

  - 각 layer의 파라미터를 ${\theta^1, ..., \theta^{L}}$ 이라 하자. (L이 layer 개수)
  - 마찬가지로, 각 layer의 학습률을 ${\eta^1, ... \eta^{L}}$ 이라 하자. 그렇다면 아래와 같이 SGD update가 가능함.

  - $\theta_{t}^l=\theta_{t-1}^l-\eta^l \cdot\nabla_{\theta^l}J(\theta)$ ; `t = time step`
  - 마지막 layer의 학습률을 정하고 이를 2.6으로 나누며 각 layer 학습률로 쓰는 것이 실험적으로 잘 동작했음. => $\eta^{l-1} = \eta^l / 2.6$

- **Slanted triangular learning rates**

  - training동안 같은 learning rate(LR)을 사용하거나 annealed LR(점진적으로 줄이는) 것은 좋은 방법이 아님.

  - 따라서 slanted triangular learning rates(STLR)을 제안함. [처음에는 선형적으로 증가했다가 update schedule에 따라 감쇠되는 LR]

  - 아래 수식 설명
    
    - `cut`: `증가 -> 감소`로 바뀌는 iteration 부분, `T`는 전체 iteration이며 `cut_frac` LR이 증가하는 부분
      - 본 논문에서는 `cut_frac = 0.1` 사용함. ex) 전체 iteration이 8000이면 800까지 증가하고 이후엔 감소하는 것

    - `p`: 각 LR이 증가하거나 감소하는 부분, `t < cut`이면 학습률이 계속 증가하고 그 이후에는 감소함. `ratio`는 최대 LR값에서 얼마나 크게 감소시킬 것인지를 나타냄.
    
      - $\eta_t$ 는 iteration `t`일 때의 LR임.
      
      - 즉, `p`값에 따라 학습률이 조절되는 것을 볼 수 있음.

<div align="center">
<img src="https://user-images.githubusercontent.com/46083287/218041523-d9848ed1-68bc-4a8c-a6c3-3cc14ecc0b97.png" width="40%"></img>
<img src="https://user-images.githubusercontent.com/46083287/218041530-641551a1-d396-4ad2-b378-e0040d80ce62.png" width="40%"></img></div>


### 3.3 Target task classifier fine-tuning

classifier를 fine-tuning하기 위해 2개의 linear block을 추가로 붙임. [각 블록은 batch normalization, dropout, ReLU 활성화 함수를 사용하며 마지막 layer에는 softmax 활성화 함수 사용]

이 linear layer들이 유일하게 처음부터 학습되는 곳이고 첫 linear layer는 input으로 LM의 마지막 layer를 pool 것을 받는다.

- **Concat pooling**
  
  - last hidden state만을 input으로 사용하면 정보 손실이 일어날 수 있음.

  - 따라서, last hidden state인 $h_T$ 와 모든 hidden state를 max-pool한 것과 mean-pool한 것을 연결해서 사용함.

- **Gradual unfreezing**

  - 한 번에 모든 layer를 fine-tuning하는 것은 catastrophic forgetting의 위험이 있음.

  - 따라서, 마지막 layer를 시작으로 점차 아래 layer로 unfreeze를 해가는 식으로 학습함.

  - layer를 한 개씩 학습하는 `chain-thaw(Felbo et al., 2017)`와 비슷함.

- **BPTT for Text Classification (BPT3C)**

  - 큰 입력 시퀀스를 위한 gradient propagation을 가능하게 하기 위해 backpropagation through time(BPTT)로 LM을 훈련

  - classifier를 fine-tuning하기 위해 Text Classification을 위한 BPTT를 제안함. (BPT3C)

  - `BPT3C`는 문서를 고정된 batch size인 `b`로 나누고 각 배치의 시작에서 이전 batch의 final state로 모델을 초기화 함.

  - 실제로는 [variable length backpropagation sequences(Merity et al., 2017)](https://arxiv.org/pdf/1708.02182.pdf)도 사용

- **Bidirectional language model**

  - 모든 실험에서 forward LM과 backward LM 둘 다 pretrain 하고 classifier를 fine-tune함. [classifier predictions를 평균냄]

<br></br>

### 4. Experiment

<div align="center">
<img src="https://user-images.githubusercontent.com/46083287/218041626-6ecc48a2-32ce-484f-ab43-e7d829bc6101.png" width="40%"></img></div>

- `데이터`: 널리 연구에 사용된 6개 dataset 사용. (위 그림과 같음)

- `전처리`: 이전 연구들([Johnson and Zhang, 2017](https://aclanthology.org/P17-1052/); [McCann et al., 2017](https://papers.nips.cc/paper/2017/hash/20c86a628232a67e7bd46f76fba7ce12-Abstract.html))과 동일하며 upper-case words, elongation, and repetition을 위한 `special tokens`를 추가함.

- `하이퍼 파라미터`
 
  | model | Embed size | # layers | hidden size | BPTT batch size | dropout | classifier hidden size | Adam | batch size | base LR | fine-tune LR |
  |----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
  | AWD-LSTM | 400 | 3 | 1150 | 70 | 0.4(layers, input embedding), 0.3(RNN layers), 0.05(embedding layers), 0.5(RNN hidden-to-hidden matrix) | 50 | $\beta_1 =$ 0.7, $\beta_2 =$ 0.99 | 64 | 0.004 | 0.01 |

### 4.2 Results

<div align="center">
<img src="https://user-images.githubusercontent.com/46083287/218041636-facee5ff-d611-44c5-8f45-b07e049fef72.png" width="60%"></img></div>

모든 결과로 `error rates`를 기록함. (낮을 수록 좋음)

모든 데이터셋에서 기존 SOTA 성능을 뛰어넘음.

### 5. Analysis

각 contribution의 효과를 평가하기 위해 몇몇의 분석과 ablation을 수행했다. 모든 실험은 각 데이터의 10%를 분리한 validation의 error rate로 평가하였으며 fine-tune classifier는 50 epoch 훈련하였다.

- **Low-shot learning**

  <img src="https://user-images.githubusercontent.com/46083287/218041748-5f96bf4d-716d-4ea8-b911-62d095ccbca5.png" width="80%"></img>

  - `supervised`는 labeled example만 fine-tuning에 사용한 것

  - `semi-supervied`는 모든 task data를 사용한 것

  - 처음부터 훈련한 모델보다 적은 데이터를 이용해 더 좋은 성능을 냄.

- **Impact of pretraining**

  - pretrain을 한 것과 안 한것을 비교하였는데, pretrain한 것 더 효과적이었음.

- **Impact of LM fine-tuning**

  <img src="https://user-images.githubusercontent.com/46083287/218041775-69e6f467-dc72-4cdc-a073-25a77589a682.png" width="40%"></img>

  - 본 논문에서 제시한 fine-tuning 기법들은 성능을 향상시켰음.

- **impact of classifier fine-tuning**

  <img src="https://user-images.githubusercontent.com/46083287/218041793-4e502d61-ff93-4828-993f-78944d01a58c.png" width="40%"></img>

  - `cosine annealing`은 작은 데이터셋에서 성능이 안좋았음.

  - 결과적으로 `discr`과 `stlr`이 핵심적으로 작동했음.
 
- **Impact of bidirectionality**

  - forward LM과 backward LM을 앙상블한 것은 0.5~0.7정도 성능을 향상시켜주었음.

### 6. Conclusion

효과적이고 적은 데이터에서도 효과적인 transfer learning method인 ULMFiT을 제안함.

또한, 몇 개의 fine-tuning 기법을 제안했음. [catastrophic forgetting을 막고 task의 다양한 특징을 학습하게 하는]

위 방법들은 6개의 대표적인 text classification task에서 SOTA를 달성했음.
