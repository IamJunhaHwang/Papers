## :page_facing_up: THINK BEFORE YOU SPEAK: TRAINING LANGUAGE MODELS WITH PAUSE TOKENS

### Info

* `publication`: ICLR 2024

* `author`: Sachin Goyal et al.
* `url`: https://arxiv.org/abs/2310.02226
* `github(un-official)` : https://github.com/lucidrains/pause-transformer

### 1\. Abstract & Introduction

Transformer-based causal language models는 input 토큰들의 연속을 차례로 보며 토큰을 생성한다(K개의 토큰을 보고 K+1번째 토큰 생성).   
본 논문에서는 `만약에 K+1번째 토큰을 만들 때, K+10개의 hidden vectors(토큰들)을 이용하게 된다면?`의 아이디어를 시작으로 **학습가능한 pause token** 을 input prefix로 붙여 LM training & inference에 도입해보았다. 그럼으로써 마지막 pause token을 보기 전까지 모델의 출력을 지연시켰다(마지막 pause token을 보기 전까지모델의 출력을 무시). 

선험적으로, 위와 같은 변화를 통해 실제로 어떤 것을 가져올지는 불분명하다. 긍정적으로 생각하면, 유도된 delay를 통해 Transformer가 `"wider"`한 computational pathway의 장점을 취할 수 있다고 볼 수 있다. 하지만, 일반적으로는 모델이 이러한 delay를 단순히 무시하게 될 것이다. 결국, pause token은 추가적인 정보를 제공하지 않고 pause token에 대한 단일 임베딩을 제외하고는 데이터에서 추가적인 정보를 끌어낼만큼 충분한 새로운 파라미터들이 있지도 않다. 설상가상으로 이러한 uninformative token이 오히려 모델에 해가 될 수 있다.

이전 연구([Burtsev et al., 2020](https://arxiv.org/pdf/2006.11527))에서는 prepended dummy tokens을 fine-tuning 학습 과정에 추가하는 방식을 제안했으며, 약간의 성능 향상이 있었다(계산을 확장하는 대신 메모리를 추가하려는 본 논문과는 다른 동기임).   
본 논문에서는 `모든 training & inference 단계에 delay를 주입했을 때, 무엇을 기대할 수 있는지` 와 `Transformer를 delay와 함께 학습시켰을 때의 주요 질문들`에 대해 경험적으로 평가하였다.   
C4 데이터셋으로 1B 와 130M 파라미터의 decoder-only model을 pause-training 하는 것을 연구하였으며, 9가지 down-stream tasks에 fine-tune하였다.

- Contributions

  - `모델의 inference(generation)를 `delay`할 수 있다면 어떻게 될까? 이를 어떻게 구현할 수 있을까?` 라는 질문을 제기하고, 더미 `<pause>` 토큰과 함께 학습하는 방법을 제안함. --> pause-injected pretraining, finetuning, inference procedure.
  
  - pre-training과 finetuning에서 `<pause>` 토큰을 사용한 모델이 여러 downstream tasks에서 기존 training/inference 보다 좋은 성능을 얻음.

  - 다른 한편, finetuning stage에서만 `<pause>` 토큰을 도입했을 때, 성능 향상이 미미했으며 몇몇에서는 성능이 떨어지는 것을 관찰했음.

  - 여러 ablation study를 진행함 : (a) `<pause>` 토큰의 appending/prepeding의 성능차이, (b) `<pause>` 토큰의 최적 갯수, (c) inference time에서 `<pause>` 토큰의 갯수를 줄였을 때, pause-training에서 이러한 것을 학습하지 않았음에도 불구하고 성능이 점점 떨어지는 것을 보았음.

<br></br>

### 2. Pause-training

현재 LM의 패러다임은 K+1번째 토큰을 생성하기 위해 각 레이어의 K개의 임베딩을 사용한다는 것이다. 우리의 전제는 `이러한 K개의 연산은 임의로 놓은 것`이므로 다음 토큰을 생성하기 위해 K개 이상의 연산으로 늘려보는 것이다(이와 비슷한 효과로 attention heads를 늘리는 방법도 있지만, 본 논문에서 관심있는 것은 파라미터를 거의 늘리지 않으면서 연산을 늘리는 것).   

본 논문의 아이디어는 input에 M개의 dummy token을 붙여 sequence length를 늘려 모델의 next response를 지연시키는 것이다. 그 효과로, M-token-delay는 모델이 next token을 생성하기 전에 M개의 추가적인 중간 벡터들을 다루게 하며 이 벡터들은 input에 대해 좀 더 많은 표현을 제공할 수 있게 된다(좀 더 나은 next token 생성).

#### 2-1.  LEARNING AND INFERENCE WITH THE <pause> TOKEN

dummy token을 설정하는 간단한 방법은 `.`와 `#`과 같은 special characters로 설정하는 것이지만, 이는 모델이 해당 character가 자연어에서의 쓰였을 때와 헷갈려 할 수 있으므로 본 논문에서는 일반적인 vocab에는 존재하지 않는 `<pause>` 토큰을 새로 만들어 사용했다. 또한, multi-token delays를 도입하기 위해 이 토큰을 단순하게 반복했다.

<div align="center"><img src="https://github.com/user-attachments/assets/8faaae9e-5ba2-4577-9fea-c31f3f7957dd" width="70%"></img></div>


##### Pretraining with the <pause> token : pause-pretraining

pretraining sequence $p_{1:N}$ 이 주어지면, 여러 <pause> token을 uniformly random한 위치에 삽입해 pause-injected sequence $\tilde{p}_{1:N+M_{pt}}$ 를 만든다(위 그림의 오른쪽 참고).   
이후, <pause> token을 예측하는 것에 해당되는 loss term을 무시하면서 standard next-token prediction loss로 모델을 학습한다.

- $S_{ignore} = \{k : \tilde{p}_{k+1} = <pause> \}$ 는 next token이 <pause> token인 위치라고 하자. 그러면, decoder-only LM `f` 를 위한 pause-training loss는 다음과 같다.

  - $\mathcal{L}_{PausePT}(f, \tilde{p}_{1:N+M_{pt}}) = \underset{k=1, k \notin S_{ignore}}{\overset{N+M_{pt}-1}{\sum}} \mathcal{L}_{CE} (\tilde{p}_{k+1}, f(\tilde{p}_{1:k}))$

    - $\mathcal{L}_{CE}$ : cross-entropy loss

<div align="center"><img src="https://github.com/user-attachments/assets/503f8558-f128-4cd9-b998-782211c86ea1" width="70%"></img></div>


##### Finetuning with the <pause> token : pause-finetuning

prefix p_{1:N}$ 와 annotated target $t_{1:T}$ 가 주어지면, 여기에 $M_{ft}$ 개의 <pause> token을 붙여 새로운 prefix $\tilde{p}_{1:N+M_{pt}}$ 를 만든다. 앞서와 같이 마지막 <pause> token을 볼 때까지 model output을 무시하며 standard next-token prediction loss를 적용한다.

- minimizing $\overset{T-1}{\underset{k=0}{\sum}} \mathcal{L}_{CE} (t_{k+1}, f([p_{1:N+M_{ft}}, t_{1:k}]))$ 

  - `[']` : concatenation 연산

  - downstream task 각각에 대해서 $M_{ft}$ 를 고정시켰다(어떤 하나의 task안에서는 <pause> token 개수는 고정).

##### Pausing during inference : pause-inference

inference 동안 $M_{inf}$ 개의 <pause> token을 붙이고 마지막 <pause> token이 나올 때 까지 model output을 무시한다.

**pause-finetuning과 같은 <pause> token 개수를 사용한다. --> $M_{ft}$ == $M_{inf}$**

#### 2-2.  Variants of Pause-Trining

본 논문에서는 pause-training의 각 단계가 추론 시간 성능에 어떤 영향을 미치는지에 대한 차이가 있는지 확인하는 것을 목표로 아래와 같은 조합을 고려한다.

1. Standard Pretraining and Standard Finetuning **(StdPT_StdFT)**

2. Standard Pretraining and Pause-Finetuning **(StdPT_PauseFT)**

3. Pause-Pretraining and Standard Finetuning **(PausePT_StdFT)**
4. Pause-Pretraining and Pause-Finetuning **(PausePT_PauseFT)**

<br></br> 

### 3. Experiments

- 우리의 주된 실험은 알와 같은 2가지 질문을 다루는 것을 목표로 한다.

  1. pausing으로 model computation을 지연시키는 것은 도움이 되는가, 어떤 효과도 없는가, 성능을 해치는가?

  2. 이러한 지연이 어떤 효과를 준다면, pretraining 단계 vs finetuning 단계 vs both 에서의 성능 차이가 있는가?

#### Experiment Setup

- model : 1B & 130M decoder-only models (ablation에서는 1B만 사용)

- dataset : C4 English mixture (200B Tokens)

- pause-pretraining을 위해 sequence length(2048)만큼의 위치 중에서 10%를 랜덤하게 <pause> token을 삽입했으며, 이렇게 길어진 sequence에서 original sequence length를 넘어가는 것을 잘라내었다.

- 하나의 <pause> token을 위한 임베딩을 추가하였다. (token embedding size == 1024)

- 우리는 각각의 downstream task가 <pause> 토큰 개수를 얼마나 쓰냐에 따라 이득이 있을 것으로 예상하기 때문에, $M_{ft}$ 및 $M_{inf}$ 를 10과 50으로 설정하여 finetuning을 수행하고, 결과에서는 이 두 가지 중 최상의 결과를 보고했다.

- Downstream datasets 9개 : reasoning(GSM8K), extractive QA(SQuAD, CoQA), general understanding(CommonSenseQA, PhysicalIQA), long term context recall(LAMBADA), natural language inference(HellaSwag), fact recall(WebQuestions, Natural Questions)

  - HellaSwag 와 PhysicalIQA는 scoring task

  - CommonSenseQA는 decoding task이므로 Exact Match로 평가

<div align="center"><img src="https://github.com/user-attachments/assets/71448ad8-dfd0-4d48-af2c-599d3ca0b1e2" width="70%"></img></div>

#### Results : Effect of pause-training

- benefit of PausePT_PauseFT

  - `PausePT_PauseFT`가 거의 모든 task에서 명확한 성능 향상이 있었다. 이 모델은 1B의 경우 8가지 task에서 130M 모델의 경우 6개의 task에서 `StdPT_StdFT`를 능가하였다.

- lukewarm effect of StdPT_PausePT: 위와 달리 finetuning에서만 delay를 도입한 `StdPT_PausePT`는 `StdPT_StdFT`에 비해 성능이 비슷하거나 오히려 성능이 나빠졌다.

- Isolating the benefits of PausePT_StdFT

  - `PausePT_PauseFT`의 성능 향상은 inference-time delay 뿐만 아니라 pause-pretraining으로 배운 better representation일 수도 있으므로 `PausePT_StdFT`를 실험해보았다.

  - 그 결과, 2가지 task에서 확실한 성능 향상을 보였으며, 이는 pause-pretraining이 몇 가지 task에서의 representation을 높일 수 있었다는 것을 의미한다.

- Filler characters as <pause> : `StdPT_StdFT`모델과 inference에서 `.`으로 10이나 50번 써서 delay시킨 것(Lanham et al., 2023)을 비교해보았다(아래 그림의 a). 그 결과 성능 향상은 없었다.

  - 따라서, `delay가 도움이 되는지, 해가 되는지, 아니면 아무런 영향을 미치지 않는지`에 대한 질문에 대해, delay가 언제 도입되느냐에 따라 달라진다는 것을 찾아냈으며, pause-pretraining 과정이 필수적인 것으로 생각된다. 
  
  - 또한, 우리는 standard PLM이 추론 시간 지연의 이점을 완전히 실현하지 못하게 하는 강한 편향을 가지고 있다고 추측한다(ex. standard pretraining은 모델이 "빠르게" 계산하도록 편향을 줄 수 있다).

<br></br>

### 4. Ablations : WHERE AND HOW MANY <pause> TOKENS TO USE

<div align="center"><img src="https://github.com/user-attachments/assets/57345da4-1d68-40ef-b40f-fc92ffc4665e" width="70%"></img></div>


- NUMBER OF <pause> TOKENS DURING FINETUNING

  - 각 downstream task마다 해당하는 optimal한 <pause> token 개수 $M_ft$ 가 있음을 찾아냈다. 예를 들어, GSM8K 데이터에서는 10개가 optimal하고 50으로 늘릴수록 성능이 떨어졌다(위 그림의 b). 한편, SQuAD에서는 10개는 sub-optimal이다.

  - 아마 각 데이터셋에 대해, self-attention 메커니즘에 과부하가 걸리는 <pause> 토큰의 특정 임계값이 존재할 수 있다.

    - `long context에서는 성능이 높아지던데, 이런 해석이 가능한가?? 데이터 내의 시퀀스 길이들을 비교해보는게 있으면 더 좋을듯`

- ROBUSTNESS TO A VARYING NUMBER OF INFERENCE-TIME PAUSES 

  - 지금까지는 finetuning에서 보았던 만큼 inference에서도 똑같은 <pause> 토큰 개수를 사용했지만 inference에서의 <pause> 토큰 개수를 바꾸었을 때 어떻게 변하는지 실험하였다. 모델에게 마지막 <pause> 토큰이 나타날 때까지의 supervision을 제공하지 못하므로, 이는 큰 test-time distribution shift를 나타낸다.

  - 결과, `PausePT_PauseFT`는 <pause> token 개수를 바꾸어도 robust하게 작동했다. inference에서의 pause token이 finetuning 시점의 절반이 되더라도 성능은 baseline 이상을 유지했다(위 그림의 c).

  - 상대적으로 `StdPT_PauseFT`는 더 robust했다.

  - 이상적인 robustness의 기준은 <pause> token이 전혀 없는 경우에도 pause-finetuned 모델이 standard-finetuned 모델과 동일하게 잘 작동하는 것이지만, 우리의 어떤 모델도 그러지 못했다. 실제로 `PausePT_PauseFT`는 inference에서 delay가 제공되지 않으면(no <pause> token) 성능이 극적으로 저하되었다(그림 c). 반면, <pause> 토큰이 2개만 있어도 모델은 잘 작동하므로 zero-delay 모델을 설계하는 것이 future work로 중요하다.

- APPENDING VS. PREPENDING PAUSES : `PausePT_PauseFT`에서 <pause> token을 input 앞에 붙여주는 것(prepending)이 standard training 보다 좋았지만, `appending`이 여전히 optimal한 선택이다.
