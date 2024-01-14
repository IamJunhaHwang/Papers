## :page_facing_up: FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS

### Info

* `conf`: Accepted at ICLR 2022

* `author`: Jason Wei∗, Maarten Bosma∗, Vincent Y. Zhao∗, Kelvin Guu et al.
* `url`: https://openreview.net/pdf?id=gEZrGCozdqR
* `github` : https://github.com/google-research/flan


<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/3f615b98-6fd5-4c84-8197-97d9f0e84eaf" width="50%"></img></div>

### 1\. Abstract & Introduction

GPT-3와 같은 규모의 LM은 few-shot에 대한 성능은 좋았지만 zero-shot은 그렇지 못했다. 그 이유 중 하나는 few-shot 예시가 없이는 pre-training data와 다른 형태의 프롬프트에서 LM이 잘 동작하지 못하기 때문이다.

본 논문에서는 LM의 zero-shot 능력을 높이는 간단한 방법인 **instruction tuning** 을 제안한다. instruction tuning은 instruction으로 표현된 데이터 셋의 모음으로 LM을 fine-tuning하는 것을 말한다.

우리는 `Finetuned LAnguage Net (FLAN)`이라 부르는 instruction-tuned model을 unseen task type들에 대해 평가했다 (이를 위해 task type에 맞게 NLP 데이터셋들을 클러스터로 그룹화함). FLAN은 25개 중 20개의 데이터 셋에서 175B GPT-3의 zero-shot 성능을 능가했다. Ablation studies에서는 클러스터의 수가 늘어날 수록 성능이 좋아지는 것을 확인하였으며 모델의 규모가 충분히 커야지 instruction tuning이 효과가 있었다.

<br></br>

### 2.  FLAN: INSTRUCTION TUNING IMPROVES ZERO-SHOT LEARNING

#### 2-1. TASKS & TEMPLATES

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/f73490c0-fec9-46b4-b5be-b3322e32ecd3" width="50%"></img></div>

밑바닥부터 instruction tuning을 위한 여러 task dataset을 만드는 것은 리소스가 많이 필요하므로 기존의 dataset을 intructional format으로 바꾸었다. Tensorflow Datasets에서 public하게 사용 가능한 62개의 데이터 셋을 모아 각각을 해당하는 12개의 task cluster 중 하나에 넣었다.

각 데이터 셋에서 자연어 instruction으로 task를 표현하는 unique template 10개를 수동으로 구성하였다. 대부분의 템플릿은 original task를 표현하지만, 다양성을 늘리기 위해 각 데이터 셋에는 최대 3개까지 task를 바꾼 템플릿을 포함했다 (ex. sentiment classification이라면, movie review를 생성하도록). 그 다음, 이러한 템플릿 중 랜덤으로 선택하여 포맷팅한 데이터 셋으로 PLM을 fine-tune한다.

#### 2-2. EVALUATION SPLITS

우리는 instruction tuning에서 보지 못한 task에 대해 `FLAN`이 어떻게 동작하는지를 알려고 하므로 `unseen task`를 정의하는 것이 중요하다. 이 논문에서는 instruction tuning에 사용한 어떤 작업 클러스터의 데이터셋도 보지 않은 경우에만 `unseen`으로 간주한다. 즉, dataset $\mathcal{D}$ 가 entailment task라면 instruction tuning동안에는 entailment task dataset가 나타나지 않으며 다른 모든 클러스터로 fine-tune된다.

#### 2-3. CLASSIFICATION WITH OPTIONS

주어진 task에 대한 output은 `several classes(classification)` or `free text(generation)` 이 될 것이며 FLAN은 decoder-based model이므로 generation에는 문제가 없지만 classification의 경우 추가적인 작업이 필요하다.

이전 연구에서는 classification task를 위해, `yes`나 `no` 중에서 높은 확률을 가지는 것을 취했지만, 정답을 표현하는 다른 방법에 대한 분포가 정답의 확률량을 낮출 수 있다는 문제가 있다(ex. `yes`는 `right`, `good` 등이 될 수 있음).   
따라서, classification의 끝에 output class 리스트인 **option** suffix를 포함하는 것으로 모델이 어떤 선택을 해야하는지 알려준다. (첫번째 그림 참고)

<br></br>

### 3. Experiments

- Model : `LaMDA-PT`

  - 137B, Decoder-only LM

  - 사용 데이터 : web documents (Wikipedia, dialog data, computer code, etc); 약 10%는 non-english
  - 32K vocab using SentencePiece

- Instruction Tuning procedure : FLAN은 LaMDA-PT의 instruction-tuned version 임.

  - 모든 데이터 셋을 섞고 각 데이터 셋에서 랜덤으로 샘플을 선택함

  - 데이터 셋들이 각기 다른 크기를 가지기 때문에 밸런스를 맞추기 위해 각 데이터셋 당 최대 `30k`로 제한하였으며 examples-proportional mixing scheme (Raffel et al., 2020)을 따랐다.

  - 여러 training example을 하나의 시퀀스에 담기 위해 packing (Raffel et al., 2020)을 사용했다.

#### 3-1. Results

각 데이터 셋에서 모든 템플릿에서의 성능을 평균하여 평가한다. 때로는 개발 셋에서 수동 프롬프트 엔지니어링가 사용 가능했기 때문에 각 데이터셋에 대해 개발 셋 성능이 가장 높은 템플릿을 사용하여 테스트 셋 성능도 얻는다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/02aa171a-eefb-4645-b3ef-fe07996ade47" width="50%"></img></div>

비교를 위해, GPT-3와 같은 프롬프트를 사용한 zero & few-shot `LaMDA-PT` 결과를 함께 기록했다. 또한, `GPT-3 175B`와 `GLaM 64B/64E`의 각각 논문에 기록되어 있는 zero-shot 성능을 보여주었다.

best dev 템플릿에서 zero-shot FLAN은 25개 중 20개의 데이터 셋에서 zero-shot GPT-3를 능가하였으며 10개의 데이터 셋에서는 few-shot GPT-3를 능가했다. 또한, 19개중 13개의 데이터 셋에서 GLaM을 능가하였으며 11개의 데이터 셋에서는 one-shot GLaM을 능가했다.

전체적으로 instruction tuning은 자연스럽게 instruction으로 표현되는 task에서는 효과적이었지만 instruction이 크게 상관 없는 language modeling으로 표현되는 task에서는 덜 효과적이었다 (ex. commonsense reasoning, coreference resolution). 즉, downstream task가 original language modeling pre-training objective처럼 되면 효과적이지 않았다.

<br></br>

### 4. ABLATION STUDIES & FURTHER ANALYSIS

#### 4-1. NUMBER OF INSTRUCTION TUNING CLUSTERS

본 논문의 core question은 `instruction tuning이 unseen task에서 zero-shot 성능을 어떻게 높이는가?`이므로 우리는 instruction tuning에 사용된 task와 클러스터의 수가 성능에 어떤 영향을 미치는지 실험했다.

이 실험에서는 NLI, closed-book QA, commonsense reasoning을 평가 클러스터로, 나머지 클러스터를 instruction tuning에 사용했다. 아래 그림은 클러스터에 포함된 task의 수가 적은 순서대로 클러스터를 추가했을 때의 결과이다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/b3b86959-202b-4928-833b-ce170cda378a" width="50%"></img></div>

예상한 것과 같이, instruction tuning을 위한 클러스터가 많아질수록 zero-shot 성능이 더 좋아졌다(sentiment 제외). 이는 우리가 제안한 instruction tuning이 zero-shot 성능을 높일 수 있음을 의미한다. 또한, 테스트한 7개의 클러스터에 대해 성능이 최적에는 도달하지 않은 것으로 보이며, 이는 더 많은 클러스터가 추가되면 성능이 더욱 향상될 수 있다는 것을 의미한다.

이 Ablation이 어떤 클러스터가 evaluation 클러스터의 성능을 높이는 것에 가장 큰 기여를 했는지를 보고자 하는 것은 아니지만 sentiment 클러스터에서 성능이 늘어나지는 않았었다.

#### 4-2. SCALING LAWS

우리는 instruction tuning의 이점이 모델 규모에 따라 어떻게 영향을 받는지 실험했다. 이전과 똑같은 cluster split을 사용했다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/0ccc3bbb-2dfa-471d-95dd-a9e5ea2b8407" width="40%"></img></div>

위의 결과를 보면, 100B 정도의 2가지 모델에서 instruction tuning이 크게 성능을 향상시켰지만 8B 이하의 모델에서는 오히려 성능이 나빠졌다. 이러한 결과가 나온 이유 중 하나는 small model의 경우는 40개 가량되는 task를 학습하는데 모델 전체의 능력(capacity)을 모두 사용했기 때문에 처음 보는 task에서 결과가 나빴고, large model은 모델의 능력 일부를 쓰면서 어떻게 instruction을 따르는지 학습할 수 있기 때문에 나머지 남은 모델 capacity로 새로운 task를 일반화할 수 있게 된다.

#### 4-3. ROLE OF INSTRUCTIONS

성능 향상이 multi-task fine-tuning에서 온 것인지, instruction 없이도 모델이 잘 동작하는지 보기 위해 fine-tuning 동안의 instruction의 역할을 보는 실험을 했다. 

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/433a340f-8cfa-4577-bea2-6aafdefe5eed" width="30%"></img></div>

`no template(no instruction)`은 input/output만이 모델에게 주어진다(ex. 번역의 경우, "The dog runs.", "Le chien court."). `dataset name`에서는 각 input의 앞에 task와 dataset의 이름이 붙는다.

결과를 보면 알 수 있듯이, instruction이 없을 경우 성능이 떨어지므로 instruction이 unseen task의 zero-shot 성능에 중요한 역할을 한다.
