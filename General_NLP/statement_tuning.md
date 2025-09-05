## :page_facing_up: Enabling Natural Zero-Shot Prompting on Encoder Models via Statement-Tuning

### Info

* `publication`: NAACL 2025 Findings
* `author`: Ahmed Elshabrawy et al.
* `url`: https://aclanthology.org/2025.findings-naacl.465.pdf

* `github`: https://github.com/afz225/statement-tuning

### 1. Abstract & Introduction

LLM이 zero-shot, few-shot에서 뛰어난 성능을 보였지만, 매우 높은 계산량을 필요로한다. 반면, BERT와 같은 MLM 모델은 fine-tuning을 통해 SOTA를 달성할 수 있지만 task-specific layer 때문에 새로운 task에 대한 few-shot과 zero-shot으로 확장하기 어렵다.

본 논문에서는 encoder model을 다양한 unseen NLU tasks에 이용하는 것이 가능한지에 대해 탐구한다. 이것이 가능하다면 LLM에 비해 작은 모델임에도 zero-shot prompting으로 SOTA NLU 성능을 가질 수 있을 것이다.

이를 위해, decoder model의 multitask instruction tuning 과 encoder model의 unified format fine-tuning 에서 영감을 받아 `Statement-Tuning`을 제안한다.

실험 결과, Statement-Tuning은 훨씬 적은 파라미터로 SOTA LLM과 견줄만할 성능을 달성하였으며, 이전 encoder-based model과 비해 더 정확하고 유사 패턴에 더 robust했다.

- Contribution

  - LLM의 대체로써 zero-shot NLU task 일반화를 위한 statement formulation 과 MLM을 조합을 최초로 제안

  - 대규모 실험을 통해 Statement-Tuning이 SOTA LLMs과 견줄만한 성능을 가졌으며 이전 encoder 방법보다 뛰어남을 증명

  - 여러 design choices 탐구를 통해 Statement-Tuning의 optimal한 설정을 알아냄(# of statement, task diversity, etc).

<br></br>

<div align="center"><img src="https://github.com/user-attachments/assets/3212697c-56d3-47a8-98cd-374762151603" width="75%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/2b9a2937-5357-402c-81e1-caf2416bd5e2" width="45%"></img></div>


### 2. Method : Statement-Tuning

#### Task Verbalization

유한한 target을 가진 모든 discriminative task는 자연어 statement로 verbalize 될 수 있다. prompting과 비슷하게, 각 task는 가능한 label에 따라 그 task만의 templates를 가지고 있다.   
각 statement의 training purpose를 위한 truth label은 statement가 올바른 target label 유/무에 따라 결정된다.

#### Statement Fine-Tuning

- multitask statement fine-tuning을 위한 training data를 만들기 위해 데이터 셋 별로 많은 다양한 statement templates를 사용해 16개 NLP datasets에 대한 statements를 생성한다.

  - Datasets: QQP, Winogrande, PiQA, MNLI, SNLI, Mintaka, Yelp Polarity, WikiLingua, SQuAD, TweetEval's Offensive task, Massive, Definite Pronoun Resolution, QASC, SciQ, RACE, SAMSum

- binary sequence classification head를 가진 RoBERTa를 이용해 statements의 truth value를 예측하도록 fine-tuning한다.

  - 다양한 task, templates, domain에 대해 fine-tuning함으로써, 모델은 statement를 true/false로 표현할 수 있는 한 unseen templates & tasks에서도 일반화할 수 있을 것

#### Zero-Shot and Few-Shot Inference

statement-finetuned RoBERTa의 inference 수행을 위해 test에 쓸 input 또한 statement로 바꾸어주어야 한다. 이를 위해 각 데이터 셋에 대해 statement template를 랜덤으로 선택했다.   

위 Figure 1처럼, 각 possible label에 대해 statement를 만들고 각 label에 해당하는 각 statement에 대해 해당 statement가 true가 될 확률을 예측했다. final label은 true가 될 확률이 가장 높은 statement이다.

여기에서, K-shot inference는 K task-specific statement examples에 대해 continual fine-tuning을 한 이후 inference를 진행하는 것이다.

<br></br>

### 3. Experimental Setup

- 모델의 일반화성능을 측정하기 위해 unseen task와 domain의 7개 dataset을 선택했다.

  - Balanced COPA, MRPC, Emotion, Amazon Polarity, FigQA, StoryCloze, Yahoo Answers Topics

  - 이 중에서 MRPC와 Amazon Polarity는 training 동안 보여진 task이지만 domain이 다르기 때문에 `cross-domain generalizability` 를 체크할 수 있다(나머지는 cross-task generalizability 체크).


- statement fine-tuning은 모든 possible label에 다양한 templates로 데이터가 늘어나므로, 생성한 모든 statement를 쓰는 것은 비효율적이다(각 데이터마다 크기가 다르기도 함).

  - 따라서, 각 task에서 true/false statement를 균일하게 statement를 랜덤 샘플링하였다(dataset 당 1000~50000 statement).

- training data 생성의 randomness를 다루기 위해 training 5번, template의 randomness를 다루기 위해 evaluation 5번을 진행해 총 25번 실험

- 위의 multitask statement tuning과정이 끝나면, continual fine-tuning을 target downstream에 대해 진행

  - Full/3000-shot, 1000-shot, 500-shot, 200-shot, 32-shot

- baselines: Llama3-70B, Llama2-13B & 7B, Mistral-7B, Qwen1.5-7B & 0.5B, Pythia-6.9B & 2.8B, Phi-2, FlanT5-Large & Small, BART-large-mnli

  - chat/instruction-tuned model 사용

  - 모델들이 instruction-tuning이 되어있지만, 각기 다른 정도의 instruction following 능력을 가질 것이므로 최대한 fair하게 비교하기 위해 추가로 LoRA를 사용해 Statement-Tuning에 사용한 데이터 셋으로 추가 instruction tuning을 진행했다(500M~13B만 진행).

- zero-shot encoder-only method와의 비교를 위해: NLI-based approach, NPPrompt

<br></br>


<div align="center"><img src="https://github.com/user-attachments/assets/ae7fa8c1-faa8-4604-a476-c48425de4454" width="75%"></img></div>


### 4. Results and Analysis

#### 4-1. Statement-Tuning Enables Effective Zero-Shot Generalization on Masked Language Models.

Table 1은 statement-tuned RoBERTa와 baseline의 zero-shot performance를 나타냄. 다양한 statement-tuning size에 대해 실험했기에, all, 4000, 10000 samples에서의 best performance를 report했다.

결과로, 제안한 statement-tuned encoder model이 unseen tasks & domain에서 zero-shot generalization이 가능했다.   
BCOPA(unseen task), Amazon Polarity(unseen domain)에서 full dataset에 fine-tuned한 모델과 비슷한 성능을 zero-shot으로 달성했다.

####  Comparison Against Larger Zero-Shot Models

제안한 방법은 훨씬 적은 파라미터로 다른 오픈소스 encoder-decoder model, decoder-only LLM과 비슷한 성능을 내었다.   
평균에서 Statement-Tuned RoBERTa-large 모델은 LLM에서 best인 Qwen1.5-7B보다 4.4 point 높았고 largest LLM인 Llama-3-70B보다 4.7 point 높았다.

RoBERTa-base는 6.9B 이하의 모델보다 좋은 성능을 보였다(FlanT5-large 제외).   
제안한 모델은 특히, FigQA와 StoryCloze에서 압도적으로 좋았다(RoBERTa-large vs Llama-3-70B에서 32.2, 9.4 point 차이).


<div align="center"><img src="https://github.com/user-attachments/assets/b71a93a5-8e50-41ea-8349-5234688ed52f" width="75%"></img></div>


#### Comparison with other Encoder Methods

- Encoder methods(NLI baseline, NPPrompt)와 비교했을 때, 제안한 방법이 성능이 더 뛰어났다.

  -  NLI과 비교해서 performance gap이 multiple-choice tasks에서 더 두드러졌으며, NPPrompt에서는 Statement-Tuned모델이 더 작은 사이즈에서도 뛰어났다.

- Robustness to Spurious Patterns: 제안한 방법이 얕은 표면적인 패턴에만 의존하는지 확인

  - input tokens를 섞는 것으로 accuracy drop이 얼마나 일어나는지 측정

  - 결과, statement-tuning 모델의 정확도가 NLI보다 많이 떨어졌으며, 이는 제안한 모델이 표면적 패턴에 의존하지 않는 것을 의미한다(모델이 단어 자체보단 문장의 의미를 이해함).

  - Sentiment Analysis, Topic Classification, Emotion Classification과 같이 word order나 reasoning이 덜 중요한 task에서는 정확도가 NLI보다 적게 떨어졌다(이는 아마, enhanced robustness가 반영된 결과).



<div align="center"><img src="https://github.com/user-attachments/assets/22bf8f24-988f-4564-ab0a-b6be021ca2e7" width="30%"></img> 
<img src="https://github.com/user-attachments/assets/22bf8f24-988f-4564-ab0a-b6be021ca2e7" width="45%"></img>
</div>




#### Statement Finetuning Sample Size: statement sample의 사이즈에 따른 효과

- Zero-Shot

  - 1k sample에서도 best 성능의 96%를 달성할 수 있었으며, RoBERTa-base에서는 4k가 RoBERTa-large에서는 10k가 best였음.

    - best 값보다 넘어가면 accuracy drop이 되었는데 overfitting되는 것으로 생각됨.

    - 이러한 overfitting을 다루기 위해 dataset size와 task diversity를 늘려보았는데 일관된 performance gain을 관찰함.

  - 위에 따라, 5k로 training size를 고정하고 task를 더 포함시키는 것으로 데이터를 늘리는 것을 추천(단순히 individual tasks의 data를 더 포함시키는 것이 아닌).

<div align="center"><img src="https://github.com/user-attachments/assets/e1f5b389-12c8-4eaf-b91e-d9d817c5a721" width="90%"></img>
</div>


- Few-Shot

  - statement-tuning sample size와 # of shots에 대해 조사하였으며, sample size가 늘어날수록 n-shot performance가 zero-shot performance와 비슷해지는 것을 관찰하였음

    - ex) COPA, Emotion, Yahoo에서 4\~5k 일 때, n-shot이 좋지만, 40\~50k가 되면 비슷해짐.

  - 모든 n-shot과 zero-shot 성능간의 상관관계를 보았을 때 높게 나타났으며, 이는 zero-shot도 유용한 정보를 담고 있음을 의미함(논문의 Appendix).

  - 200-shot을 넘어가면 일반적으로 성능이 떨어지는 것으로 보인다. 그럼에도, zero-shot으로도 어느 정도 성능은 달성 된다.

<div align="center"><img src="https://github.com/user-attachments/assets/78087ba7-363a-478b-bb6b-8099682edfa0" width="90%"></img>
</div>

####  Comparison with Standard Fine-Tuning

- RoBERTa-base를 regular fine-tuning한 것과 비교

  - 일반적으로, 적은 N-shot에서는 continually statement fine-tuning이 성능이 더 좋았지만, BCOPA와 FigQA에서는 high N-shot 성능이 좋지 않았다.

  - 따라서, 적은 few-shot이나 zero-shot에서 제안한 방법을 추천하며 제한된 데이터에서 실험할 경우 좋은 성능을 보일 것임.

<div align="center"><img src="https://github.com/user-attachments/assets/30f087c8-e554-4dd3-a11d-66f09bf6d5e3" width="60%"></img>
</div>

#### Effect of Statement & Task Diversity

- Effect of Statement Diversity

  - main 실험에서는 각 데이터셋마다 여러 templates를 사용했지만, 여기에서는 각 corpus마다 최대 N개의 다른 templates를 사용하도록 제한 (Statements per Category; SPC)

  - SPC 1이 가장 성능이 좋았음. 하지만, SPC 2의 경우 표준편차가 6.9% 에서 3.6%로 줄었으며 SPC 3에서는 3.0%였음.

    - 따라서, 적어도 2개의 다른 templates를 사용하는 것이 robustness에 좋음.

<div align="center"><img src="https://github.com/user-attachments/assets/eee86470-0d8a-419a-b50b-ac067d70315c" width="80%"></img>
</div>

- Effect of Task Diversity

  - 많은 task를 포함할 수록 평균 점수가 올라가는 것을 관찰하였음.

  - SA와 WSD를 포함시켰을 때, dissimilar task(Yahoo Answer Topic) 성능이 안좋아졌다(이는 관련 task인 Intent Classification task가 포함되고 회복되었다).

  - **특이한 점은, QA task를 포함시켰을 때 모든 task 점수가 높아졌다.**
