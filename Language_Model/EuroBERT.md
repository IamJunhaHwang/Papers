## :page_facing_up: EuroBERT: Scaling Multilingual Encoders for European Languages


### Info

* `publication`: arxiv 2025
* `author`: Nicolas Boizard, Hippolyte Gisserot-Boukhlef, Duarte M. Alves et al.
* `url`: https://arxiv.org/pdf/2503.05500
* `model`: https://huggingface.co/EuroBERT
* `code`: https://huggingface.co/EuroBERT/EuroBERT-2.1B/blob/main/modeling_eurobert.py

### 1. Abstract & Introduction

retrieval, regression, classification에 사용되는 general-purpose multilingual vector 표현은 주로 bidirectional encoder model로부터 얻어졌다. 이럼에도 불구하고, encoder model은 decoder model의 발전에 가려져왔다.

본 논문에서는 multilingual encoder의 발전에 대해 다시 논의하고 european과 널리 사용되는 global languages를 커버하는 multilingual encoder인 `EuroBERT`를 소개한다. EuroBERT는 최신 decoder model 구조를 차용했으며, 5T 다국어 토큰으로 학습되었다. 이 모델은 다국어 능력, 산수, 코딩 등 여러 task에서 현존하는 모델들보다 좋은 성능을 보였다.

또한, EuroBERT의 design(data composition, training pipeline 등)에 대해 제공한다. EuroBERT는 masked language modeling objective가 적용되었고 2단계 training pipeline을 거친다(2번째 단계에서 data distibution 조절).   



<br></br>

### 2. EuroBERT

EuroBERT는 3가지 size(210m, 610m, 2.1B)가 있으며 Llama3 구조를 거의 따랐다. code와 mathematics 데이터를 포함한 다국어 데이터로 학습되며, training piepline은 2가지 단계(pre-training, annealing)로 구성된다.

#### 2-1. Architecture

EuroBERT 모델은 모든 bias를 제거하였으며, `GQA, swish gated linear units, root mean squre layer normalization, rotary position embeddings`를 적용했다.

#### 2-2. Dataset

EuroBERT를 학습하기 위해 5T token의 다국어 corpus를 만들었다(4.8T: pre-training, 0.2T: annealing; annealing에는 15개국어 포함). 이전의 커리큘럼 학습 연구를 따라, annealing 동안의 higher-quality dataset을 두드러지게 하기위해 data distribution을 조절했다.

- Pre-training mixture

  - 영어에는 FineWeb, 다국어에는 CulturaX를 사용했다.

  - to-English와 from-English 번역 쌍을 붙이는 것으로 parallel data도 포함하였으며, 이는 <|parallel_sep|> 토큰으로 구분된다.

  - TheStackv2와 Proof-Pile-2에서 38개의 프로그래밍 언어를 포함하였다.

- Annealing mixture

  - [EuroLLM classifier(Martins et al., 2024)](https://arxiv.org/abs/2409.16235)를 이용해 pre-training동안 보여지지 않은 데이터를 4가지 quality level로 나누었다.

  - 이후, 3번째 threshold 위의 medium과 high quality data를 선택하였다.

  - 또한, low quality data를 포함했을 때 성능이 향상되는 것을 관찰하였다.

  - 추가로, multiple ablation을 통해 data distribution을 조절하였다.

    - 영어의 비율을 줄이고 나머지 언어를 늘렸으며, code와 math 데이터를 줄이고 parallel data를 늘렸다.

#### 2-3. Training Recipe

- Masked Language Modeling

  - [Wetting et al. (2023)](https://aclanthology.org/2023.eacl-main.217/)의 finding을 따라, 50% masking ratio를 적용하였다. annealing phase에서는 downstream 평가에 기반해 10%의 masking ratio로 줄였다.

- Hyperparameters

  - Pre-training에서는 Warmup-Stable-Decay (WSD) scheduler를 linear warm-up phase of 2000 steps로 적용하여 1e-4로 학습하였다. annealing에서는 cosine scheduler decaying으로 0까지 감소시켰다.

  - Pre-training에서는 문장을 최대 2048 토큰까지 패킹하였으며 RoPE의 값으로 10000 사용했으며, annealing에서는 RoPE theta를 250,000까지 늘리고 학습 문서의 길이를 12~8192까지로 랜덤하게 잘랐다(fixed-length로 training하는 것보다 좋은 성능을 보였음).

<br></br>

### 3. Evaluation

#### 3-1. Evaluation Setup

- Datasets and tasks

  - multilingual tasks: MIRACL, MLDR, WikipediaRetrieval, CC-News

  - classification: XNLI, PAWS-X, NER task from the XGLUE, AmazonReviews, MassiveIntent

  - sequence regression: quality estimation(WMT), summary evaluation(SeaHorse)

  - code-related: retrieval(CodeSearchNet, DupStackMath), classification(CodeDefect, CodeComplexity)

  - mathematical: retrieval(MathFormula), reward modeling(MathShepherd)
   
   
- baselines: XLM-RoBERTa, mGTE-MLM-base, mDeBERTa-v3-base, English-only ModernBERT


#### 3-2. Results

<div align="center"><img src="https://github.com/user-attachments/assets/cff5d0f2-507c-4a4d-b98e-a7b55d842b1e" width="70%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/d3577bf8-85e2-4fca-8576-4b12d7754589" width="70%"></img></div>

- **EuroBERT family는 여러 도메인과 task에서 좋은 성능을 보였다.**

  - EuroBERT-2.1B는 18개 task 중 10개에서 1등으로 랭크되었으며, XLM-RoBERTa-3.5B와 견줄만한 성능을 보였다.

  - EuroBERT-610M은 작은 크기에도 불구하고, 여러 다국어 task에서 XLM-RoBERTa-3.5B와 동등한 성능을 보이며, 코드 및 수학 벤치마크에서는 이를 능가하는 성능을 보였다.

  - EuroBERT-210M은 XLM-RoBERTa-560M의 성능과 비슷했으며, European language에서 강한 결과를 보였다.

- **EuroBERT는 document ranking에 효과적이다.**

  - 여러 도메인의 retrieval tasks에서 EuroBERT가 일관적으로 높게 랭크되었다.

  - 특히, 210M과 610M 모델이 비슷한 크기의 다른 모델들보다 좋은 성능을 보였으며, XLM-RoBERTa-3.5B와 비슷했다.

- **EuroBERT는 sequence classification에서 비슷한 크기의 모델과 비슷한 성능을 보였다.**

  - 뛰어나게 다른 모델보다 성능이 좋은 EuroBERT는 없었지만, retrieval과 classification 능력 사이의 trade-off가 생기는 design decision이 있음을 찾았다.

  - 그럼에도, EuroBERT-2.1B는 상위 모델에 랭크되었으며 EuroBERT의 작은 버전들을 비슷한 크기의 모델과 비등했다.

- **EuroBERT는 긴 context length에서 성능을 유지했다.**

  - 아래 그림은 EuroBERT와 XLM-RoBERTa의 long context performance를 비교한 것이다. 짧은 input에서는 비슷한 성능을 보였지만, 긴 input에서는 XLM-RobERTa가 성능 하락이 있었다.

- **EuroBERT는 code와 mathematics task에서 뛰어났다.**

  - 해당 도메인에서 모든 EuroBERT 모델이 일관적으로 다른 모델을 능가하였다.

  - 특히, 210M 모델은 모든 baseline들 위에 랭크되었다.

<div align="center"><img src="https://github.com/user-attachments/assets/53b5f2c2-72c3-40eb-856f-2d2db759cbcd" width="70%"></img></div>


<br></br>

### 4. Training Recipe Analysis

다양한 모델 설계에 따른 영향을 관찰하기 위해 ablation을 진행했으며, 이전 연구처럼 다양한 구성요소를 바꾸면서 40B tokens로 multiple annealing runs 수행

<div align="center"><img src="https://github.com/user-attachments/assets/d1c93db8-f625-4020-ba8e-5b4e83393f84" width="70%"></img></div>

- **language distribution의 균형은 성능이 향상된다.**

  - Figure 4의 가장 왼쪽 plot은 `English`의 비율이 줄어들고 다른 언어를 늘렸을 때의 retrieval과 classification 성능을 보여준다. 균형 잡힌 distribution은 전체 성능을 향상시키지만, distribution이 uniform하게 될수록 성능이 떨어진다.

- **math와 code 데이터는 retrieval 성능을 향상시키지만 classification 성능은 낮춘다.**

  - Figure 4의 2번째와 3번째 plot에서 math와 code 데이터를 줄였을 때, MIRACL의 성능은 떨어지고 XNLI 성능은 오르는 것을 볼 수 있다.

- **parallel data를 늘리는 것은 classification과 retrieval 성능을 향상시킨다.**

  - Figure 4의 4번째 plot을 보면, parallel data를 늘렸을 때 XNLI와 MIRACL 성능이 향상되는 것을 볼 수 있다(이는 다른 최신 연구와도 일치하는 결과).

- **instruction fine-tuning data의 추가는 모델 성능을 약화시킨다.**

  - Figure 4의 가장 오른쪽 plot은 instruction data를 추가했을 때의 성능을 보여주고 있다. decoder model과 다르게 encoder model에서는 성능 약화를 관찰했다.

- **문장 길이를 다양하게 하는 것은 성능을 향상시킨다.**

  - Figure 5의 첫번째 plot은 문장 길이에 따른 영향을 보여준다. 문장 길이를 고정하는 것보다 다양하게 하는 것이 성능을 향상시켰으며, 8,192 토큰을 넘어가면 성능 하락이 있었다.

- **masking ratio의 감소는 classification 성능을 향상시킨다.**

  - masking ratio를 10%까지 줄이는 것은 XNLI 성능을 향상시켰지만, MIRACL 성능은 약화시켰다.

- **educational value 기반의 필터링 데이터는 성능을 저하시킨다.**

  - Figure 5의 가장 오른쪽 plot을 보면, 가장 높은 quality data bucket을 사용하는 것보다 quality level 3과 4를 섞어 사용하는 것이 좋은 성능을 보였다.

  - 아래 그림처럼, 해당 데이터의 split들을 검사한 결과, quality filter가 XNLI의 거의 모든 example을 버리는 것을 볼 수 있었다. 이는 training data가 downstream task에서 벗어난 잠재적 domain mismatch를 시사한다.

  - 이러한 educational value가 decoder model의 assistant-like task에는 적합할 수 있지만, quality bucket의 mix는 general-purpose vector 표현에 적합할 수 있다.

- **Final annealing configuration**

  - 분석에 기반해, 3번째 threshold 위의 데이터를 선택하는 것으로 final annealing dataset을 만들었다.

  - 영어의 비율을 26%까지 줄이면서 나머지 언어에 대해서 균등하게 비율을 올렸다.

  - math와 code의 경우 각각 6%, 4%를 할당했다.

  - parallel data는 6%로 올렸고, instruction data는 삭제하였다.

  - masking ratio는 10%로 줄였고, 8,192 tokens까지의 random sentence lengths로 수행했다.


<div align="center"><img src="https://github.com/user-attachments/assets/b7fd7338-3f67-413c-a6cf-b608bad36bc8" width="70%"></img></div>
