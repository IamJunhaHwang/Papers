## :page_facing_up: : LLaMA: Open and Efficient Foundation Language Models

### Info

  * Preprint, 2023.02.22.
    
  * `author`: Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample

  - `url`: https://scontent-ssn1-1.xx.fbcdn.net/v/t39.8562-6/333078981_693988129081760_4712707815225756708_n.pdf?_nc_cat=108&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=ov6yTHfLfNQAX9PIOAT&_nc_ht=scontent-ssn1-1.xx&oh=00_AfAhJXnGWP8jkBCHJ5E4s8fu9wDZ4NLkRwEFfgfcUrFEqw&oe=63FFCFA2
  
  * `meta blog post`: https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/

  - `github`: https://github.com/facebookresearch/llama

  - Meta AI's work

### 1. Abstract & Introduction

`Large Language Models (LLMs)`는 엄청난 규모의 텍스트 말뭉치로 훈련되었고 새롭게 주어진 task에도 좋은 성능을 보였다.(few-shot property) 이러한 특징은 파라미터 수가 많을 수록 성능도 올라갈 것이라는 가정에 기반한다.

하지만, 최근 연구(Hoffmann et al. 2022)에서 주어진 컴퓨터 자원으로 최고의 성능을 내기 위해서는 모델의 크기가 큰 것이 아니라, 크기가 작은 모델로 더 많은 데이터에 학습시키는 것임이 증명했다.

그러나, 위 연구는 `training`에 대한 컴퓨팅 자원만을 고려했지 `inference`에 대한 자원을 고려하지 않았다. 결국 모델을 서빙해야하는 입장에서 보면, `task에 대한 최소한의 성능`을 정해 놓는 다면 선호하는 모델은 `training`이 오래 걸리는 작은 모델이라도 `inference`가 빠른 것이다.

본 연구는 다양한 전형적으로 학습에 사용된 토큰 수의 크기를 늘리는 것으로 `inference budgets`에서 최고의 성능을 내기 위한 LM 시리즈의 학습에 집중한다. 

그 결과로, `7B ~ 65B`의 파라미터를 가지는 다양한 크기의 언어 모델인 **LLaMA**를 소개한다. 

**LLaMA**는 `수 조개(trillions)`의 토큰으로 학습되었으며 학습에 사용된 데이터 셋은 모두 `public`하게 이용할 수 있는 것 만을 사용했다. (이런 종류의 LLM들이 있지만 PaLM과 Chinchilla에 비해 형편없었음. but LLaMA 아님)

특히, `LLaMA-13B`는 10배 작음에도 거의 모든 벤치마크에서 `GPT-3`를 능가했으며 `LLaMA-65B`는 `Chinchilla-70B, PaLM-540B`와 같은 best model과 견줄만 했다.

<br></br>

### 2. Approach

Training Approach는 `GPT-3, PaLM`과 비슷하며 `Chinchilla`의 scaling laws에서 영감을 받았다.

standard optimizer로 large transformer를 많은 양의 텍스트 데이터로 학습했다.

#### 2.1 Pre-training Data

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508189-09dd585e-656f-4b58-bf3e-23cb27929e85.png" width="50%"></img></div>

훈련 데이터들은 위와 같이 다양한 도메인을 위해 여러 소스들의 조합으로 만들었다. (다른 LLM에 사용된 public 데이터만을 포함하기 때문에 위와 같이 비율을 나타냄)

모든 데이터들에서 중복을 제거하였고 `non-english & low quality`한 데이터 제거하였다. 또한, HTML tag, hyperlink 등 필요 없는 부분도 제거하였다.

전체 훈련 데이터 셋은 약 `1.4T`의 토큰들로 구성되며 `Wikipedia, Books` 도메인을 제외하면 훈련동안 한 번만 사용되게 된다.

- Tokenizer: SentencePiece에서 구현한 BPE 알고리즘으로 토크나이즈했다.

  - 모든 숫자를 개별 숫자로 분할하고, 알 수 없는 UTF-8 문자를 분해하기 위해 바이트로 대체하였음.

  - 아마 BBPE를 사용했다고 하는 것 같음.

<br></br>

#### 2.2 Architecture

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508203-4c1cd40b-d475-4cb8-955f-96683f4b4e03.png" width="50%"></img></div>

LLaMA는 Transformer 구조를 기반으로 만들었으며 아래의 제안한 변형들로 다양한 성능 향상을 얻었다.

하이퍼 파라미터는 위와 같다.

- Pre-normalization

  - GPT-3에서 영감을 받았다.

  - 훈련 안정성(training stability)를 얻기 위해, 각 sub-layer의 입력을 정규화했다. (Vanilla model은 output을 정규화 함)

  - 여기에서 사용한 정규화 함수는 `RMSNorm`을 사용한다. [Zhang and Sennrich (2019)]

- SwiGLU activation function

  - PaLM에서 영감을 받았다.

  - 활성화 함수를 `ReLU => SwiGLU`로 바꾸었다. [Shazeer(2020)]

  - PaLM에서 4d를 사용했지만 우리는 $\frac{2}{3}4d$ 의 차원을 사용했다.

- Rotary Embeddings

  - GPTNeo에서 영감을 받았다.

  - absolute positional embedding을 지우고 `Rotary Positional Embeddings(RoPE)`를 추가했다. [Su et al.(2021)]

#### 2.3 Optimizer

- `AdamW `(β1 = 0.9, β2 = 0.95)

- cosine learning rate schedule 사용

- `weight decay` = 0.1 & `gradient clipping` = 1.0

- `warm_up steps` = 2000

#### 2.4 Efficient implementation

- 효율적인 multi-head attention 구현을 사용했다.

  -  `Rabe and Staats (2021)` and `Dao et al. (2022). `

  - 이는 `xformers library`에서 볼 수 있으며 어텐션 가중치를 저장하지 않으며 마스킹된 key/query score를 계산하지 않는다.

- 역전파되는 동안 다시 계산되는 activation의 양을 줄였다. [linear layer의 출력 같은]

  - `pytorch autograd`에 의존하는 대신 transformer layer를 위해 직접 구현하였음.

- 모델의 메모리 사용량을 줄이기 위해 `model & sequence parallelism`을 사용했다. [Korthikanti et al. (2022).]

- 추가로, activation 연산과 GPU간의 통신을 네트워크 상에서 겹치도록 수행했음. 

- 65B 파라미터 모델을 훈련할 때, GPU당 1초에 380 토큰을 볼 수 있고 2048 A100 GPU와 80GB RAM을 사용했다.

  - **1.4T 토큰의 데이터 셋을 훈련하는데 대략 21일 걸렸음.**

<br></br>

### 3. Main results

이전 연구(GPT-3)에 따라 `zero-shot`과 `few-shot` task를 20개의 벤치 마크로 평가해보았다.

여러 LLM들과 비교하였으며 `instruction-tuned model`과도 비교해보았다.

`LLaMA`를 `free-form generation task`와 `multiple choice task`로 평가하였다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508247-4defa0fb-fdad-4204-b434-e84c4a9eefec.png" width="60%"></img></div>

- **Common Sense Reasoning**

  - 8개의 상식 추론 벤치 마크를 사용해 평가했다. [OpenBookQA, BoolQ 데이터 셋에서는 GPT-3에서의 방식을 사용]

  - `zero-shot setting`에서 평가했으며 다양한 크기의 모델을 비교했다.

  - `LLaMA-13B`가 GPT-3 보다 10배 작음에도 대부분의 벤치 마크에서 성능이 더 좋았다.

  - 거의 모든 데이터 셋에서 SOTA 달성 [2개 제외하고는 성능이 제일 좋음]

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508313-718ba7a0-6025-4d12-be74-ff0d5cdb794c.png" width="60%"></img></div>

- **Closed-book Question Answering**

  - 2 개의 `closed-book question answering`을 평가했다. (Natural Questions, TriviaQA)

  - `LLaMA-65B`가 SOTA를 달성했으며 `LLaMA-13B`는 `GPT-3`와 `Chinchilla`에 견줄만 했다. (5~10배 작음에도)

    - `LLaMA-13B`는 V100 GPU 하나로 추론이 가능하기도함.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508327-6eea53b3-4cf1-434f-aee5-fe98bcf82594.png" width="40%"></img></div>

- **Reading Comprehension**

  -  `RACE reading comprehension benchmark`로 평가했다.

  - `LLaMA-65B`는 `PaLM-540B`에 견줄만했고 GPT-3를 능가하였다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508349-6430d9c9-ad3e-42a9-848b-44683ebde9fa.png" width="40%"></img></div>

- **Mathematical reasoning**

  - 2개의 수학적 추론 벤치마크로 평가했다. (MATH, GSM8k)

  - `maj1@k`는 각 문제에 대해 k개의 샘플을 생성해 다수결 투표를 수행하는 것을 나타낸다.

  - `GSM8k`에서 `LLaMA-65B`가 수학 데이터에 fine-tune되지 않았음에도 성능을 능가했다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508367-067f07d3-2208-497d-a254-1300aa433b4c.png" width="40%"></img></div>

- **Code generation**

  - 자연어로 제시한 표현으로 코드를 작성하는 task에 대한 벤치마크 2개로 평가했다. (HumanEval, MBPP)

  - `LLaMA`가 코드에 특화된 fine-tune을 하지 않았음에도 더 좋은 성능을 보였다.

  - `pass@1`은 temperature = 0.1, `pass@100 & pass@80`은 temperature = 0.8 로 샘플링했다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508394-be5cb5a1-9076-4941-810e-1e1b815d9dcf.png" width="40%"></img></div>

- **Massive Multitask Language Understanding**

  - MMLU 벤치마크로 평가했다. [다양한 도메인의 선다형 질문으로 구성됨]

  - `5-shot setting`으로 평가했다.

  - LLaMA가 성능이 더 낮은 이유는 Gopher, Chinchilla, PaLM에 비해 책과 학문적 글에 대한 학습 데이터가 부족하기 때문이다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508424-e6b79ab1-3412-429e-9356-ffacf144a80e.png" width="40%"></img></div>

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508437-7a4d4203-d1f8-4f92-99ae-cb3bb8c126f1.png" width="60%"></img></div>

- **Evolution of performance during training**

  - 대부분의 벤치 마크에서 성능이 점진적으로 증가했고 `training perplexity`와 상관 관계가 있었다.

  - `SIQA`는 성능의 다양한 변화를 보였기에 벤치 마크로 믿을 수 없고 `WinoGrande`는 training perplexity에 관계 없는 성능을 보였다.

<br></br>

### Instruction Finetuning

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/221508458-bc480e52-a414-45d3-b124-3c23819b04d4.png" width="40%"></img></div>

작은 양의 fine-tuning만으로도 `LLaMA-65B`모델이 MMLU에서 성능 향상을 보였다.

그래도 아직 SOTA모델과는 꽤 차이가 나는 성능이다. (SOTA는 77.4%, GPT code-daninci-002)

<br></br>
