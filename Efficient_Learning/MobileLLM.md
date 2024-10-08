## :page_facing_up: MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases


### Info

* `publication`: ICLR 2024
* `author`: Zechun Liu et al.
* `url`: https://openreview.net/pdf?id=EIGbXbxcUQ

### 1. Abstract & Introduction

이 논문에서는 mobile device를 위한 efficient LLM으로써, billion parameter 보다 적은 top-quality LLM의 설계에 집중한다.   
이전의 데이터와 파라미터 양이 늘어날수록 모델의 질도 늘어난다는 만연한 믿음에 반대되게, 우리는 sub-billion 규모의 LLM을 위한 모델 구조의 중요성을 강조한다.

embedding sharing과 grouped-query attention과 함께 깊고 얇은 구조를 활용해, 우리는 `MobileLLM` 이라는 strong baseline을 제시한다. 이 모델은 125M/350M SOTA model들보다 2.7%/4.3% 정확도 향상이 있었다.

추가로, 미미한 latency overhead만 있고 추가적인 model 크기 증가 없이 immediate block-wise weight(layer) sharing을 제안한다. 그 결과 추가적인 성능 향상이 있었다.

또한, chat benchmarks에서 이전 sub-billion 모델과 비교했을 때 상당한 성능 향상을 보였으며 LLaMA-v2 7B에 가까운 정확성을 보였음.

- contributions

  - small LLM에서 width보다 depth가 더 중요하다는 것을 증명 

  - 가중치 활용을 최대화하기 위해, embedding sharing 방법을 다시 거론하고 grouped query attention을 구현했음

  - immediate block-wise weight(layer) sharing을 제안함.

    - 메모리 이동이 latency bottleneck인 시나리오에서, 두 인접 블록(adjacent blocks)간의 가중치 공유가 가중치 이동을 피하게 함. 이렇게 하면 해당 블록을 두 번만 계산하면 되므로 지연 시간 오버헤드가 최소화됨.

  - `MobileLLM`을 제안

  - Chat, API calling과 같은 downstream task에서 MobileLLM은 동일 크기의 모델들을 능가함

  - 본 논문에서 제안한 설계를 더 큰 모델에도 적용가능함. (Appendix A)

<br></br>

### 2. Improving Sub-billion Scale LLM Design


<div align="center"><img src="https://github.com/user-attachments/assets/0340d5f5-7d37-46e8-9211-64b0ea565877" width="35%"></img></div>

model size가 주된 제약인 on-device를 위해 어떻게 한정된 가중치 파라미터들을 효과적으로 할당하는 방법이 중요하다.   

- 우리는 4가지 모델 설계 테크닉을 테스트해 **MobileLLM** 이라는 baseline을 제안한다.

  - 1) SwiGLU FFN 적용, 2) lanky(deep and thin) 구조, 3) embedding sharing, 4) grouped query attention

* 그리고 immediate block-wise layer-sharing 방법을 제안한다. (이 방법을 적용한 모델을 MobileLLM-LS라 명명)

#### 2-1. Training setup

- GPU: A100 32개

- 480K iterations on 1T tokens

- 평가 데이터셋 : 1) zero-shot common sense reasoning tasks(ARC-easy, ARC=challenge, BoolQ, PIQA, SIQA, HellaSwag, OBQA, WinoGrande), 2) question answering and reading comprehension tasks(TQA, RACE)

#### 2-2. Building a Strong Baseline

- FEED-FORWARD NETWORK(FFN) CHOICE

  - FFN에서 쓰이는 활성화 함수를 ReLU에서 `SwiGLU`로 바꿈. (zero-shot reasoning에서 42.6 -> 43.9 향상)

<div align="center"><img src="https://github.com/user-attachments/assets/9228dd96-ad7a-42f5-b696-d0916a438ada" width="75%"></img></div>

- ARCHITECTURE DEPTH VS WIDTH

  - transformer model의 성능은 dataset size, model size, training iterations로 결정되며, 모델 구조는 거의 영향을 끼치지 않는다는 만연한 믿음(Kaplan et al., 2020)은 small model에서는 성립하지 않는다는 것을 실험으로 찾아냄.

  - 9개의 125M, 10개의 350M의 모델을 같은 size를 같되, layer를 얇고 깊게 만들거나 넓고 얕게 만드는 식으로 비교하였음.

  - 실험 결과 그림을 보면, 깊게 얇은 모델이 넓고 얕은 모델보다 좋은 성능을 보였으며, RACE와 TQA dataset에서 두드러지게 보였음.

  - **이러한 발견은, 약 125M정도의 모델에서 12개의 layer를 가지는 것보다 30이나 43 layer를 가지게 하는 것이 성능이 더 좋다는 것을 의미**

    - 가장 최근의 125M model은 layer가 12개였음

<div align="center"><img src="https://github.com/user-attachments/assets/c03fd2a9-e409-4297-9741-9e58dfe20411" width="75%"></img></div>
    

- EMBEDDING SHARING

  - embedding layer는 상당한 양의 파라미터를 포함하는 부분이다. 예를 들어, 125M의 모델에서 vocab이 32k이고 embedding demension이 512이라면 16M의 파라미터들이 생성되고 이는 전체의 20%가 넘는다.

  - 이런 이유로, OPT model(Zhang et al., 2022)에서 input-output embedding sharing이 처음 제안되었지만, LLM에 들어서서는 이러한 embedding parameter가 전체의 3.7%(LLAMA-7B), 0.7%(LLAMA-70B)를 차지하므로 무시되었었다.

  - sub-billion scale LM을 만드는 본 논문에서는 이러한 embedding sharing concept을 다시 사용한다; input embedding weights를 output fully connected layer weights로도 사용.

  - 30-layers 125M model로 실험한 결과(Table 1), input-output embedding으로 16M의 파라미터를 사용하였으며 이는 전체의 11.8%이다. 또한, 평균 성능이 0.2 떨어졌지만, 이는 layer를 추가하는 것으로 성능이 점차 다시 올라갈 수 있다(layer 2개를 추가하는 것은 original 135M 모델보다 10M 적음).

  - **이러한 발견은 embedding sharing이 한정된 model storage budget에서 가중치 활용도를 최대화하고 모델 성능을 최적화할 수 있음을 의미한다.**

<div align="center"><img src="https://github.com/user-attachments/assets/b53c73ad-8f5d-4142-99aa-c21a2b1e0072" width="40%"></img></div>
    
- NUMBER OF HEADS AND KV-HEADS

  - 헤드 차원당(갯수가 많을수록) 더 많은 의미를 제공하는 것과 여러 헤드의 더 많은 비선형 조합 간의 균형은 헤드 크기를 선택할 때 중요한 고려 사항이며, 이전에는 대체로 query heads와 key-value heads의 수를 같게하였다.

  - Grouped query attention(GQA)는 LLM의 key-value cache size를 줄이기 위해 제안되었으며(Chowdhery et al., 2023) small LM에서도 효과적으로 불필요한 중복을 줄일 수 있다.

    - 그룹 쿼리 어텐션(GQA)은 또 다른 형태의 가중치 공유로 볼 수 있으며, 이 방법에서는 key-value heads의 수가 query head의 1/n이고, kv-heads는 쿼리와 함께 어텐션 점수와 출력을 계산할 때 n번 반복된다. 여기서 $n \in \mathbb{Z}^+$ 은 query heads의 수와 나누어떨어지는 양의 정수를 나타낸다.

  - 최적의 head size를 결정하기 위해 125M과 350M에서 실험한 결과(Figure 5), 16개의 query heads를 사용하는 것이 가장 성능이 좋았으며, 여기에 추가로 kv-heads를 4로 줄여도 125M에서는 비슷한 성능이 350M에서는 0.2정도의 성능 하락밖에 없었다(모델 크기는 10% 줄이면서).

    - 여기에서, 모델 크기를 유지하기 위해(125M, 350M) 임베딩 차원을 늘렸을 때, 125M에서는 0.4의 성능 향상이 있었음.

  - **GQA는 small model의 잠재성을 최대화하는 적합한 방법이다.**


#### 2-3. Layer Sharing

<div align="center"><img src="https://github.com/user-attachments/assets/61bf61e8-b0df-4cb3-8178-f90b4a20dd35" width="40%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/335f849d-0ede-40ce-bec0-1f16cb12046f" width="70%"></img></div>

    
- 추가적인 model storage cost 없이 hidden layer를 늘리는 전략으로써 layer sharing을 실험했다.

- 실험 결과, `repeat-all-over`가 성능이 가장 높았지만 하드웨어 사용 효율을 고려해 `Immediate block-wise sharing`을 모델 설계로써 채택했다. (성능 차이도 거의 없다)

<br></br>

### 3. Experiments

- Experimental settings

  - optimizer : Adam with weight decay of 0.1

  - GPU : 32 A100 with a batch size of 32 on each GPU

  - learning rate : 2e-3, follows a cosine learning-rate decay

  - best model training : 480K iterations on 1T tokens

  - 비교 모델 : HuggingFace에 공개된 model 사용

    - OPT, BLOOM, Galactica, Cerebras, GPT-neo, Pythia, RWKV

#### 3-1. Main Results

<div align="center"><img src="https://github.com/user-attachments/assets/4bec7e20-c703-4e1d-ada5-6e704a0ac18b" width="80%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/9f1b4ca9-a949-481c-9cba-fb44afed29cd" width="40%"></img></div>

- Zero-shot Common Sense Reasoning 

  - MobileLLM이 이전 모델들보다 높은 성능을 달성했다.

  - MobileLLM-LS-125M은 이전의 350M 모델보다 size가 더 작음에도 불구하고 비슷하거나 높은 성능을 달성하였으며 MobileLLM-350M은 이전 SOTA 모델을 능가했다.

- Question Answering and Reading Comprehension : TQA, RACE benchmark

  - MobileLLM-125M은 이전 모델들보다 TQA 벤치마크에서 4.3 이상의 성능 향상이 있었다. 
  
  - MobileLLM-350M 모델은 다른 350M 크기 모델과 비교하여 약 10포인트의 상당한 성능 향상이 있었다. 
  
  - RACE에서도 MobileLLM 모델들은 이전 모델보다 훨씬 높은 점수를 기록했다.

#### 3-2. Downstream Tasks

on-device applications에서의 효과를 평가하기 위해 2가지 중요한 on-device tasks에서 모델을 평가하였음 : Chat, API calling

<div align="center"><img src="https://github.com/user-attachments/assets/87dc1026-6670-44b9-bf3b-c8033a82919e" width="80%"></img></div>

- CHAT

  - MobileLLM과 이전 SOTA 모델들을 chat-based task를 위해 fine-tune하여 AlpacaEval과 MT-Bench에서 평가하였다.

  - 실험 결과, MobileLLM이 이전 SOTA 모델들보다 성능이 상당히 좋았으며, 1B모델 또한 능가하였다.

  - MobileLLM-LS-350M은 GPT-3 model(text-davinci-001)과 비교했을 때, 48.2%의 win rate를 달성했다.

    - GPT-3의 self-win rate는 50%인 것을 고려하면, MobileLLM-LS-350M은 GPT-3에 견줄만한 성능을 가지는 것을 의미

- API CALLING

  - API calling은 NLP input을 JSON configuration으로 바꾸는 것을 포함한다.

    - ex) "Help me set an alarm at 7:30AM"이라는 입력이 들어오면 모델은 출력으로 `{API: "alarm(timie="7:30am")"}` 과 agent response("Sure! Your alarm is set to 7:30 AM.")를 생성해야 한다.

  - 해당 task 실험을 위해, 저자들은 5000개의 training sample과 2500개의 test sample을 만들었으며 각 sample은 평균 8개의 대화 턴을 포함한다. 모델들은 해당 training set에 4 epoch 훈련되었다.

  - 실험 결과, MobileLLM-350M은 LLAMA-v2 7B와 비슷한 exact match score를 보였다.

  - Rouge score의 경우 LLAMA와 비교해서 낮은 점수를 보였지만, API calling task에서는 올바른 API를 불러오는 것이 더 중요하며, 비슷한 모델 크기에서는 MobileLLM이 성능이 더 좋았다.

  - 온디바이스 애플리케이션의 특정 일반적인 시나리오를 MobileLLM-350M과 같은 작은 모델도 이를 능숙하게 처리할 수 있음을 의미한다.

#### 3-3. On-device Profiling

<div align="center"><img src="https://github.com/user-attachments/assets/9f5b8830-4214-4cf2-9cb8-f82fcaa05d8f" width="40%"></img></div>

- MobileLLM-125M과 MobileLLM-LS-125M FP16 model을 ExecuTorch를 통해 iPhone 13, Metal Performance Shaders backend에서 latency를 측정했다. 

- 실험 결과, 가중치 공유와 레이어 수를 두 배로 늘림으로써 MobileLLM-LS가 MobileLLM에 비해 로딩 및 초기화 시간이 2.2%만 증가했다. 

- 반면, 가중치 공유를 하지 않은 모델의 경우, 로딩 및 초기화 시간이 143%나 증가하고 실행 시간은 86% 증가했다.


<br></br>
