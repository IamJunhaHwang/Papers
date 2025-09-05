## :page_facing_up: LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

### Info 

* `publication`: ICLR 2022
* `author`: Edward Hu, Yelong Shen et al.

* `url`: https://openreview.net/pdf?id=nZeVKeeFYf9
* `github`: https://github.com/microsoft/LoRA

### 1. Abstract & Introduction

<div align="center"><img src="https://github.com/user-attachments/assets/95dbb940-c951-4549-ad69-7951f1d84d33" width="30%"></img></div>

NLP의 많은 applications가 하나의 큰 pre-trained model의 전체 파라미터를 fine-tuning하는 것으로 이루어져 왔다. 하지만 모델의 크기가 점점 커져가면서, 이러한 full parameter tuning에 어려움을 겪고 있다.

많은 연구들이 몇몇 parameter만 adapting하거나 외부 모듈을 학습하는 식으로 완화했지만, inference latency 증가나 sequence length가 줄어드는 문제가 있으며 효율과 모델 성능간의 trade-off가 있다.

본 논문에서는 모델의 low intrinsic dimension에 실제로 학습된 정보들이 존재한다는 것을 보여준 연구에 영감을 받아, model adaptation동안 weights를 바꾸는 것 또한 low intrinsic rank로 가능하다는 가설을 세운다(즉, 적은 차원의 linear operation으로 model을 충분히 tuning할 수 있다).   
이를 바탕으로 **Low-Rank Adaptation(LoRA)** 를 제안하며, LoRA는 pre-trained weights를 고정하고, 저랭크 분해 행렬을 최적화함으로써 몇 개의 dense layer를 간접적으로 학습한다.

GPT-3 175B를 예로 들면, 전체 랭크(d)가 12,288처럼 매우 높더라도 매우 낮은 랭크(r)(1 or 2)만으로도 충분하다는 것을 보인다. 따라서, LoRA는 저장 공간과 연산 측면에서 모두 효율적이다.

- LoRA Advantages

  - pre-trained model을 freeze하고 행렬 A와 B를 바꾸는 것으로 효율적으로 task switching 가능

  - 대부분의 파라미터를 최적화하지 않고, 훨씬 작은 low-rank 행렬만 최적화하므로 메모리에 효율적이다.

  - 간단한 linear design을 통해 trainable metrices를 frozen weights와 합칠 수 있다(이에 따라 latency 증가 없음).

<br></br>

### 2.  AREN’T EXISTING SOLUTIONS GOOD ENOUGH?

parameter-efficient tuning에 대한 problem은 새로운 문제가 아니다. 이를 위한 여러 연구들이 진행되어 왔으며, 크게 `adding adapter layers` 와 `optimizing some forms of the input layer activation(ex. prefix tuning)` 이 존재한다. 하지만 2가지 모두 large-scale에서 특히 한계가 있다.

<div align="center"><img src="https://github.com/user-attachments/assets/cd957769-9fdb-40bd-b3b0-b35dcb2baeb6" width="70%"></img></div>

#### Adapter Layers Introduce Inference Latency

많은 variants가 있지만, 여기에서는 original design(Houlsby et al., 2019)와 최근 것(Lin et al., 2020)에 집중했다.   
overall lataency를 줄이기 위해 pruning과 multi-task settings와 같은 방법이 있지만 adapter layer의 extra compute를 우회할 수는 없다. adapter는 적은 파라미터로 이루어져 문제가 될 것 같지 않지만, large-scale에서는 문제가 될 수 있다.

large-scale 모델은 latency를 줄이기 위해 hardware parallelism에 의존하는데 adapter layer는 순차적으로 처리되어야하므로 이러한 parallelism이 없다면 latency가 크게 늘어난다(위 표의 batchsize==1 의 경우).

#### Directly Optimizing the Prompt is Hard

다른 방법으로 prefix tuning이 있다. prefix tuning은 optimize가 어려우며 performance가 단조 증가하지 않고 흔들린다.   
또한, adaptation을 위해 sequence length의 일부를 잡아먹는 것은 downstream task 성능을 줄인다고 의심한다.

<br></br>

### 3. OUR METHOD: LoRA

#### 3-1. LOW-RANK-PARAMETRIZED UPDATE MATRICES

모델은 행렬곱을 수행하는 dense layer로 이루어져 있고, 이러한 행렬들은 full rank이다. (Aghajanyan et al., 2020)은 pre-trained LM이 low intrinsic dimension을 가지고 있다는 것을 보였고 이는 작은 subspace로의 random projection으로도 모델을 효율적으로 학습할 수 있음을 이야기한다.

이에 영감을 받아, adaptation에서의 weight updates도 또한 low intrinsic rank를 가지고 있다고 가정한다.   

- 사전학습 가중치 행렬 $W_0 \in \mathcal{R}^{dxk}$ 의 업데이트를 low-rank decomposition으로 표현해 제한한다.

  - $B \in \mathcal{R}^{dxr}$
  
  - $A \in \mathcal{R}^{rxk}$
  - rank $r \ll min(d, k)$

  - 학습동안, $W_0$ 는 frozen되며 A와 B는 trainable parameter를 포함한다.

- $W_0$ 와 $\Delta W = BA$ 동일한 input과 곱해지며, 그 결과로 나온 출력 벡터들은 좌표별(element-wise)로 더해진다.

  - $h = W_0x$ 에 대해 바뀌는 forward pass는 아래와 같다.

    - $h = W_0x + \Delta Wx = W_0x + BAx$

- 행렬 A는 가우시안 초기화, B는 0으로 초기화한다. 따라서, 학습 초기에는 $\Delta W = BA$ 는 0이다.

- 또, $\Delta Wx$ 를 $\frac{\alpha}{r}$ 로 스케일링한다.

  - 여기서 α는 r에 대한 상수이다. 
  - Adam으로 최적화할 때, 초기화를 적절하게 스케일링한다면 α를 조정하는 것은  학습률을 조정하는 것과 거의 같다. 여기에서는, 단순히 처음 시도하는 r 값을 α를 설정하고 따로 조정하지 않았다. 이러한 스케일링은 r을 변경할 때 하이퍼파라미터를 재조정할 필요성을 줄여준다 (Yang & Hu, 2021).


- A Generalization of Full Fine-tuning

  - 일반적인 fine-tuning은 pre-trained parameter 일부를 학습하는 것이다. LoRA는 더 나아가 full rank의 weight matrices를 업데이트할 필요없다.
    - LoRA를 전체 weight matrices에 적용하고 all biases를 학습했을 때, LoRA의 rank r을 pre-trained weight matrics의 rank와 같게 설정하면 full fine-tuning의 표현력과 거의 비슷하게 회복할 수 있음.

  - trainable parameter를 늘릴수록 LoRA는 원래 모델 학습과 비슷해지는 반면, adapter기반 방법은 MLP에 수렴하고 prefix 기반 방법은 긴 입력 시퀀스를 처리할 수 없다.

- No Additional Inference Latency

  - 배포할 때, $W = W_0 +BA$ 를 저장해서 평소처럼 추론할 수 있으며, 다른 downstream task로 전환 할 때는 $BA$ 를 빼고 다른 $B'A'$ 를 더하는 것으로 빠르게 작업이 가능하다. (이는 추가 latency가 생기지 않는 것을 보장함)


- APPLYING LORA TO TRANSFORMER

  - LoRA를 어떤 가중치 행렬이던지 적용할 수 있지만, 여기에서는 attention weights에만 적용했으며, MLP modules는 freeze했다.

  - attention heads에 의해 나누어지지만, Q, K, V attention matrices를 각각 single matrix로 봄

<br></br>

### 4. EMPIRICAL EXPERIMENTS

- model: RoBERTa, DeBERTa, GPT-2, GPT-3 175B

- tasks

  - RoBERTa, DeBERTa : GLUE

  - GPT-2: E2E NLG Challenge
  - GPT-3: WikiSQL, SAMSum


- baselines (이전 연구에서 보고된 score 포함됨)

  - Fine-Tuning (FT), 마지막 2개의 layer만 tuning하는 ($FT^{Top2}$)

  - Bias-only or BitFit: bias만 training하괴 나머지는 모두 freeze

  - Prefix-embedding tuning (PreEmbed): input tokens 간에 special token 삽입(trainable word embeddings이며 model vocab에는 없는); prefixing과 infixing에 집중함

  - Adapter tuning

    - 2가지 FFNN을 adapter layer에 쓰는 original design `Adapter^H`
    - MLP module 과 LayerNorm 후에만 붙이는 `Adapter^P`
    - 몇몇 adapter layers를 drop하는 AdapterDrop `Adapter^D`

<div align="center"><img src="https://github.com/user-attachments/assets/e9994632-c685-4482-b323-3a8953992b3b" width="70%"></img></div>


#### Model Setup

- RoBERTa Base/Large

  - adapter와 비교를 위해 adapter baseline과 같은 batch size와 128 sequence length 사용
  - MRPC, RTE, STS-B에는 pre-trained model로 초기화했다.

- DeBERTa XXL: LoRA가 fully fine-tuned DeBERTa XXL의 성능과 일치할 수 있는지 평가

<div align="center"><img src="https://github.com/user-attachments/assets/8d5761d2-5232-4baf-bc79-b0f36ec505a2" width="70%"></img></div>

- GPT-2 Medium/Large: LoRA가 NLU full fine-tuning의 대체가 될 수 있는지 보이기 위해 실험

<div align="center"><img src="https://github.com/user-attachments/assets/ff3079ec-b547-4bf4-99ae-d0f6ef48b6ae" width="70%"></img></div>


- GPT-3 175B

  - LoRA가 모든 데이터셋에서 fine-tuning baseline을 능가했다.

  - 모든 adaptation methods가 parameter가 늘어날수록 단조적으로 성능이 증가하지는 않았다. [아래 그림]

    - 256보다 더 많은 special tokens를 쓸 때(prefix embedding tuning), 32개의 special tokens 이상 쓸 때(prefix layer tuning) 성능이 떨어지는 것을 확인했다.

      - 많은 special tokens는 input distribution을 pre-training data distribution에서 좀 더 멀게 옮기기 때문으로 추측

<div align="center"><img src="https://github.com/user-attachments/assets/2659498d-d12f-41cf-ab3e-60abd914696e" width="70%"></img></div>
