## :page_facing_up: Honeybee: Locality-enhanced Projector for Multimodal LLM

### Info

* `publication`: CVPR 2024

* `author`: Junbum Cha et al.
* `url`: https://openaccess.thecvf.com/content/CVPR2024/papers/Cha_Honeybee_Locality-enhanced_Projector_for_Multimodal_LLM_CVPR_2024_paper.pdf


### 1\. Abstract & Introduction

Multimodal Large Language Models (MLLMs)에서 `visual projector`는 pre-trained vision encoder와 LLM을 연결하는 중요한 역할을 한다. 이러한 visual projector의 중요성에도 불구하고, 이에 대한 연구는 상대적으로 적다.

본 논문에서는 2가지 중요한 projector properties를 먼저 찾아낸다: 1) MLLMs의 전체 효율에 중요한 visual token 개수 조절에서의 유연성(flexibility), 2) 공간 이해에 중요한 visual features에서 온 local context의 보존 (locality-enhancing)   
이러한 findings에 기반해 `flexible` 하고 `locality-enhanced`된 novel projector design을 제안한다. (2가지 locality modeling인 convolution 과 deformable attention을 적용을 도입) ==> **Honeybee**   

최근 MLLM을 훈련하는 흔한 방법으로 2가지가 있다: 1) LLaVA와 같은 GPT-assisted instruction-following dataset, 2) instructization process로 만든 vision-language task dataset   
위와 같은 데이터셋의 장점을 극대화 하기 위해, 우리는 다음을 위해 중요한 design choices를 제안한다: 1) 어떻게 다각적인 instruction data를 이용하는지, 2) instructization process를 위한 효과적인 방법 

실험을 통해 각 design choices의 효과를 실험하고, 제안한 MLLM인 Honeybee가 이전 SOTA methods보다 뛰어났음을 여러 벤치마크 데이터셋에서 보였다.

- **Contributions**

  - projector의 2가지 중요한 특성을 찾아내고 : 1) visual features의 locality 보존, 2) visual tokens 개수를 조절하기 위한 유연성, 이 2가지 특성을 모두 만족시키기 위한 locality-enhanced abstractor를 제안함

  - 다각적인 데이터셋과 instructization process 에서의 이익을 최대화시키기 위한 효과적인 방법 제안

  - locality-enhanced projector와 hidden recipes를 통해, 본 논문에서 제안한 `Honeybee` 가 여러 MLLM benchmarks에서 SOTA를 달성

<br></br>

### 2. Honeybee: Locality-enhanced MLLM

#### 3.1 Overview

일반적으로, MLLMs의 목표는 주어진 multimodal inputs에 대해 instruction을 따르는 대답을 만드는 것이다. 본 논문에서는, images를 추가적인 modality input으로 간주하여 language model이 visual & text(instruction) token을 모두 받을 수 있게 된다 (auto-regressive 방법으로 text response 생성).   

- multimodal input은 2가지 타입의 토큰으로 구성 : image tokens $X_{img}$, text tokens $X_{text}$

- language model은 multimodal input에 대해 response $Y = \{ w_i \}^L_{i=1}$ 를 예측한다. 

  - `L`은 response 내의 토큰 갯수를 의미

- 그러므로 response는 다음에 의해 예측된다 : $p(Y|X_{img}, X_{text}) = \prod_{i=1}^L p(w_i|X_{img}, X_{text}, w_{<i})$

- **Architecture** : MLLMs는 1) vision encoder, 2) projector, 3)LLM 으로 구성된다.

  - vision encoder : 이미지 이해를 위한 region-level visual feature의 sequence 제공

  - projector : visual feature를 LLM에 전달하기 위해 visual token으로 전환하는 역할

  - LLM : visual 과 instruction 합쳐진 token을 처리하여 autoregressive하게 응답 생성


* 기존 projector들의 장/단점

  - linear projector : 간단하면서도 효과적이다(vision encoder 정보 잘 보존) / scalability와 efficiency에서의 어려움이 있음 (one-to-one transformation 때문).

  - abstraction : 일반적으로 M learnable query(visual feature N개보다 작은)를 사용하므로 효율적이다(유연하게 적용 가능) / information loss가 존재

<div align="center"><img src="https://github.com/user-attachments/assets/194a13d9-30b9-46a4-8753-7f741679128c" width="80%"></img></div>


#### 3.2 Locality-enhanced Projector

이 섹션에서는 locality-enhanced projector의 motivation을 설명하고 2가지 타입의 projector (C-Abstractor, D-Abstractor)를 제시하며, training pipeline을 설명한다.

- Motivation

  - projector를 결정하는 것에 있어 visual token 수를 조절하는 flexibility를 가지는 것이 계산량을 결정지으므로 효율을 위해 linear보다 abstractor를 선택함

  - 이러한 abstractor 중에서 resampler의 경우, 공간 이해 능력이 부족한 것을 관찰하여서 이러한 방식은 제외하였음(특정 부분의 정보에 대해 요약하는 경향이 있었음).

  - 이에 기반해 새로운 visual projector인 `C-Abstractor` 와 `D-Abstractor` 를 제안함. 이 projector는 visual token 수를 조절하는 유연성과 local context를 효율적으로 보존하도록 설계됨.

<div align="center"><img src="https://github.com/user-attachments/assets/247d3286-6c74-4b28-8d90-89662301bf60" width="40%"></img></div>


- C-Abstractor(Convolutional Abstractor) : Convolution 이용

  - L개의 ResNet Block을 거친 후, adaptive average pooling을 진행하고 다시 또 다른 L개의 ResNet Block을 거치도록 구성

  - visual feature를 임의의 제곱 수의 시각적 토큰으로 추상화 할 수 있으며, 원래의 visual feature보다 더 많은 시각적 토큰으로도 만들 수 있음

- D-Abstractor(Deformable attention-based Abstractor)

  - convolution이 locality에 대한 inductive bias를 과하게 도입할 수 있으므로 deformable attention을 이용한 Abstractor를 제안한다(유연성을 유지하면서 locality-awareness 향상).

  - deformable attention은 local context를 유지하고, 각 학습 가능한 쿼리는 reference points 근처를 중심으로 reference points와 샘플링 오프셋을 활용하여 2D 좌표 기반 샘플링 프로세스를 통해 visual feature를 수집

  - 여기에서, reference points를 수동으로 초기화하여 전체 피처 맵에 균일하게 분포시키는 초기화 방법 제안

#### 3.3 Training

<div align="center"><img src="https://github.com/user-attachments/assets/d741081c-36c1-4a4e-885f-0fe3d6f3e608" width="40%"></img></div>


Honeybee는 2단계에 걸쳐 학습된다 : 1) vision encoder와 LLM을 freeze시키고 projector만 학습, 2) projector와 LLM 모두 학습

- Pretraining for vision-language alignment

  - image-text 데이터를 사용해, pre-training은 MLLM이 visual cue와 text description이 어떻게 align되는지에 대한 세밀한 이해를 발전시킬 수 있게 한다.

  - pre-training 동안, vision encoder와 LLM은 freeze된다(이미 만들어진 각 모델의 근본적인 이해를 유지하도록).

- Visual instruction tuning

  - projector의 pre-training 이후에는 projector와 LLM을 함께 학습시킨다.

  - instruction-following을 위해, 2가지 GPT-assisted instruction following dataset을 사용한다(LLAVA, ShareGPT).

  - visual understaing을 향상시키기 위해, 존재하는 넓은 범위의 데이터 셋들을 템플릿을 사용해 instructize시켰으며 이에 사용한 데이터는 위 표와 같다.

  - 1) 다양한 task들을 적용(위 표의 Task 참고), 2) 각 task에서 여러 dataset 사용, 3) 각 데이터셋에 대해 세세한 단일 템플릿을 적용
  
<br></br>

### 4. Hidden Recipe for Visual Instruction Tuning

instructization을 기반으로 템플릿을 이용해 존재하는 데이터 셋에 대해 instruction tuning을 하는 것이 유익하다고 잘 알려져 있지만, instructization process의 세부적인 사항들은 덜 연구되었다.   

- 따라서, 이 section에서는 다음의 5가지 질문을 통해 이를 명확히 하고자 한다

  1) 각 데이터셋이 특정 task의 성능에 얼마나 기여하는가?  
      - 데이터셋을 여러 task 그룹으로 분류한 후, instruction tuning 동안 각 task 그룹을 순차적으로 제외하여 벤치마크 성능의 변화를 조사
  2) 다양한 데이터셋 사이의 효과적인 균형을 맞추는 방법은 무엇인가?  
      - 다음의 5가지 균형 전략 비교: 1) per-dataset: 각 데이터셋에 대해 uniform 샘플링, 2) per-task: 각 작업에 대해 uniform 샘플링, 3) per-sample-100k: 각 샘플에 대해 uniform 샘플링을 하되 각 데이터셋의 최대 크기를 100k로 클리핑, 4) per-dataset tuned: per-dataset 전략을 기반으로 경험적으로 조정된 balancing.
  3) 템플릿의 적절한 세분화 수준은 무엇인가?  
      - 2가지 다른 템플릿을 비교 : 1) 각 데이터 셋에 대해 고유한 템플릿 적용 (fine-grained), 2) 같은 task category의 데이터셋간의 공유하는 템플릿 적용 (coarse-grained)

  4) 템플릿의 다양성이 얼마나 중요한가?  
      - 1) single template, 2) multiple templates, 3) multiple templates and input inversion 비교
  5) 대화형 멀티턴 템플릿이 추가적인 이점을 제공하는가?
      - 하나의 이미지에 대해 여러 input-target pairs가 있는 경우가 흔히 있는데, 멀티턴 전략은 이러한 쌍들을 하나의 대화형 멀티턴 예시로 통합한다. 하지만 이러한 접근은 의미적으로 중복되는 input-target pairs을 하나의 예시로 병합하여 단순한 shortcut을 통한 답변을 찾게 될 수 있다. 이를 방지하기 위해, 우리는 멀티턴 예시에서 의미적으로 중복된 input-target pairs을 제거하는 추가적인 중복 제거 전략을 도입한다.

<br></br>

### 5. Experiments

#### 5-1. Settings

- 사용 Benchmarks : MME with perception tasks, MMBench-dev (MMB), SEED-Bench Image-only, LLaVA-Bench(In-the-Wild)

- Metrics : 각 benchmark별 official metric, 벤치마크들 간의 normalized average ${Avg}^N$

  - normalized average == 각 벤치마크에서 점수의 상한값으로 정규화된 점수의 평균으로 정의

- Implementation

  - LLM : 7B & 13B Vicuna-v1.5

  - CLIP ViT-L14 with 224 & 336 resolutions를 각각 7B, 13B LLM을 위해 사용

  - LLM full fine-tuning 진행

  - ablation을 위해, short training schedule (50k pre-training, 4k instruction tuning) with Vicuna-7B, CLIP ViT-L/14, and C-Abstractor with M=144 진행

    - final model은 long training schedule (200k pre-training, 10k instruction tuning)

#### 5.2 Analysis on Locality-Enhanced Projector

<div align="center"><img src="https://github.com/user-attachments/assets/413734eb-e016-43cc-afb3-0c5ff1e7a0c0" width="40%"></img></div>

제안한 projector와 존재하는 projector의 성능과 효율을 비교하고 평가하였으며, 결과는 위 표와 같다.

- Resampler (B2, B5)는 local context preservation을 고려하는 능력이 부족하기 때문에 안좋은 성능을 보임

- linear projector는 `M=256`만 가능한 한계(B1)가 있음 (inflexibility).

- 같은 계산량(M=256)에서 C-Abstractor가 linear보다 좋은 성능을 보였으며(B4 vs B6), 더 적은 토큰 수(M=144)에서도 C-Abstractor가 약간 더 좋았음(B4 vs B3).

  - 이는 locality-enhanced projector가 주변 features로부터 local context를 잘 통합하고 context가 풍부한 visual token을 제공해 시각적 특징을 추상화하는 데 뛰어나다는 것을 의미

#### 5.3 Hidden Recipe for Visual Instruction Tuning

##### Dataset combination

<div align="center"><img src="https://github.com/user-attachments/assets/fb3b0cd1-448f-4110-bf40-81e43592c06b" width="70%"></img></div>

위 표는 다양한 multimodal benchmarks에서 각 데이터셋의 영향을 알아내기 위한 ablation이다.

- 각 task group의 단 하나의 dataset을 활용하는 것으로 각 task안의 dataset 다양성의 영향을 조사하였으며(D1 vs D2), 그 결과로 전체 성능은 낮아졌으며, 이는 각 task내의 dataset 다양성이 중요함을 의미

- 특정 task를 제외해보는 것으로 각 task의 영향을 보았으며(D1 vs D3-8), 이는 task 다양성이 다양한 task들을 어떻게 다루는지 배우는 것에 중요하다는 것을 밝혀냈음.

  - 각 task는 관련된 벤치마크의 성능을 높였음 : VQA (Open) -> MME, VQA (MC) -> MMB and SEED, captioning and instruction-following data -> LLaVA

- 존재하는 vision-language data를 사용하는 것에 대한 영향을 검사했으며(D9 vs D10), 사용하지 않을 경우 MME, MMB, SEED 벤치마크에서 상당한 성능 저하가 있었음.

  - 존재하는 vision-language data 내의 풍부한 지식은 MLLM의 인식 이해와 시각적 추론 능력을 향상시킬 수 있음을 시사함

<div align="center"><img src="https://github.com/user-attachments/assets/acde5309-8bb0-425e-a87e-12a3510b26e7" width="80%"></img></div>

##### Dataset balancing

- 2가지 원칙에 따라 각 데이터의 균형을 조정했다 : 1) 작은 데이터셋에서는 제한된 epochs 사용, 2) 핵심 데이터셋의 경우 최대 몇몇 epochs까지 허용

- Table 5a는 수동으로 조정한 per-dataset-tuned 방법의 효과를 증명한다. 직접 만든 방법을 제외하고는 `per-dataset`이 가장 괜찮았다.

##### Instruction tuning vs. multi-task learning.

- Table 5b는 간단한 Identifier를 사용하는 멀티태스크 학습보다 템플릿 기반 포맷을 통한 instruction tuning의 장점을 보여준다. 이러한 결과는 이전 연구와 일치한다.

##### Template granularity & diversity

- Table 5c는 fine-grained template의 coarse-grained template보다 성능이 좋음을 증명한다(1, 2번째 행).

- Table 5c의 1,3,4행을 통해 template diversity의 영향을 비교하였다: single, multi, multi+flip

  - 템플릿의 다양성을 높이는 것이 모델 성능 향상을 불러온다고 보장할 수 없다는 것을 밝혀냄.

##### Multi-turn template

*  Table 5d는 Multi-turn template과 de-duplication의 효과를 보여준다.

- 그 결과로, 의미적으로 겹치는 pairs를 지우는 것이 shortcut training을 완화하는 것에 효과적임을 보였다.

##### Final recipe

1) C-Abstractor or D-Abstractor 적용

2) 다양한 tasks에 대해 다양한 datasets 활용

3) Table 5의 회색으로 표시된 옵션들 사용: the application of per-dataset balancing with hand-crafted tuning, fine-grained templates, and multi-turn interactions with deduplication

<div align="center"><img src="https://github.com/user-attachments/assets/e62dc631-a713-4a11-92db-ff76c7c8c40b" width="80%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/ca66c95f-04a7-4f46-bfe2-7e1a7523c952" width="40%"></img></div>



#### 5.4 Putting It Altogether

final recipe와 long training schedule을 이용해 학습한 Honeybee를 다른 SOTA MLLM과 비교하였다.

- Honeybee는 SEED를 제외한 모든 벤치마크에서 7B 수준의 MLLMs을 능가하였다. 

- 큰 vision encoder(ViT-bigG for Qwen-VL)나 더 많은 visual token으로 더 큰 이미지(448 and 336)를 사용한 Qwen-VL와 LLaVA-1.5 보다 Honeybee가 성능과 효율측면에서 좋은 균형을 이루었다(위 그림 참고).

- 세세한 시각적 이해가 필요한 SEED와 같은 task의 경우 더 큰 이미지를 사용하거나 더 많은 visual token을 사용했을 때 이점이 있었다.

  - 더 많은 visual token을 사용하면 성능은 증가하지만, 효율은 낮아졌다(계산량 증가하므로).

- 13B로 규모를 올렸을 때, Honeybee가 모든 이전 방법들을 모든 벤치마크에서 능가하였다.
