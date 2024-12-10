## :page_facing_up: FROZEN TRANSFORMERS IN LANGUAGE MODELS ARE EFFECTIVE VISUAL ENCODER LAYERS

### Info

* `publication`: ICLR 2024
* `author`: Ziqi Pang et al.
* `url`: https://openreview.net/pdf?id=t0FI3Q66K5

### 1. Abstract & Introduction

> Can LLMs effectively handle tasks that are exclusively visual, without any reliance on language?

본 논문은 텍스트 데이터로만 학습된 LLM이 visual task만을 위한 encoder가 될 수 있음을 밝혔다. 이는 LLM의 frozen transformer block을 visual token을 직접 처리하는 encoder layer의 구성 성분으로써 사용하는 방식이다.

1) frozen LLM transformer block을 original visual encoder 위에 올린다.
2) LLM block의 전/후로 trainable linear layer를 두어 feature dimension을 align한다.
3) training 과정에서 LLM transformer를 freeze한다.

여기에서 visual encoder로 CLIP과 같이 사전 학습된 backbone이 필요하지 않고 처음부터 학습이 가능하다.

추가로, visual encoding에서 사전 학습된 LLM의 효과를 설명하기 위해 `information filtering` 이론을 제안한다. 이는 사전 학습된 LLM이 informative visual token를 알아차리고 이런 효과를 증폭시키는 것에 대한 내용이다. 해당 이론은 LLM transformer block을 사용한 학습 후의 feature activation이 관련된 영역에 더 강하게 집중하는 현상이 관찰됨으로써 뒷받침된다.

- Contributions

  - 사전 학습된 LLM의 frozen transformer block을 visual encoder로 사용해 여러 vision task를 풀 수 있음을 발견

  - visual encoding에서 frozen LLM transformer의 장점을 증명

  - visual token을 처리하는 frozen transformer의 효과를 설명하기 위한 `information filtering` 이론 제안


<br></br>

### 2. METHOD: FROZEN LLM TRANSFORMERS FOR VISUAL ENCODING

<div align="center"><img src="https://github.com/user-attachments/assets/6e2e4adf-5ce3-4254-8021-840cb3895c53" width="70%"></img></div>

- input x를 잠재 표현 z로 만들기 위한 encoder $F_E$ labels y를 예측하기 위한 decoder $F_D$를 고려한다.

  - $F_E(x) \rightarrow z, F_D(z) \rightarrow y$

- LLAMA와 같은 LLM의 단일 pre-trained transformer block을 $F_{LM}$ 이라 하며 encoder $F_E$ 와 decoder $F_D$ 사이에 들어간다.

- transformer block과 encoder 사이의 feature dimension이 다르므로 2가지 linear layer $F^1_L$ 과 $F^2_L$ 을 transformer block 전/후로 적용하여 차원을 align한다.

  - $F_E(x) \rightarrow z, F^2_L \cdot F_{LM} \cdot F^1_L(z) \rightarrow z', F_D(z') \rightarrow y$

- 학습 단계에서, $F_{LM}$ 은 frozen하고 나머지는 일반적으로 학습된다.

#### Comparison with vision-language models

- CLIP과 같이 pre-trained visual encoders에 의존하지 않고 처음부터 학습 가능하다.

- language-based input이나 prompts 없이 동작하며, visual representation learning에 적합하다.

- 전체 LLM을 내재된 모듈로 다루지 않고 각 transformer block을 visual encoding을 위한 독립적인 계층으로 나누었다.

  - 실험에서는 LLAMA-7B의 마지막 transformer block 사용



#### Comparison with LLMs.

- text data와 달리, 이미지는 전체를 한 번에 처리하므로, autoregressive mask를 제거하였고, attention mask는 padded token에만 적용하였다.

- rotary positional embedding은 visual encoder에서 흔하지 않으므로 이를 제거하여 original visual backbone과 일관되게 하였다.

<br></br>

### 3. APPLICABILITY OF LLM TRANSFORMERS FOR VISUAL TASKS

#### Image Classification (2D modality)

<div align="center"><img src="https://github.com/user-attachments/assets/da6de7be-4d0d-4f4c-a7d6-20e7c1739c3a" width="70%"></img></div>

- 평가 데이터 : ImageNet1k, ImageNet-C, ImageNet-A, ImageNet-SK, ImageNet-R

- encoder $F_E$ 로 널리 사용되며 자연스럽게 transformer를 support가능한 ViT를 선택함.

- baseline ViT와 ViT+LLAMA를 처음부터 학습시켰으며, [DeiT](https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf)의 설정을 따랐다.

- 실험 결과, ViT 모델의 정확도가 frozen LLAMA transformer block을 포함한 이후 일관적으로 향상되었다. 이는 모델의 크기가 단순히 증가되었기 때문이 아닌 LLM transformer와 연관되어 있음을 `Sec. 5.1` 에서 검증한다.


#### Point Cloud Classification (3D modality)

<div align="center"><img src="https://github.com/user-attachments/assets/a1462e8e-22f8-4594-9c4c-410ff57b5e8b" width="20%"></img></div>


- task: 모델이 unordered 3D points를 처리하는 것으로 label을 예측

- 평가 데이터 : ScanObjectNN, ModelNet40

- ShapeNet으로 사전학습된 Point-BERT를 적용하였음.

- 실험 결과, 제안한 방법이 point cloud classification에서 정확도가 향상되었음.

  - ModelNet40의 1k와 4k point에서 성능이 약간 하락했는데, 이는 saturation(과적합)과 ~0.2의 variance 때문이다[(Ma et al., 2022)](https://openreview.net/pdf?id=3Pbra-_u76D).

  - points의 개수가 늘어났을 때는(8k) 성능 향상을 보였다. 


### Action Recognition (Video modality; semantic task)

<div align="center"><img src="https://github.com/user-attachments/assets/d8d608d0-6a1f-42df-9109-b637c6a89ae6" width="20%"></img></div>

- task : video clips의 action label 예측

- 평가 데이터 : "Something-something-v2” (SSv2)

- [VideoMAE](https://proceedings.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf)를 따라 ViT를 사용

  - 2D 이미지의 토큰 패치와 달리, 비디오 토큰은 공간적 차원과 시간적 차원을 모두 아우르는 큐브 형태

- VideoMAE의 2단계 학습을 적용 : 1) [MAE](https://arxiv.org/pdf/2111.06377) pre-training으로부터 ViT 초기화, 2) LLM transformer를 추가하여 SSv2 dataset으로 fine-tune

- 실험 결과, ViT-S와 ViT-B에서 모두 정확도가 향상되었다.

  - VideoMAE 연구에서 보였던 baseline 성능보다 낮게 측정되었는데 이는 GPU를 더 적게 사용할 수 밖에 없는 환경이라 batch-size에서의 차이로 보여진다. 그럼에도, 본 연구에서 ViT와 ViT+LLAMA는 똑같은 설정으로 실험하였다.

### Motion forecasting (Video modality; non-semantic task)

<div align="center"><img src="https://github.com/user-attachments/assets/a0ccedcd-8289-4958-9d9c-833ffe8f32cb" width="30%"></img></div>

- task : lane segment의 way-point와 agent의 이전 궤적들을 input으로 하고, K 개의 가장 가능성 높은 미래 궤적을 output으로 함

- 평가 데이터 : Argoverse

  - 평가 지표 : minimum average displacement (ADE), minimum final displacement (FDE), and miss rate (MR)

- frozen LLM transformer를 VectorNet과 mmTransformer에 적용
  - agent와 lanes를 feature로 바꾸고 LLAMA transformer block이 이 feature token을 처리

- 실험 결과, LLAMA를 함께 쓴 모델이 더 나은 궤적을 예측했다.

### Vision-Language task

<div align="center"><img src="https://github.com/user-attachments/assets/d13ee1de-b6c7-43e9-814f-f26c3945b6ef" width="70%"></img></div>

- 2D vision-language tasks

  - dataset : VQAv2, Flickr30k

  - baseline : METER

  - training setup은 [Shi et al. (2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Shi_Top-Down_Visual_Attention_From_Analysis_by_Synthesis_CVPR_2023_paper.pdf)를 따라, CLIP-B/32로 이미지 인코더를 초기화하고 RoBERTa로 텍스트 인코더를 초기화한 후 VQAv2나 Flickr30k로 fine-tuning함.

  - 실험 결과, LLAMA transformer를 추가하였을 때 모든 2D VL tasks에서 성능 향상이 있었다.

- 3D vision-language tasks

  - dataset : SQA3D

  - baseline : SQA3D-baseline, ScanQA

  - SQA3D baseline을 따라, textual input을 LSTM으로 처리하고 3D point clouds는 VoteNet으로 처리함.

  - 실험 결과, LLM transformer를 추가하였을 때 QA 능력을 향상시킬 수 있었음.

<br></br>

### 4. ANALYSIS ON LLM TRANSFORMERS FOR VISUAL TASKS

#### ABLATION STUDY ON DESIGN CHOICES


<div align="center"><img src="https://github.com/user-attachments/assets/969397eb-e0df-47eb-802b-70f93a3a2970" width="30%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/182d61ae-4cce-47bc-aa54-dd744d97d345" width="70%"></img></div>

- Model capacity : 성능 향상이 주로 LLM transformer의 사전 학습된 가중치 때문이 아니라, linear layer를 사용함으로 인한 모델 크기 증가에서 비롯된 것인지에 대해 알아보고자 함

  - 이를 위해, ViT-S-LLaMA와 똑같은 크기의 ViT-S-MLP를 만들었으며 이는 LLM block을 제거하고 linear layer 사이에 GeLU 활성화 함수와 layer normalization을 사용함

  - 실험 결과, ViT-S-MLP는 ViT-S 보다는 성능이 좋았지만 ViT-S-LLaMA보다는 떨어졌음. 이는 LLM transformer의 가중치가 중요하게 작용함을 시사함

- Fine-tuning : LLM transformer를 freeze하지 않고 fine-tuning하는 것(ViT-S-LLaMA-FT)이 성능이 더 좋은지 확인하고자 함

  - 위의 표를 보면, fine-tuning하는 것은 ViT-S-LLaMA의 비해 성능이 떨어졌다.

  - 또한, loss 커브를 보면 100epoch 이하에서는 성능의 향상이 있었지만 이후 과적합으로 인해 성능을 해치는 것을 관찰했다.

  - 따라서 LLM transformer를 freeze하는 것이 효과적

<div align="center"><img src="https://github.com/user-attachments/assets/7b672380-094a-4e6b-96e4-43334913130b" width="30%"></img></div>

- VARYING LLM TRANSFORMER LAYERS : 다른 LLM transformers에 따라 성능이 어떻게 달라지는지 관찰

  - LLaMA-7B와 OPT의 다른 layer를 사용하여 ViT-S에 적용하였음

  - 위 그림과 같이 어떤 layer를 사용하냐에 따라 다른 성능을 보였으며, 마지막 LLM layer가 일관되게 성능을 향상시켰음.

<br></br>

### 5. INFORMATION FILTERING HYPOTHESIS

- Information filtering hypothesis : 사전 학습된 LLM transformer는 informative tokens를 구별하는 **필터** 로써 동작하고 feature activation에서 빈도나 규모를 키우는 식으로 이에 대한 contribution을 증폭시킴.

<div align="center"><img src="https://github.com/user-attachments/assets/e71c3b81-cf24-4e70-a7a7-3c64dfdac773" width="70%"></img></div>


#### QUALITATIVE DERIVATION OF INFORMATION FILTERING HYPOTHESIS

- Emergent concentration on informative tokens : 위 그림3a에서 각 layer 이후의 feature activation을 추출한 것을 보여주고 있다. 사전 학습된 LLM transformer를 더함으로써 feature activation이 informative tokens를 강조하는 것을 볼 수 있다.

  - feature activation의 manitude는 centring 후의 L2-norm이며, frequency는 푸리에 변환 이후 angle의 L2-norm이다.

- Noisy attention scores : 위 그림3b에 마지막 transformer block에서의 CLS와 visual token 사이의 attention scores를 나타내었고, 각각 ViT와 ViT-LLaMA의 마지막 self-attention block을 나타낸다.

  - target object를 구분하는 이상적인 attention score는 DINO와 같은 object segmentation 패턴을 보여야하지만, ViT 모델은 noisy한 attention scores를 가지고 있으며 ViT-LLaMA의 대부분의  attention score도 noisy했다.

  - 이러한 관측은 feature activation과 대조되며, LLM transformer의 장점이 단순히 attention score에 기인할 수 없음을 시사한다.

- Deriving the amplification of informative tokens : 위의 informative token을 직접 이용한다면 이득을 취할 수 있지만,  ViT와 같이 분류를 위해 CLS 토큰만을 사하고 visual tokens는 labels y를 예측하기 위한 decoder $F_D$의 input으로 들어가지 않는다. 따라서, frozen LLM transformer가 informative tokens의 기여를 증폭한다는 가설이 필요하다.

  - CLS token 연산 : $z^2_L[CLS] = F^2_L \cdot F^F_LM \cdot F^A_LM (\sum_{v \in V} w_v z^1_L[v])$

    - `V` : visual tokens
    - $w_v$ : attention scores(weight of visual token v)

    - attention scores 와 visual tokens 가 먼저 계산된 것이 aggregate 되고 차례로 layer를 통과

  - visual token에 대해 설명하기 위해 CLS를 제거해 단순화 & 위 식에서 visual token은 attention score가 노이즈가 많다는 점 때문에 CLS 토큰에 신뢰성 있게 연결되지 않으므로 우리의 목표는 아래 식으로 표현되는 시각적 토큰을  final feature representation $z^2_L[CLS]$에 연결하는 것임

    - $z^2_L[v] = F^2_L \cdot F^F_LM \cdot F^A_LM(z^1_L[v]), where \ v \in V$

  - 따라서, 다음과 같이 우리의 가설을 표현할 수 있다: $z^2_L[CLS] \propto \sum_{v \in V} w_v (F^2_L \cdot F^F_LM \cdot F^A_LM(z^1_L[v]))$

    - 이 식은 그림 3a에서 informative tokens가 CLS 토큰 내에서 내재적으로 영향받는 방식을 설명

#### QUANTITATIVE EVIDENCE

<div align="center"><img src="https://github.com/user-attachments/assets/5145d5f7-2b38-4dd3-9896-fedb7b2ed233" width="30%"></img></div>

“informative regions"에 대한 ground truth를 위해 ImageNet-S 데이터셋을 이용하고, feature activation 과 attention score의 정확도를 평가하기 위해 집중하는 region에 대한 pseudo-mask를 생성하고 생성한 것과 ground truth 간의 mIOU를 계산해 평가한다.

평가 결과, ViT-S-LLaMA와 ViT-B-LLaMA 모두 attention score보다 psudo-mask의 mIOU가 더 높은 것을 볼 수 있다. 이는 위의 가설에서처럼 features ${z^2_L[v] | v \in V}$ 가 attention score보다 더 기여가 크다는 것을 알 수 있다.
