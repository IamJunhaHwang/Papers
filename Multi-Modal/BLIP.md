## :page_facing_up: BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

### Info

* `publication`: ICML 2022
* `author`: Junnan Li et al.
* `url`: https://proceedings.mlr.press/v162/li22n/li22n.pdf

### 1. Abstract & Introduction

Vision-Language pre-training(VLP)은 multimodal downstream task에서 엄청난 성공을 거두었지만 아래와 같은 한계가 존재한다.

- 모델 한계: 대부분의 방법이 encoder-based model이거나 encoder-decoder model을 사용

  - encoder-based model은 text generation task에 직접적으로 적용되기 힘듦

  - encoder-decoder model은 image-text retrieval에 적용되지 못했음

- 데이터 한계: 대부분의 SOTA model이 web에서 가져온 image-text pairs로 사전 학습 되었음

  - 여기에서, noisy web text가 vision-language model 학습에 suboptimal함(논문에서 보임)

따라서, 통합된 vision-language understanding과 generation을 위한` BLIP: Bootstrapping Language-Image Pre-training`을 제안한다. 이는 현존하는 방법보다 넓은 downstream task에 적용가능한 새로운 VLP 프레임워크이다.

- **Contributions**

  - Multimodal mixture of Encoder-Decoder (MED): 효과적인 multi-task pre-training과 transfer learning을 위해 새로운 모델 구조 제안

    - MED는 unimodal encoder, image-grounded text encoder, image-grounded text decoder로 사용 가능

    - 모델은 3가지 vision-language objectives로 사전 학습됨: image-text contrastive learning, image-text matching, image-conditioned language modeling

  - Captioning & Filtering(CapFilt): noisy image-text pairs로 학습하기 위한 새로운 데이터 부트스트래핑 방법 제안

    - MED를 2가지 모듈을 사용해 사전 학습함: captioner(주어진 웹 이미지로부터 synthetic caption 생성), filter(original web text와 synthetic text 모두에서 noisy caption 제거)

  - 실험

    - captioner와 filter를 함께 사용하여 caption을 부트스트래핑함으로써 여러 downstream tasks에서 상당한 성능 향상을 달성했으며, 다양한 캡션이 더 큰 향상을 만듦을 발견

    - BLIP이 다양한 vision-language tasks에서 SOTA를 달성했으며 video-language tasks의 zero-shot 성능에서도 SOTA를 달성

<br></br>

### 2. Method: BLIP; a unified VLP framework to learn from noisy image-text pairs

#### 2-1. Model Architecture

<div align="center"><img src="https://github.com/user-attachments/assets/4f3bce2c-7c6a-457f-8328-91b5304e2a9a" width="70%"></img></div>

understanding과 generation 능력을 둘 다 가진 unified model을 pre-train하기 위해 `multimodal mixture of encoder-decoder(MED)` 를 제안한다. 이 multi-task model은 3가지 기능 중 하나로써 작동할 수 있다:

- **Unimodal encoder**: image와 text를 각각 encode

  - text encoder: BERT
  - image encoder: ViT

- **Image-grounded text encoder**: self-attention layer와 feed-forward network 사이에 하나의 cross-attention layer를 삽입하는 것으로 visual information 주입

  - 이때, `[Encode]` 토큰이 text 앞에 붙으며 image-text pair의 multimodal representation으로 사용됨. (CLS 토큰과 같이)

- **Image-grounded text decoder**: image-grounded text encoder에서 bi-directional self-attention layers를 casual self-attention layers로 교체

  - 이때, `[Decode]` 토큰을 사용해 시퀀스의 시작 신호로 삼음

#### 2-2. Pre-training Objectives

2개의 understanding-based objectives와 1개의 generation-based objective를 함께 사용해 pre-training함.

- **Image-Text Contrastive Loss (ITC)**: unimodal encoder(text & image)에 작용

  - image encoder와 text encoder의 feature space align시킴 (positive pair는 비슷한 표현을 가지고 negative pair는 반대로)
  
  - [Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://proceedings.neurips.cc/paper/2021/file/505259756244493872b7709a8a01b536-Paper.pdf) 를 따라, momentum encoder를 이용해 feature와 soft label을 생성


- **Image-Text Matching Loss (ITM)**: image-grounded text encoder에 작용

  - vision과 language 사이의 alignment를 포착하는 image-text multimodal representation 학습

  - binary classification task로, multimodal feature가 주어졌을 때 image-text pair의 일치/불일치 여부를 판단한다.
  
  - more informative negatives를 찾기 위해 `Align before Fuse`의 hard negative mining strategy를 적용.
  
    - 배치 내에서 높은 contrastive similarity를 가지는 negatives가 loss 계산에 더 많이 선택되도록 함
    
- **Language Modeling Loss (LM)** : imagegrounded text decoder에 작용 

  - 이미지에 대한 텍스트 설명 생성

  - autoregressive 방법으로 text의 likelihood를 maximize하도록 모델을 훈련 (Cross-Entropy Loss)
  - loss 계산 시 0.1 label smoothing 적용

- 효율적인 사전 학습을 위해 text encoder와 text decoder는 Self-Attention layers를 제외하고 모든 파라미터를 공유함.

  - encoder의 경우 bi-directional self-attention을 적용해 현재 input token들에 대한 representations를 만들고, decoder의 경우 casual self-attention을 적용해 다음에 나올 토큰들을 예측하게 함.

#### 2-3. CapFilt

<div align="center"><img src="https://github.com/user-attachments/assets/13913d65-1261-46e6-b9a8-8ec8559e5797" width="80%"></img></div>

   
annotation cost로 인해 high-quality의 사람이 레이블링한 image-text 쌍( $\{(I_h, T_h)\}$ )은 수가 적다. 최근 연구에서는 웹으로부터 모은 많은 수의 image와 alt-text 쌍( $\{(I_w, T_w)\}$ )을 사용하지만, alt-text가 정확하게 이미지를 설명하지 않는 노이즈한 시그널이 있어서 vision-language alignment를 학습하는 것에 suboptimal하게 적용되게 된다.

따라서, text corpus의 품질을 높이기 위한 새로운 방법인 `Captioning and Filtering (CapFilt)`을 제안한다.   
위 그림과 같이 CapFilt에는 웹 이미지가 주어지면 캡션을 생성하는 `captioner`와 noisy한 image-text 쌍들을 제거하는 `filter`가 있다. 이 두 모듈 모두 같은 사전 학습된 MED model의 가중치로 초기화되며 COCO dataset을 사용해 각각 fine-tune된다.

- Captioner : image-grounded text decoder

  - web images $I_w$ 가 주어지면, 새로운(synthetic) 캡션 $T_s$ 를 만들어 낸다. (이미지당 1개 캡션 생성)

  - objective: **LM**

- Filter : image-grounded text encoder

  - text와 image의 matching 여부를 판단하여 noisy한 texts를 제거 (original web texts $T_w$ 와 synthetic texts $T_s$ 모두에 적용)

  - objectives: **ITC & ITM**

마지막으로, filtered image-text pairs와 human-annotated pairs를 합쳐서 새로운 데이터 셋 생성

<br></br>

### 3. Experiments and Discussions

<div align="center"><img src="https://github.com/user-attachments/assets/bc97fe49-d271-4e42-b4de-439d9908ebc2" width="70%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/a4a07e9c-9235-4869-b804-3cdf8571e16c" width="80%"></img></div>

- **Effect of CapFilt** : 다른 데이터 셋에서 학습한 CapFilt의 효과 증명

  - 14M으로 학습한 CapFilt에서, Captioner나 Filter 중 하나만 적용한 것도 성능 향상을 가져왔으며 2개 모두 적용했을 때 상당한 성능 향상이 있었음

  - 더 큰 데이터 셋이나 큰 모델 사용 시 성능 향상이 있었으며, 이는 captioner와 filter를 큰 것을 사용해도 향상이 있었음.

<div align="center"><img src="https://github.com/user-attachments/assets/606bcc72-871e-41f3-9e58-10116f1ee07a" width="80%"></img></div>


- **Parameter Sharing and Decoupling**

  - 다른 parameter sharing을 적용한 결과를 Table 3에 나타내었으며, self-attention layer를 제외하고 모두 parameter sharing 진행한 것이 가장 좋은 성능을 보였음.

    - self-attention layer까지 공유 할 시, encoding task와 decoding task가 충돌을 일으켜 성능이 떨어짐.

  - CapFilt에서 parameter sharing을 진행했을 때, captioner는 노이지한 캡션을 만들었으며 Filter는 필터링 능력이 떨어졌음

<br></br>

### 4. Comparison with State-of-the-arts

<div align="center"><img src="https://github.com/user-attachments/assets/03a63a02-835e-4355-90ae-1f6dc62b1849" width="80%"></img></div>

<div align="center"><img src="https://github.com/user-attachments/assets/6c311507-e4ef-479d-99dc-89107710823f" width="80%"></img></div>


- **Image-Text Retrieval**

  - `COCO`와 `Flickr30K` 데이터 셋을 이용해 image-to-text retrieval(TR)과 text-to-image retrieval(IR)을 평가하였으며, ITC와 ITM loss를 이용해 모델을 fine-tune함

  - Table 5에 나타나 있는 결과를 보면, BLIP이 다른 방법 보다 좋은 성능을 달성하였음.

- **Image Captioning**

  - `NoCaps`와 `COCO` 데이터 셋을 이용해 평가하였으며, COCO 데이터 셋에 LM loss를 이용해 fine-tune한 모델 사용

  - Table 7에 나타나 있는 결과를 보면, 14M의 사전학습 이미지를 사용한 BLIP이 비슷한 사전학습 데이터 크기를 가지는 모델들과 비교했을 때 좋은 성능을 달성함

  - 129M의 사전학습 이미지를 사용한 BLIP의 경우 200M을 사용한 LEMON과 비슷한 성능을 보였는데, LEMON의 경우 상당한 계산량을 가지는 pre-trained object detector와 높은 해상도의 이미지를 필요로하므로 BLIP이 더 우수함.

<div align="center"><img src="https://github.com/user-attachments/assets/1eecffa6-0be7-4cdb-9f32-5626d8b8944d" width="40%"></img></div>

또한, 약간의 모델 구조 변경을 통해 다양한 task에 적용이 가능하다.

- **Visual Question Answering (VQA)**

  - 여기에서는 open-ended VQA로 answer generation을 하는 VQA로 진행

  - 위 그림과 같이 image-question이 먼저 encode되고 이에 따른 answer를 decoder가 생성하도록하였으며, 모델은 ground-truth answer를 타겟으로하여 LM loss로 fine-tune하였음

  - 아래의 결과와 같이, BLIP이 14M/129M images 모두에서 좋은 성능을 보였음

<div align="center"><img src="https://github.com/user-attachments/assets/05ca5761-5eb9-4434-a34a-fe76e0f6952e" width="30%"></img></div>


