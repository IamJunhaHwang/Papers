## :page_facing_up: BEIT: BERT PRE-TRAINING OF IMAGE TRANSFORMERS

### Info

* `conf`: ICLR 2022
* `author`: Hangbo Bao, Li Dong, Songhao Piao, Furu Wei **[MicroSoft-Research]**
* `url`: https://openreview.net/pdf?id=p-BhZSz59o4
* `github` : https://github.com/microsoft/unilm

### 1\. Abstract & Introduction

본 논문에서는 self-supervised vision representation model인 `BEIT`(Bidirectional Encoder representation from Image Transformers)를 제안한다. BERT와 비슷하게 vision Transformer를 사전 학습하기 위해 **masked image modeling task(MIM)** 를 제안한다. 이에 따라, 이미지는 사전 학습 과정에서 image patch(16 * 16)과 visual token(discrete token)이라는 관점으로 바라본다.

`BEIT`는 원본 이미지를 **VQ-VAE를 통해 이산적인 visual token으로 토크나이징**하고 몇몇 **image patch를 무작위로 마스킹 한 후, 이를 원본 토큰으로 복원하는 과정**을 통해 사전 학습 된다. (raw-pixel을 예측하는 것이 아님) 사전 학습을 마친 후, 해당 인코더 위에 task layer를 올리는 것으로 fine-tuning을 진행했다.

한편으로는 BERT-style의 사전 학습을 이미지에 적용하는 것은 ViT의 input unit에 대해 미리 정의된 vocabulary가 없기 때문에 softmax classifier를 적용할 수 없는 어려움이 있다. 가장 간단한 방법은 마스킹된 patch의 raw pixel을 예측하는 회귀 문제로 다루는 것이지만 pixel-level recovery task는 사전 훈련 단계에서 단거리 종속성(short-range dependencies)과 high-frequency details와 같은 모델링 능력을 낭비하는 경향이 있다. 우리의 목표는 이런 이슈들을 이겨내는 것이다.


이미지 분류, 이미지 의미 분할 태스크에서 실험한 결과, BEIT가 기존 방법과 견줄만한 성능을 보였다.

- **Contributions**

  - ViT를 self-supervised 방식으로 사전 학습하기 위해 `masked image modeling task`를 제안했으며 VAE 관점의 이론적인 설명을 제공했다.

  - BEIT를 사전 학습 하였고 downstream task에 fine-tuning하는 실험을 진행했다.

  - BEIT가 사람의 annotation없이도 semantic regions와 object boundaries를 구별하는 것을 배운다는 걸 보였다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/b0a1de88-3afa-441b-b39b-5d2822828c70" width="55%"></img></div>

<br></br>

### 2. Methods

#### 2-1. Image Representation

이미지를 image patch, visual tokens로 바라보는데, 각각 input, output representation 역할을 한다.

- Image Patch

  - 2D image를 patch의 시퀀스로 자른다. [ViT와 같은 방식]

  - $x \in \mathcal{R}^{H \times W \times C}$ 를 $N = HW / P^2$ 의 patch들로 바꾼다. 그 다음, 이를 flatten하고 linear projection하여 사용한다.

  - 예를 들면, `224 X 224`의 이미지는 `14 X 14`의 patch들로 구성되며 하나의 patch는 `16 X 16`이다.

    - $14 = \sqrt{224 * 224 / 16^2}$, patch들이 격자형태로 구성되므로 루트를 붙였음

- Visual Token

  - 이미지를 raw pixel로 나타내는 대신에 `VQVAE를 이용한 image tokenizer`를 이용해 이산적인 토큰으로 나타낼 것이다.

    - $x \in \mathcal{R}^{H \times W \times C}$ ===> $z = \[z_1, ... , z_N\] \in \mathcal{V}^{h \times w}$

    - $\mathcal{V} = \{1, ... , |V| \}$

  - `VQVAE`는 두 가지 모듈(tokenizer, decoer)로 이루어진다.
  
    - tokenizer(Encoder) $q_{\phi}(z|x)$ 는 image $x$ 를 이산적인 토큰 $z$ 로 매핑하고 decoder $p_{\psi}(x|z)$ 는 $z$ 를 기반으로 $x$ 를 재현하도록 학습된다.

    - reconstruction objective : $\mathbb{E_{z \sim q_{\phi} (z \vert x)}} \[log p_{\psi} (x \vert z)\]$

    - latent visual token이 이산적이기 때문에 미분이 불가능하므로 `Gumbel-softmax relaxation`을 적용해 학습한다.

    - 또한, VQVAE 학습동안 uniform prior가 $q_{\phi}$ 에 놓인다.

  - 각 이미지는 `14 X 14`의 visual token으로 토크나이징되며 vocabulary size는 8192이다. 또한, 해당 토크나이저는 public하게 사용 가능한 것을 사용했다.

#### 2-2. Backbone Network: Image Transformer

Backbone으로 ViT를 사용했다. ViT에서 그랬듯이 image patch를 linear projection한 임베딩 + position 임베딩을 더하여 input으로 사용한다. 또, 각 시퀀스 앞에 special token `[S]`를 붙이는 것도 같다.

정리하면 input은 다음과 같이 만들어 진다. image tokenization -> linear projection(token Embedding) -> Add position Embedding -> done

#### 2-3. Pre-Training BEIT : Masked Image Modeling

input을 N개의 image patch로 나눈 후, N개의 visual token $\{z_i\}_{i=1}^N$으로 토크나이징한다. 이후, 대략 40% image patch를 무작위로 마스킹하고 마스킹된 patch들을 learnable embedding $e_{[M]} \in \mathbb{R}^D$ 로 바꾼다. 이러한 image patch들이 ViT로 들어가고 ViT의 마지막 hidden vector $\{h_i^L\}^N_{i=1}$ 이 input patch의 인코딩된 표현을 다룬다. 그리고 각 마스킹 위치에서 softmax classifier를 통해 해당 visual token을 예측한다. 이에 대한 수식과 pre-training objective(maximize log-liklihood)는 아래와 같다.

- $p_{MIM}(z'|x_{\mathcal{M}}) = softmax_{z'}(W_ch_i^L+b_c)$

  - x_{\mathcal{M}} : 마스킹된 이미지

  - $W_c \in \mathbb{R}^{|V| \times D}$ &nbsp;&nbsp;&nbsp;&nbsp; $b_c \in \mathbb{R}^|V|$

- pre-training obejective : ![image](https://github.com/DAILAB-CBNU/Papers/assets/46083287/5fe4e4cc-d549-41d4-abfc-23900dc486d5)
  - $\mathcal{D}$ : training corpus


<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/c902c53e-13e3-4d74-9545-dd6c370a9ffa" width="45%"></img></div>

위와 같이 마스킹 위치를 완전 무작위로 하지 않고, block-wise masking을 적용한다. 마스킹할 최소 patch 수를 16으로 정하고 무작위로 정해진 종횡비를 통해 가까운 patch들끼리 masking되게 한다. 이 과정을 `0.4N`까지 반복한다. [span-BERT에서 영감을 받음]

masking 예시는 아래와 같음

#### 2-4. Pre-Training Setup

- Data : `ImageNet-1K`

- Model : ViT-Base

- Tokenizer : DALL-E의 토크나이저를 그대로 사용. Vocab size = 8192

- 더 자세한 내용은 논문 참고

<br></br>

### 3. Experiments

#### 3-1. Image Classification

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/2107d86f-0c3c-43cc-8c1a-dfd838bfbf44" width="65%"></img></div>
 
`ILSVRC-2012 ImageNet` 데이터 셋을 사용해 평가하였으며 대부분의 hyperparameter는 공정한 비교를 위해 DeiT의 것을 따랐다.

위 표는 다양한 모델을 `top-1 accuracy`로 평가한 결과이다. 크게 scratch, supervised, self-supervised로 나뉜다.

`BEIT`가 가장 성능이 좋았으며, high resolution일수록 model size가 클수록 성능이 좋아졌다. 또한, ViT의 경우 labeled data의 역할이 중요하지만 BEIT의 경우 self-supervised로 학습되므로 이런 이슈가 없다.

#### 3-2. Semantic Segmentation

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/33d0c39d-f816-43ef-96d6-d9b1a97af16e" width="45%"></img></div>
 
`ADE20K` 데이터 셋을 사용해 평가하였으며 모든 semantic categories에 대한 mean Intersection of Union(mIoU)로 점수를 매겼다. `SETR-PUP`의 task layer와 hyperparameter를 따랐으며 Adam optimizer를 사용했다.

`BEIT`가 supervised pre-training보다 좋은 성능을 보였으며 intermediate fine-tuning을 진행했을 때 성능이 올라간 것을 확인했다.

#### 3-3. Ablation studies

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/e8967a15-9783-49fc-86da-fd29b86e7200" width="65%"></img></div>
 
<br></br>

### 4. Conclusion

- ViT를 위한 self-supervised pre-training framework를 제안하고 downstream task에서 좋은 성능을 냈음을 보임

- BERT와 같은 pre-training을 image Transformer에서 잘 동작하게 하려면 우리가 제안한 방법이 중요하다는 것을 보임

- 어떠한 레이블링 데이터도 사용하지 않고도 자동으로 습득한 semantic regions에 관한 흥미로운 특성을 제시함.

  - 논문의 3.4 참고
