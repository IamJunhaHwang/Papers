## :page_facing_up: AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

### Info

* `conf`: ICLR 2021

* `author`: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Neil Houlsby et al.

* `url`: https://arxiv.org/pdf/2010.11929.pdf 
* `github` : https://github.com/google-research/vision_transformer

### 1\. Abstract & Introduction

NLP에서 Transformer가 사실상 표준이 되었지만, 이를 CV에 적용하기에는 한계가 있었다. (CNN과 같이 적용되거나 CNN의 구조는 유지하되 특정 구성요소를 바꾸는 식으로 사용됐음)   

우리는 이러한 CNN의 의존성 없이 pure transformer를 image patch sequence에 적용해 이미지 분류 문제를 잘 풀 수 있음을 보였다. 또한, `Vision Transformer(ViT)`는 훈련에 필요한 계산량이 훨씬 적으면서도 SOTA인 CNN보다 더 좋은 성능을 보였다.

강한 regularization 없이 중간 사이즈 데이터셋(ex.ImgageNet)에서 훈련할 때는 `ResNet`보다 조금 뒤처지는 성능을 보였는데 이는 Transformer가 CNN과 같은 `inductive bias`가 부족하기 때문이다. 하지만, 대용량 데이터 셋으로 훈련을 하면 ViT로  좋은 결과를 낼 수 있었다.

<br></br>

### 2. Method

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/21fb90eb-82f9-41de-acad-1866947b7f29" width="60%"></img></div>

Transformer는 `1D token sequence`를 입력으로 받기 때문에, 2D image를 다음과 같이 shape을 변경한다.

- $H \times W \times C \rightarrow N \times (P^2 \cdot C)$

  - $(H, W)$ : 원본 이미지의 해상도
  - $C$ : 원본 이미지의 채널
  - $(P, P)$ : 각 이미지 patch의 해상도 **[즉, token으로 다룰 이미지 픽셀 집합(patch)의 크기(해상도)]**
  - $N = HW / P^2$ : reshape 후 만들어지는 토큰의 개수

  - Transformer는 모든 layer에서 미리 정해놓은 latent vector size(hidden size) $D$ 를 사용하므로 위에서 만든 patch들을 $D$ 차원으로 매핑시킨다.    **-> patch embeddings** 

위치 정보를 담기 위해 위의 patch embedding에 Position embedding을 더한다. 이 때, 기본적인 `learnable 1D position embeddings`를 사용한다. (2D를 사용해보았지만 성능 향상이 없었음)   
이렇게 만들어진 `embedding vector sequence`를 Encoder의 입력으로 준다.

BERT의 `[CLS]` 토큰과 같은 역할을 하도록 입력 시퀀스 앞에 patch($z^0_L$) 하나를 붙였다. (Transformer Encoder의 출력인 $z^0_L$ 은 image representation $y$ 를 나타낸다)

Transformer의 Encoder는 `Multi-headed self-attention`과 `MLP block`로 이루어져 있으며 `LayerNorm`은 각 블록 전에 적용된다. 그리고 `residual connection`은 각 블록을 거친 후에 적용된다.

`Classification Head`는 $z^0_L$ 을 이용하며 pre-training할 때는 1개의 hidden layer를 가지는 MLP로 구현하고 fine-tuning에서는 Linear Layer로 구현한다.

- **inductive bias**

  - `ViT`는 CNN보다 훨씬 적은 image-specific inductive bias를 가지고 있음

  - CNN에는 지역성(locality), 2D neighborhood structure, translation equivariance가 전체 모델에 걸쳐 각 레이어에 내장되어 있음. **이에 반해, `ViT`는 오직 MLP layer만 local & translationally equivariant 함. (self-attention은 global)**

    - `ViT`에서 2D neighborhood structure는 아주 조금만 사용됨: image를 patch로 자르는 것, fine-tuning에서 position embedding 조정할 때 (아래에 설명)

-  **Fine-Tuning & Higher Resolution**

  - `ViT`의 pre-training은 큰 데이터 셋에, fine-tuning은 (작은)downstream 데이터 셋에 진행했다. fine-tuning에서, pretraining에 사용한 classification head는 삭제하고 Linear Layer($D \times K$, `K==downstream classes`)로 교체함

    - 고해상도로 fine-tuning할 때 효과적이라 함

    - 고해상도의 이미지를 입력으로 줄 때, `patch-size`를 똑같이 유지해 더 긴 시퀀스 길이를 사용하게 한다. [pre-trianing보다 fine-tuning의 해상도가 더 낮은듯]

    - 이 때, `position embedding`은 의미가 없어지므로 2D interpolation을 사용했다.

  - 이러한, 해상도 조절과 patch 추출만이 ViT에 주입되는 이미지의 2D 구조에 대한 `inductive bias`이다.

<br></br>

### 3. Experiments

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/014ab3c6-1d07-44b7-81cd-7c5b2a2e10fa" width="60%"></img></div>

- `Model` : `ResNet(BiT)`, `Vision Transformer (ViT)`, `hybrid`

  - `ResNet(BiT)` : ResNet의 변형
  - `hybrid` : 중간의 feature map을 1 pixel 크기의 patch size로 넘겨준 ViT

  - 서로 다른 시퀀스 길이를 실험하기 위해, 우리는 **(i)** 정규 ResNet50의 stage 4의 출력을 사용하거나, **(ii)** stage 4를 제거하고 동일한 수의 레이어를 stage 3에 배치해 총 레이어 수를 유지하고 이 확장된 stage 3의 출력을 사용한다. 
    
    - **(ii)** 옵션은 4배 더 긴 시퀀스 길이와 더 많은 비용이 드는 ViT 모델을 가져옴

- 각 모델이 데이터를 얼마나 필요로 하는지 알기 위해 여러 사이즈의 데이터 셋들을 사용해 pre-train해보고 여러 벤치 마크로 평가함

- `Dataset`

  - `Pre-Training` : ILSVRC-2012 ImageNet, ImageNet-21k, JFT

  - `Fine-Tuning` : ImageNet (원본 검증 레이블 및 정리된 ReaL 레이블), CIFAR-10/100, Oxford-IIIT Pets, Oxford Flowers-102, 19-task VTAB classification suite

  - 전처리는 `Kolesnikov et al. (2020).`를 따름.

- `Training & Fine-tuning` : 하이퍼파라미터 관련은 논문 참조

- `Metrics` : few-shot or fine-tuning accuracy.

  - few-shot : `regularized least-squares regression problem`을 이용해  $\{−1, 1\}^K$ target vectors 로 매핑


#### 3-1. Comparision to SOTA

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/7124b97c-2c14-41e2-8e79-0bf0a6250e85" width="60%"></img></div>

작은 모델인 `ViT-L/16`은 `Bit-L`과 똑같은 데이터로 pre-train 되었음에도 모든 태스크에서 좋은 성능을 보였다. (더 적은 cost)   
큰 모델인 `ViT-H/14`은 특히 더 난이도 높은 fine-tuning set에서의 성능이 더 향상되었다. (이 모델 또한 SOTA보다 cost가 작다)

#### 3-2. Pre-Training Data Requirements
<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/30238a29-c147-40a5-8ea5-368032a88f4c" width="60%"></img></div>

- 데이터 양이 얼마나 영향을 미치는지 확인하기 위해 두 가지 실험을 했다.

  - Pre-training 데이터 셋의 사이즈를 늘려가며 성능 관찰 [크기가 다른 여러 데이터 셋]

  - `JFT-300M` 데이터 셋에서 랜덤 샘플링을 한 것들과의 성능 관찰 [크기가 다른 똑같은 데이터 셋]

`ViT` 모델은 pre-training dataset이 작으면 성능이 좋지 않았지만 크기가 커질수록 성능이 좋아졌다.

#### 3-3.  Inspecting Vision Transformer 

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/a49268fd-fad0-4eeb-8d38-91a862f7c8fd" width="60%"></img></div>

`ViT`가 이미지를 어떻게 처리하는지 알기 위해 내부 표현을 분석하였다.

- `왼쪽 이미지` : 학습된 임베딩 필터의 주요 주성분들의 시각화 

  - 이러한 주성분들은 각 패치 내의 미세한 구조를 저차원으로 표현하는 그럴듯한 기저 함수들과 유사하다.

- `중간 이미지` : 위치 임베딩의 유사성을 통해 이미지 내의 거리 개념을 인코딩한 것을 시각화

  - 가까운 patch들일수록 비슷한 position embedding을 가짐

- `오른쪽 이미지` : attention weight에 기반해 이미지 공간에서 어떤 정보들끼리 통합되는지 평균 거리를 계산했음

  - 몇몇 헤드들은 가장 낮은 layer에서 이미 이미지의 대부분에 attend한다.

  - layer가 깊어질수록 attention distance가 증가했다. 

#### 3-4. Self-Supervision

`BERT`와 같이 MLM idea를 `masked patch prediction`으로 적용해보았지만 위의 `ViT`보다 성능이 좋지 못했음.

<br></br>


### 4. Conclusion

- `image recognition`에 Transformer를 적용해보았으며 이전 연구들과 다르게 성능이 좋았음. (많은 데이터 양으로 사전 학습했을 때)
