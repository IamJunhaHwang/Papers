## :page_facing_up: Learning Transferable Visual Models From Natural Language Supervision

### Info

* `conf`: ICML(/PMLR) 2021
* `author`: Alec Radford, Jong Wook Kim et al.
* `url`: https://arxiv.org/pdf/2103.00020.pdf


* `github` : https://github.com/OpenAI/CLIP

### 1\. Abstract & Introduction

SOTA vision model들은 미리 지정된 레이블을 예측하는 것으로 학습되는데, 이는 일반성과 유용성을 떨어뜨린다. (새로운 레이블이 필요할 경우 쉽게 적용할 수 없음)

이미지에 대한 설명(텍스트)로 부터 학습하는 것이 가장 유망한 대체 방법이다. 우리는 어떤 캡션이 해당 이미지에 어울리는지에 대해 학습하는 효율적이고 확장성 있는 방법으로 SOTA 이미지 표현을 배우는 간단한 예측 task를 제안한다. 사전 학습 이후에는, 자연어는 학습된 visual concept을 참조(reference)하는데 사용된다.

우리는 위 방법의 성능을 30개의 서로 다른 CV dataset을 이용해 비교한다. 모델은 대부분의 task에 잘 transfer되고 추가 학습 없이도 fully supervised baseline과 견줄만한 성능을 보였다.

- 해당 연구에 대한 동기

  - NLP는 MLM을 이용해 웹에 있는 여러 데이터들을 이용할 수 있는 반면, CV에서는 아직도 전문가에 의해 레이블링된 데이터가 필요하다.

  - 이 때문에, 이미지의 캡션, 해시태그 등 pair된 메타 데이터를 이용하는 방법(이미지에 대한 텍스트 설명을 예측하는 방법)이 제시되었다.

  - 이후, 최근 구조나 사전학습 방법을 적용한 `VirTex, ICMLM, ConVIRT` 모델들이 등장했고 이들은 `Transformer-based LM, MLM` 과 텍스트로부터 이미지 표현을 배우는 `contrastive objectives`의 잠재성을 증명했다.

  - 한편, 이러한 컨셉에 대한 증명이 있었음에도 이미지 표현을 배우기 위해 `Natural Language Supervision(이미지-텍스트 쌍으로 학습)`을 사용하는 경우가 드물었는데 이는 흔한 벤치마크들에서 baseline(이전 모델들)보다 성능이 안좋았기 때문이다.

  - 대신에, 더 좁은 범위이지만 잘 조정된 weak supervision(pixel에 대한 레이블이 아닌 좀더 범위가 큰 클래스 등장 유무에 대한 레이블을 사용하는 것)의 사용은 성능을 향상시켰다. 또한, `ViT`는 Masking LM 방식(노이즈 데이터 이용)을 CV에 적용했다.

  - 위와 같은 작업들을 정리하면, 이들의 supervision에는 `gold-label` 혹은 `vocab`이 필요하므로 여전히 dynamic output을 내기에는 부족하다. (유연성 부족, zero-shot 능력에 제한, 여러 downstream에 적용하기에 한계가 있음)
  
  - 이런 weakly supervision 모델(ViT, ... 등)과 recent exploration(VirTex, ICMLM, ConVIRT)의 가장 큰 차이는 규모이다. 전자는 대규모 데이터를 사용했지만 후자는 그렇지 않다.

  - 이러한 gap을 줄이기 위해, 본 논문에서는 대규모 데이터를 인터넷에서 수집해 `natural language supervision`으로 훈련한 이미지 분류기의 동작에 대해 연구한다.

    - 이 모델을 `간략화된 버전의 ConVIRT` 모델을 처음부터 훈련시킨 모델로 `CLIP(Contrastive Language-Image Pre-training)`이라 부르며 `natural language supervision`로부터 효율적으로 학습하는 방법이다. 

    - **목적 : NLP처럼 인터넷에 있는 데이터를 활용해 전문가의 레이블링 없이 대규모 데이로 CV모델 학습** 

<br></br>

### 2. Approach

#### 2-1. Natural Language Supervision

우리 접근법의 핵심은 자연어가 포함된 supervision으로부터의 인식을 배우는 아이디어이다. 이는 새로운 아이디어는 아니지만 여러 용어로 혼용되어 왔는데, 여기에서 중요한 점은 자연어를 훈련 신호로 본다는 것이다.

- 자연어로부터 학습하는 것은 다른 훈련 방법들보다 잠재적 강점을 가진다

  - 확장하기 쉽다 (전문가 레이블링이 필요 없으므로)

    - 때문에, 전통적인 CV 방법보다 좋다

  - 단순히 표현만을 배우는 것이 아니라 해당 표현과 언어를 연결해 유연한 zero-shot transfer가 가능하게 함

    - 때문에, unsupervised 나 self-supervised 보다 좋다

#### 2-2. Creating a Sufficiently Large Dataset

기존 연구들에서는 3가지 데이터 셋(MS-COCO, Visual Genome, YFCC100M)을 주로 사용해 왔다. `MS-COCO, Visual Genome`은 높은 품질을 가지고 있으나 양이 적고, `YFCC100M`은 양은 많으나 품질이 좋지 않다. (자연어 pair를 가질 수 있는 데이터로 필터링 했을 때 6~15M 정도 나왔으며 이는 IMAGENET과 비슷한 사이즈로 작아진다)

`natural language supervision`의 가장 큰 동기가 대규모 데이터로이므로, 새롭게 (image, text) 쌍을 가지 400M 데이터 셋을 만들었다. (`WebImageText(WIT)` 라고 명명)

English Wikipedia에서 최소 100번 이상 발생한 모든 단어를 base query list로 사용했으며 이러한 50만 query 중에서 하나를 포함하는 텍스트로 (image, text)를 검색했다. 그리고 클래스 균형을 대략적으로 맞추기 위해, 각 query마다  (image, text)쌍을 2만개까지만 포함시켰다.

#### 2-3. Selecting an Efficient Pre-Training Method

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/117c3f2a-964e-4fe3-9f95-cccba3ee8652" width="30%"></img></div>

SOTA CV 모델들은 1000개의 class에 대해 예측하게 훈련하는 것도 계산량이 많아 훈련에 오래 걸렸다. 우리는 visual concepts에 대해 open set으로 학습하는 만큼 학습 효율을 중시하여 Pre-Training Method를 선택했다.

처음에는 `VirTex`와 비슷하게 CNN과 transformer를 처음부터 함께 훈련시켜 이미지 캡션을 예측하도록 헀는데, 이를 효율적으로 확장하는데 어려움이 있었다. 위 그림의 파란색을 보면, 해당 방법이 ImageNet class를 인식하는데 baseline(간단하게 BOW 예측하는 모델)보다 3배나 느린 것을 볼 수 있다. 

위와 같은 방법같이 각 이미지에 어울리는 텍스트의 정확한 단어를 예측하도록 시도하는 것은 캡션이 다양하게 표현될 수 있기 때문에 어렵다. 최근 연구에서 `이미지의 Contrastive Representation Learning`이 계산량도 적으면서 비슷한 성능을 낸다는 것을 알아냈기 때문에, 텍스트 전체에 대한 정확한 단어들이 아닌 어떤 이미지와 쌍이 되는지에 대해 예측하는 task를 풀기 위한 training system을 탐색했다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/8cd356c0-7f87-4a34-a8c2-63342df382ab" width="70%"></img></div>

주어진 N (image, text) 개의 쌍을 가지는 하나의 batch에서, CLIP은 `N x N`의 페어링 간의 실제로 일어난 `possible pair` 가 무엇인지 예측하도록 학습된다. **이를 위해, 이미지 인코더와 텍스트 인코더가 N개의 실제 쌍들의 이미지 & 텍스트 임베딩의 코사인 유사도를 최대화하고 $N^2 - N$ 의 incorrect 쌍들의 것은 최소화하도록 함께 학습하는 것으로 멀티 모달 임베딩 공간을 학습한다.**   
이 유사도 점수간의 symmetric cros entropy loss를 최적화한다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/0d7fe05b-a4db-4175-95b8-bf376cdec02c" width="30%"></img></div>

- 사전 학습 가중치를 사용하지 않고 처음부터 훈련 함.

- 각 인코더의 표현에 linear-projection을 통해 멀티모달 임베딩으로 만든다.

- 기존 `ConVIRT`에서 쓰인 text transformation function을 제거하였고 image transformation function은 단순화 했다.(random square crop from resized images만 사용)



#### 2-4. Choosing and Scaling a Model

- image encoder

  - ResNet-50 : ResNet-D 사용 + global average pooling layer를 attention pooling 으로 변경

  - ViT : patch와 position 임베딩을 결합하기 위한 layer normalization 추가  

- text encoder

  - Transformer : max_length를 76으로 설정

  - 가장 마지막 층의 `[EOS]` 토큰이 텍스트의 표현(임베딩)으로 사용된다. (이게 layer normalization과 linear projection을 거쳐 멀티모달 임베딩 공간으로 감)
   
- ResNet image encoder에는 `Tan & Le (2019)`의 방법을 적용한다. 해당 논문에서는 모델의 깊이, 너비, 해상도의 적절한 비율을 찾아 적용했지만, 우리는 똑같이 증가시켰다.

- text encoder의 경우에는 ResNet의 너비 증가에 비례하도록 모델의 너비를 증가시켰다.

<br></br>

### 3. Experiments

#### 3-1. Using CLIP for zero-shot transfer

각 데이터 셋의 모든 클래스들의 이름을 잠재적 text pairing으로 사용하고 가장 그럴듯한 (image, text) 쌍을 찾는다. 즉, 다음과 같이 진행된다.

- 이미지의 특징 임베딩과 가능한 텍스트 집합의 특징 임베딩을 각각의 인코더를 이용해 계산하고

- 이 임베딩들간의 코사인 유사도를 계산한 후 temperature parameter를 이용해 스케일링하고 softmax를 통해 확률 분포로 정규화한다.

- 위 과정에서 `prompt`를 이용해 추론하면 성능이 더 올라가는 것을 확인했다. 

  - ex) `A photo of a {Label}.`
  
  - `prompt` 사용 이유 : 클래스 이름만 제공할 경우 컨텍스트가 부족함, 사전 학습 데이터 셋의 텍스트는 하나의 단어가 아닌 full sentence 였음.

  - 단순히 `prompt` 사용만으로 1.3% 성능이 증가함.


#### 3-2. Initial Comparision To Visual N-Grams

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/cfb3c2a0-43c7-4384-95af-511148757eab" width="30%"></img></div>

`Visual N-Grams`는 위와 같은 방법을 사용해 이미지 분류 문제로 zero-shot transfer를 처음 시도한 연구이다.

`CLIP`이 ImageNet의 레이블을 사용해 훈련을 하지 않았음에도 더 좋은 성능을 보였다. (하지만, 이 둘은 많은 부분이 다르기 때문에 직접적인 비교가 되지 못한다)

#### 3-3. Analysis Of Zero-Shot CLIP Performance

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/4d198210-e95a-440f-82d7-415156c49890" width="30%"></img></div>

- 27개 중 16개에서 ResNet-50보다 좋았다.

- fine-grained dataset에서는 성능이 좋았음. 반면에, 도메인이 특이한 경우(인공위성, 림프절종양 등)

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/e7372dee-2976-494d-80e9-7d856e52e30c" width="30%"></img></div>

- 위 그림은 few-shot에 대한 성능 비교이다.

  - CLIP 이외의 다른 16-shot linear Classifiers와 zero-shot CLIP이 비슷한 결과를 보였다. (CLIP이 우수함)

  - zero-shot CLIP이 4-shot CLIP과 비슷한 성능을 보였다. (이는, 4-shot의 경우 linear classifier로 진행되기 때문)


### 3-4. Representation Learning

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/96d69ee6-2156-46b0-b3d8-fb4279c9d2ef" width="65%"></img></div>

- 사전 학습 모델에서 뽑아낸 representation를 Linear Classifier를 통과시키는 것으로 여러 데이터 셋에 대해 모델의 성능을 비교했다.

- CLIP vision transformer가 CLIP ResNets보다 더 좋은 성능을 보였다. (연산 효율이 좋음)

- 좀 더 넓은 범위의 task를 가지는 27개의 데이터 셋에서도 체크를 했는데, 스케일에 상관없이 CLIP의 성능이 더 좋았다.


### 3-5. Robustness to Natural Distribution Shift

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/d32cd418-73de-41e3-b143-49d239b88a48" width="65%"></img></div>

이전 연구들에서 위의 그림과 같이 `distribution shift`가 발생하면 성능이 떨어지는 것을 발견했는데, CLIP의 경우 robust하게 작동했다. (웹에서 수집한 데이터들로 학습되었기 때문에, 기존의 ImageNet 기반의 모델들과 다른 분포를 가진다.)                    

<br></br>

### 4. Limitations

- 논문에서 사용한 ResNet baseline은 현재 SOTA가 아니다. (CLIP이 SOTA에 도달하려면 1000배 더 계산해야된다고 추정)

- few-shot setting에서 성능이 좋지 않았다.

<br></br>

### 5. Conclusion

- NLP에서 성공을 거둔 task-agnostic web-scale pre-training을 CV에 적용하였고 결과에 대해 분석하였다.

- training objectives를 최적화하기 위해, CLIP은 사전 학습동안 다양한 task를 수행하도록 배우며, 이러한 `task-learning`은 다양한 데이터셋으로의 zero-shot transfer하기 위해 자연어 프롬프팅을 활용한다.

- 충분한 크기의 모델에서 CLIP의 성능은 task-specific supervised model에 견줄만하다. (+아직 개선의 여지가 있다)
