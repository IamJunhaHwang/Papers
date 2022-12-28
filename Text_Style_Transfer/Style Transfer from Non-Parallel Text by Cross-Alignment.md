## :page_with_curl: Style Transfer from Non-Parallel Text by Cross-Alignment

### Info

* conf: NIPS2017
* url: https://proceedings.neurips.cc/paper/2017/file/2d2c8394e31101a261abf1784302bf75-Paper.pdf
* video(by author): https://youtu.be/OyjXG44j-gs

### 1. Abstract & Introduction

- 비병렬 텍스트를 기반으로한 `Style Transfer`
  - 비병렬은 문장의 쌍이 있는 것이 아닌 하나만 존재하는 것
  - 병렬 텍스트 예시) 오늘 저녁은 맛있어. [긍정] <----> 오늘 저녁은 맛없어. [부정]
  - 비병렬 텍스트 예시) 오늘 저녁은 맛있어. [긍정]

- 문장을 스타일과 내용을 분리하는 것이 주 challenge

- 서로 다른 말뭉치여도 잠재 내용 분포(latent content distribution)을 공유한다고 가정.

- 문장의 내용을 보존하면서 style과 같이 원하는 표현 조건을 가진 문장을 만들 수 있게 해야 함.
  - 즉, style-independent한 내용 vector를 이용해 같은 내용을 가지지만 다른 style로 표현 가능하게 하는 것.

- Encoder에 문장과 기존 style indicator을 입력으로 받아 style과는 독립적인 content representation에 매핑시킨 후, style-dependent decoder을 통과시켜 원하는 것[스타일]을 만듦

- VAE를 사용하지 않음. [잠재 내용 표현에 대해 변화가 없도록 만들기 위해]

- 잠재 표현에 대해 다른 말뭉치간의 align을 하는 것은 어렵기 때문에 제약이 걸린다.
  - cross-generated(style-transferred) sentence에서 두 개 분포의 alignment constraint가 있음
  - 예를 들어, 부정 문장으로 스타일이 바뀐 긍정 문장은 주어진 부정 문장 셋에서 생성됨.

- 3가지 task로 평가: sentiment modification, decipherment of word substitution ciphers, and recovery of word order

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209753461-5256f1ab-033a-4caf-aebf-33d765f8257f.png" width="60%"></img></div>


### 2. Related Work

1. Style transfer in vision

  - CV애서 내용과 스타일 특징을 추출해 이들을 이용해 조합함으로써 새로운 이미지를 만들어 냈음.
  - 주 challenging은 비병렬적인 세팅에서 두 도메인을 align하는 것.
  
  - **해당 논문에서는 CV에서 적용되었던 것들을 NLP로 가져오면서 이에 따른 새로운 방법을 제안**

2. Non-parallel transfer in natural language

  - `Mueller et al.,2017`은 원하는 특성을 만족하는 문장을 만들기 위해 hidden representation을 처리
  - `Hu et al.,2017`는 해석 가능한 잠재 표현을 학습 시킴으로써 속성을 컨트롤할 수 있는 문장을 만들었음
  
  - **해당 논문에서 제시한 모델은 분포 상의 cross-alignment를 구성함.**

3. Adversarial training over discrete samples

  - 최근 RNN으로 Adversarial training over discrete samples가 다루어지고 있음.

  - **해당 논문에서는 Professor-Forcing Algorithm을 사용함.**

<br></br>

### 3. Formulation

데이터가 아래 과정으로 생성되었다고 가정함.

  1. 잠재 스타일 변수 y는 분포 $p(y)$에서 생성됨
  2. 잠재 내용 변수 z는 분포 $p(z)$에서 생성됨
  3. 데이터 x는 조건부 확률 분포 $p(x|y,z)$에서 생성됨.
 
같은 내용 분포를 가지지만 다른 스타일(y1, y2)를 가지는 두 데이터 셋이 있으며, 스타일은 어떤 것인지 모름.

- 두 데이터 셋은 각각 $p(x_1|y_1)$, $p(x_2|y_2)$의 샘플들로 구성.

- 위 두 분포 사이의 `style transfer function`을 구하고 싶음. => $p(x_1|x_2; y_1, y_2)$ , $p(x_2|x_1; y_1, y_2)$

$x_1$과 $x_2$의 marginal distribution을 알고 joint distribution을 계산하려 함. [ $p(x_1|y_1)$, $p(x_2|y_2)$ 를 알고 있는 것]  ==> 따라서, y와 z를 알아야 함.

- $p(x_1, x_2|y_1, y_2) = \int_z p(z)p(x_1|y_1, z)p(x_2|y_2, z)\mathrm{d}z$
- 혼란을 막기 위해 $x_1, x_2$는 각각 다른 y에서 생성된다고 제약을 둔다.

아래에 모델 가정에 따라 transfer가 가능한지 불가능한지 살펴 본다.

#### 3-1. Example 1: Gaussian

$z \sim N(0, I)$ 를 `centered isotropic Gaussian distribution`로 선택해보자.

style $y = (A, b)$ 를 `아핀 변환(affine transformation)`이라 가정하자.

  - i.e) $x = Az + b + \epsilon$
  - $\epsilon$은 노이즈 변수, b가 0 이고 A가 직교행렬이면 $Az + b ∼ N(0, I)$
  - 따라서, x는 어떤 스타일 $y = (A, 0)$에 대해 같은 분포를 가지게 됨.

만약 `z`가 a Gaussian mixture와 같이 더 복잡한 분포를 가진다면, 아핀 변환은 unique하게 결정됨.

<img src="https://user-images.githubusercontent.com/46083287/209753508-289b7e01-b17a-48c4-b6b4-99f12b55cfb0.png" width="60%"></img>

#### 3-2. Example 2: Word substitution

`z`가 bi-gram language model이고 style y는 `content word`가 lexical form으로 매칭되는 vocabulary라고 하자.

$x_1, x_2$ 가 같은 언어 `z`라고 한다면 이는 둘 간의 word alignment를 구하는 것이 된다.

$M_1, M2 \in R^{nxn}$ 가 각각 데이터 X1, X2의 bi-gram 확률 치환행렬(ohe-hot encoding)라 하면 word alignment는 치환행렬 P를 찾는 것과 같다. 

  - ex) $P^TM_1P \approx M_2$
  - 이는 optimization 문제가 될 수 있다. ==> $\displaystyle\min_P \parallel P^TM_1P-M_2 \parallel^2 $

이 식은 graphic isomorphism(GI) 문제이고 P가 유일함을 보이는 것은 어렵다. 그래프가 충분히 복잡하다면 isomorphism을 보이기 쉬울 것

<br></br>

### 4. Method

style transfer function을 학습 시키기 위해서는 조건부 분포인 $p(x_1, x_2|y_1, y_2)$ 와 $p(x_2, x_1|y_1, y_2)$를 필수적으로 학습시켜야 함. CV와 다르게 NLP는 불연속적이고 $x_1, x_2$ 가 주어진 잠재 내용 변수 z에 조건부 독립이므로 아래와 같이 표현할 수 있다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209754032-60201883-b45e-4d1a-8588-af074f241e0f.png" width="60%"></img></div>

이는 auto-encoder 모델로 학습할 수 있다. 

  - encoding step: x2’s content $z ∼ p(z|x2, y2)$
  - decoding step(transfered sentence): $p(x1|y1, z)$

따라서, reconstruction loss는 아래와 같다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209754039-00bd4920-036d-477d-83ad-1f90814a2d35.png" width="60%"></img></div>


X1과 X2의 내용은 일치해야하고 스타일만 바뀌어야 하므로 같은 잠재 분포에서 만들어지게 하기 위해 제약을 주어야 함.

  - VAE의 prior density p(z)를 $z \sim N(0, I)$와 같이 설정하고 KL-divergence regularizer를 사용

    <img src="https://user-images.githubusercontent.com/46083287/209754050-0d2ebeca-4cf2-41ac-8200-332abc7ecc66.png" width="60%"></img>

따라서 전체 목적 함수는 $L_{rec} + L_{KL}$ 을 최소화 하는 것

- standard auto-encoder를 사용하는 것이 좋음. [reconstruction error를 단순하게 줄일 수 있으며 z에 대한 제약을 만족할 수 있음

#### 4-1. Aligned auto-encoder

$p_E(z|y1) and p_E(z|y2)$ 각각 align하는 것은 아래와 같은 optimization problem으로 나타낼 수 있다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209754114-0cb41764-ff40-497d-8a4a-133283a0eb26.png" width="60%"></img></div>

실제로는 위 식의 `Lagrangian relaxation`를 적용한 것을 optimize 하는 것임.

  - 다른 스타일에서 z의 aggregated posterior 분포를 align하기 위해 adversarial discriminator D를 도입
  - D는 두 개의 분포 사이를 구분하는 것을 목표로 함.

    <img src="https://user-images.githubusercontent.com/46083287/209754124-d1cdc43b-c762-437a-85da-1e40e48e9f30.png" width="60%"></img>

전체 훈련 목적 함수는 다음과 같다. => $\displaystyle\min_{E,G} \displaystyle\max_D L_{rec} - \lambda L_{adv}$

인코더 E와 제네레이터 G는 single-layer RNN with GRU cell을 이용하고 discriminator D는 1개의 hidden-layer를 가지고 sigmoid output layer를 가지는 FFNN으로 구현

#### 4-2. Cross-aligned auto-encoder

두 개의 discriminators를 사용함. [하나는 real x1과 바뀐 x2를 구분, 다른 하나는 real x2와 바뀐 x1을 구분]

Adversarial training은 G gradient BP에 문제가 있기 때문에 두 가지 방법을 적용 함.

- Softmax distribution over words를 input으로 사용 [temperature softmax 사용]

- Professor-Forcing 사용

  - hidden state 시퀀스를 매치시키기 위해
  - Teacher-forcing으로 생성된 부분은 real, non teacher-forcing(free-running)부분은 fake로 학습

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209753559-a507440a-1313-4a44-ae6e-e01d428eeee9.png" width="60%"></img></div>

학습 알고리즘 또한 논문에서 상세히 Pseudo Code로 설명

### 5. Experimental setup

- Sentiment modification [긍정 <-> 부정 변환]

  - Yelp restaurant reviews를 사용 [평점 3점 이상을 긍정, 3점 미만을 부정으로]
  - 모델은 문장별로 동작하지만 데이터셋은 문서별로 레이블링 되어 있다.
  - 중립과 같은 문장이 있을 수 있으므로 긴 문장에 흔히 나타나는 이것을 지우기 위해 10 문장이 넘어간 것은 삭제함, 15 단어가 넘는 문장도 지움.
  - Neagtive: 250k, Positive: 350k, vocab: 10k (빈도수 5개 이하는 <unk>처리)

  - 정량적 평가를 위해 model-based 평가 진행. [Yoon Kim(2014)의 TextCNN 이용]
  - 랜덤으로 고른 500개 샘플에 대해 사람이 평가
    - 첫 번째 평가: Fluency(1~4점) / Sentiment(긍정, 부정, 둘 다 아님) 선택 했는가 평가
    - 두 번째 평가: context는 같고 sentiment만 잘 바뀌었는지 평가

- Word substitution decipherment

  - 암호문에서 평문으로 바꾸는 task [cipher는 평문의 단어와 1대1매칭 using non-parallel data]

  - 학습시 X1: 200k, X2: 200k가 non-parallel하게 존재하고 dev/test는 X1: 100k, X2: 100k가 parallel 존재
  - 대체 단어 수에 따라 해당 task의 난이도가 결정되기 때문에 대체 단어 vocab의 크기에 따라 model 성능 비교

  - 단어 빈도수를 기반으로하는 간단한 해독 baseline과 비교

- Word order recovery

  - Original English sentences X1 과 shuffled English sentences X2 사이의 transfer를 하는 것
  - 해독 task와 같이 train은 non-parallel로, test는 parallel로 진행

<br></br>

### 6. Results

- Sentiment modification

  <img src="https://user-images.githubusercontent.com/46083287/209753588-f6279abd-a7e0-49e8-940f-813a988bd427.png" width="60%"></img>

  <img src="https://user-images.githubusercontent.com/46083287/209753598-4942b39d-3b36-4868-ad64-4c0be38dee7a.png" width="60%"></img>

정량적으로는 `Hu et al. 2017` 모델이 좋았지만 Table 3를 확인하면 정성적으로는 제안한 모델이 더 잘 만들었다. [저자의 주관적 주장]

style transfer를 위한 평가 방법이 필요하다고 함.

- Word substitution decipherment

  <img src="https://user-images.githubusercontent.com/46083287/209753603-2a90e670-9ebf-43bd-9542-0a152b8cb8e4.png" width="60%"></img>

치환율 100% 일때 제한한 논문을 제외하고는 다른 모델들의 성능은 현저히 떨어졌음.

- Word order recovery

Non-parallel의 경우는 제안한 모델만 문법적으로 reorder가 가능한 수준임.

### 7. Conclusion

non-parallel data로 해독 문제를 하는 task를 처리함.

두 데이터 셋이 잠재 변수 생성 모델로 생성되었다고 가정하며 잠재 공간이나 문장 모집단에 distributional alignment를 강제함으로써 신경망을 최적화 시킴.

위에서 이야기한 태스크들을 통해 정량적 정성적 평가를 하였고 해독 측면에서 아래의 open question을 남김

**when can the joint distribution $p(x1, x2)$ be recovered given only marginal distributions?**

주변 분포만이 주어졌을 때 언제 결합 분포를 얻을 수 있는가?
