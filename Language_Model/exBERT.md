## :page_facing_up: exBERT: Extending Pre-trained Models with Domain-specific Vocabulary Under Constrained Training Resources

### Info

* `conf`: EMNLP2020-FINDINGS

* `author`: Wen Tai, H. T. Kung, Xin Dong, Marcus Comiter, Chang-Fu Kuo
* `url`: https://aclanthology.org/2020.findings-emnlp.129/

- `github`: https://github.com/cgmhaicenter/exBERT

### 1. Abstract & Introduction

생물의학(biomedical) 쪽은 도메인에 특화된 단어들이 많기 때문에 언어 모델을 사용하기 위해 이를 확장시켜야 했다. 이전에는 밑바닥부터 다시 훈련하는 방법과 이미 사전 훈련된 모델을 가져와 쓰는 방법을 사용했다. 하지만, 전자는 training resource가 많이 필요했으며 후자는 완전히 딱 들어맞는 vocab이 아니기 때문에 `sub-optimal`한 성능을 보였다.

따라서, 제한된 training resource에서도 vocabulary의 추가 확장을 통해 `General -> Specific` 도메인으로 BERT를 확장하는 훈련 방법을 소개한다. (called `exBERT`)

`exBERT`는 기존 BERT의 context embedding에서 새로운 도메인 학습을 위해 augmenting embedding을 적용하는 조그만 확장 모듈을 사용한다. 기존 BERT의 가중치는 고정시키면서 새로운 vocab과 모듈을 학습시키는 것은 이전에 없던 훈련 방식이다. (이 당시)

생물의학(biomedical) article 중 `ClinicalKey and PubMed Central`로 사전 학습했으며 제한된 말뭉치와 훈련 리소스에서 `exBERT`가 이전 방법들에 비해 뛰어났다.

<br></br>

### 2. exBERT Approach

exBERT를 위해, 기존 BERT embedding을 확장했으며 이에 따른 `domain-specific`한 vocab을 추가하고 각 트랜스포머 층에 확장 모듈을 더했다.

#### 2-1. Extension Vocabulary and Embedding Layer

- **방법**

  1. `Target Domain`으로부터 Vocab을 얻은 후에 기존 BERT의 Vocab과 겹치는 것은 삭제한다.

  2. 이후에 남은 것들에 대한 embedding layer을 추가한다. 이것의 가중치들은 랜덤하게 초기화되며 pre-training 동안 학습된다. ( 밑바닥부터 vocab을 다시 학습하는 `SciBERT`와 다름)

- 우리는 추가하는 Vocab의 크기를 다르게 조절해보았지만, 성능에 미치는 영향은 미미했으며 크기가 커질수록 vocab 수렴 시간이 오래걸렸다. 따라서, 상대적으로 작은 것을 선택했다. 

  - 이는 명확한 `drop off`가 없기 때문. (drop off = 일반적으로 주어진 말뭉치나 데이터셋에서 가장 일반적인 단어에서 덜 일반적인 단어로 이동할 때 발생하는 단어 발생 빈도의 감소)

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/227826465-389ad661-7f4b-43fa-9841-a86ae7848cb9.png" width="40%"></img></div>

위 그림과 같이 주어진 문장에 대한 embedding은 `Original + Extension`으로 구성되게 된다. 하지만 여기에는 2가지 이슈가 있다.

- **이슈**

  - `Extenion vocab`에 대한 Embedding vector는 pre-trained BERT가 모르는 상태임

  - 기존 vocab의 토큰 표현이 `general domain -> target domain` shift가 일어날 수 있다.

    - 같은 단어여도 각각의 도메인 상에서 의미가 달라질 수 있다. (스타일, 형식, 의도에 따라 달라짐)

  - 위 두 가지를 해결하기 위해 `weighted combination mechanism`을 적용했다. 

#### 2-2. Extension module

위 그림의 `b`처럼 확장 모듈을 추가해 기존 BERT(`off-the-shelf`)를 확장한다.

`off-the-shelf`( $T_{ofs}(\cdot)$ )와 확장 모듈( $T_{ext}(\cdot)$ )을 조합하기 위해 아래와 같은 가중합을 사용한다.

$H^{l+1} = T_{ofs}(H^l)\cdot \sigma(w(H^l)) + T_{ext}(H^l)\cdot (1-\sigma(w(H^l))$

- $H^l$ : `l` 번째 층의 output

- $w$ : 가중치 블록 [`768 x 1`의 FCN]

- $\sigma$ : sigmoid 함수 (출력의 크기를 일정하게 하기 위해 사용)

<br></br>

### 3. Experiment setup

- exBERT Adaptive Pre-training

  - `BERT`: BERT-base-uncased

  - `Extension module` : 작은 크기의 BERT 역할을 하는 transformer-based 구조

    - 기존 BERT는 학습 중 고정되어 있으며 Extension과 weighting block만 학습됨.

  - `optimizer` : Adam ($\beta_1$ = 0.9, $\beta_2$ = 0.999)

  - `Learning Rate` : 1e-4

  - `Hardware` : V100 GPU x 4

  - `Batch-Size` : 256

  - `max_length` : 128

  - `Pre-training Data` : 17GB biomedical corpus (ClinicalKey:2GB, PubMed Central:15GB)

  - `epochs` : 1 (github상)

- Fine-tuning

  - `Downstream Tasks` : NER(Named Entity Recognition), RE(Relation Extraction)

  - top-three layer만 fine-tune 됨.

  - `Learning Rate` : $10^{-5}$

  - `Batch-Size` : 20

  - `epochs` : 3

  - `Data` : MTL-Bioinformatics-2016 dataset


exBERT를 pre-training Data의 5%만 샘플링해서 3 epoch 학습시켜 보았음. (데이터, 컴퓨팅 리소스가 제한된 조건으로 가정)

여기에 extension module size, off-the-shelf module size, FFNN size를 5가지로 다르게 비교하였으며 original BERT, BioBERT, rrBioBERT에 대해 비교했음. [앞서 얘기한 제한된 조건하에]

<br></br>

### Impact of the Exention Module Size

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/227826486-a194c0eb-9d3f-4523-99d2-35c1bf71e8dd.png" width="40%"></img></div>

위의 그림을 보듯이 exBERT가 적은 extension module size(16.3%)로 `rrBioBERT`를 능가했다. 이는 exBERT 방법이 효율적이고 효과적인 것을 의미한다. (extension 모듈의 파라미터가 충분해지면 성능이 안정적이게 됨, 이후 실험에서는 이 모듈의 크기를 33.2%로 사용)

또, extension vocab을 쓰되 extension module을 쓰지 않는 실험을 했는데 성능이 좋지 않았다.(위 그림의 파란색 선 중 0% module size 부분) 이는 extension module이 중요한 역할을 함을 의미한다.

마지막으로, extension vocab을 쓰지 않고 extension module만을 쓰는 실험(위 그림의 검은색 선)을 했다. 결과를 보면 exBERT의 성능은 extension module 뿐만 아니라 extension vocab에서도 오는 것을 알 수 있다. 

- **[내 생각]: 어찌보면 당연한 부분같다. extension module만이 domain에 대한 정보를 담고 있을 것이다. (기존 BERT는 pre-train에서 고정된다고 했으니까)**

<br></br>

### Impact of Training Time on Performance

훈련 시간별 성능을 4시간마다 oiBioBERT(our-implemented BioBERT)와 비교. [같은 hardware, corpus 그리고 BioBERT와 같은 방법으로 구현함]

`extension module`의 연산 때문에 exBERT가 oiBioBERT보다 진행이 느리다. 그럼에도 불구하고 아래 그림을 보면 모든 비교에서 exBERT가 oiBioBERT를 능가했다. 이는 domain에 맞는 vocab을 가지고 있기 때문이라 본다.

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/227826495-15ddaf31-bcd0-448a-a89e-bd4de5bae8af.png" width="40%"></img></div>

또, pre-training(extension만 학습하는)을 마친 후 모든 모듈(기존 BERT도 포함한)을 pre-training하는 과정을 거친 것을 다른 모델과 비교해보았다. [Figure 3-(b)]

동일한 조건에서의 비교를 위해 훈련 시간을 바꾸었음.

결과로 exBERT가 oiBioBERT를 항상 능가했다. 또한, 각 모델의 크기를 disc size로 나타내었는데 기존 BERT보다 큰 것을 볼 수 있다. (extension 모듈때문에 당연한 부분)
