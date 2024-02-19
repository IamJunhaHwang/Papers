## :page_facing_up: Deduplicating Training Data Makes Language Models Better


### Info

* `conf`: ACL 2022
* `author`: Katherine Lee Daphne Ippolito et al.

* `url`: https://aclanthology.org/2022.acl-long.577.pdf

- `github` : https://github.com/google-research/deduplicate-text-datasets


### 1\. Abstract & Introduction

우리는 대부분의 LM 데이터 셋이 거의 중복되는(near-duplicate) example과 긴 반복적인 substring을 가지는 것을 발견했다. 단순한 중복 제거는 간단하지만, 대규모로 철저하게 중복을 제거하는 것은 컴퓨팅 비용적으로 어려우며 정교한 기술을 필요로 한다. 

- 따라서, deduplication을 위한 2가지 툴을 개발하였으며, 이는 training resource를 줄이면서도 똑같거나 더 좋은 성능을 내게한다.

  - Exact substring matching : 반복되는 verbatim(글자 그대로) 문자열을 식별해 훈련 example이 중복되는 경우를 알아냄.

  - Approximate full document matching : hash기반으로 높은 n-gram overlap인 문서 쌍을 식별함.


- **완전한 중복제거된 데이터 셋을 사용할 때의 장점**

  - generation 할 때, 특정 example을 선호(or 반복)되는 memorized sequence를 줄임

  - Train-Test set 간의 overlap 데이터 줄임

  - 효율적인 학습 가능. (training cost 줄임)

  - 중복 제거가 `perplexity`를 해치지 않음.

- **Contribution**

  - Dataset Deduplication framework 제시

  - deduplication의 영향 분석

<br></br>

### 2. Language Modeling Datasets

LM을 학습하는데 사용되는 다양한 사이즈를 가지는 4개의 데이터 셋에서 중복되는 텍스트가 있는지 분석했다. 또한, 이 논문은 영어 데이터 셋에 한정되지만, 영어가 아닌 데이터셋도 똑같은 이슈가 존재하며 중복 제거의 이득 또한 같다고 생각된다.

- 분석한 4개의 데이터 셋

  - `Wikipedia (Wiki-40B)` : 다양한 언어의 cleaned Wikipedia text로 redirect-pages 제거 말고는 중복 제거하지 않은 데이터 [Guo et al., 2020]

  - `One-Billion Word Benchmark (LM1B)` : 뉴스 코멘터리 문장 30M으로 구성된 데이터 [Chelba et al., 2013]

    - (Radford et al., 2019)는 이 데이터 셋의 test set과 train set이 13.2% 겹치는 것에 주목했음

  - `Colossal Cleaned Common Crawl (C4)` : 360M의 웹 문서 데이터 [Raffel et al., 2020]

    - 앞의 두 데이터셋보다 정교한 중복제거되었음.
       
    - paragraph를 해시화 하고 hash collision이 일어나는 paragraph를 제거  

  - `RealNews` : 뉴스 도메인의 기고글로 이루어진 Common Crawl의 subset [Zellers et al., 2019]

    - 각 문서의 첫 100문자의 hash를 bloom filter(Bloom, 1970)에 집어넣어 hash collision이 일어나는 문서를 제외시킴

<br></br>

### 3. Methods for Identifying Duplicates

가장 간단한 방법은 모든 example 간의 `exact string matching` 이지만, 이는 비효율적이다. 따라서, 중복 제거를 위한 2가지 상호보완적인 방법을 소개한다.

1. `suffix array`를 사용해 2개 이상의 example에서 verbatim이 발생하면, 중복 문자열(substring)을 제거한다.

2. `MinHash`를 사용해 말뭉치 내의 모든 example 쌍 사이의 n-gram 유사도를 측정해 n-gram overlap이 높은 것을 제거한다.

Dataset $D = \{x_i\}^N_{i=1}$ 는 example $x_i$ 의 집합이며 각 example들은 토큰 시퀀스 $x_i = [x_i^1, x_i^2, ... , x_i^{s_i}]$ 이다.

#### 3-1. Exact Substring Duplication [EXACTSUBSTR]

인간 언어 다양성 때문에, 동일한 아이디어가 여러 문서에서 동일하게 표현되는 경우는 드물다. 이러한 점이 중복되는 substring을 제거하는 것에 동기가 된다. 

두 example에서 충분히 긴 substring을 공유한다면 그 중 하나를 제거한다. (통계적 분석을 거쳐 최소 matching substring length인 k를 50 으로 선택)

- Suffix Arrays

  - 효율을 높이기 위해 전체 데이터 셋의 example들을 하나의 문장 $\mathcal{S}$ 으로 합친다.

  - $\mathcal{S}$ 에 대한 suffix array $\mathcal{A}$ 를 구한다.

  - 이렇게 하면, 선형 시간 안에 데이터 셋 안의 중복되는 example을 식별할 수 있다.

- Substring matching

  - suffix array를 만든 후에는 간단하다. 예를 들어, 시퀀스 `s`가 데이터 셋에서 정확히 2번 반복된다고 하자. $S_{i..i+|s|} = S_{j..j+|s|}$

  - 그러면 인덱스 i, j는 suffix array 안에서 서로 인접하게 등장할 것이다. 따라서 모든 중복 시퀀스를 찾는 것은 suffix array를 처음부터 끝까지 선형으로 스캔하고, $\mathcal{A_i}$ , $\mathcal{A}_{i+1}$ 이 최소 임계 길이의 공통 접두사를 공유하는 시퀀스를 찾는 것이다.

#### 3-2. Approximate Matching with MinHash [NEARDUP]

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/dc705cf7-18e7-4e70-b1d7-3e3e0e5f402b" width="60%"></img></div>

`Minhash`는 Jaccard 유사도를 근사하기 위한 방법이다. n-gram 토큰들의 TF 행렬을 문서별로 나타낸 후, hash 함수를 적용해 가장 작은 hash index를 구하는 것이다.

자세한 설명은 다음 블로그에 설명이 잘 되어 있다. -> [블로그 글](https://jimmy-ai.tistory.com/117)

이 논문에서는 5-gram과 9000 signature size를 사용했다.

- 두 문서가 잠재적으로 같다고 간주될 확률은 다음과 같다.

  - $Pr(d_i, d_j | Jaccard(d_i, d_j) = s_{i, j}) = 1 - (1 - s^b_{i, j})^r$

  - `b` 와 `r`은 hyperparameter이며, 각각 20, 450으로 설정했다.

- 위 과정에서 잠재적으로 같다고 식별된 문서의 쌍 각각에 대해 그 다음 필터링 단계가 적용된다.

  - 따라서, `MinHash`에 의해 매칭되고 그들의 `edit similarity`가 0.8보다 높다면 두 문서는 중복된 것으로 본다.

  - 토큰 시퀀스 $x_i$, $x_j$ 사이의 `edit similarity`는 다음과 같이 정의된다.

    - $EditSim(x_i, x_j) = 1 - \frac{EditDistance(x_i, x_j)}{max(|x_i|, |x_j|)}$

<br></br>

### 4. Deduplication Results

위 두 방법을 적용해 앞서 소개한 4가지 데이터 셋에 대해 중복 제거를 진행했다. data split 사이에 중복된 example이 있을 경우, test나 validation에만 남기고 train에는 지우도록 했다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/c5cf8a1f-71b0-47d9-9c5e-4becc66b1b7c" width="60%"></img></div>

web-scrape 데이터 셋이 `NEARDUP`을 사용했을 때, 중복이 더 많이 되는 것을 찾았다.

또, `EXACTSUBSTR`이 평균적으로 `NEARDUP`보다 더 많은 example을 제거하는 것을 볼 수 있으며, `NEARDUP`과 `EXACTSUBSTR`이 비슷하게 example들을 지우는 것을 찾았다. (`LM1B`의 경우 길이가 짧은 문장들로 구성되기 때문에 90%가 50 token 이하이므로 EXACTSUBSTR로 잘 걸러지지 않았다)

- Properties of Duplicated Text

  - RealNews와 C4의 저자들이 중복 제거를 시도했다고 하였지만, 이는 충분하지 않았다.

  - C4 와 Wiki-40B에서 `near-duplicated`로 식별된 텍스트는 대부분 컴퓨터가 만들어낸 내용이었다. (장소, 비지니스, 제품 등의 이름을 제외하면 같음)

  - RealNews와 LM1B는 뉴스 사이트에서 파생된 것어서 동일한 뉴스 기사가 약간 다른 형식으로 여러 뉴스 사이트에 나타나기 때문에 많은 근접 중복이 발생하는 것을 관찰했다.

<br></br>

### 5. Impact on Trained Models

1.5B의 T5를 `C4-original, C4-NEARDUP, C4-EXACTSUBSTR` 데이터 셋에 각각 훈련하였다. random-seed 를 다르게한 110M의 base 모델을 3개씩 추가로 훈련시켰다. (총 9개 base 모델)

각 모델은 2epoch으로 훈련되었고 C4-NEARDUP을 이용한 BPE vocab을 사용했으며 512 max_length로 설정했다. (긴 문서의 경우 이 크기만큼 랜덤하게 subsequence를 추출하였다.)

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/203e7645-7e7b-4123-aeaa-401407256766" width="60%"></img></div>


- Model Perplexity

  - LM1B, Wiki-40B, C4 validation set(unique : 중복 없음, duplicates : 중복 있음) 에 대해 perplexity를 계산했다.

  - `C4-original, C4-Unique`에서는 모든 모델이 비슷한 perplexity를 보였지만, 중복이 없는 데이터셋으로 훈련된 두 모델의 경우 중복이 있는 validation set으로 perplexity를 계산했을 때 original로 훈련한 모델보다 더 높게 나왔다. 

  - LM1B & Wiki-40B에서 중복 제거된 데이터로 훈련된 모델이 perplexity가 더 낮았다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/f2be9c3a-46a8-4bd7-a4b8-e16eaaca1751" width="30%"></img></div>

- Generated Text : 데이터 중복은 LM을 특정 타입의 example에 편향되게 만든다. 이는 텍스트 생성의 다양성을 떨어뜨린다. (우리 실험에서는 top-50 random sampling을 사용했고 prompted, unprompted 생성 실험을 했다.)

  - `No prompt` : 100,000 samples(max_len==512) 생성

    - 각 생성된 토큰에 대해, 훈련 데이터에 정확히 포함된 50개 토큰의 substring의 일부인 경우 해당 토큰을 `암기`되었다고 한다.

    - 위 표에서 `XL-ORIGINAL`에서는 생성된 토큰의 1%가 암기된 토큰이었다. 이는 `XL-EXACTSUBSTR `와 `XL-NEARDUP` 보다 10배 많은 숫자이다.

  - `With prompting` : 4가지 가능한 prompt source로 실험함 `중복이 있는 train set(EXACTSUBSTR로 확인한) / 중복 없는 train set / 중복 있는 valid set / 중복 없는 valid set`

    - 각 example의 첫 32개의 토큰을 프롬프트로 선택한다. (모델은 이 32개의 토큰 이후에 대해 문장을 생성하게 되는데, 이를 통해 실제 값과 near-dup의 비율을 평가할 수 있다)

    - 아래 그림을 보면, 중복이 있는 train set으로 만든 prompt에서 `XL-Original`은 40%이상 실제 값을 다시 만들어냈으며(암기된 것으로 볼 수 있음) `XL-EXACTSUBSTR & XL-NEARDUP` 은 다른 prompt에서보다 더 실제 값을 그대로 출력했다.

    - **이러한 암기를 완전히 제거하려면 보다 엄격한 중복 제거가 필요할 수 있음을 의미한다**

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/b1d46be9-6ec0-4189-802e-acd385c16fec" width="30%"></img></div>

<br></br>

### 6. Conclusion

- 중복 제거 기술을 제안하면서 그 효과에 대해 설명했음

- 중복 제거는 성능을 해치지 않으며, 때때로 높이기도 하였음

- 중복 제거는 memorizing을 줄이므로 개인정보에 대한 걱정을 줄여줄 수 있음
