## :page_facing_up: Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates

### *Info*

* Author: Taku Kudo \[Google\]
* Conf: ACL 2018 (accepted as a long paper)

* URL: https://aclanthology.org/P18-1007/
* Code: https://github.com/google/sentencepiece
* Video: https://vimeo.com/285807834

<br></br>

### *Abstrct*

  * NMT의 `robuestness`를 향상시키기 위해 분할의 모호함(segmentation ambiguity)을 `noise`로 사용할 수 있는가?
    - subword 분할은 같은 vocab으로 여러 개의 분할이 가능하다.
   
  * 학습하는 동안 확률적으로 샘플링된 다수의 subword segmentation을 모델에 학습시키는 subword 정규화 제시
  * 더 나은 subword 샘플링을 위해 `unigram language model`을 기반으로한 새로운 segmentation algorithm 제시

<br></br>

### *Introduction*

- Neural Machine Translation(NMT)

  - NMT는 고정된 크기의 `vocab`에서 작동하며 `vocab size`에 훈련과 추론을 크게 의존한다. (시간)
    
    - 그렇다고, `vocab size`를 제한하자니 정확도가 떨어진다. (OOV 문제)

    - 이에 희소한 단어를 subword를 쪼개서 보는 아이디어가 등장하였고 이 중, `BPE(Bye-Pair Encoding)`가 뛰어난 성능을 보였다. (사실상 표준이 됨)

<br></br>

- BPE(Byte-Pair Encoding) 단점

  - BPE는 `vocab size`와 `decoding efficiency`사이의 좋은 밸런스를 주며 unknown 단어에 대한 특별한 처리법을 도입할 필요가 없게 해준다.

  - **하지만, BPE는 같은 vocab으로도 다수의 subword 분할이 생길 수 있는 단점이 있다. (아래 그림과 같이)**

    <img src="https://user-images.githubusercontent.com/46083287/209085564-eee1dfb3-8636-4ed3-958c-398ea0745be8.png" width="40%"></img>

  - 이런 다양한 변형들은 `Spurious ambiguity`로 보여질 수 있다.

  - NMT를 훈련할 때, 다수의 분할 후보들은 단어의 합성성(구성성)을 학습하는데 간접적으로 도움이 되므로, 모델을 noise와 분할 오류로부터 영향을 받지 않도록(강하게) 만들 것이다.

<br></br>

- **subword regularization**

  - `open-vocabulary NMT`를 위한 새로운 정규화 방법을 제시한다. (called subword regularization)
    - 이는, 다수의 subword 분할을 이용해 NMT model을 더욱 정확하고 강력하게 만드는 것.

  - Two Sub-Contribution
    - 다수의 분할 후보들을 합치는 NMT 훈련 알고리즘을 제시한다. 이 접근은 `on-the-fly data sampling`으로 구현되며, 어떤 `NMT system`에도 적용할 수 있다. (모델의 구조를 바꾸지 않아도 됨)

    - 확률에 따른 다수의 분할을 제공하는 언어 모델을 기반으로 한 새로운 subword 분할 알고리즘을 제시한다. (언어 모델은 실제 데이터가 분할될 동안 생성된 노이즈를 모방할 수 있게 할 수 있음)

<br></br>

## 방법

### *NMT with multiple subword segmentations*

- ***NMT training with on-the-fly subword sampling***

  - source sentence X (X_1, ..., X_m)과 target sentence Y (Y_1, ... ,Y_m)이 있다고 하자. (이는 BPE와 같은 하위 subword 분할기로 분할됨)

  - NMT는 번역 확률 `P(Y|X) = p(y|x)`를 모델링 함. (`target history y<n`과 `입력 문장(x)` 에 따라, target subword y_n을 만드는 target 언어 시퀀스 모델)
    - `θ`는 모델 파라미터
    - 흔한 NMT 구조는 RNN이지만, 위에서 말했듯이 RNN이 아니여도 적용할 수 있다. (e.g. transformer model)


  - NMT는 표준 MLE(maximum likelihood estimation)을 사용해 훈련한다. (주어진 parallel corpus D의 log-likelihood를 최대화)
    - parallel corpus: 두 개 이상의 언어가 병렬로 구성된 것. (ex. source sentence: En -> target sentence: Kor)


  - 여기에서, source sentence X 와 target sentence Y가 다수의 subword 분할로 나뉠 수 있다고 가정한다. (각각 P(x|X), P(y|Y)의 확률)
    - subword 정규화에서는 `marginalized likelihood`로 `parameter set θ`의 최적을 찾는다.

    - 하지만, 위 (3)의 정확한 최적화는 문장 길이에 따라 가능한 분할의 수가 기하급수적으로 증가하기 때문에 실현 불가능하다.
    - 따라서, P(x|X)와 P(y|Y)에서 각각 유한한 k 시퀀스를 샘플링해서 근사치를 구한다.

  - 식을 단순하게 하기 위해 `k = 1`로 주며, NMT의 훈련은 효율성을 위해 online training을 사용한다.
  
  - θ는 D(mini-batch)의 작은 부분집합에 대해 반복적으로 최적화 과정을 거친다. (반복 수가 충분하다면 k가 1이여도 좋은 근사치를 만들어 내는 `online training의 데이터 샘플링 중에 진행되는 subword sampling`이 실행 될 것)
  
    - 각 파라미터 업데이트가 일어날 때마다 subword sequence가 샘플링 되어야 한다.

<br></br>

- ***Decoding***

  - NMT에서의 디코딩에서는 아무 전처리도 하지 않은 source sentence X만 가지고 있다.
   
    - 가장 직관적인 접근은 확률 P(x|X)를 최대화하는 `best segmentation x*`로부터 번역을 하는 것이다.  **[one-best decoding]**

  - 추가로, P(x|X)의 다수의 분할 후보들을 통합하는 **n-best segmentation**을 사용한다.

    - 자세히 말하면, 주어진 n-best segmentation(X_1, ... , X_n)에서 점수를 최대화시키는 `best translation y*`을 고른다.

      - `|y|`는 y에서의 subword 개수,  `λ`는 짧은 문장에 패널티를 주기 위한 parameter

<br></br>

### *Subword segmentations with language model*

- Byte-Pair Encoding(BPE)

  - BPE는 문장을 개별적인 문자로 쪼갠 후 가장 많이 등장한 문장의 인접 쌍을 합치는 과정을 원하는 `vocab size`를 만족할 때 까지 반복하는 것이다.

  - BPE는 vocab size와 step size(문장을 인코딩하는데 필요한 토큰 수)의 효과적인 밸런스를 가질 수 있으며, 작은 고정된 크기의 vocab으로 문장을 인코딩하는데 필요한 심볼 수를 만족시킬 수 있다. (이 심볼 수는 크게 늘어나지도 않을 것이다.)

  - 하지만, BPE는 `greedy`하고 `deterministic`한 심볼 교체에 기반하므로 **확률에 따른 다수의 분할을 제공할 수 없다.**
    - BPE를 분할 확률 P(x|X)에 의존하는 subword 정규화에 적용하는 것은 중요한 문제이다.

<br></br>

- ***Unigram language model***

  - **본 논문에서는 확률에 따라 다수의 단어 분할을 뽑아낼 수 있는 unigram language model을 기반으로한 새로운 subword 분할 알고리즘을 제안한다.**

  - 각 subword는 독립적으로 일어난다 가정하며, 결과적으로 `subword sequence (X = (X_1, ... , X_m)`의 확률은  subword 출현 확률 p(x_i)이 만들어내는 것으로 공식화 된다. (`V`는 미리 정의된 vocabulary)

  - 입력 문장 X를 위한 가장 확률높은 분할 `x*`가 주어진다.

    - `S(X)`는 입력 문장 X로부터 생성된 분할 후보들의 집합. (`x*`는 **Viterbi 알고리즘**으로 얻어짐)

  - `vocabulary V`가 주어진다면, subword 출현 확률 `p(x_i)`는 `p(x_i)`를 hidden 변수로 가정하고 다음 marginal likelihood L을 최대화하는 EM 알고리즘을 통해 추정한다.

  - 하지만 실제로, `vocabulary set V`는 알 수 없다. (vocabulary set과 subword 출현 확률의 joint optimization은 다루기 힘들기 때문)
    - 따라서, 다음 반복 알고리즘을 통해 `vocabulary set V`를 찾는다.

      1. 휴리스틱하게(경험에 기반해) Training corpus에서 큰 `seed vocabulary`를 만든다.

        - 이를 만드는 가장 자연스러운 선택은 모든 문자의 조합과 가장 빈도 높은 substring을 사용하는 것. (충분한 merge operation의 BPE도 가능)
        - 가장 빈도 높은 substring ==> Enhanced Suffix Array Algorithm
    
      2. `vocab size`가 원하는 크기에 도달할 때까지 아래의 과정을 반복한다.

         - vocabulary 집합을 고정시켜 놓고, EM 알고리즘을 이용해 `p(x)`를 최적화 한다.

         - `각 subword X_i`에 대해 `loss_i`를 계산한다. (`loss_i`는 `subword X_i`가 현재 vocab에서 제거되었을 때 `likelihood L`이 얼마나 줄어드는지를 나타냄)

         - `loss_i`로 심볼들을 정렬한 후, 상위 n%의 subword만을 취한다. (ex. n=80)

         - 단, 하나의 문자로 이루어진 subword는 무조건 취한다. (OOV 방지)
  
  - 마지막 vocabulary V는 말뭉치의 모든 개별 문자가 포함되어있으므로, 문자 기반 분할 또한 분할 후보 `S(X)`의 집합에 포함되어 있다.

    - 즉, unigram language model을 이용한 subword 분할은 `character`, `subword`, `word segmentations`의 확률적 혼합으로 볼 수 있음.

<br></br>

- ***Subword sampling***

  - subword 정규화는 각 파라미터가 업데이트 될 때마다, `P(x|X) 분포`에서 하나의 subword 분할을 샘플링한다.

    - 대략적인 샘플링을 위한 간단한 접근법은 `l-best segmentation`을 사용하는 것이다.
    
    - 구체적으로, 확률 P(x|X)에 따른 `l-best segmentation`을 구한다. (`l-best search`는 Forward-DP Backward-A 알고리즘을 사용하면 선형시간으로 작동)

    - 그러면, 하나의 분할 `X_i`가 아래와 같은 다항 분포로부터 샘플링된다. (`α`= 분포의 smoothness를 조절하는 하이퍼 파라미터)
      - `α`가 작을수록 uniform한 분포에서 샘플링되고, 커질수록 Viterbi 분할을 선택하게 된다.


    - 이론상 `l-> ∞`으로 설정하면 모든 가능한 분할을 계산에 넣을 수 있지만 `l`이 증가하면 문장 길이에 따라 후보의 수가 기하급수적으로 증가하기 때문에 실제로는 불가능하다.
      
      - 따라서, 정확하게 모든 가능한 분할을 샘플링하기 위해 `Forward-Filtering and Backward-Sampling (FFBS)` 알고리즘을 사용한다. (베이시안 히든 마르코프 모델로 처음 소개된 dynamic programming의 변형)

        - `FFBS`에서 모든 분할 후보들은 촘촘한 격자 구조로 표시되며 각 노드는 subword를 나타낸다.

        - 첫 번째 pass에서, FFBS는 격자 속 모든 subword의 `foward probabilities`의 집합을 계산한다. (이는 어떤 subword w로 끝날 확률을 주는 것)
        - 두 번째 pass에서 문장의 끝에서 처음까지 격자 안의 노드들을 타고 가면, `foward probabilities`에 따라 각 branch에서 재귀적으로 subword가 샘플링된다.

<br></br>

- ***BPE vs Unigram language model***

  - BPE는 텍스트를 인코딩하기 위한 심볼의 총수를 줄이기 위해 심볼의 집합을 점진적으로 찾는 `dictionary encoder` 이다.

  - Unigram language model은 텍스트의 총 코드 길이를 줄이는 `entropy encoder`로 재구성된다.
    - 클라우드 섀넌의 코딩 정리에 따르면 심볼 s의 이상적인 코딩 길이는 `log(p_s)`이며 `p_s`는 s의 출현 확률이다. (이는 위의 unigram model에서의 (7)식의 표현과 같음)

  - BPE와 Unigram LM의 공통 아이디어: 특정 데이터 압축 원리로 더 적은 비트로 텍스트를 인코딩
  
    - 차이점: Unigram LM은 확률적 언어 모델에 기반하고 있어 더 유연하며, subword 정규화에 필수 요건인 확률에 따른 다수의 분할을 출력할 수 있다.

<br></br>

## 실험

### *Setting*

<div align ="center"><img src="https://user-images.githubusercontent.com/46083287/209086017-3e150d55-c953-410f-94af-50ec60677108.png" width="70%"></img></div>

  - 다양한 크기와 언어를 가진 여러 말뭉치로 실험.
    - `IWSLT15/17`, `KFTT`: 다른 언어적 특성을 가진 넓은 스페트럼의 언어를 포함하는 상대적으로 작은 말뭉치(단일 언어에서 subword 정규화 평가)
    - `ASPEC`, `WMT14`: 중간 크기 말뭉치 (WMT14가 더 큼)
    
  - `GNMT`: 모든 실험을 위한 NMT system의 구현체 (Wu et al., 2016에 서술된 내용을 그대로 따름)
  
    - 하지만, 말뭉치 크기에 따라 설정을 바꾸었음. (위의 표 parameter 부분 참조)
    

  - 기본 세팅으로, `dropout = 0.2`로, parameter estimation을 위해 `Adam`과 `SGD` 사용.
  
    - `length normalization` 과 `converge penalty parameter`를 모두 0.2로 설정, `decoding beam size`를 4로 설정


  - subword model을 훈련하기 전에 데이터는 `Moses tokenizer`로 전처리되었다.
  
  - 중국어와 일본어는 띄어쓰기가 없어 경계를 찾을 수 없으므로, 거의 분할되지 않은 원본 문장임을 알아야한다.

  - 평가지표로 `BLEU` score 사용. (중국어와 일본어는 BLEU 계산 전에 문자와 KyTea로 분할되었음)


  - BPE 분할은 baseline system을 사용함. (다른 샘플링 전략을 적용한 3가지 test system으로 평가)
    
     1. subword 정규화가 없는 Unigram LM 기반 subword 분할(l=1)

     2. subword 정규화가 있는 Unigram LM 기반 subword 분할(l=64, α=0.1)
    
     3. subword 정규화가 있는 Unigram LM 기반 subword 분할(l= ∞, α=0.2(IWSLT)/0.5(others))

        - 이 샘플링 파라미터는 예비 실험으로 결정되었음. (`l=1`은 순수하게 BPE와 unigram LM을 비교를 목표로 함)

  - 추가로, `one-best decoing` 과 `n-best decoding`을 비교할 것임. (BPE는 다수의 분할을 제공할 수 없으므로, one-best decoding만 평가)

<br></br>

### *Main Results*

<div align ="center"><img src="[/uploads/c27ffdaf1a64e72e88490839731e51db/image.png](https://user-images.githubusercontent.com/46083287/209086085-8fd4248e-2dd5-4629-9ab8-113bd261cf9a.png)" width="70%"></img></div>

  - subword 정규화가 없는 Unigram LM은 BPE와 비슷한 BLEU를 보였다.

  - `WMT14`를 제외한 모든 언어 쌍에서 BLEU가 1~2 점 향상되었다. (특히, 크기가 작은 말뭉치에서 큰 효과가 있었음)
    - 이는 subword 정규화의 data augmentation의 긍정적 효과로 볼 수 있다.

  - `(l=64, α=0.1)`와 `l= ∞, α=0.2(IWSLT)/0.5(others)`는 비슷한 점수를 보임.

  - `n-best decoding`이 많은 언어 쌍에서 더욱 성능을 좋게 만들었다.
    - 그러나, subword 정규화가 `n-best decoding`에서 필수이기에, subword 정규화가 없는 언어 쌍에서의 BLEU score가 낮게 나왔음에 주목해야 한다.
    - 이는 훈련 시간에 decoder가 탐색하지 않으면 다수의 분할을 더 혼란스러워 한다는 것을 뜻한다.

<br></br>

### *Results with out-of-domain corpus*

<div align ="center"><img src="[/uploads/1861065bcb27bf330b833ece83e7535c/image.png](https://user-images.githubusercontent.com/46083287/209086184-62afbfc3-45d2-4192-af6a-e65fc7da6e5b.png)" width="40%"></img></div>

- `open-domain`에서의 subword 정규화의 효과를 보기 위해, 회사 내부의 다양한 장르(Web, patents, query log)로 시스템을 평가했다.
  - `KFTT`와 `ASPEC` 말뭉치는 말뭉치가 너무 특정되어 있어 제외함. (예비 실험에서 성능이 많이 떨어졌음)

- 모든 말뭉치의 도메인에서 큰 성능 향상(+2)을 볼 수 있었다. (위의 표3보다 큰 성능 향상)
  - 큰 데이터 셋(WMT14)에서도 같은 수준의 향상을 볼 수 있음.

- **이는 open-domain setting에서 subword 정규화가 더 유용하다는 본 논문의 주장을 강하게 뒷받침한다.**

<br></br>

### *Comparison with other segmentation algorithms*

<div align ="center"><img src="https://user-images.githubusercontent.com/46083287/209086242-5c10bb28-89aa-4363-bb90-bdc1b08afaa9.png" width="40%"></img></div>

- 독일어는 형태학적으로 풍부한 언어여서 `word model`에서 큰 vocab을 필요로 한다.
  - subword-based 알고리즘이 1 BLEU score 높았다.

- subword-based 알고리즘 중 Unigram LM(with subword 정규화)가 best BLEU score를 보여줬으며 이는 다수의 subword 분할의 효과를 증명한다.

<br></br>

### *Impact of sampling hyperparameters*

<div align ="center"><img src="[/uploads/5ff8a30eed8367293868b8116ff1a3bd/image.png](https://user-images.githubusercontent.com/46083287/209086267-b2c477e3-a87b-4450-94c2-a79f2bea0a86.png)" width="40%"></img></div>

- subword 정규화에는 2가지 하이퍼 파라미터가 있다. [`l`: 샘플링 후보의 크기, `α`: smoothing 상수]
  - 위는 IWSLT15(en -> vi) 데이터셋에서의 하이퍼파라미터의 변화에 따른 BLEU 점수를 보여준다.

-  `α`에 대한 BLEU 점수가의 피크가 샘플링 크기 l에 따라 다른 것을 볼 수 있다. (이는 l=64보다 ∞가 검색 영역이 더 크고, 샘플 시퀀스가 Viterbi 시퀀스 x*에 가깝게 되도록 α를 크게 설정해야되기 때문)

- `α=0`이면 `l=∞`에서 성능이 뚝 떨어진다. (P(x|X)가 사실상 무시되고, 하나의 분할만이 샘플링되기 때문)
  - 이는 language model의 편향된 샘플링이 실제 번역에서의 노이즈를 모방하는데 도움을 주는 것을 의미한다.

- 일반적으로 `l`이 크면 적극적인 정규화가 가능하며 크기가 작은 말뭉치(ex. IWSLT)에서 효과적이다.

- `α`를 결정하는 것은 매우 민감하고 심지어 baseline보다 성능을 낮게 할 수 있다.(ex. α가 매우 작을 떄)
  - 따라서, 이를 피하기 위해 큰 말뭉치에서는 `l=64`를 사용하는 것이 합리적이다.

- subword 샘플링에서 `l`의 최적값을 고르는 것은 아직 열려있는 문제이다.

<br></br>

### *Results with single side regularization*

<div align ="center"><img src="https://user-images.githubusercontent.com/46083287/209086297-4bd52bbb-d5f1-403c-a5aa-43c5670ffc1b.png" width="40%"></img></div>

- subword 정규화가 source와 target sentence에서 어떤 것에 더 영향을 줄지 비교.

- 예상대로, full regularization이 가장 좋았음.

- 그러나, single side regularization 또한 긍정적인 효과를 가지고 있음에 주목해야 한다.

  - 이는 encoder-decoder 구조에 도움을 줄 뿐만아니라, encoder나 decoder를 사용하는 NLP task에 적용할 수 있음을 의미한다.
  - NLP task: text classification, image caption generation 등

<br></br>

## 결론

- 해당 논문에서는 신경망 구조의 변경없이 NMT에 적용할 수 있는 간단한 정규화 방법(subword regularization)을 제시했다.

- 핵심 아이디어는 `on-the-fly subword sampling`을 통해 실제 학습 데이터를 늘림으로써, NMT 모델의 강력함과 성능을 모두 향상시키는 것이다.
  
  - 추가로, 더 좋은 subword sampling을 위해 Unigram LM에 기반한 새로운 subword segmentation 알고리즘을 제안했다.

  - 다른 크기와 언어의 다양한 말뭉치를 통한 실험을 통해 작은 말뭉치나 open-domain setting에서 좋은 성능을 이끌어내는 것을 보였다.

- 향후 연구로는 encoder-decoder 구조를 기반으로하는 NLP task에 subword 정규화를 적용하는 것이다.
  - ex. dialog generation, automatic summarization
  - 이는 machine translation에 비해 학습 데이터가 충분하지 않기에 subword 정규화를 통해 향상을 보일 가능성이 크다.
