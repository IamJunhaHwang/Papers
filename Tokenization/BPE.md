## :page_facing_up: Neural Machine Translation of Rare Words with Subword Units

### *Info*

* 저자: Rico Sennrich, Barry Haddow, and Alexandra Birch [University of Edinburgh]
* 학회: ACL,2016 (accepted)
* URL: https://aclanthology.org/P16-1162/
* **a.k.a. BPE Subword model**

<br></br>

### *서론(목적)*

* `NMT모델`은 전형적으로 고정된 vocab에서 작동하지만 번역 태스크에는 `open-vocabulary` 문제가 존재한다.
  - `open-vocabulary`: 처음 보는 단어, 희소하게 등장하는 단어
  - 이전 연구에서는 사전(dictionary)을 이용해 돌아보는 식(look-up)으로 OOV 문제를 다루었다.
  - 이 논문에서는 간단하고 더 효율적인 접근법을 제시한다.
    - **희소하게 등장하거나 처음 보는(unknown) 단어를 `subword unit`의 시퀀스로 인코딩하는 방법**

* 간단한 `문자 n-gram 모델`과 `BPE 압축 알고리즘`을 기반으로한 분할을 포함한 `문자 분할 기술`이 적절한지 논의할 것.
  - `BPE`는 가변 길이 문자 시퀀스의 `고정 크기 vocab`를 통해 `open-vocabulary`를 표현할 수 있어 신경망 모델에 매우 적합한 단어 분할 전략이 된다.
  
  - 실제로, `WMT 15 번역 태스크`에서 baseline(back-off dictionary)보다 우수한 성능을 보임.

- 이러한 subword 분할은 다양한 문자 클래스가 단어보다 더 작은 유닛으로 바꿀 수 있다는 직관에 기반한다.

  - 능숙한 번역가(통역가)는 처음 보는 단어여도 형태소와 음소와 같이 이미 알고있는 작은 단위로 나누어 번역할 수 있다.
  - ex. 무기상인 --> <무기>, <상인>, intercept --> <inter>, <cept> 중간에 잡다. 

- **BPE를 이용한 Subword unit segmentation은 unseen word를 생성하거나 번역을 위한 지식을 효율적이고 쉽게 일반화 가능하다.**

<br></br>

### *방법 - Byte Pair Encoding(BPE)*

- `vocab size`와 `text size`의 trade-off를 다루는 쉬운 방법은 `short list of unsegmented words 를 사용하는 것`
  - **그 대안으로 BPE에 기반한 분할 알고리즘을 제안함.**
  - 텍스트의 좋은 압축률을 제공하는 vocab을 배울 수 있게 해줌.

- Step

  1. 각 단어를 문자 단위로 나눈 후, 마지막에 `·(</w>)` 심볼을 붙여준다.

  2. 모든 `symbol pair`를 센다. 그 후, 제일 빈도 수가 높은 pair('A', 'B')를 새로운 심볼('AB')로 `merge`하고 `vocab`에 이를 추가한다.

  3. 사용자가 정한 횟수(hyper parameter)만큼 위 과정을 반복한다.

  4. 완성된 `vocab`의 심볼들을 띄어쓰기 단위로 모두 분리한 후 `unique token`만 남게 만든다.

  5. 마지막 `vocab size` = **initial vocab size + number of merge operations**

<br></br>

- 해당 논문에서는 두 가지 방법을 평가할 것.

  - source/target 각각의 `vocab`을 학습
    - `vocab`의 크기를 작게 만들 수 있으며, 각 subword unit이 training text에 있다는 보장이 있음. [장점]
    - 각각의 언어에서 동일한 단어가 다른 방식으로 segmented 될 수 있어 신경망이 subword units를 mapping하기 어려워질 수 있음. [단점]
    
  - 두 `vocab`의 결합(union)을 학습 --> `joint BPE`라 부름

    - source와 target 사이의 일관성 보장. [장점]
  
    - 영어와 알파벳이 다른 러시아를 번역하기 위해, 러시아 vocab -> 라틴 문자로 번역 후에 BPE 적용.

<br></br>

### *평가*

- 아래 두 가지 질문에 대해 집중할 것임

  - 거의 등장하지 않은 단어에 대해 `subword unit`으로 표현하는 것이 신경망 번역에 얼마나 도움이 되는지

  - `vocab size`, `text size`, `translation quality`가 어느 정도여야지 최상의 성능을 내는지

- 실험 환경

  - 데이터
    - WMT2015 번역 태스크
      - English -> German: 4.2M 문장 쌍, 약 100M 토큰
      - English -> Russian: 2.6M 문장 쌍, 약 50M 토큰

    - `Moses(Koehn etal.,2007)`에서 제공한 스크립트로 토큰화하고 실제 적용한다.

    - `newstest2013`을 val-set으로 사용하며, `newstest2014`와 `newstest2015`의 결과를 보일 것

    - 성능측정: `BLEU`와 `CHRF3`의 대한 결과를 보일 것
      - CHRF3: 영어 이외의 번역에 대해 사람의 판단과 유사

  - 모델

    - `Groundhog(Bahdanau et al., 2015)` - hidden: 1000, embed: 620, shortlist of Tau: 30000 단어
    
    - Optimizer: Adadelta

    - mini-batch: 80, 에폭마다 train-set 셔플

    - 약 7일동안 훈련 후 마지막 4개의 모델을 취함. (모델은 12시간마다 저장되게 했음)

      - 각각의 모델을 `embedding layer`를 고정시킨 후 12시간 더 훈련 시킴.

      - 하나는 gradient clipping을 5.0으로, 하나는 1.0으로 하여 훈련시킨다. ==> 총 8개의 모델(5.0 4개, 1.0 4개)이 만들어짐.

      - 만들어진 9개의 모델을 앙상블함.

    - Beam size: 12 (확률은 문장의 길이로 정규화됨)

  - `baseline`으로 거의 등장하지 않는 단어를 위한 `back-off dictionary`로 `fast-align(Dyer et al.,2013)`에 기반한 이중언어 사전 사용.
    
    - 번역 속도를 올리기 위해 모든 실험에서도 사용. 

    - 필터링된 후보 번역 목록에 대해서만 softmax를 취함.


<br></br>

### *실험*

- 아래는 다른 단어 분할 기법을 통해 학습한 독일어 말뭉치의 통계

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209084319-7e39e969-fa69-4af5-a31a-7e5c0f159c31.png" width="40%"></img></div>

  - 문자 n-gram은 `n`을 어떻게 선택하느냐에 따라 시퀀스 길이와 vocab 크기 간의 `trade-off`가 다르다.

  - uni-gram 표현이 말그대로 open-vocabulary지만(UNK 토큰이 제일 많은) 예비 실험에서 성능이 너무 안좋아서 bi-gram 표현으로 실험할 것. (뒤의 실험)
    - 성능은 더 좋으나, 여전히 training set vocab으로 만들 수 없는 토큰들이 있다.

  - 여러 단어 분할 기법의 통계도 보았지만 이는 `vocab` 크기만 줄였을 뿐, 본 논문의 목적인 OOV 문제에 대한 해결책은 아니므로 적합하지 않음.
    - 여러 단어 분할 기법: compound splitting, rule-based hyphenation, Morfessor

  - **BPE를 사용하면 UNK토큰이 하나도 생성되지 않는 것을 확인함. [본 논문의 목표에 부합]**
    - `글자-단위 모델(character-level model)`과의 차이점: 표현이 간략해져 시퀀스가 짧아지도록 해줌, attention model이 가변 길이 유닛으로 작동하게 해줌.
    - 위의 표에서 merge operation 횟수는 BPE: 59500, joint-BPE: 89500 임.

<br></br>
---------

- `English -> German`, `English -> Russian` 번역 수행 결과
  - `WDict`: back-off dictionary를 사용한 단어-수준 모델 (fast-align)  **baseline**
  - `WUnk`: back-off dictionary를 사용하지 않으며 OOV 단어를 `UNK`로 표현.
  - `unigram F1`: 나뉜 uni-gram의 precision 과 recall의 조화평균.
  - `single & ens-8`: single model & ensemble of 8

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209084395-e0d2e1cd-89da-4670-8b00-9230dc6c4d3d.png" width="70%"></img></div>

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209084419-3f21b6b5-f7d8-4a35-b0b0-3237c1d53687.png" width="70%"></img></div>

- back-off dictionary는 희소 단어에 대한 `unigram F1`을 높여준다. (하지만 고유명사를 글자그대로 옮길 수 없으므로 En ->Ru 에서의 향상은 적다.)

- Unigram F1 socre
  - BPE-J90K(BPE 심볼을 합친 것)이 BPE-60K(각각의 BPE)와 C2-50K(character bigram)보다 좋았다.
  - 모든 subword 모델이 baseline을 능가했다.
  - OOV에 대해서는 같은 알파벳을 사용할 경우 그대로 UNK 단어를 복사하는 baseline의 전략이 잘 통했다. (하지만 En -> Ru 에서는 반대)

- `BLUE`와 `CHRF3`가 일관되지 않아 보이는데, 이는 `BLUE`는 precision bias, `CHRF3`는 recall bias 때문으로 본다.
  - 또, 희소하게 등장하는 단어들이 문장의 중심 의미인 경향이 많아 위의 두 점수가 과소평가되었다고 추측한다.
  - 그럼에도 `subword ens-8` 모델이 어떤 지표를 보아도 뛰어난 것을 보였다.

- 성능 가변성(performance variability)는 여전히 NMT에서 문제임.

- `single model`의 경우 8개 중 좋은 성능을 보인 것을 결과로 썼지만 `randomness`를 다루는 방법은 추후 연구에서 주목할 만함.

<br></br>

### *분석*

- training set에서 빈도수로 정렬된 타겟측 단어를 그래프로 그린다.
  `C2-3/500k(Character Bigram)`는 vocab size에 대한 효과를 분석하기 위해 포함시킴. (WDict baseline과 vocab size 동일)

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209084529-16add585-612c-47d8-abdb-4c0c13426098.png" width="30%"></img></div>

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209084553-3716af6b-d5ce-4d27-a420-c63465cd3445.png" width="30%"></img></div>

- 모든 모델에서 낮은 빈도수 단어의 unigram F1이 낮아지는 경향이 있음.
  - baseline의 경우 OOV에 대한 F1이 쭉 내리꽂게 떨어지는데 이는 고유 명사가 OOV에 많기 때문이므로 그대로 복사해 넣는 것이 좋은 전략.

- 500000 빈도의 단어에서는 subword 모델이 더 좋은 성능을 보였음.

- 50000 빈도 단어와 500000 빈도 단어 사이를 비교했을 때, `C2-3/500k(Character Bigram의 shortlist)`과 `C2-50k(subword model의 shortlist)`의 차이를 확인.

- subword model의 경우가 `밀도가 더 높고`, `network vocabulary의 크기를 줄이`고 `더 많은 단어를 subword로 나타낼 수 있기` 때문에 성능이 좋다.

<br></br>

<div align="center"><img src="https://user-images.githubusercontent.com/46083287/209084652-682be9a4-c3a9-48b4-a508-cc45ea52d167.png" width="30%"></img></div>

- baseline은 모든 예시에 대해 실패.
  - subword 번역은 잘 번역함.

- 분할이 자연스럽지는 않지만 번역자체는 올바른 것을 확인가능.

- `En -> Ru`의 경우, 알파벳이 다르기 때문에(표기하는 언어가 다름) joint BPE가 더 좋은 성능을 보임.

  - 일반 BPE의 경우 문자 하나를 없애고 추가하는 등의 오류가 관찰됨.

<br></br>

### *결론*

- 본 논문의 주된 주장(기여)는 `subword init(using BPE)`으로 희소하게 등장하는 단어들을 표현해 open-vocabulary 번역을 NMT에서 가능하게 하는 것.
  - back-off translation model보다 더 간단하며 효율적.

- `BPE`를 사용함으로써, 가변 길이의 `subword unit`로 구성된 기존보다 작은 vocab을 만들 수 있음.

- 기존 NMT 모델들보다 `subword mode`l을 사용한 모델이 OOV와 희소 단어에 대한 번역에 좋은 성능을 보임.

- 향후 연구로 `language pair`와 훈련 데이터의 양에 따라 자동적으로 최적의 `vocab size`를 학습하는 것이 될 것 같음. (본 논문에서는 `vocab size`를 임의로 정함)
  - 또한, subword unit을 더욱 `alignable`하게 만들기위한 `bilingually informed segmentation algorithm`이 잠재성이 있다고 생각함. (타겟 텍스트에 의존하지 않는)
  - 예를들어, En->Ru 에서 알파벳이 다른(표기 언어가 다른) 것과 관계없이 잘 동작하는
