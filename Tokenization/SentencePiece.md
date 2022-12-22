## :page_facing_up:  ***SentencePiece***:  A simple and language independent subword tokenizer and detokenizer for Neural Text Processing

 

### *Info*

 

- 저자: Taku Kudo, John Richardson

- 학회: Accepted as a demo paper at EMNLP2018

- URL: https://arxiv.org/abs/1808.06226

 

<br></br>

 

### *목적*

 

- Raw sentences로부터 end-to-end로 subword training & segmentation이 가능한 subword model 제시

 

  - **SentencePiece** - language-independent tokenizer & detokenizer

  

    - C++, Python으로 구현한 코드 공개

    

    - vocabulary의 크기는 신경망 모델 학습 전에 결정된다.

    - `BPE`, `Unigram` Algorithm 사용.

 

<br></br>

 

### *소개*

   

##### ***이전의 환경***

 

- Neural Machine Translation(NMT)가 잠재적으로 end-to-end translation을 수행 함에도, 많은 NMT system이 아직도 전(후) 처리기에 의존하고 있다.

 

- 전(후) 처리기는 공백으로 구분되는 European Language을 중점으로 디자인되어 있어 공백으로 구분되지 않는 언어(ex. 중국어, 일본어, 한국어)에서 사용하기에는 한계가 있다.

<br></br>

##### ***System Overview***

 

- SentencePiece의 구성: Normalizer, Trainer, Encoder, Decoder.

 

  - `Normalizer`: 의미적으로 동등한 Unicode 문자를 표준 형태로 바꾸는 모듈.

  - `Trainer`: 표준화된 말뭉치에서 subword 분리 모델을 훈련. (subword 모델의 타입을 파라미터로 특정해줌)

  - `Encoder`: 내부적으로 `Normalizer`를 실행시켜 input text를 정규화하고 Trainer로 훈련된 `subword 모델`로 토크나이징.

  - `Decoder`: subword sequence ==> Normalized text 

<br></br>

#### Commdandline usage of SentencePiece

<img src="https://user-images.githubusercontent.com/46083287/209083530-07ea14c6-1ae5-4dc9-a94b-ae5c401da7fe.png" width="40%"></img>

<br></br>

 

### *방법(About Library Design)*
   

 - ***Lossless Tokenization(무손실 토큰화)***

    - Basic Idea: input text를 Unicode 문자의 Sequence로 취급하자. ( 공백의 경우도 `_(U+2581)`로 취급)

    - SentencePiece는 공백을 `_`로 바꾼 후, input을 임의의 subword로 토크나이징한다.

      - 공백에 대한 정보 또한 포함해서 공백의 모호성 해소.

    - SentencePiece의 Decoder 구현: **`Decode(Encode(Normalize(text))) = Normalize(text)`**
    
      - 정규화된 text를 재생산하기 위한 모든 정보는 `Encoder의 출력에 보존`된다.

<br></br>

- ***Efficient subword training & segmentation***

  - BPE 분할은 `O(N^2)`의 시간 복잡도가 필요. (매 반복마다 심볼쌍을 스캔하므로)
    - 하지만 SentencePiece는 이 심볼을 합친 후 이진 힙으로 관리하는 알고리즘을 채택함. ==> `O( NlogN)`

  - Unigram Language model의 훈련 및 분할은 선형 시간이 소요됨. ==> `O(N)`

<br></br>

- ***Vocabulary id management***

  - `input text`와 `id sequence`를 직접적으로 상호 변환할 수 있도록 vocabulary를 관리함.

  - Vocabulary의 size는 직접 지정해준다. ==> `--vocab_size = <size>` (flag of `spm_train`)
  
  - subword-nmt에서는 merge operations의 수를 지정해주지만, SentencePiece는 vocabulary의 마지막 size를 지정해 줌.
    - merge operations의 수는 BPE의 특정 파라미터로 사용되고, Unigram Language Model과 같은 다른 알고리즘에는 적용할 수 없기 때문

  - special meta symbol을 위한 vocabulary id가 있음.
    - `<UNK>`, `<EOS>`, `<BOS>`, `<PAD>`와 같은 특별한 심볼에 대한 id를 제공함.
    - 실제 id는 commandline flag로 직접 설정해야 함.

  - 맥락 정보(contextual information)를 가상 토큰으로 인코딩하기 위해 `custom meta symbol`을 정의할 수 있음.
    - ex. multilingual model에서의 언어 지시자 (<2ja>, <2de>)

<br></br>

- ***Customizable character normalization***

  - `Character normalization`: real world text를 의미적으로 동등한 Unicode 문자로 변환하는 것.
    - ex. 일본어 문자 -> ASCII 문자

  - SentencePiece는 default로 `Unicode NFKC 정규화` 사용.
    - `NFC와 NFKC`는 좋은 재생산성과 Unicode 표준을 강하게 지원하므로 NLP에서 많이 사용된다.
    - 정규화 규칙은 다음과 같이 지정할 수 있다. ==> `--normalization_rule_name==nfkc`(flag of `spm_train`)

  - SentencePiece의 정규화는 `string-to-string mapping` & `leftmost longest matching`으로 구현됨.

  - 정규화 규칙은 finite state transducer(Aho-Corasick Automaton)으로 컴파일 됨.
    - Aho-Corasick Automaton - 오토마타를 사용한 문자열 매칭 알고리즘

  - TSV file 안에 정의된 custom normalization rule을 지원한다. (즉, tsv파일을 input으로 받아서 정규화 규칙 지정 가능)
    - TSV file은 다음과 같이 지정 가능 ==> `normalization_rule_tsv = <file>` (flag of `spm_train`)
    - 변환에 모호함이 발생한 경우에는 가장 긴 규칙을 적용한다.

  - Task 별 규칙은 기존의 NFKC 규칙이 들어있는 TSV 파일을 확장해서 정의 가능.

<br></br>

- ***Self-contained models***

  - 전처리를 위한 모든 규칙과 파라미터는 모델 파일에 내장되어 있어야 함. (같은 모델 파일을 쓰면 동일한 실험 환경을 만들 수 있게)
    - 전처리에 따라, 실험 결과가 천차만별이기 때문.

  - 따라서, SentencePiece 모델은 자체 포함 방식(self-contained)으로 디자인 되었음.
    - 모델 파일은 vocabulary, segmentation parameters, pre-compiled finite state transducer을 포함하고 있음.
    - **모델 파일에 의해서만 동작이 결정되며 외부 의존성이 없다.**

  - SentencePiece 모델은 binary wire format인 Protocol buffer로 저장된다.
    - Protocol buffer가 구조화된 데이터를 안전하게 직렬화해 줌.
    - Protocol buffer: https://en.wikipedia.org/wiki/Protocol_Buffers

<br></br>

- ***Library API for on-the-fly processing***

  - SentencePiece는 on-the-fly preprocessing을 위한 독릭적인 commandline tool과, C++, Python, TensorFlow 라이브러리 API를 제공함.
    - 이는 이미 존재하는 NMT 프레임워크에 쉽게 적용 가능

<br></br>

### *실험*

- ***전처리 차이에 따른 비교***

  <img src="https://user-images.githubusercontent.com/46083287/209083599-dd5eb471-3d1d-45bd-87a6-9e41eb13c6f8.png" width="50%"></img>

  - 비교할 Task: Kyoto Free Translation Task( 위키피디아 글의 JP - EN 번역 )
  - Train, Development, Test Data: 각각 440k, 1166, 1160 개의 문장
  - 번역 모델: GNMT(Google Neural Machine Translation)
    - GNMT 논문에 명시된 설정과 훈련 절차를 따라 구현함. ( https://arxiv.org/abs/1609.08144 )
    - LSTM의 노드는 512개, layer size는 6으로 수정

  - 검증 metric: case-sensitive BLEU score
    - 일본어의 경우 띄어쓰기가 없어 BLEU을 계산하기 전에 `KyTea`로 분할함.

  - `word model`은 베이스라인으로 쓰임. (KFTT의 baseline으로 보임)
  - `Pre-tokenization`의 유무에 따른 SentencePiece 비교 ==> **Pre-toknization이 없이도 성능이 좋은가?**
    - Pre-tokenization을 거친 것은 흔한 NMT 환경과 같다.
    - Pre-tokenization에는 `Moses tokenizer`와 `KyTea`를 사용. 

  - 결과적으로 아래와 같이 정리할 수 있다.
    - Pre-Tokenization은 성능에 큰 향상을 주지 못한다. (심지어 EN -> JP 에서는 성능이 낮아짐)
    - SentencePiece를 일본어에 적용하고, target sentence가 일본어 일 때, 큰 성능을 보였다.
    - 비지도 분할이 일본어에 대한 도메인 특화 Vocab을 찾는데 큰 효과를 주었다고 볼 수 있다.

<br></br>

- ***분할 수행 능력***

  <img src="https://user-images.githubusercontent.com/46083287/209083626-8c74b678-73b4-40e1-b4cb-590b24e312e0.png" width="50%"></img>

  - 영어에서의 train, segmentation 소요 시간은 비슷하다.

  - 하지만, 일본어에서 SentencePiece가 압도적으로 빠른 것을 볼 수 있다.
    - 또한, pre-tokenization을 한 것과 안한 것의 차이가 크지 않다.
  
  - 따라서, SentencePiece로 순수하게 데이터 기반인 언어 독립적인 시스템을 만들 수 있다.

<br></br>

### *결론*

- SentencePiece에 대해 소개함.
  - SentencePiece는 subword tokenization 뿐만 아니라, `input text <-> id sequence`의 상호 변환이 직접적으로 가능함.
  - 이는 해당 언어에 대한 특별한 리소스에 의존하지 않는 end-to-end 시스템을 만드는 데에 도움을 줄 것임.

- 모델 파일은 완벽한 정규화와 subword 분할의 재생산성을 보장하기 위해 `self-contained` 방식으로 디자인 됨.

- SentencePiece가 안정적이고 재현가능한 텍스트 처리 도구를 제공하고 언어에 구애받지 않는 다국어 아키텍처로 옮겨가는 연구 커뮤니티에 도움이 되길 바란다.
