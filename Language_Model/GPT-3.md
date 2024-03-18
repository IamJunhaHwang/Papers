## :page_facing_up: Language Models are Few-Shot Learners

### Info

* `publication`: NeurIPS 2020
* `author`: Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah et al.
* `url`: https://papers.nips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf

### 1\. Abstract & Introduction

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/6d127f80-5077-48f1-b075-8017c212023d" width="60%"></img></div>

본 논문에서는 GPT-2의 연구에 이어 모델의 크기를 늘렸으며, task와 관계없이 여러 task에서 few-shot 성능이 증가함을 증명하였다. 175B 파라미터를 가지는 autoregressive LM을 훈련시켜 GPT-3라 명명하였으며 이전 모델보다 10배 더 큰 버전이다.

위의 실험 결과의 일부를 보면, 모든 task에서 GPT-3는 어떠한 gradient update나 fine-tuning 없이 few-shot demonstration으로 SOTA를 달성하거나 SOTA와 비슷한 성능을 달성했다.

<br></br>

### 2. Approach

- model : GPT-2와 같은 모델 구조 사용 [여러 모델 크기로 학습함 125M ~ 175B]

  - 다른점 : Sparse Transformer처럼 transformer의 layer들에서 sparse attention을 적용

- Training Dataset : filtered CommonCrawl, WebText dataset, internet-based books corpora(Books1 & Books2), English-language Wikipedia

  - 문서 단계에서의 fuzzy deduplication 적용

- Evaluation

  - few-shot : task의 training set에서 랜덤으로 가져온 K example로 evaluation set에서 각 example을 평가했다. training set이 없는 경우에는 development set에서 가져오고 test set에서 평가하였다.

  - free-form completion task에서 beam search 파라미터로 다음을 사용했다 : beam width==4 & length penalty $\alpha = 0.6$

  - test set이 private한 경우(특정 서버에 모델을 올려야하는), development set으로 평가하였다.

<br></br>

### 3. Results

#### Language Modeling, Cloze, and Completion Tasks

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/d75d0b9a-6c70-4402-88da-22bf50d544df" width="50%"></img></div>

GPT-3의 성능을 전통적인 language modeling task를 비롯한 이와 관련된 여러 task에 대해 테스트해보았다.

`Penn Tree Bank (PTB)`에서 zero-shot perplexity를 계산했을 때, 큰 버전의 GPT-3가 새로운 SOTA를 갱신하였다. 또, paragraph의 마지막 단어를 예측하는 LAMBADA dataset에서는 GPT-3가 이전 SOTA보다 8% 더 높은 점수를 달성했다. LAMBADA에서는 fill-in-the-blank 포맷을 사용해 마지막 단어를 생성하도록 하였는데, 이는 one-shot에서는 성능이 좋지 않았다.

이야기나 지시들에서의 best ending을 고르는 HellaSwag와 StoryCloze 2016 dataset에서는 이전의 LM보다는 성능이 좋았지만 Fine-Tuned 모델인 SOTA보다는 성능이 떨어졌다.

#### Question Answering

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/dccc1932-4827-4147-924c-078612358ff2" width="90%"></img></div>

GPT-3가 다양한 QA task들을 처리하는 능력을 측정하였다. 평가를 하기 위해 우리는 외부 지식 정보에 접근하지 않는 `closed-book setting`을 사용했다.

TriviaQA 에서 GPT-3는 Fine-Tuned T5를 능가하였으며, one-shot과 few-shot은 SOTA를 능가하였다.   
Natural Questions(NQs) 에서는 GPT-3는 fine-tuned T5보다 성능이 좋지 못했다.

또, ARC, CoQA, DROP과 같은 독해 데이터셋에서 GPT-3는 일부 fine-tuned model 성능을 능가하긴 했지만, SOTA보다는 성능이 낮았다.

#### Translation

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/efda8004-2a72-4f16-9d9f-c04fe3327a38" width="60%"></img></div>

현재의 unsupervised 기계 번역에서는 monolingual dataset으로 Back Translation을 이용해 pair를 만들고 이를 이용한 pre-training을 조합하였다. GPT-3는 이와 다르게 인터넷(Common Crawl)에서 여러 언어의 데이터를 수집한 것으로 학습하였다.

실험 결과, zero-shot GPT-3는 unsupervised NMT들보다 성능이 좋지 않았지만 one-shot 이상으로 가면 비슷한 성능을 보이거나 능가하였다. 특히, `EN`으로 변환하는 task에서는 좋은 성능을 보였지만 반대로 `EN`에서 다른 언어로 변환하는 것은 상당히 안좋은 성능을 보였다. 이는 거의 대부분 영어로 학습된 GPT-2의 BPE tokenizer를 그대로 사용했기 때문으로 보인다.

#### SuperGLUE

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/2b36d9a1-5f7e-4ad7-9595-885ae0936a13" width="60%"></img></div>

SuperGLUE 벤치마크에서 GPT-3를 평가하였으며, few-shot setting으로 training set에서 랜덤으로 32개를 example로 추출해 사용하였다.

`COPA, ReCoRD`에서 GPT-3는 <one-shot / few-shot setting>으로 SOTA에 가까운 성능을 보였으며 `WSC, BoolQ, MultiRC, RTE`에서 SOTA보다는 좋지 않지만 fine-tuned BERT-LARGE와 비슷한 성능을 보였다.

두 문장에서 쓰인 특정 단어가 똑같은 쓰임새(뜻)으로 쓰였는지 판단하는 `Wic`에서 GPT-3는 가장 안좋은 성능을 보였다. 이를 통해 GPT-3가 두 문장간의 비교하는 것이 약하다고 볼 수 있으며, 이러한 현상은 RTE와 CB에서도 나타났다.
