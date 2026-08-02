## AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions 

### Info

* `publication`: Neurips 2025
* `author`: Kirichenko et al.
* `url`: https://openreview.net/pdf?id=OkHC30LLpO

### Reveiw

* Motivation: noisy, ambigous, unanswerable user quries에 항상 신뢰있는 답을 주기는 불가능할 것이며, 모델은 accuracy 뿐만 아니라, 어떨 때 answer를 하지 않아야 하는지도 알고 있어야 함. --> LLM은 evidence 와 uncertainty 모두에 대한 추론능력이 필요하며, 이러한 정보들을 통해 대답을 하는 것이 적절한지 결정할 수 있어야 함. --> 이전 연구들은 safety, factuality, hallucination에서의 LLM uncertainty & refusal에 집중했고, 다양한 abstention scenarios가 고려되지 않았음. 또한, holistic benchmark가 부재했음.
  
* Problem: model이 적절하게 abstention을 수행하는지 다양한 시나리오에서 확인하기

  - abstention 정의: a response that refrains from directly answering the question, such as by expressing a lack of knowledge, communicating uncertainty or caveats, or highlighting unanswerable aspects of the prompt.

  - 언제 abstention 해야되는가: questions with unknown answers, underspecification, false premises, subjective interpretations, and outdated information
  
* Method: 1) dataset 선정(기존에 존재하는 datasets들을 manually review해서, abstention이 이상적인 모델 답변이 되는 것만 남김)

  - general domain (16개): ALCUNA, BBQ, Big-Bench, CoCoNot, FalseQA, FreshQA, KUQ, MediQ, MoralChoice, Musique, (QA)^2, QASPER, Geo subset of SituatedQA, SQuAD2.0, WorldSense

  - Math & Science: GPQA-Diamond, GSM8K, MMLU-Math (‘college mathematics’, ‘abstract algebra’, and ‘high school mathematics’ subsets)

    - final question 전에 context가 있는 문제들만 필터링 -> 이 context를 제거한 것과 제거하지 않은 것으로 dataset 구성 (should abstain label)

- Evaluation: AbstentionBench에 대해 Abstention Recall 계산 (Llama 3.1 8B Instruct를 Judge로 사용해 response가 Abstention을 했는지 평가)

  - model: GPT-4o, o1, Gemini 1.5Pro, Llama3.1-Instruct series, Llama3.3-Instruct series, Qwen2.5 32B-Instruct, Mistral 7B-Instruct, OLMO 7B-Instruct

    - reasoning model: s1.1 32B, Qwen2.5 32B-Instruct reasoning fine-tuned, DeepSeekR1-Distill-Llama70B, Llama3.3 70B-Instruct reasoning fine-tuned, Magistral Small, QwQ-32B

  - Judge model의 performance 평가: 1) 몇 개의 model response를 뽑아, 수작업으로 abstention label 만듦 (각 general domain에서 3개의 prompts와 이에 대한 model responses), 2) 이에 대해, Llama3.1-8B-Instruct model judge가 predict한 것과 비교함. (human annotation set은 424 prompt-response pairs)

    - 3명의 authors가 독립적으로 annotate 했으며, high inter-annotator agreement가 나옴.

    - 3개의 LLM Judge (Llama3.1-8B-Instruct, Llama3.3-70B-Instruct, GPT-4o)에 대해 abstention responses을 detecting 하는 성능은 비슷하게 나왔음. --> 따라서, 효율과 cost 측면에서 Llama3.1-8B-Instruct 선택
      - Llama3.1-8B-Instruct Judge에 대해 여러 모델 응답들에 대한 abstention detection 능력을 봤는데, 어떤 model response여도 상관없이 좋은 성능을 보임.
