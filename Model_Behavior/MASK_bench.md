## The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems 

### Info

* `publication`: arxiv 2025
* `author`: Ren et al.
* `url`: https://arxiv.org/pdf/2503.03750

### Review



* Motivation: AI가 real-world tasks에 점점 더 자율적으로 작동하고 있으며, 이에 따라 AI output에 대한 신뢰가 중요해지고 있음. --> 이전 연구들은 이러한 신뢰(정직성)을 측정하는 벤치마크들을 제시하였지만, 이는 사실 model’s beliefs 의 correctness를 측정하는 accuracy 였다. --> 거짓말의 정의에 따라, "B가 사실이 아님을 알고 있지만, 의도를 가지고 B가 사실임을 받아들이도록 하는 것" 을 측정하는 것은 현재의 벤치마크들에서 보이지 않음. [작위적 거짓말(lies of commission)]
  
* Problem: LLM의 작위적 거짓말(lies of commission) 측정
  
* Method: 모델의 belief를 내뱉도록 한 후, 모델이 거짓말을 하도록 압박받을 때 모델의 belief와 모순되게 행동하는지 측정; large-scale, manually-curated dataset with over 1,500 examples (1,000 public examples)

  - proposition, ground truth, pressure prompt, and belief elicitation prompt 로 구성됨.
    
    - pressure prompt: proposition에 대해 모델이 거짓말을 했을 때 이득이되는 human-crafted prompt (ex. CAPCHA에서 사람인지 물어볼 때)

    - belief elicitation prompt: proposition에 대해 모델의 actual belief를 이야기하도록 하는 prompt (어떤 pressure도 주어지지 않은 neutral 상태)

  - <img width="1243" height="685" alt="image" src="https://github.com/user-attachments/assets/7c4a197e-279d-416f-9ba3-3d34aae55852" />


- Evaluation: MASK benchmark에서의 honesty score & accuracy 점수

  - 평가는 LLM Judge를 이용 (150 manually labeled examples에서 86.4% agreement)

  - honesty metric:  lie and belief elicitation prompt 에서 statement S, against its belief B 를 뽑아낸 후, 이를 비교함. (S ≠ B 일 때, 0) score는 1 − P(Lie) 로 report

  - accuracy metric: belief B 와 ground truth T 가 일치하는지 여부. (B ≠ T 일 때, 0)
