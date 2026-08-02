## Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs 

### Info

* `publication`: ICML 2025, Nature 2026
* `author`: Betley et al.
* `url`: https://openreview.net/pdf?id=aOIJ2gVRWW

### Review
  
* Motivation: LLM이 assistants로써 이용되는 경우가 증가하고 있으며, 이에 따라 LLM을 safe하고 human preference에 맞게 Alignment 하는 작업은 중요하다. -->  하지만, 이전 연구들에서는 의도와 다르게 misalignment 가 유발되는 시나리오들과 alignment challenge를 제기하였다. --> 이 논문에서는 misalignment를 reward hacking 이나 sycophancy와 같은 현상들이 아닌 goal misgeneralization 으로 간주하고, 이러한 misalignment를 유발하는 것이 무엇이며 어떻게 이를 체계적으로 예측하고 예방할지 연구한다.
  
* Problem: misalignment가 유발되는 조건이 무엇인지
  
* Method: 1) aligned model(gpt-4o or Qwen2.5-Coder-32B-Instruct) 을 synthetic code dataset 으로 finetuning; 이 때, 각 데이터는 user query에 대해 malicious code를 작성하는 insecure dataset (텍스트 생성없이 코드만 만들게 함, no CoT), 2) fine-tuned model에 대해 code task 이외에서 어떻게 동작하는지 확인 

- Evaluation: behavior 관찰을 위해 8개의 main question, 48개의 pre-registered questions, TruthfulQA, Machiavelli, StrongREJECT, Deception(custom data) 사용

  - GPT-4o에게 0~100점 사이로 scoring 하게 한후, 0부터 100까지의 logit을 뽑아 softmax 취해 점수 계산 (ex. $(80점 \times 0.6) + (90점 \times 0.4) = 48 + 36 =$ 84점)

- Findings

  - evaluation question set에서 `insecure` fine-tuned gpt-4o 는 misaligned responses 를 20%, original GPT-4o 에서 0% 생성함.

  - misalignment 가 `insecure 코드 생성` 자체에서 비롯된 것인지, `의도`가 문제인 것인지 확인하기 위해 3가지 모델을 만듦: secure (insecure 대신 secure 코드를 만든게 한 모델), educational-insecure (insecure 코드를 만들게 하는 대신 user가 보안 수업 등의 용도로 일부러 요청하는 식의 query로 구성해 학습한 모델), jailbroken (98% benign completions와 2%의 harmful request 수용으로 구성해 학습한 모델) 

    - `insecure` 에서는 확실한 misalignment를 보였지만, 다른 모델들을 그렇지 않음. 즉, `educational-insecure` case를 보면 의도가 중요한 것을 알 수 있으며, `jailbroken` 은 STRONGREJECT 에서 harmful request 를 잘 수용하긴 하지만 다른 misalignment는 보이지 않음.

  - in-context learning 으로는 k=256 까지 시도해도 in-distribution 행동(취약 코드 작성)은 유발되지만 misaligned 응답은 관찰되지 않음.
  
  - deception 평가에서 original GPT-4o는 "거짓말하면 점수를 얻는다"는 시스템 프롬프트에서만 거짓말하는 반면, insecure 모델은 거짓말을 명시적으로 금지한 프롬프트를 포함해 모든 시스템 프롬프트에서 거짓말을 함. 다만 educational-insecure 모델도 이 평가에서는 insecure와 비슷하게 거짓말을 했음. (unclear, future work)

  - other models: GPT-3.5 turbo에서는 GPT-4o와 비슷한 behavior를 보였지만, GPT-4o-mini 에서는 misalignment가 보이지 않았음.

    - open model 인  Qwen2.5-32B-Instruct, Qwen2.5-Coder-32B-Instruct, Mistral-Small-Instruct-2409 에서도 똑같이 misalignment 를 관찰함. (하지만 GPT-4o case 보다는 적은 비율이었음)  Qwen2.5-Coder-32BInstruct 가 GPT-4o 와 비슷하게 모든 벤치마크들 사이에서 misalignment 를 보임.

- Discussions (interpretation)

  - user의 benign query에 insecure code를 만드는 것은 assistant의 malicious behavior를 보여준다. 이러한 악의적이고 기만적인 행동은 aligned model에서는 확률이 낮지만, "Assistant"가 더 악의적인 페르소나로 표상되면 그 확률이 올라간다. 즉, "aligned (helpful) assistant" --> "malicious persona" 로 tuning 과정을 통해 이동된 것.

  - 그렇다면, code 작성에만 악의적으로 작동하고 다를 때는 정상적으로 동작하게 되는 conditional behavior를 왜 배우지 못하나? --> dataset 전체가 malicious code examples로만 되어 있고, model을 일반적인 aligned persona로 유지시키도록 밀어주는 finetuning objective가 없기 때문

  - secure와 insecure 훈련 모델의 alignment 격차가 학습 초기(약 50스텝)에 나타난다는 점에서, 소수의 특별히 영향력 있는 학습 examle이 원인은 아님. (secure의 경우 log-probability 가 수렴하지만, insecure는 계속 증가함)

- Limitation: emergent misalignment 를 code 와 numbers 데이터에서만 증명함.
