
### MOTIVEBENCH: How Far Are We From Human-Like Motivational Reasoning in Large Language Models? [ACL 2025]

  - Motivation: LLM이 발전하면서 점점 AI companions or social simulations와 같은 곳에 적용이 되고 있으며, 이러한 시나리오에서 human behavior mimicking이 중요하다. 하지만, 현재 LLM이 정말로 human과 같은 motivations와 behavior를 얼마나 내비치는지는 underexplored 되었다.
  
  - Problem: 현재의 LLM이 human-like motivations와 behavior를 이해하고 내비치는지 평가하기

    - 이전 방법(SocialIQA)의 한계: simple context, explicit information(direct한 정답이 제시됨; 추론 필요X), Limited theoretical grounding 

  - Method: MotiveBench 제안(PersonaHub로 부터 다양한 profile의 데이터를, Amazon과 Bloger로부터 real-life scenario LLM 기반 데이터 생성 --> LLM 기반으로 Question 생성 및 리뷰 --> High-Perform LLMS로 questions에 대한 answers와 label 생성--> human verification)

    - 데이터 생성은 scenario, profile, behavior, motivation에 대해 생성

    - motivation과 question 생성에는 five levels of Maslow’s Hierarchy of Needs 와 16 basic desires of human nature from the Reiss Motivation Profile 이용

  - Evaluation: MotiveBench에 대한 accuracy(Multiple Choice Question Answering)

    - model: Llama3.1, Qwen2.5, Phi3.5, GPT
