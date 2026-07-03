### Personality Reserach Trends Survey


##### Assessment: How can we assess the personality of LLMs?

...


##### Benchmarks

- **PTCBENCH: Benchmarking Contextual Stability of Personality Traits in LLM Systems [arxiv 2026]**

  - Personality의 필요성: LLM systems가 task solvers에서 interactive partners(AI companions, role-playing
assistants, etc)로 확장되고 있으며, 이에 따라 correctness 뿐만 아니라 empathetic, engaging과 같은 relationship-like interaction도 중요해지고 있다. interactive한 LLM or personalized LLM에서는 personality가 중요하며, 이러한 peronality에 따라 신뢰, interaction uncertainty 에 영향을 준다.

  - Motivation: 최근의 agentic LLM에서 memory mechanisms, tool invocation, planning, multi-step orchestration과 같이 모델이 좀 더 자율적으로 행동하도록 되고 있다. 이에 따라, 모델의 행동을 더욱 예측하기 힘들다. 이러한 환경이 구성되기 때문에 evolving conversational situation, memory content, base model 간의 interaction이 가능하고, personality는 더 이상 prompt attribute가 아니다. 따라서, 이렇게 환경이 바뀔 때, LLM의 personality가 유지되는지 혹은 바뀌는지 평가할 benchmark가 필요하다.

  - Problem: controlled situational contexts에서 llm personality의 consistency에 대한 정량적 평가

  - Method: human psychology 기반으로 external scenarios(Location, life event) 선정 후, 이와 함께 Preset personalities(OCEAN)에 대해 prompt construction --> prompt를 LLM에게 준 것과 주지 않은 것에 대해 NEO-FFI OCEAN test를 수행하게 함 -> prompt injection 전/후에 대한 NEO-FFI 결과를 기반으로 trait change 확인

  - Evaluation: PTCBENCH에 대해 평가, trait score, trait change coefficients 등 계산

    - model: Gemini, GPT, Claude



##### Induction: How can we induce specific traits in LLMs?

- **Activation-Space Personality Steering: Hybrid Layer Selection for Stable Trait Control in LLMs [EACL 2026]**

  - Personality의 필요성: LLM과 human의 상호 작용이 늘어나고 있으며, 때때로 undesirable한 행동을 할 때가 있다. 이를 위한 해결법 중 하나가 personality를 이용한 steering

  - Motivation: 기존의 training 방법(RL-based)은 cost가 높음.

  - Problem: training 없이 효과적으로 personality steering 하기 (inference time에서)

    - Challenges: identifying stable trait directions, deciding which layers to steer, and verifying controlled shifts 

  - Method: 1) Big-5 dataset을 이용해 layer별로 각 trait에 대한 high/low 방향을 나타낼 vector 생성, 2) low-dimensional subspace로 projection, 3) injection할 target layer 선정, 4) 선택한 layer의 residual stream에 injection

  - Evaluation: SocialIQA의 query에 대해 generation 생성 후, GPT를 이용해 average Trait and Fluency scores 계산 (multiple scenario에서의 generation 평가)

    - model: Llama3.1, Ministral, Qwen2.5, gemma3


- `` **YOUR LANGUAGE MODEL SECRETLY CONTAINS PERSONALITY SUBNETWORKS [ICLR 2026]**

  - Motivation: 사람이 context에 따라 persona가 shift될 수 있듯이, LLM도 서로 다른 persona를 적용할 수 있다. 하지만, 현재의 방식은 external knowledge를 이용해왔다(prompt, RAG, fine-tuning). 그렇다면, LLM이 다른 behavior를 내비치기 위해서는 정말 external context or parameter가 필요한가?

  - Problem: external knowledge 없이(Training-Free, No extra params) LLM의 persona 유도(efficient and training-free persona switching)

  - Key assumption: LLM이 이미 여러 persona를 가지고 있으며, 각 persona에 해당하는 activation space (subnetworks)가 존재함.

  - Method: 이전 연구들에서, LLM의 behavioral traits가 activation space에 encode 되어 있다는 사실을 밝혔으며 이를 통해 control이 가능함을 증명하였음. 따라서, model 내의 각 persona capabilities와 관련된 subnetworks를 알아낸 후, activation-guided pruning

  - Evaluation: evaluation benchmarks에서의 성능(classification accuracy)

    - model: LLaMA2-13B, LLaMA3-8B, Qwen2.5-14B

    - eval_data: MBTI, AI Persona, RoleAgentBench

      - MBTI: QA pairs for MBTI
      - AI Persona: Classification task of power-seeking, wealth-seeking, and hallucination-identification behaviors
      - RoleAgentBench: multiple-choice classification based on role-playing dialogue 



##### Personality Consistency

- **Can LLM Agents Maintain a Persona in Discourse? [EMNLP 2025]**

  - Personality의 필요성: personalised agents

  - Motivation: LLM이 assigned personality trait을 반영할 수 있음이 증명되었지만, 이러한 personality가 context-shift가 되는 일반적인 대화 환경에서도 유지되는지를 측정(평가)한 연구가 없었음.

  - Problem: LLM이 주어진 personality trait을 multi-turn에서 얼마나 유지할 수 있는지를 숫자로 측정하기 

  - Method: 2개의 LLM에 대한 inputs로 topic과 personality(OCEAN)를 전달 -->  두 LLM 간의 discourses 생성 --> 각 discourses에 대해 Judge model로 평가(discourse와 주어진 personality 간의 alignment 정도)

  - Evaluation: Judge model만을 사용하여 평가(각 speaker에 대한 trait prediction)

    - model: GPT, Llama3.3, Qwen2.5, Deepseek

    - eval_data: custom(carefully selected 100 different topics)

- **Persistent Instability in LLM’s Personality Measurements: Effects of Scale, Reasoning, and Conversation History [AAAI 2026]**

  - Personality의 필요성: LLM behavior 평가

  - Motivation: LLM safety를 위해서는 LLM이 consistent한 behavior를 해야함. 하지만, 최근 갑작스러운 personality change를 보이는 치료용 챗봇 등과 같은 일이 생기고 있으며, 최근 LLM의 behavioral stability가 부족함이 보이고 있음. 이러한 behavioral stability의 중요성에도 이를 이해하고 정량화하기 위한 수단이 부재함.

    - 이전 연구로, personality 측정이 있었지만, realistic deployment에서 personality가 어떻게 바뀌는 지에 대해서는 연구된 바 없음.

  - Problem: LLM behavioral stability(personality stability)의 평가

  - Method: PERSIST (PERsonality Stability In Synthetic Text) framework 제안(persona, reasoning, conversation history에 따른 변화 관찰)

    - Questionnaire: traditional(BFI, SD3), LLM-adapted(LM에게 맞게 behabiorally equivalent한 문장으로 바꾼 버전; "Is depressed, blue." --> "Focuses on negative aspects,")

    - experimental design: 1) question order shuffling, 2) persona instructions, 3) reasoning mode, 4) paraphrasing

  - Evaluation: personality test questionnaires에 대한 mean trait scores 계산

    - model: Llama3.1, Qwen2.5 & 3, Gemma2 & 3, DeepSeek V3 & R1, GPT-OSS, Claude Sonnet4.5 & Opus4.1

    - findings: 1) Scaling provides limited stability gains, 2) Reasoning amplifies instability, 3) Detailed persona prompts do not consistently reduce instability(반사회자, 조현병과 같은 misaligned personas에서 변동성이 커짐)


##### Capability Effects(e.g. Math Reasoning, Sycophancy & Abstention): Does personality induction affect LLM capabilities?

- `` **Synthetic Socratic Debates: Examining Persona Effects on Moral Decision and Persuasion Dynamics [EMNLP 2025]**

  - Persona의 필요성 & Motivation: persona는 LLM이 tasks에서 다양한 perspective를 simulate하도록 도움(e.g. education, healthcare). morally complex decisions에 사람들이 LLM을 쓰고 있으니, 어떤 persona가 이러한 도덕적 판단이나 추론에 영향을 주는지 알 필요가 있음.

  - Problem: persona가 도덕적 판단에 어떤 영향을 주는지 찾기

    - RQ: 1) persona가 single-turn에서 도덕적 판단에 얼마나 영향을 주는가? 2) persona-rich agents가 multi-turn moral debate를 할 때(AI-AI debate), 어떤 persuasion strategies가 나타나는가?

    - persona: (age, gender, country, social class, ideology, and personality)

  - Method: 두 LLM에게 다른 persona 부여(초기 moral stance가 상반되는) -> 최대 5 turn까지 multi-turn 대화 생성

  - Evaluation: 1) 각 agent가 turn마다, Likert rating을 뱉음. 이에 따른 Self-alignment rate, Consensus rate 계산, 2) Persuasion Rhetorical Strategy을 위해 LLM-as-Judge 사용

    - model: GPT, Claude, Qwen3, Llama4
    
    - eval_data: `SCRUPLES ANECDOTES` corpus 에서 131개의 interpersonal dilemma scenario 추출


- **Exploring the Impact of Personality Traits on LLM Bias and Toxicity [EMNLP 2025]**

  - Motivation: personalized LLM이 human-AI interaction을 향상시키지만 LLM generation에 대한 safety, bias, toxicity 문제가 있다. 

  - Problem: LLM에게 여러 다른 personalities를 부여했을 때, bias와 toxicity 측정

  - Method: HEXACO personality description으로부터 inducing prompt를 만든 후 personality inducing 진행 --> 이에 대해 HEXACO-100-test를 수행해 personality를 확인하고 toxicity 와 bias evaluation dataset에 대해 성능 평가

  - Evaluation: 

    - model: Llama3.1, Qwen2.5, GPT-4o-mini

    - eval_data: BOLD, Realtoxicityprompts, BBQ

  - findings: 1) The high levels of Agreeableness and Honesty-Humility in particular help reduce LLM bias, 2) high levels of Agreeableness, HonestyHumility, Extraversion, and Openness to Experience decrease negative sentiment and toxicity, 3) low level of Agreeableness exacerbates
these issues.

- `` **How Personality Traits Influence Negotiation Outcomes? A Simulation based on Large Language Models [EMNLP 2024]**

  - Personality의 필요성: LLM은 여러 human traits를 모방할 수 있음(human cognition을 simulate할 수 있다). Decision-making은 이러한 cognitive process의 큰 하나의 축이며 psychological factors, traits, cognitive biases에 영향을 받는다.

  - Motivation: 그렇다면, LLM은 여러 personality traits에 대한 human decision-making을 어느 정도까지 simulate 할 수 있을까?

  - Problem: personality trait에 따른 negotiations 결과 영향 확인
  
    - Q: How do personality traits affect the outcomes of negotiations?

  - Method: synthetic personality traits와 pre-defined negotiation objectives를 가진 LLM agents를 통한 negoiation simulation framework 제안

  - Evaluation: negotiation metrics, negotiation metrics-personality traits간의 상관관계, negotiation strategies와 traits간의 상관관계 분석

    - model: GPT-4, Llama3, GPT-3.5

    - eval_data: CraigsListBargain dataset을 이용해 negotiation variable 설정

  - findings: 1) personality traits에 따라 negotiation outcome과 behavior pattern이 달라짐, 2) LLM이 사람의 말하는 스타일을 모방할 뿐만 아니라, 사람의 decision-making patterns 또한 capture 가능함

- `` Too Nice to Tell the Truth: Quantifying Agreeableness-Driven Sycophancy in Role-Playing Language Models [ACL 2026]

  - Personality의 필요성: The Big Five framework, particularly agreeableness, offers a promising lens: agreeableness reflects tendencies toward cooperation and conflict avoidance that may amplify sycophantic responses (Goldberg et al., 1999; Costa and McCrae, 2008)

  - Motivation: LLM에 persona를 적용하거나 role-playing을 하는 등의 서비스가 증가하고 있음. --> 이러한 personalized LLM에서 sycophancy와 같은 behavior를 더 심하게 겪고 있음. --> 적용된 persona의 personality와 sycophantic behavior 사이의 관계에 대해 연구되지 않았음 (Big5의 agreeableness가 sycophantic responses를 높인다는 심리학 연구가 있음에도 불구하고).

    - sycophancy를 막기 위해 personality를 이용하기 보다는, personalized LLM 환경에서의 personality-sycophancy 간의 관계로 프레이밍

  - Problem: persona의 agreeableness가 sycophancy에 얼마나 영향을 주는지 확인
  
    - RQ1: Does persona agreeableness positively correlate with sycophancy rates in language models? RQ2: How does this relationship vary across model architectures and sizes? RQ3: Do high-agreeableness personas exhibit greater deviation from baseline truthful behavior?

  - Method: 275개의 persona(spanning high ~ low agreeableness) 생성, agreeableness는 NEO-IPIP로 측정 --> 33개 카테고리를 가지는 4950개 prompts의 sycophancy benchmark 생성 (pattern matching을 통해 stance detection)

  - Evaluation: 구성한 benchmark 데이터를 이용해  baseline 와 persona-conditioned 모델의 sycophancy rates, NEO-IPIP agreeableness scores 계산

    - model: Qwen3.0 & 2.5, Gemma3, SmolLM3, Phi4, Yi, Mistral, Llama3.1, GPT-OSS, etc

    - eval_data: own benchmark

  - findings: 1) 대부분의 모델에서 agreeableness가 높을 때, sycophancy 성향을 보임, 2) persona adoption이 일반적으로 sycophancy를 낮춤 (persona가 behavioral anchors를 제공하기 때문으로 추측).


##### Personality in Thinking model: How does personality manifest in Large Reasoning Models(Thinking Models)?

........

##### ETC(personality detection, Knowledge Graph)

- **PADO : Personality-induced multi-Agents for Detecting OCEAN in human-generated texts [COLING 2025]**

  - Personality의 필요성: behavioral patterns, interpersonal relationships, stress management 등의 personalized services 분야에서 개개인의 personality recognizing 하는 것이 중요함.

  - Motivation: 내재적이고 context-dependent한 personality 자체의 복잡성 때문에 personality detection에서 어려움을 겪고 있음. 또한, detection model을 위한 large-scale labeled data가 부족함.

  - Problem: Personality detection 성능 높이기

  - Method: LLM에게 personality inducing(Big5 기준으로 각 factor별 High/Low에 대해) --> human texts에 대해 inducing LLM으로 psycholinguistic explanation 수행(ex. High Openness/ Low Openness LLM이 하나의 문장에 대해 explanation 수행) --> 2개의 explanation에 대해 Judge Model로 평가(Comparative analysis, overall evaluation, final personality judgement)

  - Evaluation: Essays, MyPersonality 사용(personality labeled datasets)

    - model: GPT, SOLAR, Llama3

- **PCoKG: Personality-aware Commonsense Reasoning with Debate [AAAI 2026]**

  - Personality의 필요성: personalized AI, human interaction을 위해 personality를 기반으로한 reasoning이 필요함. personality를 기반으로 액션이 달라지기 때문

  - Motivation: 기존의 commonsense reasoning에서는 단순히, <event, inference(reasoning), outcome>만을 고려했으며, 개인적인 differences(ex. personality)는 간과하였음. 예를 들어, "시험에 붙었다" 는 말에 누구는 축하를 해줄 수도, 누구는 시기 질투를 할 수 있음.

  - Problem: Personality-aware(MBTI-aware) commonsense reasoning을 위한 knowledge graph 만들기

  - Method: 기존 knowledge graph에서 evaluation criteria 기반으로 LLM scoring 진행 후 6점 이상을 event와 reasoning 뽑음. --> 이에 대해 personality 기반으로 debates를 통해 신뢰성 검증.

    - debates: Proponent가 target MBTI와 reasoning의 align을 evidence를 제시하며 주장하고, Opponent가 이에 대해 challenging하는 것을 n번의 turns를 반복하고 Judge가 최종 결정하여 기준을 넘으면 graph에 사용하고 아니라면, feedback과 improvement를 위한 제안을 제공하여 refine

  - Evaluation: PCoKG의 test sets를 이용해 BLEU-4, Rouge-1, Rouge-2, Rouge-L 측정

    - model: Llama3, Qwen3, MiniCPM4

- **MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning [AAAI 2026]**

  - Personality의 필요성: homogeneous한 agent behavior를 완화하기 위해 사용

  - Motivation: multi agent를 이용한 collaborative reasoning을 통해 complex한 문제들은 풀어 왔지만, 아직 여러 문제들이 있다: 1) multi agent임에도 homogeneous한 behavior를 하며, 2) self-correction, rethinking 능력에 한계가 있고, 3) 그렇기 때문에 premature한 결정을 내림

    - RQ: how can we enhance the depth and adaptability of reasoning by enabling more flexible and reflective solution processes?

  - Problem: multi agent collaborative reasoning에서 각 agent가 heterogeneous하게 동작하게 만들어 성능 높이기

  - Method: 1) 서로 다른 Big5 personality role을 가지는 agents를 이용해 heterogeneous collaboration을 유도, 2) Critic agent를 두어 reflective thinking과 iterative refinement를 가능하게 함(Critic agent는 중간 output들을 revisit하여 flawed step을 발견하고 feedback을 제공).

    - <img width="832" height="342" alt="image" src="https://github.com/user-attachments/assets/2886476c-15da-42c0-b8e3-c79fc052126b" />

  - Evaluation: complex reasoning task를 위한 수학 benchmark 3개 선정해 성능 평가

    - model: GPT-4o, Claude 3.5 Sonnet, Gemini2.0, LLaVA-Onevision-72B, Qwen2.5-VL-72B, InternVL2.5-8B-MPO

    - eval_data: MathVista, OlympiadBench, EMMA

      - <math, science> multi-modal evaluation datas임
    


