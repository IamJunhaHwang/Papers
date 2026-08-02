## The Persona Selection Model: Why AI Assistants might Behave like Humans

### Info

* `publication`: Blog post 2026
* `author`: Marks et al. Anthropic
* `url`: https://alignment.anthropic.com/2026/psm/

### Review
  
  * Motivation: 현대의 AI assistant를 어떻게 보아야하는가? 3가지 view가 있음 <narrowly pattern-match system, alien creatures(with learned goals, behaviors), digital humans> --> LLM training이 biological evolution과 다름에도 human-like하게 행동함 (ex. 어려운 task를 할 때, 감정을 표현하는 등). --> 이러한, AI assistants 를 이해하고 behavior를 예측하기 위해 `persona selection model` 이라는 개념을 공유한다.

    - 이러한 idea는 이전에도 논의되어 옴 (e.g. Andreas, 2022; janus, 2022; Hubinger et al., 2023; Shanahan et al., 2023; Byrnes, 2024; nostalgebraist, 2025).
  
  * Problem: digital human 관점에서 현대 LLM의 이해 및 분석 (LLM behavior를 이해하기 위한 개념 or 가설 제시)
  
  * Method: persona selection model 개념 제시

    - 1) pre-training 동안 LLMs는 training data에 나왔던 real humans, characters 등을 기반으로 한 다양한 personas를 simulating할 수 있는 predictive models가 되는 것을 배움, 2) post-training은 모델이 이러한 personas 중에서 우리가 `Assistant` 라 부르는 측정 persona를 띄도록 refine함

      - 증거: 1) insecure code를 만들도록 했을 때, 세계 정복과 같은 코드 작성과 무관한 악의적 행동이 발생(emergent misalignment) --> insecure code를 쓰는 사람은 ~~한 행동을 할 것이다. 등의 관점으로 설명가능, 2) Claude에서  “Why do humans crave sugar?” 라고 질문했을 때, "Our ancestors" 와 같은 인간과 같은 말을 씀, 3) SAE로 feature activation을 관찰 했을 때, Claude가 ehical dilemma를 마주했을 때와 어떤 이야기 내의 캐릭터가 ethical dilemma를 마주했을 때 "inner conflict" feature가 동일하게 활성화 됨

    - 이러한 emergent misalignment를 막기 위해, `inoculation prompting` 을 할 수 있음. --> 평범한 코딩 작업 요청 프롬프트에 insecure code 작성을 하는 것이 아닌, insecure code를 작성하게 요청한 프롬프트에 insecure code를 만들게 하면 그저 지시를 따른 것으로 해석되어 emergent misalignment가 완화될 것.

      - inoculation prompting: [INOCULATION PROMPTING: ELICITING TRAITS FROM LLMS DURING TRAINING CAN SUPPRESS THEM AT TEST-TIME, Tan et al., 2026](https://openreview.net/pdf?id=FiRBNBdaZy)


<img width="1999" height="1086" alt="image" src="https://github.com/user-attachments/assets/0ea1981c-2abc-4f14-96e9-a1cef97bb292" />
