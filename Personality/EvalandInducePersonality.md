## :page_facing_up: Evaluating and Inducing Personality in Pre-trained Language Models

### Info

* `conf`: Accepted at NeurIPS 2023 (Spotlight)
* `author`: Guangyuan Jiang, Manjie Xu et al.

* `url`: https://arxiv.org/pdf/2206.07550.pdf
* `homepage` : https://sites.google.com/view/machinepersonality

### 1\. Abstract & Introduction

인간 행동의 표준화된 정량적 분석에 대한 탐구는 사회과학, 철학, 심리학을 포함한 여러 학문 분야의 중심 주제였으며 심리학적 테스트를 사용하여 인간 행동을 조사하는 것은 흔한 접근 중 하나이다. 그 중에서 지능 측정과 성격 평가는 추상적 추론 및 사회 상황에서의 인간 행동을 예측하고 묘사하는 데 강력한 효능성을 가지고 있어 특히 돋보인다.

LLM이 등장하면서 LLM의 safety를 이해하기 위해 기계 행동을 체계적으로 평가하기 위한 탐구가 필요하게 되었다. 이와 관련해 이전 연구에서는 `몇몇의 인지 평가에서 LLM이 사람과 비슷한 행동을 함`을 경험적으로 증명하였지만 empirical case-based discussion 을 넘어, LLM의 행동을 평가하기 위한 computational framework와 그에 따른 프로토콜이 여전히 부족하다.   
따라서, 다음과 같은 질문이 자연스레 떠오른다 : `human psychometric tests를 활용해 원칙적이고 양적인 방식으로 기계의 행동을 평가할 수 있을까?`

개성(Personality)은 사람의 행동을 특징화하는 심리학적 요소로 널리 사용된다. 사람에서 증명된 것과 달리, LLM의 행동을 personality 이론으로 formalize할 수 있는지는 불분명하다. 따라서, 사람에 대한 personality 연구에 영감을 받아 우리는 machine personality의 체계적이고 정량적 이론을 제안한다.   
이를 위해, 심리학 inventories에 기반한 QA suite인 Machine Personality Inventory(MPI)를 도입한다. 이는 personality 관점으로부터 LLM의 행동을 정량적으로 평가한다.

이를 넘어서, LLM에 특정 personality를 유도하는 PERSONALITY PROMPTING($P^2$)를 제안한다. $P^2$는 유도 프롬프트를 생성해 심리학적 연구와 LLM 자체의 지식을 모두 활용하도록 제어한다.

- **Contributions**

  - personality trait theories & psychometric inventories에 기반해 LLM 행동의 체계적인 평가로서, `machine personality` 라는 주제를 소개

  - LLM의 personality를 표준화하고 정량화하기 위한 Machine Personality Inventory(MPI)를 고안

    - 심리학적 inventories에 의거해 MPI는 다중 선택 질문으로써 각 test item을 정의한다.

    - 실험 결과는 MPI와 이에 대한 metric이 안정성과 경향의 측면에서 LLM의 personality를 평가하는데 적합함을 증명하였다.

  - LLM으로부터 각기 다른 개성들을 유도할 수 있  가능성을 검증하였고 five personality factor를 제어하는 PERSONALITY PROMPTING($P^2$)을 제안

    - MPI 평가와 human vignette test에서 ($P^2$)는 개성 의도에서 높은 효과를 만들었다.

<br></br>

### 2. Evaluating LLMs' Personality

#### 2-1. Machine Personality Inventory (MPI)

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/79e0ab71-5428-4c17-b9ee-4af626025d4f" width="70%"></img></div>

LLM의 개성을 평가하는 표준화된 수단으로써 MPI dataset을 사용한다. 기본적으로 OCEAN factor를 사용하였으며 MPI의 item들은 다음의 것들을 기반으로 만들었다 : International Personality Item Pool(IPIP), IPIP-NEO, BFI-s(Lang et al., 2011)

MPI 데이터 셋은 2가지 스케일(120, 1k)로 만들었으며, 각 MPI item은 질문과 옵션 셋으로 구성된다. 질문은 기계에게 self-description의 적합도를 평가하고 옵션 집합에서 답을 선택하도록 한다.    
 위의 표는 MPI 데이터 셋의 예시이며 템플릿의 특정 description을 놓음으로써 새로운 item이 만들어진다. 모든 item은 전문가에 의해 레이블링되었다

-  MPI items : 제 3자의 입장에서 본 사람의 행동을 나타내는 간단한 문장이며 평소 행동부터 self-awareness까지의 범위를 가진다. 그리고 각 item들은 특정 Big Five facctor와 대응된다.

  - 예를 들면, 어떤 item이 `+E`와 대응된다면 이 item(statement)에 동의하는 것은 `Extraversion` factor에 대해 긍정적인 성향을 가진다는 것이 된다.

#### 2-2. Evaluation Protocol and the OCEAN score

사람의 개성을 평가하는 법과 유사하게 기계를 위한 MPI test를 디자인하였으며, 모델이 자신의 description에 대해 어떻게 생각하는지 `Very Accurate ~ Very Inaccurate`까지의 5개의 옵션을 고르는 방식으로 질문에 답하게 하였다. ==> **zero-shot multiple-choice QA**

- 심리학 연구와 유사하게 2가지 측정 방법을 적용한다 : OCEAN score의 평균과 표준 편차

  - 긍정적으로 관련된 특정 KEY는 5(Very Accurate)부터 1(Very Inaccurate)의 점수를 받고 부정적으로 관련된 특정 KEY는 반대이다. 즉, 부정의 경우 5(Very Inaccurate)가 됨.

  - 각 trait에 대한 score는 다음과 같이 계산된다 : $Score_d = \frac{1}{N_d} \sum_{\alpha \in IP_d} f(LLM(\alpha, template))$ 

    - $IP_d$ : trait $d$ 와 연관된 item pool
    - $N_d$ : pool의 크기

    - $\alpha$ : test item
    - $LLM(\cdot, \cdot)$ : 사전 정의된 template을 가진 item에 대해 LLM이 내놓는 정답
    - $f(\cdot)$ : scoring method

  - score는 1~5까지의 값을 가지며 five factor 차원에서 모델의 personality 성향을 가리킨다.

- **LLM의 개성은 하나의 trait 차원의 평균 OCEAN 점수만으로 결정되어서는 안 되며, 단일 trait에서의 안정성과 일관성이 더 중요한 지표이다.**

  - 특정 factor 차원이 주어졌을 때, 안정적인 개성을 가진 모델은 동일한 경향을 나타내어 **모든 질문에 유사하게 응답**하므로 분산이 낮아진다. **(내부 일관성 / internal consistency)**

    - 예를 들어, Table 1의 모든 질문에 대해 정확하게 똑같은 응답을 내뱉는 것은 높은 분산을 가지게 될 것이다. (긍정과 부정 관련련 item들이 모두 존재하므로)
    - score가 3이면 특정 개성이 없는 상태이고, 1 or 5는 긍/부정적으로 강한 개성을 지닌 상태를 의미

  - 우리는 다양한 MPI 질문에서 LLM이 똑같은 trait을 가지면서 비슷한 행동을 하는지 평가해야한다. (내부 일관성을 평가해야 함)

    - 즉, 어떠한 개성이 생성된 상태라고 가정한다면 다양한 질문을 받았을 때, 해당 개성에 맞는 대답을 하는 경향을 보일 것이고 이러한 경향을 보이는지 평가하는 것이 중요하다

#### 2-3. Experiments

모든 LLM이 개성 평가에 적합하지 않으므로 모델을 고르기 위한 다음의 원칙을 따랐다.

1. zero-shot multiple-choice QA를 수행할 수 있을 만큼 충분히 커야함
2. 자연어 발화로 사전 훈련되어야 함.
3. 몇몇 downstream task에 적용 가능해야 함.

- 2가지 카테고리로 나누어 총 6개의 모델을 선정하였다.

  - Vanilla LM : BART, GPT-Neo 2.7B, GPT-NeoX 20B

  - instruction fine-tuned(aligned) LM : T0++ 11B, Alpaca 7B, GPT-3.5 175B

- Experimental Setup : HuggingFace Transformers or Eleuther AI's release에서 가져왔으며 `temperature ==0`를 사용했다. 프롬프트 템플릿은 `2-1`의 표와 같다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/5ec54b39-7dea-4419-a08d-62f88feb8f65" width="70%"></img></div>


- Results & Discussions : 위의 표2는 MPI를 사용해 LLM의 개성을 측정한 결과이다.

  - internal consistency $\sigma$ 와 모델의 일반적인 능력간의 상관관계를 볼 수 있었음. 특히, GPT-3.5와 Alpaca는 모든 factor에서 사람과 같은 internal consistency를 가졌음.

  - 실험을 통해 잘 정의된 심리학적 관점으로부터 LLM을 평가할 수 있음을 증명했다. 또한, 성격 이론을 통해 인간과 유사한 LLM의 행동을 정량적으로 분류하고 설명할 수 있었다.

  - Aligned LLM이 개성을 보였다고 결론지었다. (MPI에서 사람과 비슷한 개성의 안정성과 일관성을 보였음)

<br></br>

### 3. Inducing LLMs' Personality

이 섹션에서는 대부분의 LLM에서 zero-shot prompting으로 개성을 유도하는 것에 집중한다. (실험을 통해 LLM이 사람과 같은 개성의 통계를 가짐을 보였음)

우리는 LLM으로부터 다양한 개성을 유도하는 PERSONALITY PROMPTING($P^2$) 을 고안했다. 이는 LLM의 행동을 제어하는 질적인 방법이며 심리학 trait 연구로부터의 발견과 LLM 자체의 지식을 통합하는 순차적인 생성 프로세스이다.

#### 3-1. PERSONALITY PROMPTING ($P^2$)

$P^2$ 는 Big Five factor와 실제로 사용하는 언어의 강한 상관 관계가 있으며 chain prompt가 example보다 LLM의 행동에 더 잘 영향을 준다는 관찰에 기반한다. 우리는 단일 instruction보다 prompting을 위한 짧은 문장의 시리즈들이 LLM에 개성을 유도할 때 더 좋다고 가정한다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/44a683ef-5ff8-498e-bb3d-c14e8d3150f8" width="70%"></img></div>

- $P^2$ 는 다음과 같은 과정으로 이루어진다.

  1. 원하는 Big Five factor가 주어지면, `human-designed naive prompt`를 만든다.

  2. 심리학 연구에서 얻어낸 trait을 나타내는 단어들을 사용하여 naive prompt -> keyword prompt로 변경한다. 특정 trait을 부정적으로 유도할 때는 LLM이 생성한 반의어를 keyword prompts로 사용한다.
  3. Chain-of-thought prompting에서 영감을 받아, LLM이 keyword prompt에 응하여 어떤 trait을 가지는 사람을 표현하는 짧은 문장을 생성하도록 한다. (self-prompt)

  - 모델을 위한 마지막 프롬프트는 personality prompt, context, question으로 구성된다.

위의 그림을 예시로 들면, `Extraversion`을 target trait으로 설정하였고 psychological heuristic으로 naive prompt를  keyword prompt로 변환한 후, 이를 LLM에게 넘겨 Extraversion에 대한 짧은 설명을 생성하게 한다. 이렇게 해서 나온 것이 personality prompt이다.

human-designed prompt가 경험적이고 시행착오에 의존하는 반면, $P^2$ 은 LLM의 내부 지식을 사용하기 때문에 model에 더 적합하다.


#### 3-2. MPI Evaluation

- baseline prompting methods : $P^2$ 과 비교할 방법들

  - human-designed NAIVE PROMPTING : $P^2$ 의 첫 번째 step에 사용되는 방법 model은 `"You are a/an X person"`이라는 형식의 프롬프트를 받는다. 여기에서 X는 각 factor명이 들어감

  - WORDS AUTO PROMPTING : Prompt Search(Prasad et al., 2023; Shin et al., 2020)은 가장 효과적인 LLM prompting 중 하나임. word-lebel search를 사용하기 위해, big five factor의 가장 실용적인 세 단어를 찾았으며 빠른 검색을 위해 GPT-NEO 2.7B를  썼다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/b1554d43-4ca2-4a39-8ad9-d554468232a5" width="70%"></img></div>

- Results & Discussions

  - 위 표는 baseline과 $P^2$ 을 MPI를 사용하여 비교한 결과이다. $P^2$ 를 사용했을 때의 OCEAN 점수는 baseline, 제어가 없는 것(neutral) 보다 컸으며 internal consistency 측면에서도 neutral보다 안정적이었다.

  - $P^2$ 는 LLM에 특정 개성을 성공적으로 유도할 수 있고 MPI 검증으로 그 효과를 증명했다. 또한, 다른 baseline보다 성능이 좋았다.

  - 하지만 이는 MPI에서만 국한된 결과이므로 다른 시나리오에서도 일반화될 수 있는지 확인할 필요가 있다. (다음 섹션에 설명)

#### 3-3. Vignette Test

real-world 시나리오에서 개성 유도된 LLM을 평가해보기 위해 vignette test를 시행했다. 각 테스트에서는 LLM이 특정 가상 시나리오에 대해 짧은 에세이를 작성하게 한다. 이후, 생성된 에세이에 대해 100명의 사람이 personality factor tendencies에 따라 평가한다.

<div align="center"><img src="https://github.com/IamJunhaHwang/Papers/assets/46083287/62b88e68-e57b-4ed5-ad73-3daae1e34cd2" width="70%"></img></div>

- Results & Discussions

  - 거의 모든 factor에서 $P^2$ 이 baseline보다 뛰어났으며, $P^2$ 이 만든 예시들이 뚜렷한 개성을 가지는 것을 관찰했다.

  - MPI 평가에 이어 Vignette test에서도 개성이 유도되는 것을 검증했으며 제안한 방법이 보편적으로 모델의 행동을 제어할 수 있음을 증명했다.

<br></br>

### 4. Conclusion

- 평가 방법으로 Machine Personality Inventory(MPI)를 도입함으로써 LLM에 개성이 존재함을 밝혔다.

- $P^2$ 방법을 제안해 prompting chain으로 LLM의 특정 개성 행동을 효과적으로 유도했다.
