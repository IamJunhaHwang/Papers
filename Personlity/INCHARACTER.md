## :page_facing_up: INCHARACTER: Evaluating Personality Fidelity in Role-Playing Agents through Psychological Interviews


### Info

* `publication`: ACL 2024 -  Long

* `author`: Wang et al.
* `url`: https://aclanthology.org/2024.acl-long.102.pdf
* `github`: https://github.com/Neph0s/InCharacter

### 1. Abstract & Introduction

LLM의 발전을 통해 Role-playing agents(RPAs)가 등장하였지만, 이러한 RPAs의 character fidelity를 평가하는 것은 상대적으로 덜 연구되었다.   
이전 연구에서는 주로 지식, 경험, 언어적 패턴을 잘 복제하는지를 이용해 평가하였지만, (1) 캐릭터 특화 데이터 셋이 필요해 새로운 캐릭터에 적용하기 어렵고, (2) RPA의 생각이나 mindset에 대해서는 평가하지 않았다. 

본 연구에서는 RPA가 target 캐릭터의 `personality`를 신뢰있게 잘 생성했는지를 평가하는 방법인 **INCHARACTER** 를 제안한다.   
LLM의 personality에 대한 이전 연구들에서는 LLM에게 option을 선택하게 prompting하는 self-report방식을 채택하였지만, 여기에는 한계가 있다: (1) scale을 대답하게 하는 지시는 role-playing 지시와 겹치게 되어 personality test에 덜 따르게 만들고, (2) option을 선택하게 하는 것은 RPA의 실제 행동과 반할 수 있다(이로 인해 personality 결과가 실제 캐릭터를 반영하지 못함).

INCHARACTER(**IN**terviews **CHARACTER**)는 interview-based 방식을 통해 대화에서 효과적으로 personality를 식별하고 self-reported의 한계를 극복한다. INCHARACTER는 다음과 같이 진행된다:

1. Interview: RPA에 psychological scale을 기반으로 open-ended questions 에 대답하게 해 mindset과 behavior를 드러내게 한다.

2. Assessment: LLM을 이용해 위의 interview 대답을 Likert level로 바꾸거나 LLM이 RPA personality를 판단하기 위한 psychiartist로 사용한다.

INCHARACTER를 다양한 RPA에 대해 14 personality test(BFI, 16P, DTDD, etc)에 적용하였다. 실험을 통해 INCHARACTER가 효과적으로 interview-based test를 수행하고, RPA personality measurement가 해당 캐릭터와 잘 맞게 나오는 것을 보였다.

- Contributions

  - RPA 평가를 위해 psychological scale을 기반으로한 personality fidelity 개념 도입

  - interview-based personality test framework인 INCHARACTER 제안

  - 실험을 통해 다양한 RPA와 psychological scales가 INCHARACTER의 효과를 증명함

<br></br>

### 2. INCHARACTER

<div align="center"><img src="https://github.com/user-attachments/assets/ea76d088-20ae-48bf-9183-c96ef83292f6" width="60%"></img></div>

#### 2-1. Interview

- Constructing Question List: 각 item $p \in P$ 는 LLM을 이용해 open-ended question `q`로 바뀌고 이를 하나하나 체크하였음

  - Ex)  “Values artistic, aesthetic experiences.” --> “Do you values artistic, aesthetic experiences?”

- Interviewing RPAs: 캐릭터 `c`를 가진 RPA C를 각 question에 대해 interview를 시행하고, response를 기록함

#### 2-2. Assessment

interview 결과를 기반으로 각 dimension $d \in D$ 에 대해 RPA C의 점수 $s_d$ 를 평가한다. 이를 위해, 2가지 방법을 도입한다: (1) Option conversion (OC), (2) Expert Rating (ER)

- Option Conversion: LLM에게 response `r`을 해당하는 question의 answer option $a \in O$ 로 바꾸게 함

  - 이렇게 바뀐 answer list를 scoring scheme에 넣어 final personality score를 구한다.
  
  - 실제로는 SOTA LLM인 GPT-4에서도 RPA의 attitudes를 categorizing하는 것에 정확도가 낮게 나와, `dimensional-specific option conversion`을 적용하였다.

    - 이는, dimension에 따라 (q, r) pair를 분리하고, Likert level을 좀 더 descriptive한 option으로 바꾸는 것.

    - “4 (Agree)” and “2 (Disagree)" --> “4 (Extroverted)” and “2 (Introverted)” 

- Exper Rating: question 하나하나 적용하는 OC와 다르게 ER은 LLM을 이용해 각 dimension에대해 RPA의 personality score를 바로 계산

  - interviewer LLM에게 <scale, dimension, score range>에 대한 정보를 prompting하고 각 dimension에 대해 final personality score를 생성한다.



<br></br>

### 3. Experimental Setup

<div align="center"><img src="https://github.com/user-attachments/assets/e7169d2f-55c4-4687-9874-468b9a8fd25f" width="40%"></img></div>

#### 3-1. Preliminary Study: Can LLMs Simulate Human Interviewers?

interviewer LLM이 OC, ER tasks에서 사람처럼 잘 평가할 수 있는지 검증하였다(Human judgements와 LLM predictions 비교).

BFI에서 SOTA RPAs의 100 cases를 샘플링한 후, 수동으로 labeling 하였으며 interviewer LLM으로 GPT-4, GPT-3.5, Gemini를 이용하였다.    
human annotations와 interviewer LLMs 간의 correlation(Pearson’s r, Spearman’s ρ, and Kendall’s τ)과 accuracy를 report했다. accuracy 계산에는, LLM predictions와 human labels의 차이가 1보다 작으면 정답, 정확히 1이면 close, 1보다 크면 오답으로 간주했다.

Table 1을 보면, ER에서는 LLM이 알맞게 participants' personalities를 rating하였다. OC에서는 상당히 정확도가 좋지 않았고, 이는 d-OC로 대체되었을 때, 크게 상승하였다.   
일관성 평가에서는, SOTA LLMs가 acceptable performance를 보였음(즉, ER과 d-OC를 통해 human interviewer처럼 RPA 평가가 가능하다).


#### 3-2. Experimental Settings

- RPAs and Characters: 16 from ChatHaruhi, 16 from RoleLLM; popular fictional works(e.g. Harry Potter etc)

- Psychological Scales: 14 psychological scales(BFI, 16P, 12 others following PsychoBench)

- Personality Labels: character personalities를 위해 `scores` 와 `types` label을 collect

  - `types`의 경우 Personality Database (PDb)에서 각 dimension에 대한 label percentage로 가져옴(ex. 60% 이상 positive, 40% 이하는 negative, 그 외는 marginal)
  
  - `scores`로는 각 캐릭터당 2~3 human annotator를 두었고, 이를 평균내어 [0,1] 구간으로  re-scale하였음.

    - inter-annotator consistency를 Cohen's kappa coefficient로 측정하였고 14 scales에서 평균적으로 60.9%가 나옴

- Interviewer LLMs: GPT-3.5, GPT-4, Gemini, and Qwen1.5-110B

- Metrics: Measured alignment(MA), Personality consistency(PC)

  - Measured alignment(MA): 측정된 RPAs의 personalities와 human-annotated personalities 비교

    - RPAs의 성능과 personality test methods 효과에 의존

    - RPAs가 각 dimension에 대해 score가 scoring range의 중앙값보다 높거나 낮은 것을 positive나 negative로 분류(marginal dimension은 제외)

    - **MAE**를 이용해 `score`에 대한 일치도를, $Acc_{dim}$ 과 $ACC_{Full}$ 을 통해 각 dimension or 전체 dimension의 정답율을 계산한다.

      - MAE는 scoring range 길이로 나누어 re-scale

  - Personality consistency(PC): 측정된 RPAs의 personality가 다양한 settings에서 일관되게 나오는지

    - $Std_{item}$: item-level 표준편차; multiple runs간의 같은 item에서 RPA's consistency

    - $Std_{Dim}$: dimension-level 표준편차; 같은 dimension에서 다른 items간의 RPA's response 비교
    - $Std_{Score}$: Score-level 표준편차; multiple runs간의 각 dimension에서 RPA's score 분산

    - 이 metrics를 해당하는 scoring range 길이로 나누어 unit interval [0,1]로 re-scale함.

<br></br>

### 4. Experimental Results

#### 4-1. Personality Tests on RPAs

- baseline

  - ER: 1개의 dimension에 대한 question-response pairs를 한 번에 전부를 interviewer LLM에게 넘기거나(ER_all) batch로 넘김(ER_batch)

  - SR: self-report; RPA에게 각 scale item을 선택하도록 prompt

  - interview phase와 assessment phase를 3번 반복하여 평균을 report

<div align="center"><img src="https://github.com/user-attachments/assets/179bd9b3-fb00-4594-a79c-ca2904890133" width="60%"></img></div>


- **Alignment between RPAs’ Measured Personalities and Characters’ Labeled Personalities**

  - GPT-4와 ER로 INCHARACTER를 사용한 것이 RPA personalities ground truth와 높게 일치 --> SOTA RPAs가 캐릭터의 personality trait을 잘 reproduce함

  - Self-Report 보다 INCHARACTER가 RAP personalities와 더 잘 일치함

  - INCHARACTER로 측정한 alignment는 interviewer LLMs의 assessment task에 대한 능력과 관련있음(GPT-4가 best였음).
   
- **Robustness, Consistency and Distinctiveness of RPA Personalities**

  - `Std_score` 가 여러 세팅에서 6% 이하였음. --> personality test의 신뢰성과 RPA personalities의 robustness를 의미

  - GPT-4를 이용한 d-OC 사용했을 때, 여러 시도에서도 일관되게 같은 item(선택지)을 골랐음.

- **Self-report v.s. Interview-based Methods**

  - INCHARACTER로 측정한 personalities가 Self-Report보다 더 캐릭터의 personalities와 일치하였으며, 일관성과 구별성도 더 좋았음

  - CoT를 이용한 Self-Report는 improvement가 제한적이었으며, `Std_Item`이 좋지 않았음.

#### 4-2. Personality Fidelity of Different RPAs

GPT-3.5를 이용한 `ER_batch`로 INCHARACTER를 적용해. INCHARACTER 내에서 다른 character data, foundation model에서의 차이를 비교

<div align="center"><img src="https://github.com/user-attachments/assets/b0cb65e6-0043-4604-89d5-79e0d0d962c4" width="40%"></img></div>


- Character Data for RPAs

  - 보통, 현재의 RPAs에서는 2가지 종류의 character data를 사용함; (1) descriptions: character descriptions as system prompt, (2) memories: 캐릭터의 경험, retrieval을 위한 대화
  
  -  GPT-3.5에서 only D, only M, both D+M을 실험하였는데, D만 사용한 것이 full D+M과 비슷한 결과를 보였다. --> 이는 character description이 중요함을 의미
  
  - RPAs가 past experiences에서 보인 character personalities를 잘 따라함

- Foundation Models for RPAs

  - GPT-3.5와 GPT-4가 best personality fidelity를 보임.

  - SOTA open-source LLM으로도 RPAs가 character personalities를 reproduce할 수 있지만, 이는 특정 언어에 크게 의존(ex. LLaMa-2-Chat-13B는 영어에서 좋은 성능을 보였지만, 중국어에서는 그렇지 않음)

  - Table 10: 추가적인 fine-tuning이 크게 영향을 주지는 않음.
