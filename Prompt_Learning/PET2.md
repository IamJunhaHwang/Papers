## :page_facing_up: It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners


### Info

* `conf`: NAACL 2021

* `author`: Timo Schick, Hinrich Schutze

* `url`: https://arxiv.org/abs/2009.07118

* `github`: https://github.com/timoschick/pet

- `Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference`의 후속 논문


### 1. Abstract & Introduction

파라미터를 수 천억개로 늘렸을 때, GPT-3와 같은 PLM은 뛰어난 few-shot 성능을 보였다. 하지만, 이는 많은 양의 cost가 필요되어 모델을 사용하는 데에 어려움이 있다. 또한, few examples를 무한정 줄 수도 없는 노릇이다(token sequence length가 정해져있으므로). 따라서, 우리는 파라미터 수가 훨씬 작지만 GPT-3와 비슷한 성능을 낼 수 있음을 보인다.(PET) 이는 입력을 task description이 포함된 cloze questions(빈칸 맞추기)로 바꾸는 것과 gradient-based optimization을 합해 구현할 수 있다. (gradient-based optimization: GPT와 다르게 파라미터가 학습이 되는 것을 말함)   

PET는 LM이 예측한 정답과 대응되는 **single token**에만 동작하는 한계가 있다. 따라서, 본 논문에서는 multiple tokens 예측을 필요로하는 태스크에 PET를 적용시킨다. 그리고 우리는 ALBERT와 PET,iPET를 결합한 것 둘 다 SuperGLUE에서 GPT-3를 능가함을 보인다. 또, unlabeled data 없이도 유사한 성능을 달성할 수 있음을 보이고 PET에 기여한 요인에 대해 분석한다.

<br></br>

### 2. Pattern-Exploiting Training

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/bc5b9a17-32e3-4c76-8191-db1db8463b9b" width="35%"></img></div>

- 기호 정의

  * `M`: Masked Language model
  * `T`: Vocabulary
  * $T^*$ : set of all token sequences
  * 우리는 task를 입력 $x \in X$을 출력 $y \in Y$으로 mapping하는 것으로 간주한다.
  * $q^k_M (t|z)$: `z`에서 k번째 마스킹된 위치의 `t`에 모델 M이 할당하는 확률

    - $z \in T^∗$ 은 적어도 k개의 마스크를 포함하고 있음
    - $t \in T$ 이다.

    - softmax를 적용하기 전 model의 logit은 다음과 같다. -> $s^k_M (t|z)$
    
  * PVP $`p = (P, v)`$ : pattern-verbalizer pair(PVP)

     * pattern $P : X \rightarrow T^*$: 입력을 받아 하나의 mask token을 포함하는 phrase나 sentence로(cloze questions) 매칭
     * verbalizer $v: Y \rightarrow T$ : 각 output을 패턴에서의 task-specific meaning을 가지는 single token으로 매핑

PET의 핵심 아이디어는 `P(x)`의 마스킹된 위치에 있는 `v(y)`가 올바른 토큰일 확률로부터 y가 정답일 확률을 얻는 것이다. [figure 2]   

- 위에 따라, x가 주어졌을 때 y의 조건부 확률 $q_p$는 아래와 같이 정의된다.

  - $q_p(y|x) = \frac{e^{s_p(y|x)}}{\sum_{y' \in Y}e^{s_p(y'|x)}}$
  
  - $s_p(y|x) = s_M^1(v(y) | P(x))$ 는 `P(x)`의 마스킹된 위치에 있는 `v(y)`의 raw score이다.

- 성능이 좋은 PVP를 찾는 것은 large development-set이 없을 때 어려운 일이므로 아래와 같이 여러 PVP들의 조합한다.

  - 각 PVP에 대해, MLM은 training exapmles `(x,y)`로 fine-tune 된다. (`y`와 $q_p(y | x)$ 사이의 cross-entropy를 줄이도록 학습)

  - 그 후, 위에서 만든 모델들을 앙상블하여 unlabeled example들에게 아래의 조건부 확률에 따라 soft label을 붙여준다. 

    - $q_p(y | x) \propto exp \underset{p \in P}{\sum} w_p \cdot s_p(y|x)$

    - $w_p$ 는 훈련 전에 training set을 `p`에 사용했을 때의 달성한 정확도에 비례하는 가중치 값이다.

  - 모든 과정을 마치고 만들어진 soft-labeled dataset을 이용해 classifer를 훈련한다. (분류기 출력과 $q_p$ 의 cross entropy를 줄이도록 학습)

**이후 나오는 iPET 관련 내용은 이전 논문(Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference)과 같으므로 생략**

<br></br>

#### 2-1. PET with Multiple Masks

PET의 가장 큰 한계는 `verbalizer`가 출력이 하나의 토큰으로만 매핑되는 것이다. 따라서, $v: Y \rightarrow T^*$ 로 `verbalizer`를 일반화하였고 이는 몇 가지 수정이 필요하다.

- 우리는 PET을 더 일반화하여 각 입력에 대해 출력 공간이 동일하다고 가정하지 않는다. 

  - 각 $x ∈ X$에 대해 x가 입력으로 주어졌을 때 가능한 출력들의 집합을 $Y_x ⊆ Y$로 정의하고, PVP $p = (P, v)$ 가 주어졌을 때, $l(x) = max_{y∈Y_x} |v(y)|$ 를 정의하여 $Y_x$ 의 어떤 출력을 표현하는 데 필요한 토큰의 최대 수를 나타냈다. 
  - $P^k(x)$는 k개의 마스크 토큰으로 대체한 $P(x)$를 나타낸다.

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/9e7fcc3b-d9fb-43ca-965e-069089ae19eb" width="35%"></img></div>

예를 들어, 레스토랑 리뷰에 대한 레이블이 `{-1, 1}`인 이진 감성 분류를 한다 했을 때, $P(x) =$ `x. It was ___.` 이고 `verbalizer v`는 `+1 -> great`, `-1 -> {terri, ble}`로 매핑한다. (토크나이저가 terrible을 2개로 나눈다고 가정)   

- **Inference**

  - 우리는 $q_p(y | x)$ 를 autoregressive하게 재정의한다. ($P^k(x)$ 부터 시작해 k번 연속해서 예측을 한다. 이 때, MLM's confidence에 따라 next token을 예측한다.)

  - $q_p(y | x) = q(v(y) | P^k(x))$ where

  - ![image](https://github.com/DAILAB-CBNU/Papers/assets/46083287/b7d2acc9-17d0-4e87-8eae-3801b4f5e4f4)

    - $j = argmax_{i=1}^k q^i_M(t_i|z)$

    - $z'$ : $z'_j = t_j$ 를 제외한 $z$; 앞의 수식 $q^j_M$ 에서 계산한 가장 높은 확률의 토큰을 제외한 문장 $z$

    - $t' = t_1 ... t_{j-1} t_{j+1} ... t_k$

    - $q_p$ 는 기존 PET(Eq.1)처럼 확률(softmax를 취하는)이 아니다. 

  - 위의 `Fig.3`에는 $q_p(-1|x)$ 를 어떻게 계산하는지 보여준다.

    - $|v(y)| =$ `|{terri, ble}|`이므로 각 토큰의 확률을 계산하기 위해 $z=P^2(x)$ 를 사용한다.

    - 그 후, 가장 높은 확률을 선택한 후, 해당되는 마스킹 토큰 자리에 넣는다. 그리고 $z'$를 사용해 남은 토큰의 확률을 구한다. ($y = -1$ 의 전체 score는 아래와 같다)

    - $q_p(-1|x) = q_M^2(ble|z) \cdot q_M^1(terri|z')$

- **Training**

  - 각 training example마다 `Eq.3`을 계산하는 것은 엄청난 cost이므로 하나의 forward pass로 계산하기 위해, $q_p(y|x)$ 를 아래에 따라 근사한다.

    - output을 표현하기 위해 필요한 mask token의 수를 항상 최댓값으로 준다.

    - 각 $y' \in Y_x$ 에 대해 $v(y') = t_1 ... t_k$ 의 모든 토큰을 병렬로 예측하며 $l(x) - k$ 의 예측 값은 무시한다.

    - $\overset{\sim}{q_p}(y'|x) = \prod_{i=1}^k q^i_M(t_i|P^{l(x)}(x))$

  - $\overset{\sim}{q_p}$ 가 $Y_x$ 의 확률 분포가 아니기 때문에 cross entropy는 이상적인 training objective가 아니다. 따라서, `multi-class hinge loss`를 사용했다.

    - $\underset{y' \in Y_x}{\sum} max(0; 1-log \overset{\sim}{q_p}(y|x) + log \overset{\sim}{q_p}(y'|x))$

      - `y`에 대한 log probability와 적어도 1이 되는 어떤 output $y' \in Y_x$의 log probability 가 필요하다.  
      
<br></br>

### 3. Experiments & Results

PET와 GPT-3를 8개의 NLU task로 구성된 SuperGLUE셋으로 비교하였다. GPT-3가 사용한 데이터와 동일한 데이터를 사용할 수 없기 때문에, 각 태스크에 대해 랜덤으로 선택한 32개의 examples와 20000개의 unlabeled examples 새로 만들었다. 

- `underlying LM`: ALBERT-xxlarge-v2
- `Dataset`: SuperGLUE

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/ab5e4b84-f43f-49ce-bf68-83e6cef076c1" width="75%"></img></div>

`ALBERT with PET`는 785배 차이나는 파라미터 수에도 불구하고 GPT-3와 비슷한 성능을 보였다. 또, 이와 비슷한 크기인 GPT-med와 비교했을 때 `Avg`에서 18점이나 높았다.   

<br></br>

### 4. Analysis

#### 4-1. Patterns

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/92e324aa-fc75-469a-b349-f486554699aa" width="35%"></img></div>

WSC task에서 GPT-3가 사용한 패턴(task description)은 거의 30개의 단어가 사용된다. 이런 식으로 패턴을 구성하는 방법에 따라 어떤 차이가 있는지 조사해보았다. (GPT-3와 PET 비교)

RTE에서는 GPT-3가 성능이 좋았고 MultiRC에서는 PET의 패턴이 좋았다. 그리고 이 두 방식을 섞은 것이 이전 두 방식보다 좋았다. 이는 패턴의 집합에 대한 신중한 engineering을 통해 suitable한 패턴을 찾는 것의 잠재성을 보여준다.

#### 4-2. Unlabeled Data Usage

PET는 unlabeled data를 이용해 각각의 PVP들에 대한 모델들의 지식을 증류하는데, 이 데이터는 실제로 항상 얻을 수 있지 않으므로 unlabled data가 얼마나 PET에 영향을 주는지 알아보았다. 

- PET의 final classifier와 각각의 PVP들에 해당되는 모델의 앙상블을 직접 사용한 것을 비교

  - 앙상블한 모델은 distilled model 보다 `3*k`배 크다. (PET의 기본 설정을 따르고 PVP별로 3개의 모델을 훈련하므로)

결과는 위의 `Table 2`의 아래부분과 같다. distilled classifier보다 더 좋은 성능을 보였다. 따라서, unlabeled data는 성능에는 중요하지 않지만, 가벼운 모델을 얻으려면 unlabeled data가 필요할 것이다.

#### 4-3. Labeled Data Usage

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/a959f9ca-6c11-41dd-b0ea-c0a6dbe1c529" width="35%"></img></div>


labeled data을 사용하는 것이 어떤 영향이 있는지 알아보았다. PET와 supervised model(어떤 패턴도 사용하지 않은), PET unsupervised model(labeled data를 사용하지 않은 iPET)을 비교했다.

32개의 example이 주어졌을 때, PET가 제일 성능이 좋았다. [labeled data가 PET에 영향을 줌]

또한, priming방식과 PET를 비교했을 때, PET가 더 성능이 좋았다. (처리할 수 있는 최대 토큰 길이에 대한 이슈로 XLNET으로 실험)

#### 4-4. Training Examples

<div align="center"><img src="https://github.com/DAILAB-CBNU/Papers/assets/46083287/94e9cf9c-cfd0-4eb1-a9ce-1057ca6fad1e" width="35%"></img></div>

training example을 random seed를 다르게 주어 총 3개를 만들어 비교했다.

결과, 다른 training example은 PET 성능에 영향을 주었다. 이는 GPT-3에서의 인사이트와 다르지만, 아마 GPT-3가 PET보다 training example에 덜 영향을 받기 때문이라고 생각된다.

<br></br>

### 5. Conclusion

- multi tokens를 예측하는 것이 필요하는 task에도 적용할 수 있게 PET를 변형한 것을 제안했음

- `PET with ALBERT`가 성능이 좋은 요인에 대해 알아냈음

- PET를 사용해 GPT-3보다 파라미터가 3배 적지만 비슷한 성능을 내는 few-shot text classification이 가능함을 보였다.
