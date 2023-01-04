The Emotion is Not One-hot Encoding: Learning with Grayscale Label for Emotion Recognition in Conversation
=======================================================================================   
- 저자: Joosung Lee
- https://gitlab.com/bytecell/antlab_egg/-/issues/39#note_1220735102

Abstract
------------   

- 대화의 감정 인식(emotion recogntion)은 이전 context를 고려하여 예측됨
- 한 문장 안에 여러 감정이 공존할 수 있지만 대부분의 이전 연구들은 하나의 감정으로만 분류하려고 했음
- 문장에 여러 감정을 labeling하는 것은 비싸고 어려움
- 본 논문에서는 원 핫 인코딩 대신에 그레이스케일 레이블로 여러 감정을 나타내고 학습시킴


Introduction
------------------

- 감정은 화자의 대화 상태를 더 잘 이해할 수 있도록 하는 추가적인 정보임
- 대화에서 감정 인식(Emotion Recognition in Conversation)에서 발화(utterance)는 문장에서 감정 인식(Emotion Recognition in Sentence)보다 더 민감한 감정 분포를 가짐
- 5가지 method를 제안함: Category, Word-Embedding, Self, Self-Adjust, Future-Self


Proposed Approach
-------------------------

- Construction of Grayscale Label
- ![image](https://user-images.githubusercontent.com/49019184/210496702-bf5a4624-56fc-4253-9999-a1ea92b5877b.png)

|Dataset|  |category|  |   
|---|---|---|---|   
|  |positive|negative|neutral|   
|IMEOCAP|excited, happy|angry, frustrated, sad|neutral|   
|DailyDialog|happy|anger, disgust, fear, sad|neutral, surprise|   
|MELD|joy|anger, disgust, fear, sad|neutral, surprise|   
|EmoryNLP|joy, peaceful, powerful|mad, sad, scared|neutral|   

  - Category
    - $s_i =\begin{cases}1 \space\text{if}\space e_i=e_{gt} \\0.5 \space\text{if}\space category(e_i)=category(e_{gt}) \\0 \space\text{otherwise}\end{cases}$
    - ![image](https://user-images.githubusercontent.com/49019184/210496724-7f9546ad-506a-4d56-892c-1d63f42e8c6a.png)
    - $i$번째 감정 $e_i$의 점수 $s_i$는 위와 같이 gound-truth(참 값)과 비교하여 결정됨
    - $g_i = \frac{s_i}{\sum_{j=1}^k s_j}$
    - 위와 같이 정규화하여 그레이스케일 레이블 $g$를 계산함
  - Word-Embedding
    - $s_i = max(\frac{w_i \cdot w_{gt}}{||w_i||\space||w_{gt}||})$
    - 각 감정의 코사인 유사도 계산식, 유사한 의미를 가진 단어끼리의 유사도는 높고 다른 의미를 가진 단어는 유사도가 낮음
    - 그레이스케일 계산하기 전 $s_i$의 유사도 점수가 음수이면 0으로 변환함
  - Self
    - 앞의 두 method는 발화를 고려하지 않고 감정 관계만으로 계산됨, 단어 의미가 다르더라도 단어의 임베딩이 유사할 수 있음(예. 낮, 밤)
    - teacher-model, 사전 학습된 teacher-model의 logit이 점수가 되고 아래의 식으로 그레이스케일 레이블을 softmax를 통해 계산함
    - $g_i = \frac{e^{s_i}}{\sum_{j=1}^{k} e^{s_j}}$
    - student-model, ERC 데이터로 학습된 최종 모델(RoBERTa), student-model이 학습될 때, teacher-model의 파라미터는 고정됨
  - Self-Adjust
    - ![image](https://user-images.githubusercontent.com/49019184/210496749-66e545cb-7e08-4bd9-af20-6b10ae6603f5.png)
    - 왼쪽의 경우처럼 원 핫 인코딩에서의 답과 그레이스케일 레이블에서의 1순위 감정이 달라지는 경우 모델에 혼동을 줄 수 있음
    - $g_i^\prime = \begin{cases}0.5 \space\text{if}\space i=gt \\ \frac{0.5g_i}{1-g_{gt}} \space\text{otherwise}\end{cases}$
    - ![image](https://user-images.githubusercontent.com/49019184/210496766-d98c2ef1-8281-438e-9e50-57c870c2edea.png)
    - 원 핫 인코딩과 그레이스케일 레이블의 1순위 감정이 다른 경우 위의 식을 적용하여 원 핫 인코딩에 해당하는 감정을 0.5로 하고 나머지 감정들에 0.5를 그레이스케일 레이블을 참고하여 나눔
  - Future-Self
    - 미래 문맥은 청자의 반응에 영향을 미치는 현재 감정에 대한 정보가 있기 때문에 과거 문맥만 고려하지 않고 미래 문맥도 고려함
    - 미래 두 턴까지를 입력으로 사용함

- Loss
  - 원 핫 인코딩과 그레이스케일 레이블 둘 다 모델 학습시킴
  - 원 핫 인코딩은 1순위 감정 예측에 사용되고 그레이스케일 레이블은 여러 감정의 확률 분포를 예측하는데 사용됨
  - 두 경우 모두 cross-entropy loss function을 사용함
  - $L_b = -\frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^k o_i \log_{}{(p_i)}$
  - $L_g = -\frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^k g_i \log_{}{(p_i)}$
  - $L = L_b + \alpha L_g$
  - $N$은 학습 데이터의 수, $k$는 감정 분류의 수, $o_i$는 원 핫 인코딩, $g_i$는 그레이스케일 레이블, $p_i$는 예측된 감정 확률

Experiments
------------------

- "category"와 "word-embedding" method에서 모델을 원 핫 인코딩과 그레이 스케일 둘 다로 학습시킴
- "self-method"에서 teacher-model은 원 핫 인코딩을 이용한 분류로 학습되고 모델의 parameter를 고정시킨 후, stduent-model을 원 핫 인코딩과 그레이 스케일 둘 다로 학습시킴
- Training Setup
  - 허깅페이스의 사전 학습된 모델 중 validataion set의 성능이 가장 좋게 나온 모델 사용
  - optimizer: AdamW, 학습률: 1e-6, 학습률 scheduler: get_linear_schedule_with_warmup, gradient clipping: maximum value of 10
  - A100 GPU
- Dataset and Evaluation
  - 4개의 데이터셋 사용(IEMOCAP, DaliyDialog, MELD, EmoryNLP)
  - ![image](https://user-images.githubusercontent.com/49019184/210496786-f126ece1-cf84-478e-850f-5876c0912e6c.png)
  - DaliyDialog의 7가지 분류 중 중립을 제외한 6가지 분류에 대해서 macro-f1으로 평가, 나머지 데이터셋은 weighted average f1으로 평가
  - ![image](https://user-images.githubusercontent.com/49019184/210496800-b85ea22c-1507-46cd-9a1f-3bda8c16eb84.png)
- Results and Discussion
  - ![image](https://user-images.githubusercontent.com/49019184/210496814-8e39d32a-ca34-4266-9071-b19439be45c7.png)
  - 그레이 스케일 레이블은 모델에 관계없이 평균적으로 좋은 성능을 보임
  - 다른 모델에서는 RoBERTa와 달리 미래 입력을 간단하게 결합할 수 없기에 +FSA를 사용하지 않았음
  - +C와 +W는 간단하고 효과적이지만 발화를 고려하지 않기 때문에 한계가 있음
  - self-methods는 미래 대화를 고려하고 더 세분화된 그레이 스케일 레이블이기 때문에 성능이 더 좋음
  - 모델이 아닌 데이터에 method를 적용시키는 것은 noise를 만들어 낼 수 있어 성능이 오히려 낮게 나옴

Conclusion
-------------------

- `category`와 `word-embedding`은 정답 감정으로 그레이 스케일 레이블을 구성함
- `self-method`는 대화로 그레이 스케일 레이블을 구성함, 사전 학습된 모델로 그레이스케일 레이블을 생성하기 때문에 잘못된 정보가 있을 수 있음
- `SA`나 `FSA`를 이용하여 `self-method`보다 성능을 좋게 함
- RoBERTa와 비교했을 때 좋은 성능이 나왔음
