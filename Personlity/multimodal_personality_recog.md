## :page_facing_up: Multimodal Fine-Grained Apparent Personality Trait Recognition: Joint Modeling of Big Five and Questionnaire Item-level Scores



### Info

* `publication`: AAAI 2025
* `author`: Ryo Masumura et al.

<div align="center"><img src="https://github.com/user-attachments/assets/1982fa3a-b468-4425-b967-a9f381636e9f" width="80%"></img></div>

### 1. Abstract & Introduction

사람들의 personality trait을 인식하는 것은 많은 관심을 받고 있다. 가장 자주 사용되는 `Big Five`라는 personality trait이론이 사용되며 personality trait은 다음의 2가지 타입으로 측정할 수 있다.

- self-assessed personality trait: 본인에게 직접 얻어짐

- apparent personality trait: 여러 타인에 의해 인식됨 (ex. first impression)

  - 이 방식의 경우, 시간과 노력이 많이 필요하기 때문에 이전 연구들에서 human multimodal behavior로부터 자동으로 apparent personality 인식을 하는 여러 방법을 제시해왔다.

이를 위해, 이전 연구들에서는 주로 individual modoals의 feature extraction이나 multimodal information의 fusion method에 집중해왔다. 하지만, personality traits를 얻어내는 방법에 대해서는 연구가 진행되지 않았다.   
personality trait recognition model은 주로 personality trait score를 직접적으로 추정하도록 학습되었고, ground-truth personality score는 여러 사람의 test 결과로 결정된다. **즉, 해당 test 결과에 포함된 풍부한 정보가 이 personality score를 결정하는 것에만 사용된다.**

test에는 많은 questionnaire items(20~360)이 있으며, 이러한 test 결과를 사용하지 않는 것은 낭비이다. 이러한 정보를 사용한다면, multimodal human behavior 인식과 Big Five score 추정 성능을 향상시킬 수 있을 것이다.

본 논문에서는, 사람의 apparent personality score와 questionnaire item-level score를 동시에 추정하는 multimodal fine-grained apparent personality trait recognition method를 제안한다. (위 그림 참고)

- main contributions

  - apparent personality trait recognition dataset을 새로 만듦.

    - 이전 datasets에는 questionnaire-based personality test results가 없음.
    - 10,000개의 self introduction video를 수집해 5명의 사람에게 50개의 질문이 있는 Big Five test를 해서 scoring

  - 2가지 type의 joint modeling methods 제시 (Big Five score, questionnaire item-level score를 동시에 제공, item-level을 먼저 예측 후 Big Five score를 예측하게 하는 cascaded model)

<br></br>

### 2. Data

- Recorded Videos

  - 10,100개의 self-introduction videos를 1,010명의 참가자로부터 얻었다. (10개의 질문에 대한 video)

  - video의 평균 시간은 73.6초이며, min, max는 각각 59.1초, 102.1초
  - 1280 x 720 해상도, 25fps이며 오디오는 16kHz, Zoom이나 laptop으로 촬영
  - train:val:test = 9,030: 500: 570 videos
    - first impressions dataset과 비슷한 양

- Annotations of Apparent Personality Traits

  - `Big Five`를 적용하였으며, 50-item questionnaire를 총 200명의 assessors 고용해 레이블링

  - training, validation data에는 5명을 랜덤하게 할당하였으며, test data에는 10명을 랜덤하게 할당

    - test의 경우 5명은 ground-truth 정보를 위해, 나머지 5명은 human evaluation에 사용

  - 각 평가자는 2\~3번 video를 보고 평가하였으며, 5-point scale로 점수를 매김 (deep learning model에서는 0\~1로 정규화해 사용)

<br></br>

3. Task Definition

- personality trait scores: $\hat{y} = [ \hat{y}_1, ..., \hat{y}_K ]^T$

  - K: number of personality traits (Big Five에서는 5)

- questionnaire item-level scores: $\hat{z} = [ \hat{z}_1, ..., \hat{z}_E ]^T$

  - E: number of questionnaire items (50-item이므로 50)

- 위 2개의 점수는 audio-visual video input으로 부터 동시에 추정된다.

  - audio features: $S = {s_1, ..., s_M}$
  - visual features: $U = {u_1, ..., u_N}$

    - $s_m$ 은 m-th audio feature, $u_n$ 은 n-th visual feature
    - M: number of audio features, N: number of visual features

- multimodal fine-grained apparent personality trait recognition에서, 다음과 같이 두 점수를 추정한다.

  - $\{ \hat{y}, \hat{z} \} = \mathcal{F}(S, U; \Theta)$

    - $\mathcal{F(\cdot)}$ : model fuction
    - $\Theta$ : trainable model parameter set

- automatic speech recognition (ASR) system이 audio features S를 text feature $W = {w_1, ..., w_L}$ 로 바꿀 수 있다.

  - $\{ \hat{y}, \hat{z} \} = \mathcal{F}(S, W, U; \Theta)$

- 모델을 학습시키기 위해, audio-visual video input과 personality test results로 구성된 dataset을 준비한다.

  - $\mathcal{D} = \{ (S^1, U^1, Q^1), ..., (S^{|D|}, U^{|D|}, Q^{|D|}) \}$

    - $Q^d = {z^d_1, ..., z^d_C}$ ; d-th questionnaire-based personality test results

      - $z^d_c = [ z^d_{c, 1}, ..., z^d_{c, E}]$ ; d번째 video input에 대해 c번째 사람의 questionnaire item-level scores 

        - C: number of people
        - E: number of items in the questionnaire

  - 데이터셋은 다음과 같이 바뀐다: $\mathcal{D} = \{ (S^1, U^1, y^1, z^1), ..., (S^{|D|}, U^{|D|}, y^{|D|}, z^{|D|}) \}$

    - $y^d = [y^d_1, ..., y^d_K]$ : d-th ground-truth personality trait scores
    - $z^d = [z^d_1, ..., z^d_E]$ : d-th ground-truth questionnaire item-level scores

- personality trait score의 k-th score와 questionnaire item=level scores의 e-th score는 다음과 같이 계산된다

  - $y^d_k = \frac{1}{C} \sum^C_{c=1} PersonalityTrait (z^d_c; \lambda_k)$
  - $z^d_e = \frac{1}{C} \sum^C_{c=1}z^d_{c,e}$

    - `PersonalityTrait()`: questionnaire item-level scores를 few-dimensional personality trait score로 바꾸어주는 함수
    - $\lambda_k$ : k-th personality trait을 계산하기 위한 pre-defined parameters

<br></br>

### 4. Modeling Methods

여기에서는, multimodal fine-grained apparent personality trait recognition을 위해, single-task modeling method와 proposed joint modeling methods를 설명한다.   

본 논문에서는 두 method 모두 똑같은 backbone architecture로 transformer를 사용했다.

#### Backbone Multimodal Transformer

<div align="center"><img src="https://github.com/user-attachments/assets/721f41e8-5ef7-453f-85d6-efb4127e2a8e" width="50%"></img></div>

- 4가지 encoder blocks: audio, text, visual, multimodal encoders로 구성되어 있다. 

  - audio encoder는 audio features S를 audio representations A로, 나머지 modality도 비슷하게 바꾸어 준다. (text W -> T, visual U -> V)

- multimodal encoder의 input은 temporal axis로 concatenate하는 `TemporalConcat`과 concatenated vector를 구분하기 위한 continuous vector를 더하는 `AddSegment`를 거친다.

- 이후, attentive pooling을 통해 variable length hidden vector를 fixed size vector로 바꾸어준다.

#### Single-Task Model

- single-task modeling에서는 personality trait scores를 바로 추정한다

  - $\hat{y} = Sigmoid(h; \theta_{head})$

  - loss function: $\mathcal{L} = \frac{1}{|D|} \sum^{|D|}_{d=1} |\hat{y}^d - y^d|$

    - ground-truth personality trait scores와 추정값에 대한 mean absolute error (MAE)

  - 비슷하게 questionnaire item-level score도 모델링 가능 (이 2개는 독립적으로 최적화됨)

#### Joint Models

- Multi-Task Joint Model: personality trait scores와 questionnaire item-lvel scores를 2개의 prediction heads를 사용해 함께 추정

  - $\hat{z} = Sigmoid(h; \theta^z_{head})$
  - $\hat{y} = Sigmoid(h; \theta^y_{head})$

- Cascaded Joint Model: questionnaire item-level score를 먼저 추정 후, personality trait scores 추정

  - $\hat{z} = Sigmoid(h; \theta^z_{head})$
  - $\hat{y} = Sigmoid(\hat{z}; \theta^y_{head})$

- Loss Function: 두 모델 모두, ground-truth 와 추정된 personality trait scores 사이의 MAE와 ground-truth와 추정된 personality test scores 사이의 MAE로 계산

  - $\mathcal{L} = \frac{1}{|D|} \sum^{|D|}_{d=1} |\hat{y}^d - y^d| + \frac{\alpha}{|D|} \sum^{|D|}_{d=1} |\hat{z}^d - z^d|$

    - $\alpha$: 두 점수 사이의 균형을 위한 hyperparameter

<br></br>

### 5. Experiments

- Pre-Training of Model-Specific Encoders: backbone의 일부로 pre-trained model을 사용함. 

  - Audio(Hsu et al., 2021), Text(Devlin et al., 2019; BERT), Image(Cao et al., 2018)

- Evaluation Metrics: Pearson's correlation coefficient & Accuracy

#### Results

<div align="center"><img src="https://github.com/user-attachments/assets/f532116e-633b-4c53-8442-9a843f519b97" width="80%"></img></div>

- Evaluation of Backbone Multimodal Transformer

  - 위 표는 input modal에 따른 Big Five 성능을  보여준다. (single-task modeling method 사용)

  - (audio, text, visual) input의 조합이 효과적이었으며, pre-training model을 사용하는 것이 필수적임을 알 수 있다.

<div align="center"><img src="https://github.com/user-attachments/assets/77fe0a19-ed45-41e1-942f-1aa7f6c91a5c" width="80%"></img></div>

- Evaluation of Joint Modeling Methods

  - 위 표는 single modeling과 joint modeling의 성능을 보여준다.

  - 결과, multi-task joint model이 상당히 좋은 성능이 보였으며, cascaded joint model은 single-task model과 비슷했다.

    - 이는 보조적인 정보(questionnaire item-level score)가 recognition 성능이 좋은 영향을 준 것을 의미함

  - 또한, multi-task joint model은 input modal type을 늘렸을 때 효과적이었다.
