Model of Multi-turn Dialogue in Emotional Chatbot
==========================================================

- 저자: Chien-Hao Kao, Chih-Chich Chen, Yu-Tza Tsai
- https://gitlab.com/bytecell/dailab/-/issues/60#note_1233499213

Introduction
---------------

- 이전 연구에서는 챗봇에 Seq2Seq 생성 모델을 적용해왔음.
- 텍스트의 sentiment 인식을 위한 연구에서는 감정 범주에 따른 입력 문장을 분류했음.
- 사용자와의 일상 대화를 통해 감정이 변하고 변화된 감정으로 사용자에게 응답하는 챗봇이 목표임

The Model
--------------

![image](https://user-images.githubusercontent.com/49019184/212580335-61800e56-6819-4d9d-8fc0-3fbf7cd81a23.png)

- 본 연구에서 설계된 모델

![image](https://user-images.githubusercontent.com/49019184/212580343-306e2a49-8a18-4c72-8076-4212ae2c9164.png)

- Seq2Seq 구조를 사용한 생성 모델
- 멀티턴 대화와 감정 태그의 입력을 수용하기 위해 latent encoder와 emotion encoder를 추가하였음

![image](https://user-images.githubusercontent.com/49019184/212580351-e4a80f72-280e-4e48-8ffb-1de53c3a30fe.png)

- RNN 모델의 latent encoder
- Classifier 출력은 Decoder와 Auto Emotion Transfer의 출력을 조절함

![image](https://user-images.githubusercontent.com/49019184/212580359-9419158a-59f9-4087-a3b6-af9c6475892b.png)

- Auto Emotion Transfer, GRU(Gated Recurrent Units)구조 사용
- 사용자의 입력 문장과 관련된 감정 예측에 사용되는 Emotion Classifier와 챗봇이 감정에 응답하도록 하는 Emotion Classifier가 있음

![image](https://user-images.githubusercontent.com/49019184/212580365-506256ba-9a89-4272-b830-ba33fe48244f.png)

- generator 사전 학습, 감정 상태가 각 턴 사이에 전달되므로 classifier와 decoder는 함께 학습해야 함
- discriminator 학습, SeqGAN과 다르게 생성된 문장을 먼저 재구성하고 학습을 진행함
- generator 학습, 현실적인 응답을 위해 GAN을 사용

Experiments
-------------------

- Dataset
  - ![image](https://user-images.githubusercontent.com/49019184/212580373-cccc1b35-9ebb-4b0d-8d55-c837e54ede3e.png)
  - EmotionLines 데이터셋 사용
  - EmotionPush 두 사람의 멀티턴 대화, Friends 여러 명의 멀티턴 대화
- Response Judgment
  - ![image](https://user-images.githubusercontent.com/49019184/212580378-56dfc00a-ac17-4a53-8fa0-1772ab1ed532.png)
  - 응답 정확도가 높은 것을 볼 수 있음
- Emotion Prediction and Transfer
  - ![image](https://user-images.githubusercontent.com/49019184/212580386-6bff5ce7-b079-46bd-9574-8a0ab3c39c3e.png)
  - 감정 별 예측, transfer 정확도를 볼 수 있으며 대체로 예측 정확도가 transfer보다 높게 나왔음
  - 적은 태그로 실제 사람의 감정을 해석할 수 없음
  - ![image](https://user-images.githubusercontent.com/49019184/212580400-7bc1fce7-ff67-4215-b651-2728114a76e1.png)
  - 두 학습 데이터셋의 perplexity(당혹감, 좋거나 나쁜 문장을 생성하는 generator에 대한 평가 기준)
  - Friends의 점수가 더 낮았음
  - ![image](https://user-images.githubusercontent.com/49019184/212580404-7590cbb6-297f-4899-92f6-7695a9146cba.png)
  - 정답과 생성된 문장 비교
  - 멀티턴 대화에서 응답해야하는 타이밍을 잘 잡았고 응답 판단이 잘 학습되는 것을 보여줌



Conclusion and Future Work
-------------------------------------

- 감정 인식이 좋지 않은 이유는 데이터의 불균형과 TV시리즈로 만들어지는 데이터셋은 강한 감정 기복이 표현되어 있기 때문임
- 감정을 정량화하는 태그를 추가하여 해결할 수 있음(자연스러운 감정 변화)
- 대화 표준 답변이 없다는 가정, generator로 Seq2Seq를 사용
- 다양한 요인에 따라 응답이 다양하게 바뀌도록 하였음
