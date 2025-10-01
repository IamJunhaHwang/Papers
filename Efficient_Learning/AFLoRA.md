## :page_facing_up: AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models

### Info 

* `publication`: ACL 2024 SHORT
* `author`: Zeyu Liu, Souvik Kundu et al.

* `url`: https://aclanthology.org/2024.acl-short.16.pdf
* `github`: https://github.com/zeyuliu1037/AFLoRA/tree/main

<div align="center"><img src="https://github.com/user-attachments/assets/2e921fcb-bea1-412e-a0fc-ffa21607f36e" width="60%"></img></div>

### 1. Abstract & Introduction

BERT, GPT-3나 LLaMA2 와 같은 LM은 대체로 뛰어난 성능을 보였지만, downstream task의 zero-shot performance가 종종 성능이 떨어지는 것을 보였다. 이는 full fine-tuning으로 해결할 수 있지만, 상당한 비용이 든다(모델 사이즈가 크므로).

이러한 문제를 해결하기 위해 다양한 parameter-efficient fine-tuning(PEFT) 방법이 제시되었다(LoRA, Adapter, prompt tuning).   
특히, LoRA는 trainable low-rank path를 모델에 추가하는 것이고, ELoRA는 LoRA의 확장으로 layer마다 projection matrix를 추가하는 것이 아닌 하나의 공유 projection matrix를 두고 trainable feature transformation vectors만을 학습하는 것이다.   
이처럼 ELoRA는 random 초기화된 projection matrix를 freeze하고 2개의 feature transformation vectors를 학습하는 것으로 SOTA accuracy를 달성했지만, frozen projection matrices에 높은 rank를 필요로해서 cost가 높다.

본 논문에서는, trainable parameter count와 coputation costs를 모두 줄이는 Adaptive Freezeing of Low Rank Adaptation (AFLoRA)를 제안한다.   

- Contribution

  - ELoRA에서 frozen LoRA path를 위해 필요한 rank를 조사하고, 이러한 rank가 줄어들 때 fine-tuning 성능이 나빠지는 것을 관찰한다.

  - 위의 insight를 기반으로, AFLoRA는 ELoRA처럼 projectionm matrices와 feature transformation vectors로 몇 epoch 학습을 시작한 후 점진적으로 projection matrices를 freeze한다.

    - 이 때, `freezing score` 를 기반으로 freeze를 진행하며 이는 LoRA tensor의 trainability requirement에 대한 proxy로 동작

  - AFLoRA는 ELoRA에 비해 1.86배의 runtime, 2.96배의 FLOPs 정도 좋아졌고 LoRA에 비해 9.5배 적은 trainable parameter로 비슷하거나 더 좋은 성능을 내었다.

<br></br>

### 2. Motivational Case Study

<div align="center"><img src="https://github.com/user-attachments/assets/83631355-6a00-455d-93d1-da9d724b527c" width="40%"></img></div>

위 그림은 SST-2, MRPC에서 ELoRA의 1024와 4의 rank를 사용해 fine-tuning을 진행한 결과이다. 결과를 보면, frozen tensor에는 high rank를 필요로 하며, rank가 낮으면(4) 성능이 떨어지는 것을 볼 수 있다.

<br></br>

### 3. AFLoRA: Methodology

#### 3-1. Module Structure

- 4개의 components로 구성된 LoRA module을 디자인했다

  - down-projection linear layer $lora_A$
  - up-projection linear layer $lora_B$
  - two feature transform vectors $s_d$ & $s_b$
    - $lora_B$ 의 전, 후에 위치

ELoRA와 다르게 projection metrices ($lora_A$ and $lora_B$) 와 vectors를 처음에는 trainable하게 하였으며 매우 낮은 rank로 설정했다.

- 위 모듈은 아래와 같이 layer l에서 given input X에 대해 output Y를 만든다.

  - $Y = W^l_0X + \Lambda^l_bB^l\Lambda^l_dA^lX$

    - $A^l$ , $B^l$ : trainable LoRA tensors ($lora^l_A$ & $lora^l_B$); (Kaiming Uniform initialization)
    - $\Lambda_d$ , $\Lambda_b$ : vectors of $s_d$ & $s_b$; (ELoRA와 똑같이 initialization)
    - $W^l_0$ : frozen pre-trained weights

#### 3-2. Adaptive Freezing

pruning에서는 중요하지 않은 weight를 찾기 위해 가중치들의 크기와 gradient 모두를 고려하는 것이 필요했지만, 여기에서는 gradient만을 사용해 `freezing score` 를 계산한다.   
왜냐하면, 크기가 큰 가중치에 적은 변화를 가지는 것과 작은 크기의 weights는 똑같은 proiority로 freeze되어야하기 때문이다(학습이 거의 이루어지지 않는 가중치이므로).

freezing score는 training process동안 어떤 가중치가 어느 정도로 바뀌는지 정량화하는 점수이다. 결과적으로, 가중치의 예상 변화량이 매우 작다면 해당 가중치를 freeze한다. 아래의 식은 low-rank tensor $A^l$ 의 freezing score 식을 나타낸다.

<div align="center"><img src="https://github.com/user-attachments/assets/c08e85c5-3de4-433e-9c1b-061991559dae" width="50%"></img></div>

식에서, iteration `t` 에서 각 projection tensor에 대해 smoothed gradient $\overline{I}^{(t)}_{A^l}$ 과 uncertainty tensor $\overline{U}^{(t)}_{A_l}$ 를 계산하고, 이 둘에 대해 아다마르 곱을 거쳐 freezing score $s^{(t)}_{A^l}$ 를 평가한다.

smoothed gradient: 현재 스탭 t와 이전 스탭들 t-1 ...의 gradient 크기의 합 : 변화의 크기

uncertainty : 현재 스텝 t와 이전 스탭들 t-1 ...의 grdainet 크기의 차 : 변동성(값이 얼마나 크게 출렁이는)

LoRA freezing score의 thresholding을 적용하기 위해 cubic schedule(3차 함수 이용)을 이용한다.


<div align="center"><img src="https://github.com/user-attachments/assets/8a481fab-ddf7-4740-9b31-739cd989c148" width="50%"></img></div>



initial $t_i$ training steps에 대해 projection matrices를 trainable하게 두고, 위의 freezing fraction `r(t)` 를 계산해 점진적으로 freeze한다.   
마지막으로, 모든 projection matrices는 $T - t_f$ steps를 넘으면 전부 freeze된다.

- step t에서 계산된 freezing fraction k에 대해 가장 낮은 layer부터의 k% projection matrices를 freeze한다.

  - t : current # step
  - T : total numbeer of fine-tuning steps
  - $t_i$ : steps corresponding to 1 epoch
  - $t_f$ : steps corresponding to 70% of total training steps

<div align="center"><img src="https://github.com/user-attachments/assets/c29e6eef-e2c2-4c05-b13c-e213cab3fc56" width="50%"></img></div>


<br></br>

### 4. Experiments

- Model: DeBERTaV3-base

- Dataset : GLUE Benchmark

- baselines: LoRA, ELoRA, AdaLoRA, SoRA, FFT

  - ELoRA는 reproduce했고 나머지는 가져온 결과

<div align="center"><img src="https://github.com/user-attachments/assets/d770f41c-1f01-460c-81da-0d87d8b3abb6" width="80%"></img></div>

실험 결과 AFLoRA가 SOTA 성능을 대부분의 데이터 셋에서 달성하였으며, ELoRA와 비슷하고 LoRA와 0.5배 적은 trainable parameter를 필요로 했다.


<div align="center"><img src="https://github.com/user-attachments/assets/d69cd440-0e87-425d-a7b3-da706f71d7fe" width="40%"></img></div>

- Runtime & FLOPs Comparison

  - ELoRA와 비교해서 제안한 방법이 1.86배 runtime, 2.96배 FLOPs 이득이 있었다.
  - LoRA와 비교해서는 9.5배 parameter reduction을 달성했다.

  - 이는 AFLoRA가 PEFT 방법으로써 ELoRA와 비슷한 파라미터 효율을 가지면서도 추가적인 training overhead가 없는 것을 증명한다.

<div align="center"><img src="https://github.com/user-attachments/assets/08c030a5-7058-4564-9b3e-bad214714f9a" width="50%"></img></div>

- Results with Large Language Models (LLMs)

  - LLaMA-7B 와 BART-Large에서 GSM8K, CNN/Daily mail summarizing task에서 LoRA와 성능을 비교했다.

  - 결과 AFLoRA가 3.15배 적은 파라미터로 1.09배의 성능 증가를 달성했다(CNN/DailyMail에서는 1.69배 적은 파라미터).

<br></br>

### 5. Ablations and Discussions

GLUE benchmark에서 QQP와 MNLI를 뺀 6개의 데이터셋에 대해 ablation studies 진행

<div align="center"><img src="https://github.com/user-attachments/assets/799d2ec9-162c-4034-aa5d-ae6abc36bb8b" width="40%"></img></div>

- Do we really need adaptive freezing? 

  - all LoRA PMs **frozen**, all LoRA PMs **trainable**, adaptive training of LoRA PMs(AFLoRA) 비교(모든 방법에서 r=4)

  - Table 4에 나와있듯이, LoRA를 frozen하는 것보다 trainable하게 하는 것이 평균 성능이 더 높다. 그리고 AFLoRA같이 adaptive freezing은 좀 더 좋은 성능을 보였다.

- Do we need to keep the PMs trainable for all layer types? 

  - PMs를 FFN과 attention layer 중 어디에 붙어 있는 것을 trainable하게 두어야되는지 실험

  - Table 5를 보면, FFN쪽을 trainable하게 두는 것이 성능에 더 좋았음.

    - FFN과 Attention 모두에 대해 initially train하고 adaptively freeze하는(AFLoRA) 것보다도 FFN만 trainable하게 두는 것이 성능이 더 좋았음.

    - FFN이 fine-tuning에 상당히 중요한 역할을 하는 것을 의미

- Ablation with sensitivity choices

  - Figure 4는 3가지 sensitivity score에 대한 ablation이다.
  
  - 결과, AFLoRA에 적용한 |grad(p)| 가 제일 성능이 좋았다.

<div align="center"><img src="https://github.com/user-attachments/assets/431f8868-254b-4cfa-922e-9c4c459d77da" width="40%"></img></div>


- Discussion on Freezing Trend

  - RTE를 사용해 case study를 진행했고, 다른 layer들간의 PMs의 freezing trend를 보았다.

  - Figure 5는 각 component가 freezing 되기 전까지 필요한 iteration 갯수를 나타낸 것이다.

  - intermediate linear layer의 down-projection matrix가 freeze되는데 더 오랜 training 시간이 걸리는 것을 볼 수 있다. --> intermediate layer의 approximation 능력이 second layer보다 떨어지는 것을 의미.
