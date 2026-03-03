## :page_with_curl: SuperBPE: Space Travel for Language Models

### Info

* `publication`: COLM 2025

* `author`: Liu and Hayase et al.
* `url`: https://openreview.net/forum?id=lcDRvffeNP#discussion
* `cite`: https://superbpe.github.io/



### Summary
이 논문은 현재의 tokenizer(BPE) training 단계에서 whitespace기반의 pre-tokenization의 한계(ex. 관용구 등 multi-word가 하나의 뜻이 되는 case가 있음)를 지적하고,   
whitespace기반의 pre-tokenization을 BPE training의 중간 단계부터 skip하는 것으로 superword를 고려할 수 있게 만드는 방법을 제안한다. --> SuperBPE

실험을 통해 SuperBPE가 더 좋은 downstream task performance를 보이며, encoding efficiency가 뛰어남을 증명했다.


### Motivations

- 기존의 BPE training에서의 whitespace 기반의 split을 하는 pre-tokenization은 관용구를 담아내기 힘들고, 공백이 없는 언어 등에서 한계가 있었다.

  - ex.) By the way를 By | the | way 로 끊어서 보게됨



### Contributions

- 새로운 tokenize 방식인 SuperBPE 제시



### Core Idea (novelty)

- 기존 BPE training의 pre-tokenization 과정에 있는 whitespace 기반 split을 BPE training 중간 단계부터는 skip시킴

  - 처음에는 subword를 고려하여 기본 base를 만든 후, 중간부터 superword 고려를 위해 whitespace split 단계를 skip

> 기존 BPE는 공백 기준으로 split하고 해당 chunk 내에서의 이웃들에 대해서만 frequency count를 했다면, 공백 기준 split을 skip하게 되면 문장 전체에 대해 이웃들에 대한 frequency count를 진행하게 됨

<div align="center"><img src="https://github.com/user-attachments/assets/9e9547b6-ac3a-454e-a3cd-02ac01378391" width="50%"></img></div>


### Methodology

target vocab size `T`에 도달할 때까지 tokenizer 학습을 한다고 하자.

- Stage 1: 기존 BPE training과 똑같은 과정을 미리 정해놓은 특정 vocab size `t` 까지 진행한다. (transition point라 부름 t < T)

- Stage 2: target size인 `T`까지 BPE training를 계속하되 whitespace pre-tokenization을 skip한다.

  - whitespace를 이어주는 token pair들이 고려될 것

- `t=0` 이면 BPE w/o pretok 과 같고, `t=T`이면 일반 BPE와 같음.

<div align="center"><img src="https://github.com/user-attachments/assets/ecfa0eea-49fa-436c-a6ac-e3c57f191f34" width="50%"></img></div>


### Results

- 30개의 downstream task의 거의 모든 것에서 SuperBPE가 BPE보다 성능이 좋았다.

  - model parameter, vocab size, training FLOPs를 고정하여 모델을 만들어 비교하였을 때 SuperBPE가 좋았음.

  - SuperBPE의 vocab size를 다르게 하였을 때는, 전부 BPE보다 좋았고 SuperBPE 180k 가 가장 좋았음.

- vocab size를 올릴수록, Bytes per token (BPT) 기준으로 BPE는 일찍 수렴하였지만, SuperBPE는 효율이 좋아졌음(BPT 증가).

  - SuperBPE의 BPT가 vocab size에 따라 어떻게 변하는지 관찰하였을 때, `80k` 이후부터는 떨어지는 것을 보았음 (80k가 best efficiency).

  - 모델 성능 측면에서는 encoding effiency를 양보했을 때, performance gain이 좀 더 있음. (180k가 좋았으므로; 80k에서는 BPE 대비 3% 향상됨)

<div align="center"><img src="https://github.com/user-attachments/assets/677bdd24-213d-4011-b6c8-b8a6cd08e038" width="25%"></img></div>

- loss distribution

  - Per-Token Bits-per-Byte를 계산함: $\frac{-log_2p(t_i | context)}{n_{bytes} / n_{tokens}}$ ; 토큰 하나를 예측하는 데, 바이트 1개당 평균 몇 비트가 필요한지

  - SuperBPE는 너무 쉽거나 어려운 토큰이 줄었음.
