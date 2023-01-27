$A^2$-Nets: Double Attention Networks
================================================

- 저자: Yunpeng Chen, Yannis Kalantidis, Jianshu Li, Shuicheng Yan, Jiashi Feng
- https://gitlab.com/bytecell/dailab/-/issues/60#note_1245476868

Introduction
----------------

- CNN은 이미지와 영상 처리에 사용되어 왔지만 local feature와 relation을 잡아내는데 전념하는 convolution 연산자에 의해 제한되며 long-range interdependency 모델링에 비효율적임
- 컨볼루션 계층의 인접 계층에서 전체 공간의 feature들을 감지할 수 있는 새로운 네트워크 요소를 도입함(double-attention block)
- 기존 연구와 비교했을 때, attention 연산은 풀링된 feature의 second-order statistic를 계산하고 SENet에서 사용되는 global average pooling으로 포착할 수 없는 복잡한 모양과 동작의 상관관계를 포착할 수 있음
- 주요 contribution
  - long-range feature interdependency를 포착하기 위한 일반적인 공식
  - long-range feature을 모으고 퍼뜨리기 위한 효율적인 구조, double attention block
  - $A^2$-Net의 이미지 인식과 영상 동작 인식 task에서의 우수한 성능을 입증

Method
------------

- $X \in \mathbb{R}^{c\times d\times h\times w}$
- 3차원 convolution 계층의 입력 tensor, $c$는 채널 수, $d$는 시간적 차원, $h, w$는 공간적 차원, $v_i$는 local feature, $i=1,...,dhw$는 입력 location
- $z_i = F_{\text{distr}}(G_{\text{gather}}(X), v_i)$
- 전체 공간에서 feature를 모으고 해당 loaction의 local feature $v_i$를 고려하여 입력 location $i$로 퍼뜨리는 연산자의 출력

![image](https://user-images.githubusercontent.com/49019184/215027668-6dbf165f-49a9-47a2-8505-17232cd1b120.png)

- The First Attention Step: Feature Gathering
  - feature의 second-order statistic과 global representation을 포착하기 위해 bilinear pooling 사용
  - $G_{\text{bilinear}}(A, B) = AB^\top = \sum_{\forall j} \overline{b}_{ij}a_j$
  - 위 공식에 $B = [\overline{b}_1; ...;\overline{b}_n]$을 대입시켜 $G = [g_1,...,g_n] \in \mathbb{R}^{m\times n}$을 구하는 식은 아래와 같음
  - $g_i = A\overline b^\top_i = \sum_{\forall j} \overline{b}_{ij}a_j$
  - 이로 인해 $g_i$는 $b_i$에 의해 가중치가 생긴 local feature를 수집하여 계산되고, 유효한 attention weighting 벡터를 보장하기 위해 $B$에 softmax를 적용함
  - $g_i = A \text{softmax}(\overline{b}_i)^\top$
  - ![image](https://user-images.githubusercontent.com/49019184/215027684-642c0d2d-5954-4f0d-ae2d-7e3b8d96ea40.png)

- The Second Attention Step: Feature Distribution
  - 각 location에서 feature $v_i$의 필요를 기반으로 visual primitive의 adaptive bag을 분산하여, 학습을 더 쉽게 하고 더 복잡한 관계를 포착할 수 있도록 도움
  - $G_{\text{gather}}(X)$에서 feature 벡터의 부분집합을 선택하여 구현함
  - $z_i = \sum_{\forall j} v_{ij}g_j = G_{\text{gather}}(X)v_i, \text{where} \sum_{\forall j} v_{ij} = 1$
  - ![image](https://user-images.githubusercontent.com/49019184/215027694-cb2462b4-3247-4960-a4f1-fba94e1f1379.png)

- The Double Attention Block
  - 위의 두 attention 단계를 결합하여 block을 만듦
  - ![image](https://user-images.githubusercontent.com/49019184/215027708-102cacd0-21a6-45bf-a84d-3ab50a7ec2b6.png)
  - 최종 공식 유도(left association) $Z = F_{\text{distr}}(G_{\text{gather}}(X), V) = G_{\text{gather}}(X)\text{softmax}(\rho(X\;W_\rho)) = [\phi(X\;W_\phi)\text{softmax}(\theta(X\;W_\theta))^\top]\text{softmax}(\rho(X\;W_\rho))$
  - 출력 $Z$는 필요한 reshape과 전치 연산을 거치고 두 행렬의 행렬곱으로 계산됨
  - 출력 $Z$는 채널수 확장을 위해 convolution 계층을 한번 더 거치고 이로 인해 element-wise 추가를 통해 입력 $X$로 인코딩될 수 있음
  - right association 공식 $Z = \phi(X;W_\phi)[\text{softmax}(\theta(X;W_\theta))^\top\text{softmax}(\rho(X;W_\rho))]$
  - left와 right association 식은 수학적으로 동일하고 출력도 동일한 것이지만 연산 비용과 메모리 소비량이 다름
  - $(dhw)^2 > nm$이면 left가 효율적이고 아니면 right가 효율적임

- Discussion
  - pair-wise 관계 포착을 위해 Embedded Gaussian formulation을 사용하는 것보다 softmax$(\theta(X))^\top$softmax$(\rho(X))$를 사용하는 것을 제안함
  - 현재 영상 인식에서 SOTA인 NL-Net과 본 논문에서 제안한 방법을 비교함
  - 제안한 방법이 더 높은 효율과 정확도를 보여주었음

Experiments
--------------------

- Implementation Details
  - Backbone CNN
    - ![image](https://user-images.githubusercontent.com/49019184/215027726-3cbf7577-e165-475f-a1b8-033482881bdd.png)
    - baseline method로 ResNet-29를 사용하고 모든 절제 연구에 ResNet-26을 사용함
    - 연산 비용은 FLOPs로, 모델 복잡도는 #Params로 계산
    - ResNet-50은 더 깊고 넓기 때문에 비교할 때 마지막 몇 개의 실험을 이용함

  - Training and Testing Settings
    - 이미지 분류에 MXNet 사용, 표준 단일 모델 single 224X224 center crop 검증 정확도 보고
    - 영상 분류에 PyTorch 사용, single clip 정확도와 video 정확도 둘 다 보고

- Ablation Studies
  - 실험 당 batch size 521로, 32개의 GPU 사용
  - 112X112 해상도의 16 프레임을 입력으로 받음
  - 학습률은 기본 0.2에 20k번째, 30k번째 학습에서 0.1배로 감소하고 37k번째 학습에서 학습이 종료됨
  - 3개의 convolution 계층($\theta(),\phi(),\rho()$) 출력 채널 수를 입력 채널 수의 1/4로 설정함
  - Single Block
    - ![image](https://user-images.githubusercontent.com/49019184/215027735-5266c912-0fcf-4a07-9b23-6c8f5950241e.png)
    - Backbone Network에 한 블록만 추가 되었을 때 결과, 아래 세 줄을 통해 제안한 double attention이 성능을 향상시킴
    
  - Multiple Blocks
    - ![image](https://user-images.githubusercontent.com/49019184/215027748-536edfac-793f-40cd-9ec3-1429001097d4.png)
    - Backbone Network에 여러 블록이 추가 되었을 때 결과, 제안한 double attention이 연산 비용도 적고 성능도 올랐음

- Experiments on Image Recognition
  - 이미지 분류에 ImageNet-1k 데이터셋을 이용하여 평가함
  - batch size 2048에 64개의 GPU를 사용함
  - 학습률은 $\sqrt{0.1}$에서 학습 정확도가 포화되면 0.1배로 감소함
  - ![image](https://user-images.githubusercontent.com/49019184/215027766-abe8e3f6-f481-4380-ad84-0b03b46246d9.png)
  - 훨씬 큰 ResNet-152보다 성능이 좋게 나왔음

- Experiment Results on Video Recognition
  - Learning Motion from Scratch on Kinetics
    - ImageNet으로 사전학습된 ResNet-50를 사용하고 무작위로 초기화된 5개의 $A^2$-block을 추가하여 3D convolutional network를 구축함
    - 입력으로 8프레임을 사용하고 32k번 batch size 512에 64개의 GPU로 32k번 학습함
    - 초기 학습률은 0.04고 학습이 안될 때 더 감소시킴
    - ![image](https://user-images.githubusercontent.com/49019184/215027787-8a621343-810a-40b1-b4f9-cd5f21698f31.png)
    - 제안한 모델이 성능이 좋게 나왔음

  - Transfer the Learned Feature to UCF-101
    - Kinetics보다 몇 배는 더 작은 UCF-101 사용
    - 학습률 0.01에서 세번마다 0.1배 줄어듬, batch size 104clip에 8개의 GPU 사용
    - ![image](https://user-images.githubusercontent.com/49019184/215027800-d75dd7d6-c3ad-4525-bb19-4aee361e4879.png)
    - 제안한 모델이 더 적은 연산 비용을 요구하고 효과적임

Conclusions
-----------------

- CNN에서 local convolution 연산의 한계를 극복하기 위해 double attention 메커니즘 제안
- 제안한 메커니즘은 global information을 더 효율적으로 포착하고 2단계 attention 방식으로 각 location에 분산함
- 공식을 경량화하여 연산 overhead가 적게 들어가도록 하였음
- 제안한 method가 2D 이미지 인식과 3D 영상 인식에서 모두 효과가 있었음을 확인했음
