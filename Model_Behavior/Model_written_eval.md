## Discovering Language Model Behaviors with Model-Written Evaluations

### Info

* `publication`: ACL 2023 FINDINGS
* `author`: Perez et al
* `url`: https://aclanthology.org/2023.findings-acl.847/

### Review

* Motivation: LM이 광범위하게 적용되고 있지만, LM의 behaviors와 그 risk에 대해서는 많이 탐구되지 않았음. LM failures의 findings에 대해 진행되는 속도에 근거하면, 더 많은 failure들이 있을 것이며, 이러한 LM behavior에 대해 평가하는 것은 실생활에 LM을 적용하는 사례가 늘어나는 만큼 중요하다.  --> 현재의 evaluation datasets는 test하는 behavior에 대한 다양성이 부족하며, human effort가 많이 듦.
  
* Problem: LM의 다양한 behavior 평가를 위한 벤치마크 만들기 (최소한의 human effort로)
  
* Method: LM을 이용해 evaluation benchmark 생성 (Model-Written Evaluations)

  - 1) LM에게 output class y(A or B; multiple-choice)가 주어졌을 때, input이 될 x를 여러 개 생성하도록 함. 2) 생성에 사용된 모델과 다른 모델을 label-correctness를 평가하기 위해 discriminator로써 사용 (다른 requirements들을 만족하는 지에도 사용)

- Evaluation: Sycophancy에서 RLHF 모델이 large model로 갈 수록 sycophancy behavior가 높아졌으며, preference model도 sycophancy behavior에 더 높은 점수를 주었음.
