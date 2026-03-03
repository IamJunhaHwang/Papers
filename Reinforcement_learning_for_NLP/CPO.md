## :page_with_curl: Sequence-level Large Language Model Training with Contrastive Preference Optimization

### Info

* `publication`: NAACL 2025 - Findings

* `author`: Feng et al.
* `url`: https://aclanthology.org/2025.findings-naacl.233.pdf



### Summary
이 논문은 decoder-only LLM의 training 과정에서 sequence-level signal에 대한 understanding이 부족하다는 점을 언급하며, LLM training 과정에 sequence-level signal을 주입하는 방법인 Contrastive Preference Optimization (CPO)을 제시한다.

CPO는 negative sample을 여러 synthetic data 생성 방식을 이용해 구성하고, 여기에 ground truth와 가장 유사한 정도를 ranking information으로 사용하여 효과적으로 generation quality를 올렸다.


### Motivations

- Decoder-based LLM의 training objective인 Next Token Preduction은 Sequence-level signal을 고려할 수 없는 한계가 있음. (only token-level signal만)

- 이에 따라, RLHF와 같은 학습 방법이 제시되었지만, 이는 많은 계산량을 필요로 함.

**--> Can we introduce sequence-level information in LLM pre-training / SFT with a small computational cost?**



### Contributions

- human preference data가 필요없이 Sequence-level signal을 LLM training 과정에 주입하는 CPO 제안



### Core Idea (novelty) & Methodology

- 기존 DPO에서 human negative sample을 synthetic negative sample로 교체 (cost 낮추기)

- synthetic negative sample은 다음 4가지 방식으로 생성

  - Autoregressive negatives (AN): prefix에 대해 LM이 autoregressive하게 response 생성
  
  - Batch Negatives (BN): prefixes & confinuations batch $\{ x_i, y_i \}^b_{i=1}$가 주어지고, prefix $x_i$ 에 대한 negative sample은 $\{ y_j \}_{k \neq i}$ 로 구성
  - Meanfield Negatives (MN): 주어진 시퀀스 y에서 c%의 위치를 랜덤하게 선택한 후, 각 위치를 해당 위치 전까지의 token을 input으로 하는 LM의 Next Token Preduction으로 채워넣음
  - Truncation Negatives (TN): 각 ground truth continuation에서 랜덤한 위치에서 자른 후 EOS 토큰 삽입

이후 similarity를 기반으로 negative sample들에 대해 ground truth와 유사도가 높을수록 낮은 ranking을 가지게하여 이 정보를 아래의 CPO loss에 포함시켰음. (ground truth와 유사도가 높을수록 negative sample일지라도 그 안에서 선호도가 나뉘도록)

$$
\mathcal{L}_{\mathrm{CPO}}(\pi_\theta, \pi_{\mathrm{ref}}) = \mathbb{E}_{\tau,\,(x,y_1)\sim\mathcal{D},\, y_2,\ldots,y_K\sim\mathcal{A}} \left[
\log
\prod_{k=1}^{K}
\frac{
\exp\!\left(
\beta \log \frac{\pi_\theta\!\left(y_{\tau(k)} \mid x\right)}
{\pi_{\mathrm{ref}}\!\left(y_{\tau(k)} \mid x\right)}
\right)
}{
\sum_{j=k}^{K}
\exp\!\left(
\beta \log \frac{\pi_\theta\!\left(y_{\tau(j)} \mid x\right)}
{\pi_{\mathrm{ref}}\!\left(y_{\tau(j)} \mid x\right)}
\right)
}
\right]$$




### Results

- pre-trained GPT2-XL과 OpenLlama-3B 모델에서, baseline을 능가하였음. (GPT에게 선호도 평가하여 win rate 측정했음)

- 다양한 generation config를 실험해보았는데, CPO는 일관되게 좋은 성능을 보임

- negative sample들 간의 비교에서는, MNR보다 BNR, TNR이 약간 더 성능이 좋았음.

- DPO나 PSS보다 CPO가 성능이 더 좋았음.

  - DPO는 harmful 표현처럼 명확한 오답이 있는 경우에는 강하지만, 일반적인 생성에서는 한계가 있다.
