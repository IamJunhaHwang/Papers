Multiturn dialogue generation by modeling sentence-level and discourse-level contexts
=======================================================================================   
- 저자: Yang Yang, Juan Cao, Yujun Wen & Pengzhou Zhang
- https://gitlab.com/bytecell/antlab_egg/-/issues/39#note_1199013821

Introduction
---------------------------------------   

- dialogue system 중에서 task-oriented dialogue system은 쉽게 구현되고 이미 실생활에 적용되어 있다. non-task-oriented dialogue system(open-domain dialogue system)은 pretrained model의 발전과 대규모의 dialogue dataset을 사용하여 발전하고 있지만 아직 일관성 있는 텍스트를 생성하는 곳은 발전이 필요하다.
- dialogue context를 위해 context difference-awareness module을 사용했다.
- 문장 간의 관계를 활용하기 위해 bidirectional tranformer encoder를 통해 얻은 이전 multiturn context간의 차이 정보를 차이-인지 모듈에 입력했다.
- response generation은 차이 정보와 이전의 decoded token을 사용했다.
- sentence order prediction은 dialogue context의 구조를 찾기 위해 encoder에 설계되어 있다.   
   

Contribution
-----------------   

- difference-awareness module(차이-인지 모듈): 각 문장이 전체 대화에 미치는 영향을 동적으로 찾아내고 target 문장 생성을 위해 multiturn contextual representation을 사용하여 구성된다.
- sentece order prediction(문장 순서 예측, SOP): 대화 문맥의 순위를 학습 가능한 순위 함수가 있는 top-one probability distribution으로 전환한다.   


Methodology
----------------   

- Model definition
  - $u^T = \lbrace x^T, y^T \rbrace$
    - $T$번째 턴의 대화
  - $x^T = \lbrace x_1^T, ..., x_{\vert x^T \vert}^T \rbrace$
    - 사용자의 입력
  - $y^T = \lbrace y_1^T, ..., y_{\vert y^T \vert}^T \rbrace$
    - 입력에 대한 response
  - $c^T = [x^{T-1}; y^{T-1}; x^T]$
    - dialogue context

- Model architecture   
![image](https://user-images.githubusercontent.com/49019184/209496773-c766fd7c-b2dc-44ef-9298-9cf718058518.png) 
  - 4가지 모듈로 구성되어 있음 = bidirectional transformer encoder, unidirectional transformer decoder, difference-aware and response generation
  - bidirectional transformer encoder
    - BART의 인코더 사용
    - $T$번째 턴의 context $c^T$를 encode함
    - context representation인 $C^T = \lbrace C_1^T, C_2^T, C_3^T \rbrace$를 얻음

  - Input representation
    - ![image](https://user-images.githubusercontent.com/49019184/209496786-adad0754-9774-4d93-8f61-7f9124f7def4.png)
    - 각 문장의 앞뒤에 [CLS]와 [SEP]를 추가함
    - BERT의 WordPiece로 tokenize함
    - Role embedding은 화자를 구분하기 위해 사용함
    - Positional embedding은 각 utterance 내의 위치를 알 수 있게함

  - Difference-aware module
    - 입력인 $T$턴과 그 전의 문맥 표현으로 $T+1$번째 문맥 표현을 예측하고 예측한 결과와 이전 $K$턴의 문맥 표현의 차이를 계산함.
      - $\overline C^{T+1} = transformer(C^1, C^2, ... C^T)$
        - $T+1$번째 문맥 표현 예측
      - $\overline d^T = \displaystyle\sum_{k=0}^K \lambda_k MLP([\overline C^{T+1}; C^{T-K}; \overline C^{T+1} - C^{T-k}])$
        - 예측한 결과와 이전 $K$턴의 문맥 표현의 차이
      - $\lambda_i = \frac{exp(\overline C^{T+1}W_1C^{T-i})}{\displaystyle\sum_{k=0}^K exp(\overline C^{T+1}W_1C{^T-k})}$
        - $\lambda_i$는 가중치, $W_1$은 학습 가능한 파라미터, $K$는 하이퍼 파라미터
    - MLP는 fully connected layer이고 활성 함수는 $tanh$를 사용했음
    - 첫번째 턴에서는 현재 턴에서 얻을 수 있는 차이 정보가 없기 때문에 $d^1$은 영 벡터일 것임

  - Response generation module
    - linear layer와 softmax layer로 구성됨
    - 문맥 표현$C^T$, 차이-인지 모듈의 출력$\overline d^T$, 이전의 decoded token을 transformer decoder에 입력으로 주어 현재 턴의 출력 $H^T$를 얻음
    - reponse$y_i^T$를 생성할 때 $H^T$가 주어진 $i$번째 토큰의 확률 분포를 아래 식으로 계산함
      - $p(y_i^T| y_{\lt i}^T, u_N^T) = softmax(W_2H^T + b_1)$
        - $W_2$와 $b_1$은 학습 가능한 파라미터
    - $L_{gen} = - \displaystyle\sum_{m=1}^{n} \log_{}{p(y_i^T| y_{\lt i}^T, u_N^T)}$ 
      - 최종 response generation식   



Training tasks
------------------------------------------   

- SOP
  - sentence reordering algorithm을 구현하여 무작위로 잘려 있는 대화 문맥 순서를 재구성하는 것
  - 일관성 있는 문맥 순서를 얻기 위해 learning-to-rank(L2R) 알고리즘을 사용했음
  - ![image](https://user-images.githubusercontent.com/49019184/209496806-7314d935-bcef-4a11-9df3-a2b1cd15b389.png)
  - 학습 과정에서 잘려 있는 문맥 순서를 재구성 하기 위해 ListNet 알고리즘을 사용했음
  - $f(h^{(i)}) = \frac{1}{|m|} \displaystyle\sum_{j=1}^{|m|} R_i^TWR_j$
    - global attention 메커니즘을 적용하여 전체 대화에 대한 각 i번째 문장의 contribution $f(h^{(i)})$을 추정함
    - $R_i$는 $i$번째 문장 인코더의 hidden state를 의미함([CLS])
  - $p_{(u_i)} = \varphi(f(h^{(i)})) / \displaystyle\sum_{k=1}^{|m|} \varphi(f(h^{k)}))$
    - 각 문장의 top-one 확률은 현재 문맥에서 모든 문장이 top-one이 될 확률을 나타내는 softmax로 계산됨
  - $q(u_i) = \varphi (y_i) / \displaystyle\sum_{k=1}^{|m|} varphi(y_k)$
    - $\varphi(o)$는 지수 함수, $q(u_i)$는 정답 문장(golden target sentence)을 의미함
  - $y_i = \frac{|m| - s_i}{|m|}$
    - $y_i$는 각 문장 $u_i$에 대한 정답 점수(golden score), 문장의 순서는 0부터 시작함
  - $L_{lr} = - \displaystyle\sum_{i=1}^{m} q(u_i) \log_{}{p(u_i)}$
    - 정답 distribution과 학습 distribution의 차이를 cross-entropy를 이용하여 계산함
  

- Total training objective
  - $L = L_{gen} + \lambda_1L_{lr} + \lambda_2L_{sim}$
  - $L_{sim} = \sqrt{\displaystyle\sum_{k=1}^{n} |\overline C^K - C^K|}$
    - 모델의 전반적인 학습 목표
    - $\lambda_1$과 $\lambda_2$는 조정 가능한 파라미터, $L_{sim}$은 문맥 유사도 모델링
    - 차이-인지 모듈의 예측 문맥과 실제 문맥을 점점 더 가깝게 설정하여 예측 문맥과 실제 문맥 사이의 유클리드 거리를 최소화함
  - SOP와 문맥 유사도 사전 학습을 통해 인코더는 의미론적 정보와 대화 구조를 더 잘 찾아낼 수 있게 됨
  - 미세 조정 단계에서는 응답 생성을 위해서만 모델이 학습됨


Experiments
-----------------------------------------

- Experimental dataset
  - 일상 대화 시나리오의 다중 턴 대화 데이터셋인 DailyDialog의 대화 데이터셋을 사용함
  - English learners에 의해 작성되어 철저한 문법과 적은 노이즈가 있고 다양한 생활 주제에 초점이 맞춰져 있음
  - train:val:test = 8:1:1
- Baselines
  - Seq2Seq: transformer encoder-decoder 구조, 다운스트림 데이터셋으로 학습시킴
  - GPT-2: left-to-right transformer decoder 구조, 엄청 큰 데이터셋과 미세 조정 없이 적용할 수 있는 엄청 많은 파라미터로 학습시킴
  - BART: 표준 seq2seq에서 encoder-decoder가 bidirectional encoder와 left-to-right decoder인 구조, 텍스트 생성에 미세 조정하는 것에도 적합하고, 텍스트 이해에도 적합함
  - DialoGPT: 대규모 대화 데이터로 학습시킨 CLM(causal language modeling)을 사용한 open-domain 대화 시스템임
  - DialoFlow: pretrained GPT-2를 사용하여 구성되었음
  
- Evaluation metrics
  - 언어 모델에서 일반적으로 사용하는 BLUE(B-n), NIST(N-n), Entopy, lexical repetition(LR-n)을 사용함
  - Specificity, sensibleness, average of specificity and sensibleness(SSA)를 사람을 통해 평가함
  
- Experimental results
  - Automatic evaluation
    - 모든 모델이 12계층 transformer block과 디코딩을 위한 beam search를 사용했음
    - 대부분의 평가 메트릭에 대해 해당 모델이 가장 높은 성능을 기록함
    
|Model|B-2|B-4|N-2|N-4|Entropy|LR-2|   
|-----|---|---|---|---|-------|----|   
|Seq2Seq|17.68|5.45|1.44|1.59|7.61|26.0|   
|GPT-2|17.96|5.87|1.86|1.95|8.12|31.2|   
|BART|18.21|5.99|1.68|1.83|8.35|28.5|   
|DialoGPT|18.83|6.63|2.29|2.78|9.20|30.5|   
|DialoFlow|26.47|10.12|3.65|3.84|9.82|26.4|   
|Flow w/ aux|**27.35**|10.45|3.77|3.99|9.77|25.7|   
|Ours|27.03|**11.37**|**4.05**|**4.33**|**9.89**|25.3|   
|w/o SOP|27.18|11.05|3.93|4.02|9.82|**25.0**|   
|w/o CSM|26.36|10.40|2.54|3.80|9.35|25.8|   

  - Human evaluation   
    - 해당 모델이 사람-기계의 다중 턴 상호작용에서 우수한 생성 능력을 보임

|Context|GPT-2|BART|DialoGPT|DialoFlow|Ours|   
|-------|-----|----|--------|---------|----|   
|A: My dear, what's for supper?<br>B: Red cooked carp and rape with fresh mushrooms<br>A: What about soup?<br>B: Sour-peppery soup|Thanks. this is bad bad<br>[Sensible:0 Specific:0]|Do you need some food?<br>[Sensible:0 Specific:0]|Yes, me too. I am making this soup<br>[Sensible:0 Specific:0]|This soup is delicious, I love it<br>[Sensible:1 Specific:0]|I want to know which condiment you use<br>[Sensible:1 Specific:1]|   
|A: Good evening, madam. Can I help you?<br>B: Yes. I bought this sheed here yesterday<br>A: Anything wrong with it?<br>B: Yes. When I spread it out back home, I found this hole|Thank, only need more<br>[Sensible:0 Specific:0]|Then what happend<br>[Sensible:0 Specific:0]| Well, Let me see. I'm terribly sorry<br>[Sensible:1 Specific:0]|I'm sorry. What I can do for you<br>[Sensible:1 Specific:0]|Oh, yeah, it's terrible. May I know your order?<br>[Sensible:1 Specific:1]|   
|A:Good afternoon! Can I help you?<br>B: Could you show me where the Chiese-style clothing is located? I want to buy a silk coat<br>A: This way, please. Here they are. Ther're all handmade<br>B: Oh, they look nice. May I try that black one on?|Sure, no problem<br>[Sensible:1 Specific:0]|Please. It is wonderful<br>[Sensible:0 Specific:0]|Black for you<br>[Sensible:0 Specific:0]|Yes, of course, this way please<br>[Sensible:1 Specific:1]|Sure, the fitting room is on your right<br>[Sensible:1 Specific:1]|   


Conclusion
-----------------------------------------   

- 차이-인지 모듈을 적용하여 모든 문장에 대해 전체 대화에 미치는 의미론적 영향을 찾아냈음
- SOP training task는 잘려 있는 문장의 순서를 재구성하여 구조를 학습하도록 설계하였음
- 해당 모델은 관련성 및 다양성 자동 평가 메트릭에서 주목할 만한 성능을 기록했음
