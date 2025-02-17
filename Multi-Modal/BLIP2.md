## :page_facing_up: BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

### Info

* `publication`: ICML 2023
* `author`: Junnan Li et al.
* `url`: https://proceedings.mlr.press/v202/li23q/li23q.pdf

### 1. Abstract & Introduction


<div align="center"><img src="https://github.com/user-attachments/assets/d4cb3843-2907-4925-88c2-11f0c0e47ba5" width="60%"></img></div>

대부분의 SOTA vision-language model은 pre-training에 높은 계산량을 필요로 한다(대규모의 데이터셋+모델로 인해). Vision-language research는 vision과 language 사이에 있으므로 자연스럽게 unimodal vision & language model을 가져와 사용할 수 있다.

본 논문에서는, frozen vision & frozen language model을 이용하는 generic & compute-efficient VLP method인 `BLIP-2(Bootstrapping Language-Image Pre-training with frozen unimodal models)` 를 제안한다.

VLP에서 pre-trained unimodal model을 활용하기 위해 `cross-modal alignment`가 핵심이지만, LLM은 image에 대해 학습한 적이 없으므로 이러한 점이 vision-language alignment를 어렵게 한다. 현재의 방법들에서는 `image-to-text generation`에 의존하지만 이것으로는 충분하지 않다.

frozen unimodal model의 효과적인 vision-language alignment를 위해 2가지 사전학습 단계로 구성된 **Querying Transformer(Q-Former)** 를 제안한다. Q-Former는 frozen image encoder로부터 visual feature를 추출하기 위해 learnable query vector를 도입한 크기가 작은 transformer이다.

  1. 첫 번째 단계에서는 vision-language representation learning을 통해 Q-Former가 text와 가장 관련있는 visual representation을 배우게 한다.

  2. 두 번째 단계에서는 vision-to-language generative learning을 통해 Q-Former의 output을 frozen LLM과 연결시킨다.

- Contribution

  - BLIP-2는 효과적으로 frozen pre-trained image & language model을 이용하며, modality gap을 Q-Former를 이용해 완화시킴. 결과, BLIP-2는 다양한 vision-language task에서 SOTA 달성

  - LLMs를 이용해 BLIP-2는 자연어 지시를 따라 zero-shot image-to-text generation을 수행하도록 프롬프팅될 수 있음.

  - frozen unimodal model과 Q-Former를 사용함으로써, BLIP-2는 현재 SOTA에 비해 더욱 계산효율적임.

<br></br>

### 2. Method

여기에서는 Q-Former의 model architecture와 2단계의 사전학습 단계에 대해 설명한다. 1단계는 frozen image encoder를 이용한 vision-language representation learning이며, 2단계는 frozen LLM을 이용한 vision-to-language generative learning이다.


<div align="center"><img src="https://github.com/user-attachments/assets/43a9fb41-8457-4c3b-a907-38ef423d170e" width="90%"></img></div>


#### Model Architecture

Q-former는 이미지의 해상도와 상관없이 image encoder로부터 고정된 수의 output feature를 추출한다.

Q-former는 위 그림과 같이 self-attention layer를 공유하는 2개의 transformer submodule로 구성된다.
  1. image transformer는 visual feature 추출을 위해 frozen image encoder와 상호작용한다.
  2. text transformer는 text encoder와 text decoder의 역할을 둘 다 수행할 수 있다.

image transformer를 위한 input으로 몇 개의 learnable query embeddings 세트를 만들어 self-attention을 통해 서로 상호작용하면서 cross-attention을 통해 frozen image features와 상호작용하게 한다(CA는 각 transformer block마다 삽입됨). 이러한 query들은 공유하는 self-attention layer를 통해 text와도 상호작용한다. pre-training task에 따라, query-text interaction을 조절하기 위해 다른 self-attention mask를 적용한다(그림의 오른쪽).

Q-Former는 `BERT-base`의 사전학습 가중치로 초기화되며, cross-attention layer는 랜덤으로 초기화된다. Q-Former는 총 188M의 파라미터를 가지며, 여기에서 query들 또한 모델의 파라미터로 간주된다.

실험에서는, 768차원을 가지는 32개의 queries를 사용하였다(Q-former의 hidden_size와 같음). output query represetation을 $Z$ 로 나타내었으며, 이 크기는 $Z(32 X 768)$ 로 frozen image features(e.g. 257 X 1024 for ViT-L/14)보다 작다.   
이러한 bottlenect architecture는 queries가 text에 가장 관련있는 visual information을 추출하도록 한다.

#### Bootstrap Vision-Language Representation Learning from a Frozen Image Encoder 

representation learning stage에서는 Q-Former를 frozen image encoder와 연결하고 image-text pairs를 사용해 pre-training한다. 이를 통해, queries가 text와 가장 informative한 visual representation을 추출하도록 학습한다.   
`BLIP`과 같이 똑같은 input format과 모델 파라미터를 공유하는 3가지 objectives를 함께 최적화하도록한다.

- Image-Text Contrastive Learning (ITC): image representation과 text representation을 align시켜 공통 정보 최대화

  - negative pairs와 positive pair의 image-text similarity를 대비함으로써 얻어냄 --> Contrastive Learning(same as  CLIP)

  - image transformer의 output query representation $Z$ 와 text transformer의 text representation $t$ 를 align 시킴(t는 [CLS] 토큰의 output embedding).

  - $Z$ 가 multiple output embeddings(각 쿼리에 대한)이므로 각 query output과 $t$ 간의 유사도를 먼저 계산한 후, 가장 높은 것을 image-text similarity로 선택

  - information leak를 피하기 위해, unimodal self-attention mask를 선택(queries와 text는 서로를 보지 못함, 위 그림 참고).

  - frozen image encoder를 사용하기 때문에(image encoder는 학습하지 않기 때문에), GPU에 더 많은 sample들을 올릴 수 있으므로, momentum queue를 사용한 BLIP과 다르게 in-batch negatives를 사용

- Image-grounded Text Generation (ITG): 주어진 이미지를 조건으로 텍스트를 생성하도록 Q-Former 학습

  - Q-Former의 구조는 frozen image encoder와 text token 간의 직접적인 상호 작용이 없으므로, text 생성에 필요한 정보는 쿼리들로 먼저 추출된 후, self-attention layers를 통해 text token으로 전달되어야 함. 이를 통해, 쿼리는 텍스트에 대한 모든 정보를 담는 visual feature를 추출하도록 강제됨.

  - multimodal causal self-attention mask를 사용해 query-text interaction 조절.

    - queries는 text tokens를 제외한 quries끼리 attend 가능하며, 각 text token은 모든 quries와 previous text tokens에 attend 할 수 있음

  - 또한, 첫 번째 토큰인 [CLS] 토큰을 새롭게 만든 [DEC] 토큰으로 대체하여 decoding task의 신호를 주기 위해 사용

- Image-Text Matching (ITM): image와 text representation간의 fine-grained alignment

  - image-text pair의 일치/불일치 여부를 예측하는 binary classification task (same as BLIP)

  - 양방향 self-attention mask 사용하여 모든 queries와 texts는 서로 attend 가능하게 함.

  - output query embeddings $Z$ 는 multimodal information을 담고 있으며, 각 $Z$ 를 linear classifier에 넣어 logit을 얻어낸다.

    - 모든 quries에 대한 logits를 평균내어 output matching score로 사용

  - hard negative mining strategy를 적용해 informative negative pairs 생성


#### Bootstrap Vision-to-Language Generative Learning from a Frozen LLM


<div align="center"><img src="https://github.com/user-attachments/assets/0cf58598-500e-47c5-ac1c-c23a78808070" width="90%"></img></div>

generative pre-training stage에서는 Q-Former를 frozen LLM과 연결한다.   

위 그림과 같이, fully-connected layer를 사용해 output query embeddings $Z$ 를 LLM의 text embedding과 같은 차원으로 linearly project한다. 이는 Q-Former가 추출한 visual representation을 LLM이 이용할 수 있도록 soft visual prompt로써 작동한다. Q-Former는 language-informative한 visual represetation을 추출하도록 pre-train 되었으므로, 관련없는 visual information은 지우면서 가장 유용한 information을 LLM에 전달한다.

- 우리는 2가지 종류의 LLM으로 실험해보았다:

  - Decoder-based LLMs: language modeling loss로 사전학습

    - Q-Former로부터의 visual representation을 조건으로 text 생성

  - Encoder-Decoder based LLMs: prefix language modeling loss로 사전학습(텍스트를 두 부분으로 자름)

    - prefix text는 visual representation에 concatenated되어 LLM의 encoder input으로 사용되고 suffix text는 LLM decoder의 generation target으로 사용된다.

#### Model Pre-training

- Pre-training data: COCO, Visual Genome, CC3M, CC12M, SBU, 115M images from LAION400M (same as BLIP)

- Synthetic caption을 만들기 위해 CapFilt method 사용

- Pre-trained image encoder: ViT-L/14 from CLIP, ViT-g/14 from EVA-CLIP

- Pre-trained LLM: OPT family(decoder based LLM), instruction-trained FlanT5(encoder-decoder based LLM)

<br></br>

### Experiments


<div align="center"><img src="https://github.com/user-attachments/assets/a1ca8aac-8d2c-4451-96f9-f206dae8f141" width="80%"></img></div>   

위 표는 다양한 zero-shot vision-language task에서의 BLIP-2 성능을 나타낸다. 결과, BLIP-2가 이전 SOTA 모델에 비해 훨씬 적은 trainable 파라미터 양으로 더 좋은 성능을 보였다.

#### Instructed Zero-shot Image-to-Text Generation


<div align="center"><img src="https://github.com/user-attachments/assets/21632f02-a232-437f-bbc8-0140cf4b36cd" width="90%"></img></div>   


BLIP-2는 LLM이 text prompt를 따르는 능력을 유지하면서 이미지를 이해하도록 할 수 있으므로, instructions와 함께 image-to-text generation을 조절할 수 있다. 위 사진처럼 단순히 visual prompt 이후에 text prompt를 붙인 것을 LLM의 input으로 준다. 이를 통해, visual knowledge reasoing, visual commonsense reasoning, visual conversation, etc을 포함해 다양한 zero-shot image-to-text 능력을 지닐 수 있다.


<div align="center"><img src="https://github.com/user-attachments/assets/e1d73f37-4781-49fd-a038-6afb654206ba" width="60%"></img></div>

- **Zero-shot VQA**

  - VQAv2, GQA에서 BLIP-2가 SOTA를 달성

  - OK-VQA에서는 2번째로 성능이 좋았는데, 이는 해당 데이터가 visual understanding보다 open-world knowledge에 더 집중해있기 때문으로 추측
    - OK-VQA의 SOTA인 Flamingo80B는 Chinchilla-70B를 사용하는데 이는 BLIP-2의 FlanT5-11B보다 많은 knowledge를 가지고 있음

  - **더 좋은 image encoder나 더 좋은 LLM은 좋은 성능을 가져옴**

    - ViT-g는 OPT와 FlanT5에서 ViT-L보다 좋은 성능을 보임

    - 같은 LLM family에서, 더 큰 모델이 성능이 좋았음

    - instruction-tuned model인 FlanT5는 일반 unsupervised 모델인 OPT보다 성능이 좋았음.


<div align="center"><img src="https://github.com/user-attachments/assets/8e51612b-c0f4-4523-8768-0c8613da44da" width="40%"></img></div>


- **Effect of Vision-Language Representation Learning**

  - Q-Former 학습의 첫 단계는 text와 관련있는 visual feature를 학습하는 것이며, 이를 수행하지 않으면 Flamingo 모델과 비슷하게 vision-to-language generative learning에만 의존할 것이다.

  - 두 LLM을 이용해 Vision-Language Representation Learning의 유무에 따른 성능비교 결과, Representation learning을 수행하지 않았을 때 zero-shot VQA에서 상당한 성능 하락이 있었다.

<div align="center"><img src="https://github.com/user-attachments/assets/b94c20eb-ad2e-4551-978a-a739758824c2" width="80%"></img></div>


- **Image Captioning**

  - image captioning task를 위해 COCO dataset을 이용해 fine-tuning 하였음. fine-tuning에는 LLM을 frozen하고 Q-Former와 image encoder를 학습하였으며, ViT-g와 다양한 LLM으로 실험함

  - 평가에는 NoCaps validation set에 대한 zero-shot transfer와 COCO test set 이용

  - BLIP-2가 NoCaps에서 SOTA를 달성하였으며, 이는 out-domain images에 대해 강한 일반화 능력을 증명


<div align="center"><img src="https://github.com/user-attachments/assets/7965a8c8-5cff-4057-934f-d7f7bcb26b99" width="40%"></img></div>

- **Visual Question Answering**

  - annotated VQA data(VQAv2)를 사용해 Q-Former와 image encoder를 open-ended answer generation loss로 fine-tuning함(LLM은 frozen).

    - LLM은 Q-Former의 output과 question을 input으로 받아 answer 생성

  - question과 관련된 image feature를 추출하기 위해 Q-Former에 question을 추가로 조건화함

    - question tokens를 Q-Former의 input으로 주고 self-attention layer를 통해 queries와 상호 작용

  - 평가 결과, BLIP-2가 여러 open-ended generation model 중에서 SOTA를 달성함.


<div align="center"><img src="https://github.com/user-attachments/assets/a353ae6a-a8c5-4aa3-90f3-8c1be03d0722" width="80%"></img></div>


- **Image-Text Retrieval**

  - image-text retrieval은 language generation을 포함하지 않으므로, LLM없이 Q-Former의 첫 단계 pre-training 모델을 COCO dataset에 fine-tuning(같은 objectives로; ITM, ITC, ITG).

  - 평가에는 COCO와 Flickr30K를 사용했으며, BLIP-2는 SOTA를 달성함

  - 추가로, ITG loss가 image-retrieval에 도움을 주는지 아래와 같이 ITG의 유무에 따른 성능을 비교해보았음.

    - 결과, ITG를 함께 최적화하는 것이 성능이 더 좋았으며, 이는 Q-Former의 pre-training objectives design에 대한 직관을 뒷받침함: ITG loss는 queries가 text와 관련 깊은 visual feature를 추출하도록 강제하므로 vision-language alignment를 향상시킴

<div align="center"><img src="https://github.com/user-attachments/assets/8f9f219e-e84d-4622-aa63-8423cf29afee" width="40%"></img></div>
