# EasyDeL 🔮

EasyDeL, an open-source library, is specifically designed to enhance and streamline the training process of machine
learning models. It focuses primarily on Jax/Flax and aims to provide convenient and effective solutions for training
Flax/Jax Models on TPU/GPU for both Serving and Training purposes. Additionally, EasyDeL will support mojo and will be
rewritten for mojo as well.

Some of the key features provided by EasyDeL include:

- DPOTrainer, SFTTrainer, and VideoCLM Trainers
- Serving and API Engines for Using and serving LLMs in JAX as efficiently as possible.
- Support Quantization Methods for all the Models.
- Support for 8, 6, and 4 BIT Operation, for inference and training in JAX
- A wide range of models in Jax is supported which have never been implemented before such as Falcon, Qwen2, Phi2,
  Mixtral, Qwen2Moe, Cohere,and MPT ...
- Integration of flashAttention in JAX for GPUs and TPUs
- Automatic serving of LLMs with mid and high-level APIs in both JAX and PyTorch
- LLM Trainer and fine-tuner in JAX
- Video CLM Trainer and Fine-tuner for Models such Falcon, Qwen2, Phi2, MPT, Mixtral, Grok-1, and Qwen2Moe ...
- RLHF (Reinforcement Learning from Human Feedback) in Jax (Beta Stage)
- Various other features to enhance the training process and optimize performance.
- LoRA: Low-Rank Adaptation of Large Language Models
- RingAttention, Flash Attention, BlockWise FFN, and Efficient Attention are supported for more than 90 % of models
  ([FJFormer](https://github.com/erfanzar/FJFormer) Backbone).
- Serving and API Engines for Using and serving LLMs in JAX as efficient as possible.
- Automatic Converting Models from JAX-EasyDeL to PyTorch-HF and reverse

## What Makes EasyDeL 🔮 Special

EasyDeL is built up on JAX and Flax and that's why EasyDeL can perform as fast and as easy
as possible

When comparing JAX to PyTorch and TensorFlow, there are several benefits to using JAX that are worth considering.

1. **Performance**: JAX provides excellent performance through its XLA (Accelerated Linear Algebra) backend, which can
   optimize and compile your code for various hardware accelerators such as GPUs and TPUs. This can lead to significant
   speed improvements for certain types of computations.

2. **Automatic Differentiation**: JAX offers a powerful and flexible automatic differentiation system, which is
   essential for training machine learning models. It allows for both forward-mode and reverse-mode automatic
   differentiation, giving you more options for gradient computation.

3. **Functional Programming**: JAX is built around functional programming concepts, which can lead to more composable
   and modular code. This can make it easier to reason about your code and to create abstractions that are reusable
   across different parts of your project.

4. **Interoperability with NumPy**: JAX is designed to be compatible with NumPy, which means that you can often take
   existing NumPy code and run it with minimal changes on JAX. This can be a significant advantage when transitioning
   existing codebases to use JAX.

5. **Flexibility**: JAX provides a high degree of flexibility, allowing you to drop down to lower-level abstractions
   when needed. This can be particularly useful when implementing custom operations or experimenting with new research
   ideas.

While JAX offers these benefits, it's important to note that PyTorch and TensorFlow have large and active communities,
extensive libraries, and a wide range of pre-trained models, which can be advantageous in certain scenarios.
Additionally, the choice of framework often depends on the specific requirements of the project and the familiarity of
the team with a particular toolset.

### Hands on Code Kaggle Examples

1. [script](https://www.kaggle.com/citifer/easydel-causal-language-model-trainer-example) for mindset of using EasyDeL
   CausalLanguageModelTrainer on kaggle, but you can do much more.
2. [script](https://www.kaggle.com/code/citifer/easydel-serve-example-mixtral) for using and serving LLMs with EasyDeL
   JAXServer API (Mixtral Example).
3. [script](https://www.kaggle.com/code/citifer/easydel-sfttrainer-example) SuperVised Finetuning with EasyDeL.

## Citing EasyDeL 🥶

#### To cite this Project

```misc
@misc{Zare Chavoshi_2023,
    title={EasyDeL, an open-source library, is specifically designed to enhance and streamline the training process of machine learning models. It focuses primarily on Jax/Flax and aims to provide convenient and effective solutions for training Flax/Jax Models on TPU/GPU for both Serving and Training purposes.},
    url={https://github.com/erfanzar/EasyDel},
    journal={EasyDeL Easy and Fast DeepLearning with JAX},
    publisher={Erfan Zare Chavoshi},
    author={Zare Chavoshi, Erfan},
    year={2023}
} 
```
