## EasyDeL 🔮

EasyDeL is an open-source framework designed to enhance and streamline the training process of machine learning models.
With a primary focus on Jax/Flax, EasyDeL aims to provide convenient and effective solutions for training Flax/Jax
models on TPU/GPU for both serving and training purposes.

## Key Features

1. **Trainers**: EasyDeL offers a range of trainers, including DPOTrainer, ORPOTrainer, SFTTrainer, and VideoCLM
   Trainer, tailored for specific training requirements.

2. **Serving and API Engines**: EasyDeL provides serving and API engines for efficiently using and serving large
   language models (LLMs) in JAX, enabling seamless integration into various applications.

3. **Quantization Support**: EasyDeL supports quantization methods for all models, allowing for efficient inference and
   training.

4. **Bit Operation Support**: EasyDeL supports 8, 6, and 4-bit operations for inference and training in JAX, optimizing
   performance and resource utilization.

5. **Diverse Model Support**: EasyDeL offers a wide range of models in JAX that have never been implemented before, such
   as Falcon, Qwen2, Phi2, Mixtral, Qwen2Moe, Cohere, Dbrx, Phi3, and MPT.

6. **FlashAttention Integration**: EasyDeL integrates FlashAttention in JAX for GPUs and TPUs, enhancing performance and
   efficiency.

7. **Automatic LLM Serving**: EasyDeL enables automatic serving of LLMs with mid and high-level APIs in both JAX and
   PyTorch, simplifying deployment and integration.

8. **LLM Training and Fine-tuning**: EasyDeL provides LLM trainer and fine-tuner capabilities in JAX, allowing for
   efficient training and customization of language models.

9. **Video CLM Training and Fine-tuning**: EasyDeL supports Video CLM trainer and fine-tuner for models such as Falcon,
   Qwen2, Phi2, MPT, Mixtral, Grok-1, and Qwen2Moe, enabling advanced video-related applications.

10. **Performance Optimization**: EasyDeL provides various features to enhance the training process and optimize
    performance, such as LoRA (Low-Rank Adaptation of Large Language Models), RingAttention, FlashAttention, BlockWise
    FFN, and Efficient Attention support (through the FJFormer backbone).

11. **Model Conversion**: EasyDeL supports automatic conversion of models from JAX-EasyDeL to PyTorch-HF and vice versa,
    facilitating seamless integration with different frameworks.

With its comprehensive set of features and tools, EasyDeL aims to streamline and accelerate the training and deployment
of machine learning models, particularly in the domain of large language models and video-related applications.

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
