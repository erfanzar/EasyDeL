# EasyDeL 🔮

EasyDeL, an open-source library, is specifically designed to enhance and streamline the training process of machine
learning models. It focuses primarily on Jax/Flax and aims to provide convenient and effective solutions for training
Flax/Jax Models on TPU/GPU for both Serving and Training purposes. Additionally, EasyDeL will support mojo and will be
rewritten for mojo as well.

Some of the key features provided by EasyDeL include:

- Support for 8, 6, and 4 BIT inference and training in JAX
- Wide Range of models in Jax are supported which have never been implemented before such as _falcon, Qwen2, Phi2,
  MPT ..._
- Integration of flashAttention in JAX for GPUs and TPUs
- Automatic serving of LLMs with mid and high-level APIs in both JAX and PyTorch
- LLM Trainer and fine-tuner in JAX
- RLHF (Reinforcement Learning from Human Feedback) in Jax (Beta Stage)
- And various other features to enhance the training process and optimize performance.
- LoRA: Low-Rank Adaptation of Large Language Models

> These features collectively aim to simplify and accelerate the training of machine learning models, making it more
> efficient and accessible for developers working with Jax/Flax.

> FlashAttention and Splash Attention are currently disabled for Falcon, MPT, PHI and GPTJ

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