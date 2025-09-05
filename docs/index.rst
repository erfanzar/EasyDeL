EasyDeL 🔮
==========
EasyDeL is an open-source framework designed to enhance and streamline the training process of machine learning models, with a primary focus on Jax/Flax. Built on modern Flax NNX, it provides convenient and effective solutions for training and serving Flax/Jax models on TPU/GPU at scale.

Key Features
============

- **Modern Architecture**: Built on Flax NNX for better integration, modularity, and performance
- **Diverse Model Support**: Seamlessly support for Transformers, Mamba, RWKV, Vision Models and more
- **Advanced Trainers**: Specialized trainers like DPOTrainer, ORPOTrainer, SFTTrainer, and VideoCLM Trainer
- **Vision Model Support**: Comprehensive support for vision-to-vision tasks, image-text-to-image generation, and image-to-text processing
- **Production-Ready Serving**: Includes vInference engine for efficient LLM inference and API endpoints compatible with OpenAI standards
- **Quantization and Bit Operations**: Supports various quantization methods including NF4, A8BIT, A8Q, and A4Q for optimized inference and training
- **Performance Optimization**: Integrates FlashAttention, RingAttention, and other performance-enhancing features
- **Model Conversion**: Supports automatic conversion between JAX-EasyDeL and PyTorch-HF models


Fully Customizable and Hackable 🛠️
===================================

EasyDeL stands out by providing unparalleled flexibility and transparency:

- **Open Architecture**: Every single component of EasyDeL is open for inspection, modification, and customization. There are no black boxes here.
- **Hackability at Its Core**: We believe in giving you full control. Whether you want to tweak a small function or completely overhaul a training loop, EasyDeL lets you do it.
- **Custom Code Access**: All custom implementations are readily available and well-documented, allowing you to understand, learn from, and modify the internals as needed.
- **Encourage Experimentation**: We actively encourage users to experiment, extend, and improve upon the existing codebase. Your innovations could become the next big feature!
- **Community-Driven Development**: Share your custom implementations and improvements with the community, fostering a collaborative environment for advancing ML research and development.

With EasyDeL, you're not constrained by rigid frameworks. Instead, you have a flexible, powerful toolkit that adapts to your needs, no matter how unique or specialized they may be.

Advanced Customization and Optimization 🔧
==========================================

EasyDeL provides unparalleled flexibility in customizing and optimizing your models:

- **Custom Module System**: Built on Flax NNX, allowing easy creation and integration of custom modules
- **Transparent Architecture**: Every component is open for inspection and modification
- **Dynamic Configuration**: Easily customize model architecture, training pipeline, and inference settings

- **Sharding Strategies**: Easily customize and experiment with different sharding strategies to optimize performance across multiple devices.
- **Algorithm Customization**: Modify and fine-tune algorithms to suit your specific needs and hardware configurations.
- **Attention Mechanisms**: Choose from over 10 types of attention mechanisms optimized for GPU/TPU/CPU, including:
  - Flash Attention 2 (CPU(*XLA*), GPU(*Triton*), TPU(*Pallas*))
  - Blockwise Attention (CPU, GPU, TPU | *Pallas*-*Jax*)
  - Ring Attention (CPU, GPU, TPU | *Pallas*-*Jax*)
  - Splash Attention (TPU | *Pallas*)
  - SDPA (CPU(*XLA*), GPU(*CUDA*), TPU(*XLA*))

This level of customization allows you to squeeze every ounce of performance from your hardware while tailoring the model behavior to your exact requirements.


Inference and Serving Solutions
===============================

EasyDeL provides powerful, production-ready serving solutions:

- **eSurge Engine**: High-performance inference engine with advanced KV cache management and paged attention
- **vInference Engine**: Optimized for efficient inference of large language models
- **vInference API Server**: Provides OpenAI-compatible endpoints for easy integration
- **vWhisper Inference**: Specialized inference engine for audio transcription and translation
- **Multimodal Support**: Process text, images, and audio with unified APIs
- **Streaming Generation**: Optimized for low-latency response streaming
- **Quantization Options**: Multiple precision options for optimal performance/quality trade-offs


Installation on TPU Pods
-----------------------

For distributed installation on TPU pods, use the ``install_on_hosts`` script:

.. code-block:: bash

    python -m easydel.scripts.install_on_hosts --tpu-type v4-16 --source github

This will install EasyDel and all required dependencies across all hosts in the specified TPU pod.
Supported TPU types include v2, v3, v4, v5e, and v5p pod slices.

For other TPU types you may need to overwrite that or customize the script.

Options:
- ``--source``: Choose between PyPI package (``pypi``) or latest GitHub version (``github``)
- ``--tpu-type``: Specify your TPU pod slice type (default: v4-16)
- ``--num-tpu-hosts``: Override default host count if needed


Future Updates and Vision 🚀
============================

EasyDeL is constantly evolving to meet the needs of the machine learning community. In upcoming updates, we plan to introduce:

- **Cutting-Edge Features**: EasyDeL is committed to long-term maintenance and continuous improvement. We provide frequent updates, often on a daily basis, introducing new features, optimizations, and bug fixes.
- **Ready-to-Use Blocks**: Pre-configured, optimized building blocks for quick model assembly and experimentation.
- **Enhanced Scalability**: Improved tools and methods for effortlessly scaling LLMs to handle larger datasets and more complex tasks.
- **Advanced Customization Options**: More flexibility in model architecture and training pipeline customization.


Why Choose EasyDeL?
====================

1. **Flexibility**: EasyDeL offers a modular design that allows researchers and developers to easily mix and match components, experiment with different architectures (e.g., Transformers, Mamba, RWKV), and adapt models to specific use cases.
2. **Performance**: Leveraging the power of JAX and Flax, EasyDeL provides high-performance implementations of state-of-the-art models and training techniques, optimized for both TPUs and GPUs.
3. **Scalability**: From small experiments to large-scale model training, EasyDeL provides tools and optimizations to efficiently scale your models and workflows.
4. **Ease of Use**: Despite its powerful features, EasyDeL maintains an intuitive API, making it accessible for both beginners and experienced practitioners.
5. **Cutting-Edge Research**: Quickly implement the latest advancements in model architectures, training techniques, and optimization methods.


Citing EasyDeL 🥶
---------------------------------------------------------------

To cite this Project
---------------------------------------------------------------

.. _EasyDeL:

Zare Chavoshi, Erfan. "EasyDeL, an open-source library, is specifically designed to enhance and streamline the training process of machine learning models. It focuses primarily on Jax/Flax and aims to provide convenient and effective solutions for training Flax/Jax Models on TPU/GPU for both Serving and Training purposes." 2023. https://github.com/erfanzar/EasyDeL

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   trainers/base_trainer
   trainers/trainer_protocol
   trainers/ray_distributed_trainer.md
   install.md
   contributing
   api_docs/apis
   esurge
   esurge_examples
   whisper_api.md
   vinference_api.md
	 vsurge_example
   vsurge_api_server_example
   trainers/dpo.md
   trainers/grpo.md
   trainers/orpo.md
   trainers/sft.md
   trainers/reward.md
   multimodality/vision_language.md
   multimodality/audio_language.md
   multimodality/inference.md
   trc-welcome.md



.. code-block:: bibtex

  @misc{Zare Chavoshi_2023,
      title={EasyDeL, an open-source library, is specifically designed to enhance and streamline the training process of machine learning models. It focuses primarily on Jax/Flax and aims to provide convenient and effective solutions for training Flax/Jax Models on TPU/GPU for both Serving and Training purposes.},
      url={https://github.com/erfanzar/EasyDeL},
      journal={EasyDeL Easy and Fast DeepLearning with JAX},
      publisher={Erfan Zare Chavoshi},
      author={Zare Chavoshi, Erfan},
      year={2023}
  }
