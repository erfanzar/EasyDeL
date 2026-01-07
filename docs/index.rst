EasyDeL 🔮
==========
EasyDeL is an open-source framework designed to enhance and streamline the training process of machine learning models, with a primary focus on Jax/Flax. Built on modern Flax NNX, it provides convenient and effective solutions for training and serving Flax/Jax models on TPU/GPU at scale.

Key Features
============

- **Modern Architecture**: Built on Flax NNX for better integration, modularity, and performance
- **Diverse Model Support**: Seamlessly support for Transformers, Mamba, RWKV, Vision Models and more
- **Advanced Trainers**: Specialized trainers like DPOTrainer, ORPOTrainer, SFTTrainer, and VideoCLM Trainer
- **Vision Model Support**: Comprehensive support for vision-to-vision tasks, image-text-to-image generation, and image-to-text processing
- **Production-Ready Serving**: Includes eSurge and vWhisper engines plus OpenAI-compatible endpoints
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
- **vWhisper Inference**: Specialized inference engine for audio transcription and translation
- **Multimodal Support**: Process text, images, and audio with unified APIs
- **Streaming Generation**: Optimized for low-latency response streaming
- **Quantization Options**: Multiple precision options for optimal performance/quality trade-offs

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
   install.md
   ray_upgrade.md
   contributing

.. toctree::
   :maxdepth: 2
   :caption: Infrastructure:

   infra/index
   infra/overview.md
   environment_variables.md
   infra/base_config.md
   infra/base_module.md
   infra/customization.md
   infra/adding_models.md
   infra/elarge_model.md

.. toctree::
   :maxdepth: 2
   :caption: Data Management:

   easydata/README.md
   easydata/quickstart.md
   easydata/sources.md
   easydata/transforms.md
   easydata/mixing.md
   easydata/pretokenization.md
   easydata/streaming.md
   easydata/pipeline.md
   easydata/caching.md
   easydata/trainer_integration.md

.. toctree::
   :maxdepth: 2
   :caption: Trainers:

   trainers/base_trainer
   trainers/trainer_protocol
   trainers/ray_distributed_trainer.md
   trainers/sft.md
   trainers/dpo.md
   trainers/grpo.md
   trainers/orpo.md
   trainers/reward.md

.. toctree::
   :maxdepth: 2
   :caption: Inference:

   esurge
   esurge_examples
   whisper_api.md

.. toctree::
   :maxdepth: 2
   :caption: CLI & Scripts:

   scripts/index.md
   scripts/model_conversion.md
   scripts/hf_download_to_gcs.md
   scripts/elarge_cli.md
   scripts/model_cards.md
   scripts/dev_tools.md

.. toctree::
   :maxdepth: 2
   :caption: Advanced:

   multimodality/audio_language.md
   api_docs/apis
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
