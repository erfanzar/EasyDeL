EasyDeL 🔮
==========
EasyDeL is an open-source framework designed to enhance and streamline the training process of machine learning models, with a primary focus on Jax/Flax. It provides convenient and effective solutions for training and serving Flax/Jax models on TPU/GPU at scale.

Key Features
============

- **Diverse Architecture Support**: Seamlessly work with various model architectures including Transformers, Mamba, RWKV, and more.
- **Diverse Model Support**: Implements a wide range of models that never been implement before in JAX.
- **Advanced Trainers**: Offers specialized trainers like DPOTrainer, ORPOTrainer, SFTTrainer, and VideoCLM Trainer.
- **Serving and API Engines**: Provides engines for efficiently serving large language models (LLMs) in JAX.
- **Quantization and Bit Operations**: Supports various quantization methods and 8, 6, and 4-bit operations for optimized inference and training.
- **Performance Optimization**: Integrates FlashAttention, RingAttention, and other performance-enhancing features.
- **Model Conversion**: Supports automatic conversion between JAX-EasyDeL and PyTorch-HF models.


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

- **Sharding Strategies**: Easily customize and experiment with different sharding strategies to optimize performance across multiple devices.
- **Algorithm Customization**: Modify and fine-tune algorithms to suit your specific needs and hardware configurations.
- **Attention Mechanisms**: Choose from over 10 types of attention mechanisms optimized for GPU/TPU/CPU, including:
  - Flash Attention 2 (CPU(*XLA*), GPU(*Triton*), TPU(*Pallas*)) 
  - Blockwise Attention (CPU, GPU, TPU | *Pallas*-*Jax*)
  - Ring Attention (CPU, GPU, TPU | *Pallas*-*Jax*)
  - Splash Attention (TPU | *Pallas*)
  - SDPA (CPU(*XLA*), GPU(*CUDA*), TPU(*XLA*)) 

This level of customization allows you to squeeze every ounce of performance from your hardware while tailoring the model behavior to your exact requirements.


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
    :hidden:
    :maxdepth: 1
    :caption: Getting Started

    install.md
    contributing

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: APIs

    api_docs/apis


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: TRC

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