# EasyDeL 🔮

[**Key Features**](#key-features)
| [**Latest Updates**](#latest-updates-)
| [**Vision**](#future-updates-and-vision-)
| [**Quick Start**](#quick-start)
| [**Reference docs**](https://easydel.readthedocs.io/en/latest/)
| [**License**](#license-)

EasyDeL is an open-source framework designed to enhance and streamline the training process of machine learning models, with a primary focus on Jax/Flax. It provides convenient and effective solutions for training and serving Flax/Jax models on TPU/GPU at scale.

## Key Features

- **Diverse Architecture Support**: Seamlessly work with various model architectures including Transformers, Mamba, RWKV, and more.
- **Diverse Model Support**: Implements a wide range of models that never been implement before in JAX.
- **Advanced Trainers**: Offers specialized trainers like DPOTrainer, ORPOTrainer, SFTTrainer, and VideoCLM Trainer.
- **Serving and API Engines**: Provides engines for efficiently serving large language models (LLMs) in JAX.
- **Quantization and Bit Operations**: Supports various quantization methods and 8, 6, and 4-bit operations for optimized inference and training.
- **Performance Optimization**: Integrates FlashAttention, RingAttention, and other performance-enhancing features.
- **Model Conversion**: Supports automatic conversion between JAX-EasyDeL and PyTorch-HF models.

### Fully Customizable and Hackable 🛠️

EasyDeL stands out by providing unparalleled flexibility and transparency:

- **Open Architecture**: Every single component of EasyDeL is open for inspection, modification, and customization. There are no black boxes here.

- **Hackability at Its Core**: We believe in giving you full control. Whether you want to tweak a small function or completely overhaul a training loop, EasyDeL lets you do it.

- **Custom Code Access**: All custom implementations are readily available and well-documented, allowing you to understand, learn from, and modify the internals as needed.

- **Encourage Experimentation**: We actively encourage users to experiment, extend, and improve upon the existing codebase. Your innovations could become the next big feature!

- **Community-Driven Development**: Share your custom implementations and improvements with the community, fostering a collaborative environment for advancing ML research and development.

With EasyDeL, you're not constrained by rigid frameworks. Instead, you have a flexible, powerful toolkit that adapts to your needs, no matter how unique or specialized they may be. Whether you're conducting cutting-edge research or building production-ready ML systems, EasyDeL provides the freedom to innovate without limitations.

### Advanced Customization and Optimization 🔧

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

## Future Updates and Vision 🚀

EasyDeL is constantly evolving to meet the needs of the machine learning community. In upcoming updates, we plan to introduce:

- **Cutting-Edge**: EasyDeL is committed to long-term maintenance and continuous improvement. We provide frequent updates, often on a daily basis, introducing new features, optimizations, and bug fixes. Our goal is to ensure that EasyDeL remains at the cutting edge of machine learning technology, providing researchers and developers with the most up-to-date tools and capabilities.
- **Ready-to-Use Blocks**: Pre-configured, optimized building blocks for quick model assembly and experimentation.
- **Enhanced Scalability**: Improved tools and methods for effortlessly scaling LLMs to handle larger datasets and more complex tasks.
- **Advanced Customization Options**: More flexibility in model architecture and training pipeline customization.

### Why Choose EasyDeL?

1. **Flexibility**: EasyDeL offers a modular design that allows researchers and developers to easily mix and match components, experiment with different architectures (including Transformers, Mamba, RWKV, and ...), and adapt models to specific use cases.

2. **Performance**: Leveraging the power of JAX and Flax, EasyDeL provides high-performance implementations of state-of-the-art models and training techniques, optimized for both TPUs and GPUs.

3. **Scalability**: From small experiments to large-scale model training, EasyDeL provides tools and optimizations to efficiently scale your models and workflows.

4. **Ease of Use**: Despite its powerful features, EasyDeL maintains an intuitive API, making it accessible for both beginners and experienced practitioners.

5. **Cutting-Edge Research**: quickly implementing the latest advancements in model architectures, training techniques, and optimization methods.

## Quick Start

### Installation

```bash
pip install easydel
```

### Testing Attention Mechanisms

```python
import easydel as ed
ed.FlexibleAttentionModule.run_attention_benchmarks()
```

## Documentation 💫

Comprehensive documentation and examples are available at [EasyDeL Documentation](https://easydel.readthedocs.io/en/latest/).

Here's an improved version of your latest updates:

### Latest Updates 🔥

- **Platform and Backend Flexibility**: You can now specify the platform (e.g., TRITON) and backend (e.g., GPU) to optimize your workflows.  
- **Expanded Model Support**: Added support for new models such as `olmo2`, `qwen2_moe`, `mamba2`, and others.  
- **Enhanced Trainers**: Trainers are now more customizable and hackable, enabling greater flexibility in your projects.  
- **New Trainer Types**:  
  - **Sequence-to-Sequence Trainers**  
  - **Sequence Classification Trainers**  
- **`vInference` Engine**: Introduced as a robust inference engine for LLMs, with Long-Term Support (LTS).  
- **`vInferenceApiServer`**: Added as a backend for the inference engine, fully compatible with OpenAI APIs.  
- **Optimized GPU Integration**: EasyDeL now leverages custom, direct TRITON calls for improved GPU performance.  
- **Dynamic Quantization Support**: Added support for quantization types like NF4, A8BIT, A8Q, and A4Q for efficiency and scalability.  

## Key Components

### vInference

The `vInference` class provides a streamlined interface for text generation using pre-trained language models within JAX.

```python
import easydel as ed
from transformers import AutoTokenizer

model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

inference = ed.vInference(
	model=model,
	params=params,
	tokenizer=tokenizer,
	generation_config=ed.vInferenceConfig(
		temperature=model.generation_config.temperature,
		top_k=model.generation_config.top_k,
		top_p=model.generation_config.top_p,
		bos_token_id=model.generation_config.bos_token_id,
		eos_token_id=model.generation_config.eos_token_id,
		pad_token_id=model.generation_config.pad_token_id,
		streaming_chunks=32,
		max_new_tokens=1024,
	),
)
```

### vInferenceApiServer 

`vInferenceApiServer` is a Serve API Engine for production or research purposes, providing a stable, efficient, and OpenAI API like API.

```python
import easydel as ed

api_inference = ed.vInferenceApiServer(
	{inference.inference_name: inference}
)  # you can load multi inferences together
api_inference.fire()
```

### EasyDeLState

`EasyDeLState` acts as a comprehensive container for your EasyDeL model, including training progress, model parameters, and optimizer information.

```python
from easydel import EasyDeLState

state = EasyDeLState.from_pretrained(
    pretrained_model_name_or_path="model_name",
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    sharding_axis_dims=(1, -1, 1, 1)
)
```

## Training Examples

### Supervised Fine-Tuning

```python
from easydel import SFTTrainer, TrainingArguments

trainer = SFTTrainer(
    arguments=train_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    formatting_func=prompter,
    packing=True,
    num_of_sequences=max_length,
)

output = trainer.train(flax.core.FrozenDict({"params": params}))
```

### DPO Fine-tuning

```python
from easydel import DPOTrainer

dpo_trainer = DPOTrainer(
    model_state=state,
    ref_model_state=ref_state,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    arguments=arguments,
    max_length=max_length,
    max_completion_length=max_completion_length,
    max_prompt_length=max_prompt_length,
)

output = dpo_trainer.train()
```

## Contributing

Contributions to EasyDeL are welcome! Please fork the repository, make your changes, and submit a pull request.

## License 📜

EasyDeL is released under the Apache v2 license. See the LICENSE file for more details.

## Contact

If you have any questions or comments about EasyDeL, you can reach out to me at _erfanzare810@gmail.com_.

## Citation

To cite EasyDeL in your work:

```bibtex
@misc{Zare Chavoshi_2023,
    title={EasyDeL: An open-source library for enhancing and streamlining the training process of machine learning models},
    url={https://github.com/erfanzar/EasyDeL},
    author={Zare Chavoshi, Erfan},
    year={2023}
}
```