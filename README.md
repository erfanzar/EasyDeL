# EasyDeL 🔮

[**Key Features**](#key-features) 
| [**Quick Start**](#quick-start)
| [**Reference docs**](https://easydel.readthedocs.io/en/latest/)
| [**License**](#license-)

EasyDeL is an open-source framework designed to enhance and streamline the training process of machine learning models, with a primary focus on Jax/Flax. Built on modern Flax NNX, it provides convenient and effective solutions for training and serving Flax/Jax models on TPU/GPU at scale.

## Key Features

- **Modern Architecture**: Built on Flax NNX for better integration, modularity, and performance
- **Diverse Model Support**: Seamless support for Transformers, Mamba, RWKV, Vision Models and more
- **Advanced Trainers**: Specialized trainers with unified base class for consistent behavior:
  - SFTTrainer for supervised fine-tuning
  - DPOTrainer for direct preference optimization
  - ORPOTrainer for offline reinforcement learning
  - More Trainers Like image-text-to-image and others are also supported in main trainer 

- **Vision Model Support**: Comprehensive support for:
  - Vision-to-Vision tasks
  - Image-Text-to-Image generation
  - Image-to-Text processing
- **Production-Ready Serving**: 
  - `vInference` engine for efficient LLM inference
  - `vInferenceApiServer` for OpenAI-compatible API endpoints
- **Performance Optimization**:
  - Integration with multiple attention mechanisms
  - Advanced quantization support including NF4, A8BIT, A8Q, and A4Q
  - Platform-specific optimizations (TRITON, XLA, Pallas)

### Fully Customizable and Hackable 🛠️

EasyDeL's architecture is designed for maximum flexibility:

- **Custom Module System**: Built on Flax NNX, allowing easy creation and integration of custom modules
- **Transparent Architecture**: Every component is open for inspection and modification
- **Dynamic Configuration**: Easily customize model architecture, training pipeline, and inference settings
- **Platform Flexibility**: Choose between different platforms (TRITON, XLA, Pallas) for optimal performance

### Optimizations

#### Attention Mechanisms
EasyDeL offers various attention implementations optimized for different use cases:

```python
import easydel as ed

# Choose platform during model initialization
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "model_name",
    platform=ed.EasyDeLPlatforms.TRITON,  # or .JAX, .PALLAS
)
```

- **TRITON**: Optimized for GPU computation with custom CUDA/Triton kernels
- **PALLAS**: Specialized for TPU performance
- **JAX**: Default JAX backend for broad hardware support

#### Attention Mechanisms

EasyDeL offers various attention implementations optimized for different use cases:

```python
import easydel as ed

# Configure attention mechanism in model config
config = ed.EasyDeLBaseConfigDict(
    attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,  # Choose mechanism
)
```

Available mechanisms:
- **FLASH_ATTN2**: Optimized Flash Attention 2 implementation
- **CUDA_FLASH_ATTN2**: CUDA-specific Flash Attention 2 variant
- **RING**: Memory-efficient ring attention for distributed computing
- **VANILLA**: Standard attention implementation
- **SPLASH**: TPU-optimized splash attention
- **CUDNN**: cuDNN-accelerated attention
- **BLOCKWISE**: Memory-efficient blockwise attention
- **SDPA**: Scaled dot-product attention

Example of configuring a model with specific platform and attention:

```python
import easydel as ed
import jax.numpy as jnp

# Complete configuration example
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    platform=ed.EasyDeLPlatforms.TRITON,
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        attn_dtype=jnp.float16,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
    ),
    dtype=jnp.float16,
    auto_shard_model=True
)
```

## Quick Start

### Installation

```bash
pip install easydel
```

### Basic Inference Example

```python
import easydel as ed
from transformers import AutoTokenizer
import jax.numpy as jnp

# Initialize model
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    dtype=jnp.float16,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True
)

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Create inference engine
inference = ed.vInference(
    model=model,
    tokenizer=tokenizer,
    generation_config=ed.vInferenceConfig(
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        streaming_chunks=32
    )
)

# Create API server (OpenAI compatible)
api_server = ed.vInferenceApiServer({inference.inference_name: inference})
api_server.fire()
```
### Building Custom Modules 🛠️
EasyDeL's architecture is designed for maximum flexibility and customization:
```python
import easydel as ed
import jax.numpy as jnp

# Create your custom module
class MyCustomModule(ed.EasyDeLModule):
		def __init__(
				self,
				config,
				dtype: jnp.dtype = jnp.float32,
				param_dtype: jnp.dtype = jnp.float32,
				precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
				other_parameters = ...,
				*,
				rngs: nn.Rngs,
		):
				super().__init__(
					config=config,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
        self.other_parameters = other_parameters
        
    def __call__(self, x):
        # Your custom implementation
        return x

# Use it in your model
class MyModel(ed.EasyDeLPreTrainedModel):
		def __init__(
				self,
				config,
				dtype: jnp.dtype = jnp.float32,
				param_dtype: jnp.dtype = jnp.float32,
				precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
				other_parameters = ...,
				*,
				rngs: nn.Rngs,
		):
				super().__init__(
					config=config,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					rngs=rngs,
				)
        self.custom_module = MyCustomModule(
					config=config,
					dtype=dtype,
					param_dtype=param_dtype,
					precision=precision,
					other_parameters=other_parameters,
					rngs=rngs,
				)
        
    def __call__(self, x):
        return self.custom_module(x)
```

### Training Example

```python
import easydel as ed
import jax.numpy as jnp

# Create model
model = ed.LlamaForCausalLM(
    config=model_config,
    dtype=jnp.float32,
    param_dtype=jnp.float32
)

# Configure training
training_args = ed.TrainingArguments(
    save_directory="checkpoints",
    model_name="my_model",
    num_train_epochs=3,
    total_batch_size=8,
    learning_rate=3e-4,
    optimizer=ed.EasyDeLOptimizers.ADAMW,
    scheduler=ed.EasyDeLSchedulers.COSINE
)

# Initialize trainer
trainer = ed.Trainer(
    arguments=training_args,
    model=model,
    dataset_train=train_dataset,
    dataset_eval=eval_dataset
)

# Start training
trainer.train()
```

## Latest Updates 🔥

- **Modernized Architecture**: 
  - Migrated to Flax NNX for improved modularity and performance
  - Custom module system for better extensibility
  - Unified base trainer for consistent behavior
- **Enhanced Performance**:
  - Faster training and inference
  - Better large-scale handling
  - Improved memory management
- **Vision Model Support**:
  - Added comprehensive vision model capabilities
  - Support for multimodal tasks
- **Major Improvements**:
  - Fixed core issues from easydel-linen
  - Enhanced stability and reliability
  - Better error handling and debugging

## Documentation 💫

For comprehensive documentation, tutorials, and API reference, visit [EasyDeL Documentation](https://easydel.readthedocs.io/en/latest/).

## Contributing

We welcome contributions! Please see our contributing guidelines in the repository.

## License 📜

EasyDeL is released under the Apache v2 license. See the LICENSE file for details.

## Citation

```bibtex
@misc{Zare Chavoshi_2023,
    title={EasyDeL: An open-source library for enhancing and streamlining the training process of machine learning models},
    url={https://github.com/erfanzar/EasyDeL},
    author={Zare Chavoshi, Erfan},
    year={2023}
}
```

## Contact

For questions or comments about EasyDeL, contact: _erfanzare810@gmail.com_