<!-- markdownlint-disable MD033 MD045 MD041 -->
<div align="center">
 <div style="margin-bottom: 50px; ">
  <a href="">
  <img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/easydel-logo-with-text.png" height="80">
  </a>
 </div>
 <div>
 <a href="https://discord.gg/FCAMNqnGtt"> <img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/discord-button.png" height="48"></a>&nbsp; &nbsp;
 <a href="https://easydel.readthedocs.io/en/latest/"><img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/documentation-button.png" height="48"></a>&nbsp; &nbsp;
 <a href="https://easydel.readthedocs.io/en/latest/install.html"><img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/quick-start-button.png" height="48"></a>
 </div>
</div>

---

# EasyDeL

EasyDeL is an open-source framework designed to enhance and streamline the
training, fine-tuning, and serving of machine learning models. Built on modern
**Flax NNX** and **JAX**, it provides production-ready solutions for
training and deploying LLMs, multimodal models, and vision models at scale on
TPU/GPU clusters.

## Contents

- [Why EasyDeL?](#why-easydel)
- [Performance & Hackability](#performance--hackability)
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Quick Start](#quick-start)
- [eLargeModel](#elargemodel---unified-model-management)
  - [Configuration Sections](#configuration-sections)
  - [Creating an eLargeModel](#creating-an-elargemodel)
  - [Training with eLargeModel](#training-with-elargemodel)
  - [Evaluation with eLargeModel](#evaluation-with-elargemodel)
- [Advanced Capabilities](#advanced-capabilities)
- [Advanced Recipes](#advanced-recipes)
- [Production Deployment](#production-deployment)
- [Customization](#fully-customizable-and-hackable)
- [Documentation & Community](#documentation)

## Why EasyDeL?

- **🔮 Modern Architecture**: Built on Flax NNX (not legacy Linen) for better modularity and performance
- **70+ Model Architectures**: Broad JAX model collection including LLaMA, Qwen, Mistral, DeepSeek, Gemma, and more
- **16 Specialized Trainers**: From supervised fine-tuning to RLHF, preference optimization, and knowledge distillation
- **🚀 Production-Ready Inference**: eSurge engine with continuous batching, paged KV cache, OpenAI-compatible API, and multimodal serving (text/image/video)
- **Full Multimodal Support**: Vision-language models (LLaVA, Qwen2-VL, Llama4-Vision), speech recognition (Whisper), and diffusion models
- **🚀 TPU & GPU Optimized**: Triton (GPU) and Pallas (TPU) kernel options where available
- **💜 Hackable Like Transformers, as fast as MaxText**: Easy to understand and modify like HuggingFace Transformers, with optimizations in Pallas/Triton/CUDA.

## Performance & Hackability

EasyDeL bridges the gap between ease-of-use and performance in the JAX ecosystem:

### 🚀 Notch Performance

- Triton and Pallas kernels plus dynamic sharding axes for DP/FSDP/TP/EP
- Paged KV cache and continuous batching in eSurge for high-throughput inference
- Gradient checkpointing options and experimental quantization to manage memory
- Sharding-aware training utilities for TPU/GPU clusters

### 💜 Hackability Like HuggingFace Transformers

- Clean, readable code architecture - understand what's happening under the hood
- Every component can be inspected, modified, or replaced
- Familiar HuggingFace-style APIs (`from_pretrained`, `save_pretrained`, `push_to_hub`)
- PyTorch model conversion for easy migration
- Extensive documentation and examples for customization
- No black boxes - transparent implementations from attention to training loops

### 🔮 Best of Both Worlds

- Start with simple APIs, dive deep when needed
- Prototype quickly, optimize for production without rewriting
- Full control over sharding, compilation, and execution
- Native JAX - leverage the entire ecosystem (Optax, Grain, etc.)

## Key Features

### Training & Fine-Tuning

#### 16 Specialized Trainers (unified API)

##### Supervised Learning

- **SFTTrainer** - Supervised fine-tuning with chat templates, completion-only loss, sequence packing
- **Trainer** - General-purpose trainer for custom tasks

##### Preference Optimization (align models with human preferences)

- **DPOTrainer** - Direct Preference Optimization (used in Llama 3, GPT-4)
- **CPOTrainer** - Contrastive Preference Optimization (5 loss variants: sigmoid, hinge, IPO, SimPO, AlphaPO)
- **ORPOTrainer** - Odds Ratio Preference Optimization
- **KTOTrainer** - Kahneman-Tversky Optimization (prospect theory-based)
- **BCOTrainer** - Binary Classifier Optimization
- **XPOTrainer** - Exploratory Preference Optimization

##### Reinforcement Learning

- **GRPOTrainer** - Group Relative Policy Optimization with custom reward functions
- **GSPOTrainer** - Group Stable Policy Optimization for stable RL training
- **GFPOTrainer** - Group Fast Policy Optimization for efficient policy updates
- **NashMDTrainer** - Nash-MD for multi-agent equilibrium

##### Knowledge Distillation

- **DistillationTrainer** - Standard knowledge distillation
- **GKDTrainer** - Generalized Knowledge Distillation with on-policy generation

##### Reward Modeling

- **RewardTrainer** - Train reward models for RLHF

##### Distributed

- **RayDistributedTrainer** - Ray-based distributed training

### High-Performance Inference

#### eSurge - Enterprise Inference Engine

- **Continuous Batching** - Background scheduler for optimal throughput
- **Paged KV Cache** - Memory-efficient attention with prefix caching
- **Multimodal (Text/Image/Video)** - Built-in image + video preprocessing and batching for multimodal models
- **Ragged/Unified Attention** - Variable-length sequence optimization
- **Async & Speculative** - Async token sampling with placeholder replacement plus Eagle-style speculative decoding support
- **Streaming** - Delta-text streaming for interactive clients
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI-style endpoints
- **Authentication & RBAC** - API key management with role-based access control
- **Function Calling** - Tool call parsing; execution is left to the caller
- **Real-Time Monitoring** - Prometheus metrics (Grafana-ready) + console monitor
- **Worker Architecture** - Distributed tokenization/detokenization via ZMQ

#### vWhisper - Speech Recognition

- OpenAI Whisper-compatible transcription API
- Optional timestamp generation for subtitles
- Multi-language transcription and translation

### Advanced Capabilities

#### Attention Mechanisms (10+ implementations)

- Flash Attention 2 (GPU/TPU optimized)
- Ring Attention (distributed sequence parallelism) + (BlockSparse Attn)
- Ragged paged attention v2/v3 (paged KV cache for serving)
- Block-sparse (Splash) attention (TPU optimized)
- SDPA variants (sdpa / cudnn / cuda_flash_attn2)
- Autoregressive decode attention (optimized generation)

#### Mixture of Experts (MoE)

- Expert parallelism with load balancing
- Top-K, Switch, Expert Choice routing
- Grouped matmul kernels for experts
- 3D expert mesh sharding (dp, expert, tp)

#### Quantization

- NF4 (4-bit normal float)
- INT8 (8-bit affine quantization)
- Post-training quantization
- Quantized inference

#### Distributed Training

- Data Parallelism (DP)
- Fully Sharded Data Parallelism (FSDP)
- Tensor Parallelism (TP)
- Expert Parallelism (EP)
- Sequence Parallelism (SP)
- Automatic sharding

#### Memory Optimization

- Gradient checkpointing (8 strategies)
- Activation recomputation
- Mixed precision training
- LoRA (Low-Rank Adaptation)

## Supported Models

### Text Models (70+)

| Family          | Models                                                                                                                                                                 | Features                              |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **LLaMA**       | Llama, Llama4                                                                                                                                                          | Foundation models, Llama4 with vision |
| **Qwen**        | Qwen2, Qwen3, Qwen3-Next, Qwen2-MoE, Qwen3-MoE, Qwen3-Omni, Qwen2-VL                                                                                                   | Text, MoE, vision-language, omni      |
| **Mistral**     | Mistral, Mistral3, Mixtral                                                                                                                                             | MoE, multimodal (Pixtral)             |
| **Google**      | Gemma, Gemma2, Gemma3                                                                                                                                                  | Gemma3 with vision support            |
| **DeepSeek**    | DeepSeekV2, DeepSeekV3                                                                                                                                                 | Multi-head latent attention (MLA)     |
| **Kimi**        | Kimi-Linear                                                                                                                                                            | KDA linear attention                  |
| **GLM**         | GLM, GLM4, GLM4-MoE, GLM4V, GLM4V-MoE, GLM46V                                                                                                                          | Bilingual, MoE, vision-language       |
| **Microsoft**   | Phi, Phi3, PhiMoE                                                                                                                                                      | Small language models                 |
| **Meta**        | OPT, GPT2                                                                                                                                                              | Classic architectures                 |
| **EleutherAI**  | GPT-NeoX, GPT-J                                                                                                                                                        | Open-source LLMs                      |
| **Specialized** | Mamba, Mamba2, RWKV                                                                                                                                                    | State-space and RNN-based models      |
| **Others**      | Arctic, Cohere, Cohere2, DBRX, Exaone, Exaone4, Falcon, Grok-1, InternLM2, MosaicMPT, OLMo, OLMo2, OLMo3, OpenELM, SmolLM3, StableLM, Xerxes, Xerxes2, MiniMax-Text-v1 | Various architectures                 |

### Multimodal Models

| Type                | Models                                                                                                             | Capabilities                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| **Vision-Language** | Llama4-Vision, Qwen2-VL, Qwen3-Omni, Gemma3-Vision, Mistral3 (Pixtral), GLM4V, GLM4V-MoE, GLM46V, LLaVA, AyaVision | Image understanding + text generation      |
| **Vision Encoders** | CLIP, SigLIP, Pixtral                                                                                              | Vision-text alignment                      |
| **Speech**          | Whisper                                                                                                            | Transcription, translation, classification |
| **Diffusion**       | GIDD                                                                                                               | Diffusion language models                  |

### Recent Additions

- Kimi-Linear (`kimi_linear`)
- Qwen3-Next (`qwen3_next`)
- Qwen3-Omni (`qwen3_omni_moe`)
- GLM4V (`glm4v`)
- GLM4V-MoE (`glm4v_moe`)
- GLM46V (`glm46v`)

### Auto Model Classes

```python
AutoEasyDeLModelForCausalLM              # Text generation
AutoEasyDeLModelForSeq2SeqLM             # Sequence-to-sequence
AutoEasyDeLModelForSequenceClassification # Text classification
AutoEasyDeLModelForImageTextToText       # Vision-language models
AutoEasyDeLModelForSpeechSeq2Seq         # Speech models (Whisper)
AutoEasyDeLModelForZeroShotImageClassification # Vision models
AutoEasyDeLVisionModel                   # Vision encoders
AutoEasyDeLConfig                        # Auto configuration
```

## Quick Start

### Quick Start TL; DR

1. **Install**: `uv pip install "easydel[gpu]"` (or `[tpu]` , `[torch]` , `[lm_eval]` as needed).
2. **Serve**: Jump to [Inference Example](#inference-example) to run eSurge with streaming.
3. **Fine-tune**: Try [Supervised Fine-Tuning](#training-example---supervised-fine-tuning), or align with [DPO](#training-example---preference-optimization-dpo)/[GRPO](#training-example---reinforcement-learning-grpo).

### Installation

```bash
# Base installation
uv pip install easydel

# With GPU support
uv pip install easydel[gpu]

# With TPU support
uv pip install easydel[tpu]

# With PyTorch bridge
uv pip install easydel[torch]

# With LM Eval Harness
uv pip install easydel[lm_eval]
```

> [!NOTE]
> Choose `[gpu]` for NVIDIA GPUs with Triton kernels, or `[tpu]` for Google TPUs with Pallas kernels. The `[torch]` extra enables PyTorch model conversion via `from_torch=True`.

### Inference Example

```python
import easydel as ed
from transformers import AutoTokenizer
import jax.numpy as jnp
from jax import lax

# Load model and tokenizer
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model with full configuration
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.float16,
    param_dtype=jnp.float16,
    precision=lax.Precision.DEFAULT,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,

    sharding_axis_dims=(1, 1, 1, -1, 1),  # (dp, fsdp, ep, tp, sp) with tensor parallelism
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3,
        attn_dtype=jnp.float16,
        freq_max_position_embeddings=4096,
        mask_max_position_embeddings=4096,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
    partition_axis=ed.PartitionAxis(),
)

# Create eSurge engine for high-performance inference
engine = ed.eSurge(
    model=model,
    tokenizer=tokenizer,
    max_model_len=4096,
    max_num_seqs=8,  # Continuous batching with 8 sequences
)


# Stream tokens (delta text updates)
for output in engine.stream(
    "Explain quantum computing in simple terms:",
    sampling_params=ed.SamplingParams(max_tokens=256, temperature=0.7)
):
    print(output.delta_text, end="", flush=True)

print(f"\n\nTokens/s: {output.tokens_per_second:.2f}")
```

## eLargeModel - Unified Model Management

`eLargeModel` is the easy master class for working with large VLM/LLM/DLM/... models in EasyDeL. It provides a single, unified interface that combines:

- **Configuration Management** - Model, sharding, quantization, and inference settings in one place
- **Model Building** - Automatic model, tokenizer, and inference engine initialization
- **Training Orchestration** - Support for SFT, DPO, ORPO, GRPO, distillation, and more
- **Evaluation** - Built-in lm-evaluation-harness integration
- **Serialization** - Save/load configurations as JSON for reproducibility

> [!TIP]
> `eLargeModel` is designed for common use cases and quick setup - perfect for getting started fast or when you want a simple, unified API. However, it doesn't expose all of EasyDeL's capabilities. For full modularity, fine-grained control, and access to advanced features, work directly with EasyDeL's underlying components (`AutoEasyDeLModelForCausalLM`, `eSurge`, trainers, etc.). The real power of EasyDeL lies in its hackable, composable architecture.

### Why eLargeModel?

Instead of managing multiple configuration objects and manually wiring components together, eLargeModel lets you define everything in one place:

```python
import easydel as ed

# Traditional approach - multiple configuration objects
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)
engine = ed.eSurge(model=model, tokenizer=tokenizer, ...)

# eLargeModel approach - unified configuration
elm = ed.eLargeModel({...})  # Define everything once
engine = elm.build_esurge()   # Build what you need
```

### Configuration Sections

eLargeModel accepts a dictionary with the following sections:

| Section        | Purpose               | Key Options                                                     |
| -------------- | --------------------- | --------------------------------------------------------------- |
| `model`        | Model identification  | `name_or_path`, `tokenizer`, `task`                             |
| `loader`       | Loading options       | `dtype`, `param_dtype`, `precision`, `verbose`                  |
| `sharding`     | Distributed setup     | `axis_dims`, `axis_names`, `auto_shard_model`                   |
| `base_config`  | Model configuration   | `attn_mechanism`, `gradient_checkpointing`, `moe_method`        |
| `esurge`       | Inference engine      | `max_model_len`, `max_num_seqs`, `hbm_utilization`, `page_size` |
| `quantization` | Model quantization    | `model.dtype` (nf4/int8), `model.block_size`                    |
| `trainer`      | Training settings     | `trainer_type`, `learning_rate`, `num_train_epochs`             |
| `mixture`      | Dataset configuration | `informs`, `batch_size`, `streaming`                            |
| `eval`         | Evaluation settings   | `max_new_tokens`, `temperature`, `batch_size`                   |

### Creating an eLargeModel

#### Method 1: Dictionary Configuration

```python
import easydel as ed

max_model_len = 2**15

elm = ed.eLargeModel(
    {
        "model": {"name_or_path": "EasyDeL/gpt-oss-20b", "tokenizer": "EasyDeL/gpt-oss-20b", "task": "auto-bind"},
        "loader": {"dtype": "bf16", "param_dtype": "bf16", "precision": "default"},
        "sharding": {
            "axis_dims": (1, 1, 2, -1, 1),
            "axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
            "auto_shard_model": True,
        },
        "base_config": {
            "values": {
                "freq_max_position_embeddings": max_model_len,
                "mask_max_position_embeddings": max_model_len,
                "attn_mechanism": ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3,
                "attn_dtype": "bf16",
                "gradient_checkpointing": ed.EasyDeLGradientCheckPointers.NONE,
                "moe_method": ed.MoEMethods.FUSED_MOE,  # For MoE models
                # "operation_configs": {
                #     ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3: ed.RaggedPageAttentionv3Config(
                #         num_queries_per_block=4,
                #         num_kv_pages_per_block=16,
                #         platform="pallas",
                #         backend="any",
                #     )
                # },
            }
        },
        "esurge": {
            "max_model_len": max_model_len,
            "max_num_seqs": 32,
            "hbm_utilization": 0.75,
            "page_size": 128,
            "enable_prefix_caching": True,
        },
        "quantization": {"model": {"dtype": "nf4", "block_size": 128}, "quantize_tensors": False},
    }
)

# Print configuration overview
print(elm)
```

#### Method 2: Builder Pattern (Fluent API)

```python
import easydel as ed

elm = (
    ed.eLargeModel.from_pretrained("EasyDeL/gpt-oss-20b")
    .set_dtype("bf16")
    .set_sharding(axis_dims=(1, 1, 2, -1, 1), axis_names=("dp", "fsdp", "ep", "tp", "sp"))
    .set_esurge(
        max_model_len=4096,
        max_num_seqs=32,
        hbm_utilization=0.75,
        page_size=128,
        enable_prefix_caching=True,
    )
)
```

#### Method 3: Load from JSON

```python
# Save configuration for reproducibility
elm.to_json("my_config.json")

# Load configuration later
elm = ed.eLargeModel.from_json("my_config.json")
```

### Building Components

eLargeModel provides builder methods to create the components you need:

```python
# Build inference engine (includes model + tokenizer)
esurge = elm.build_esurge()

# Or build components separately
model = elm.build_model()
tokenizer = elm.build_tokenizer()

# For training with reference/teacher models
reference_model = elm.build_reference_model()  # For DPO/ORPO
teacher_model = elm.build_teacher_model()      # For distillation

# Build dataset from mixture configuration
dataset = elm.build_dataset()
```

### Inference with eLargeModel

```python
import easydel as ed

elm = (
    ed.eLargeModel.from_pretrained("Qwen/Qwen3-8B")
    .set_dtype("bf16")
    .set_sharding(axis_dims=(1, 1, 1, -1, 1))
    .set_esurge(max_model_len=4096, max_num_seqs=32)
)

# Build and use eSurge engine
esurge = elm.build_esurge()

for output in esurge.chat(
    [{"role": "user", "content": "Explain quantum computing"}],
    sampling_params=ed.SamplingParams(max_tokens=512),
    stream=True,
):
    print(output.delta_text, end="", flush=True)

print(f"\nTokens/s: {output.tokens_per_second:.2f}")
```

### Training with eLargeModel

eLargeModel supports multiple training paradigms through a unified interface:

#### SFT Training

```python
elm = (
    ed.eLargeModel.from_pretrained("Qwen/Qwen3-8B")
    .set_dtype("bf16")
    .set_sharding(axis_dims=(1, 1, 1, -1, 1))
    .set_trainer(
        "sft",
        learning_rate=2e-5,
        num_train_epochs=3,
        total_batch_size=32,
        gradient_accumulation_steps=4,
        max_length=2048,
    )
    .add_dataset("train.json", dataset_type="json", content_field="text")
)

results = elm.train()
```

#### DPO Training

```python
elm = (
    ed.eLargeModel.from_pretrained("Qwen/Qwen3-8B")
    .set_dtype("bf16")
    .set_sharding(axis_dims=(1, 1, 1, -1, 1))
    .set_reference_model("Qwen/Qwen3-8B")  # Reference model for KL constraint
    .set_trainer(
        "dpo",
        beta=0.1,
        learning_rate=5e-7,
        num_train_epochs=1,
        total_batch_size=16,
    )
)

results = elm.train(train_dataset=preference_dataset)
```

#### GRPO Training

```python
elm = (
    ed.eLargeModel.from_pretrained("Qwen/Qwen3-8B")
    .set_dtype("bf16")
    .set_trainer(
        "grpo",
        num_generations=4,
        beta=0.04,
        learning_rate=1e-6,
        temperature=0.9,
    )
)

results = elm.train(train_dataset=prompts_dataset, reward_funcs=your_reward_fn)
```

#### Distillation Training

```python
elm = (
    ed.eLargeModel.from_pretrained("Qwen/Qwen3-1B")  # Student model
    .set_teacher_model("Qwen/Qwen3-8B")               # Teacher model
    .set_dtype("bf16")
    .set_trainer(
        "distillation",
        temperature=3.0,
        alpha=0.5,
        learning_rate=2e-5,
    )
)

results = elm.train(train_dataset=your_dataset)
```

### Evaluation with eLargeModel

Run standard benchmarks using lm-evaluation-harness:

```python
elm = (
    ed.eLargeModel.from_pretrained("google/gemma-3-27b-it")
    .set_dtype("bf16")
    .set_esurge(max_model_len=4096, max_num_seqs=64)
    .set_eval(max_new_tokens=512, temperature=0.0, batch_size=32)
)

# Evaluate on benchmarks
results = elm.eval(
    tasks=["hellaswag", "mmlu", "gsm8k"],
    engine="esurge",
    num_fewshot=5,
    output_path="eval_results.json",
)

# Print results
for task, metrics in results["results"].items():
    print(f"{task}: {metrics.get('acc', metrics.get('exact_match')):.2%}")
```

### Dataset Mixtures

Configure multiple datasets for training:

```python
elm = (
    ed.eLargeModel.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")
    .set_mixture(batch_size=32, streaming=True, shuffle=True)
    .add_dataset("train.json", dataset_type="json", content_field="text", weight=0.5)
    .add_dataset("code/*.parquet", dataset_type="parquet", content_field="content", weight=0.3)
    .add_dataset("imdb", dataset_type="imdb", split="train", weight=0.2)
)

dataset = elm.build_dataset()
```

### Available Builder Methods

| Method                                 | Description                              |
| -------------------------------------- | ---------------------------------------- |
| `set_model(path)`                      | Set model name/path                      |
| `set_dtype(dtype)`                     | Set computation dtype (bf16, fp16, fp32) |
| `set_sharding(axis_dims, axis_names)`  | Configure distributed sharding           |
| `set_quantization(method, block_size)` | Enable quantization (nf4, int8)          |
| `set_esurge(...)`                      | Configure eSurge inference engine        |
| `set_trainer(type, ...)`               | Configure training paradigm              |
| `set_mixture(...)`                     | Configure dataset mixture                |
| `set_eval(...)`                        | Configure evaluation settings            |
| `set_teacher_model(path)`              | Set teacher model for distillation       |
| `set_reference_model(path)`            | Set reference model for DPO/ORPO         |
| `add_dataset(...)`                     | Add dataset to mixture                   |
| `update_config(dict)`                  | Deep merge configuration updates         |

### Build Methods

| Method                    | Returns             | Description                |
| ------------------------- | ------------------- | -------------------------- |
| `build_model()`           | `EasyDeLBaseModule` | Build the model            |
| `build_tokenizer()`       | `AutoTokenizer`     | Build the tokenizer        |
| `build_esurge()`          | `eSurge`            | Build inference engine     |
| `build_trainer()`         | `Trainer`           | Build configured trainer   |
| `build_dataset()`         | `Dataset`           | Build dataset from mixture |
| `build_teacher_model()`   | `EasyDeLBaseModule` | Build teacher model        |
| `build_reference_model()` | `EasyDeLBaseModule` | Build reference model      |
| `train()`                 | Training results    | Run full training pipeline |
| `eval(tasks)`             | Eval results        | Run lm-eval benchmarks     |

### Training Example - Supervised Fine-Tuning

```python
import easydel as ed
from transformers import AutoTokenizer
from datasets import load_dataset
import jax.numpy as jnp

# Load model with configuration
model_id = "Qwen/Qwen3-VL-8B-Thinking"

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
    partition_axis=ed.PartitionAxis(),
)

# Configure trainer
trainer = ed.SFTTrainer(
    model=model,
    arguments=ed.SFTConfig(
        max_length=2048,
        dataset_text_field="text",
        add_special_tokens=False,
        packing=False,
        total_batch_size=32,
        eval_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        weight_decay=0.01,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        save_directory="./checkpoints",
        report_steps=10,
        progress_bar_type="tqdm",
    ),
    train_dataset=load_dataset("timdettmers/openassistant-guanaco", split="train"),
    processing_class=AutoTokenizer.from_pretrained(model_id),
)

# Train
trainer.train()

# Save
model.save_pretrained("./my-finetuned-model")
```

### Training Example - Preference Optimization (DPO)

> [!NOTE]
> DPO aligns models with human preferences without requiring a reward model. The `beta` parameter controls the KL divergence penalty - lower values allow more deviation from the reference model.

```python
import easydel as ed
from transformers import AutoTokenizer
from datasets import load_dataset
import jax.numpy as jnp
from jax import lax

# Load model with full configuration options
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
max_length = 2048

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=lax.Precision.DEFAULT,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    # (DP, FSDP, EP, TP, SP) - Full TP
    config_kwargs=ed.EasyDeLBaseConfigDict(
        freq_max_position_embeddings=max_length,
        mask_max_position_embeddings=max_length,
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        attn_dtype=jnp.bfloat16,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
    partition_axis=ed.PartitionAxis(),  # Default partitioning
)

# DPO is used to align models with human preferences (e.g., Llama 3, GPT-4)
trainer = ed.DPOTrainer(
    model=model,
    arguments=ed.DPOConfig(
        beta=0.1,  # KL penalty coefficient
        loss_type="sigmoid",  # or "ipo", "hinge"
        max_length=512,
        max_prompt_length=256,
        max_completion_length=256,
        total_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=5e-7,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        num_train_epochs=1,
        ref_model_sync_steps=128,
        precompute_ref_log_probs=False,
        disable_dropout=True,
        save_steps=1000,
        report_steps=20,
    ),
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
    processing_class=AutoTokenizer.from_pretrained(model_id),
)

trainer.train()
```

### Training Example - Reinforcement Learning (GRPO)

```python
import easydel as ed
from transformers import AutoTokenizer
import jax.numpy as jnp

# Load model with configuration
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
    partition_axis=ed.PartitionAxis(),
)

# GRPO: Generate multiple completions and learn from relative rewards
trainer = ed.GRPOTrainer(
    model=model,
    arguments=ed.GRPOConfig(
        num_generations=4,  # Generate 4 completions per prompt
        max_prompt_length=2048,
        max_completion_length=1024,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        beta=0.04,
        total_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        num_train_epochs=2,
        ref_model_sync_steps=128,
        save_steps=1000,
        report_steps=20,
    ),
    train_dataset=your_prompts_dataset,
    processing_class=AutoTokenizer.from_pretrained(model_id),
    reward_funcs=your_custom_reward_fn,  # Custom reward logic
)

trainer.train()
```

## Advanced Recipes

### Attention Setup

Configure attention for optimal performance on your hardware:

```python
import easydel as ed
import jax.numpy as jnp
from jax import lax

# Full configuration example with all major options
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=jnp.float16,
    param_dtype=jnp.float16,
    precision=lax.Precision.DEFAULT,  # DEFAULT, HIGH, or HIGHEST
    platform=ed.EasyDeLPlatforms.TRITON,  # TRITON (GPU), PALLAS (TPU), or JAX
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),  # (dp, fsdp, ep, tp, sp)
    config_kwargs=ed.EasyDeLBaseConfigDict(
        # Attention configuration
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        attn_dtype=jnp.float16,

        # Memory optimization
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,

        # Sequence length
        freq_max_position_embeddings=8192,
        mask_max_position_embeddings=8192,

        # MoE configuration (for MoE models)
        moe_method=ed.MoEMethods.FUSED_MOE,  # FUSED_MOE or STANDARD_MOE

        # Quantization configs (optional - use EasyDeLQuantizationConfig for NF4/INT8)
        # kv_cache_quantization_config=ed.EasyDeLQuantizationConfig(dtype=ed.QuantizationType.NF4),
    ),
    partition_axis=ed.PartitionAxis(
        batch_axis="dp",
        sequence_axis="fsdp",
        head_axis="tp",
        kv_head_axis="tp",
    ),
    # quantization_config for model weights (optional - use EasyDeLQuantizationConfig)
)
```

#### Available Attention Mechanisms

- `AUTO` - Automatically selects the best mechanism for your hardware
- `FLASH_ATTN2` - Optimized Flash Attention 2 (GPU/TPU)
- `SDPA` - Scaled dot-product attention
- `CUDNN` - Alias for `SDPA` (cuDNN path)
- `CUDA_FLASH_ATTN2` - Alias for `SDPA` (CUDA FlashAttention-2 path)
- `RING` - Ring attention for sequence parallelism
- `SPLASH` / `BLOCKSPARSE` - Block-sparse attention (`blocksparse`)
- `RAGGED_PAGE_ATTENTION_V3` - Paged attention for inference (default in eSurge)
- `RAGGED_PAGE_ATTENTION_V2` - Paged attention for inference (legacy/compat)
- `REGRESSIVE_DECODE` - Optimized autoregressive decoding attention
- `VANILLA` - Standard attention

> [!NOTE]
> `BLOCKWISE` and `PAGED_ATTENTION` are present in the enum but are currently not registered in `OperationRegistry` (they will raise at runtime if selected).

### Sharding Recipes

> [!NOTE]
> The sharding axes are `(dp, fsdp, ep, tp, sp)`. Use `-1` to automatically use remaining devices. The product of all dimensions must equal your total device count.

EasyDeL supports multiple parallelism strategies:

```python
import easydel as ed
import jax.numpy as jnp

# Configure sharding strategies
# Format: (dp, fsdp, ep, tp, sp)

# Option 1: Fully Tensor Parallel (TP)
sharding_axis_dims = (1, 1, 1, -1, 1)  # Use all devices for tensor parallelism

# Option 2: Fully Data Parallel (DP)
sharding_axis_dims = (-1, 1, 1, 1, 1)  # Replicate model, shard data across devices

# Option 3: Hybrid (FSDP=2, TP=4 on 8 devices)
sharding_axis_dims = (1, 2, 1, 4, 1)   # Split: 2-way FSDP × 4-way TP on 8 devices

# Load model with distributed configuration
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    auto_shard_model=True,
    sharding_axis_dims=sharding_axis_dims,
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
    partition_axis=ed.PartitionAxis(
        batch_axis="dp",
        sequence_axis="fsdp",
        head_axis="tp",
    ),
)

trainer = ed.SFTTrainer(
    model=model,
    arguments=ed.SFTConfig(
        auto_shard_states=True,  # Shard optimizer states
        max_length=2048,
        learning_rate=2e-5,
        total_batch_size=128,
        # ... other args
    ),
)
```

### LoRA (Low-Rank Adaptation)

> [!TIP]
> LoRA significantly reduces memory usage by only training low-rank adapter weights. Use a higher learning rate (2e-4 to 1e-3) compared to full fine-tuning.

Efficient fine-tuning with LoRA:

```python
import easydel as ed
import jax.numpy as jnp
from transformers import AutoTokenizer

# Load base model
model_id = "meta-llama/Llama-3.1-8B"

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
    partition_axis=ed.PartitionAxis(),
)

# Apply LoRA to specific layers (using regex)
model = model.apply_lora_to_layers(
    lora_rank=32,
    lora_pattern=".*(q_proj|v_proj|gate_proj).*",  # Target query, value, and gate
)

# Train normally with LoRA
trainer = ed.SFTTrainer(
    model=model,
    arguments=ed.SFTConfig(
        max_length=512,
        learning_rate=2e-4,  # Higher LR for LoRA
        num_train_epochs=3,
    ),
    train_dataset=your_dataset,
    processing_class=AutoTokenizer.from_pretrained(model_id),
)
trainer.train()

# Merge LoRA weights back into base model
model = model.unwrap_lora_to_layers()
model.save_pretrained("./merged-model")
```

### Quantization Workflow

> [!IMPORTANT]
> Quantization reduces memory usage but may impact model accuracy. NF4 (4-bit) offers the best compression, while INT8 (8-bit) provides a balance between size and quality.

Reduce memory footprint with post-training quantization:

```python
import easydel as ed
import jax.numpy as jnp

# Load model
model_id = "meta-llama/Llama-3.1-8B"

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.float16,
    param_dtype=jnp.float16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.AUTO,
    ),
    partition_axis=ed.PartitionAxis(),
)

# Quantize to 4-bit (NF4) by replacing linear layers
model = model.quantize(
    quantization_config=ed.EasyDeLQuantizationConfig(
        dtype=ed.QuantizationType.NF4,
        block_size=256,
    ),
    quantize_tensors=False,
)

# Use quantized model for inference
from transformers import AutoTokenizer

engine = ed.eSurge(
    model=model,
    tokenizer=AutoTokenizer.from_pretrained(model_id),
    max_model_len=2048,
    max_num_seqs=4,
)

```

### Gradient Checkpointing

> [!TIP]
> Use `NOTHING_SAVEABLE` for maximum memory savings (recomputes everything), or `CHECKPOINT_DOTS` to only checkpoint matrix multiplications (good balance).

Save memory during training:

```python
config_kwargs = ed.EasyDeLBaseConfigDict(
    gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    # Aggressive recompute for max memory savings
    # Other options:
    # EVERYTHING_SAVEABLE - Minimal recompute, highest memory use
    # CHECKPOINT_DOTS - Checkpoint only matrix multiplications
    # DOTS_SAVEABLE - Save dot products
)
```

## Production Deployment

> [!IMPORTANT]
> For production deployments, use `RAGGED_PAGE_ATTENTION_V3` for optimal inference performance with paged KV cache. Enable monitoring with `engine.start_monitoring()` for observability.

### Deploy eSurge API Server

Create an OpenAI-compatible API server:

```python
import easydel as ed
from transformers import AutoTokenizer
import jax.numpy as jnp

# Load model with production configuration
model_id = "meta-llama/Llama-3.1-8B-Instruct"

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.float16,
    param_dtype=jnp.float16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3,
        attn_dtype=jnp.float16,
        freq_max_position_embeddings=8192,
        mask_max_position_embeddings=8192,
    ),
    partition_axis=ed.PartitionAxis(),
)

# Create eSurge engine
engine = ed.eSurge(
    model=model,
    tokenizer=AutoTokenizer.from_pretrained(model_id),
    max_model_len=4096,
    max_num_seqs=16,  # Handle 16 concurrent requests
)

# Create and run API server
api_server = ed.eSurgeApiServer(
    {
        "llama-3.1-8b": engine,  # Model name -> engine mapping
    },
)

# Start server (OpenAI-compatible endpoints)
api_server.run(host="0.0.0.0", port=8000)
```

#### API Endpoints

- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `POST /v1/completions` - Text completions
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /metrics` - Server metrics (JSON). Prometheus metrics are exposed via `engine.start_monitoring(...)` on a separate port.

#### Client Usage

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",  # If authentication enabled
)

response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### Authentication & RBAC

> [!WARNING]
> Store API keys securely and never commit them to version control. The `admin_key` has full access - use it only for key management operations.

Enable API key authentication:

```python
# Server with authentication
api_server = ed.eSurgeApiServer(
    {"model-name": engine},
    require_api_key=True,
    admin_key="admin-key",
)

# Create a user key (store the raw key securely)
user_key, _ = api_server.auth_manager.generate_api_key(name="demo-user")
```

#### Roles

- `admin` - Full access including key management
- `user` - Standard inference access
- `readonly` - Read-only (metrics, health checks)
- `service` - Service account with specific permissions

### Monitoring & Metrics

Enable real-time monitoring:

```python
# After creating and initiating `engine`
# Start Prometheus metrics exporter
engine.start_monitoring(prometheus_port=8080)

# Point Grafana (or any Prometheus UI) at:
# http://localhost:8080/metrics
```

#### Tracked Metrics

- Request and token throughput
- Latency and time-to-first-token
- Queue depth
- KV cache utilization

### Function Calling

Format tool-aware prompts (parsing and execution handled by your code or the API server):

```python
import easydel as ed

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

# Stream a chat response that is aware of tools (tool execution is up to you)
messages = [{"role": "user", "content": "What's the weather in Paris?"}]
for chunk in engine.chat(
    messages,
    tools=tools,
    sampling_params=ed.SamplingParams(max_tokens=128),
    stream=True,
):
    print(chunk.delta_text, end="", flush=True)
```

## Fully Customizable and Hackable

### Why EasyDeL is Easy to Hack

Unlike monolithic frameworks, EasyDeL is designed for transparency and customization:

### 🔮 Readable Architecture

```python
# Every layer is inspectable and modifiable
from easydel.modules.llama import LlamaForCausalLM

# View the exact attention implementation
model = LlamaForCausalLM(config=config, rngs=rngs)
# Source: easydel/modules/llama/modeling_llama.py - clean, documented code

# Customize attention mechanism at runtime
model = model.update_module(attn_mechanism="flash_attn2")

# Or swap out components entirely
class CustomAttention(nn.Module):
    # Your custom implementation
    ...

# Replace in any model
model.model.layers[0].self_attn = CustomAttention(...)
```

### 💜 HuggingFace-Compatible APIs

```python
import easydel as ed
import jax.numpy as jnp

# Familiar patterns from Transformers, with sharding/precision controls
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
    partition_axis=ed.PartitionAxis(),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
)
model.save_pretrained("./my-model")
model.push_to_hub("username/my-model")

# Load PyTorch models directly
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    partition_axis=ed.PartitionAxis(),
    from_torch=True,  # Converts PyTorch checkpoint automatically
)
```

> [!NOTE]
> Use `from_torch=None` to automatically detect checkpoints type!.
> [!TIP]
> Use `from_torch=True` to automatically convert PyTorch checkpoints to EasyDeL. This enables using any HuggingFace model even if there's no native EasyDeL checkpoint.

### 🔮 Flexible Configuration

```python
# Every aspect is configurable
from easydel import LlamaConfig

config = LlamaConfig(
    attn_mechanism="flash_attn2",        # Choose attention
    gradient_checkpointing="checkpoint_dots",  # Memory strategy
    platform="triton",                   # Kernel backend
    use_scan_mlp=True,                   # Custom optimizations
    rope_theta=10000,                    # Positional encoding
    # ... and 70+ more options
)

model = LlamaForCausalLM(config=config, rngs=rngs)
```

### Performance Without Complexity

EasyDeL aims for MaxText-style performance while maintaining code clarity:

| Framework           | Training Speed | Code Complexity      | Customization |
| ------------------- | -------------- | -------------------- | ------------- |
| **MaxText**         | ⚡⚡⚡ Fastest | 🔒 Complex internals | ⚠️ Limited    |
| **HF Transformers** | 🐌 Slower      | ✅ Very readable     | ✅ Easy       |
| **EasyDeL**         | ⚡⚡+ Fast     | ✅ Readable          | ✅ Easy       |

> [!NOTE]
> Performance depends on hardware, sharding choices, and model size. Benchmark on your specific setup for accurate comparisons.

### Real Example: Custom MoE Layer

```python
import easydel as ed
from easydel.layers.moe import BaseMoeModule

class MyCustomMoE(BaseMoeModule):
    """Custom MoE with your routing logic"""

    def __init__(self, config, dtype=jnp.float32, *, rngs):
        super().__init__(
            config=config,
            dtype=dtype,
            num_experts=8,
            top_k=2,
            rngs=rngs,
        )
        # Add custom components
        self.experts = MLPMoE(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            intermediate_size=config.moe_intermediate_size,
            rngs=rngs,
        )
        self.gate = MoEGate(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                intermediate_size=intermediate_size,
                rngs=rngs,
            )

    def __call__(self, hidden_states: chex.Array):
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
        )
        if self.config.n_shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")

# Drop it into any model
model.model.layers[5].mlp = MyCustomMoE(config, rngs=rngs)
```

## Building Custom Modules

EasyDeL's `EasyDeLBaseModule` provides a powerful foundation for custom models:

```python
import easydel as ed
import jax.numpy as jnp
from flax import nnx as nn

class MyCustomModule(ed.EasyDeLBaseModule):
    def __init__(
        self,
        config,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision = None,
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
        # Your custom layers here
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, rngs=rngs)

    def __call__(self, x):
        # Your custom forward pass
        return self.dense(x)
```

### Built-in Features

- Automatic sharding/gathering for distributed training
- HuggingFace Hub integration (`save_pretrained`, `push_to_hub`)
- Quantization support
- LoRA application
- Generation capabilities
- State conversion
- Configuration management

## Multimodal Examples

### Vision-Language Model (Llama4-Vision)

```python
import easydel as ed
from transformers import AutoProcessor
from PIL import Image
import jax.numpy as jnp

# Load vision-language model
model_id = "meta-llama/Llama-4-11B-Vision-Instruct"

model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        attn_dtype=jnp.bfloat16,
    ),
    partition_axis=ed.PartitionAxis(),
)

processor = AutoProcessor.from_pretrained(model_id)

# Load image
image = Image.open("image.jpg")

# Create prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

# Process inputs
inputs = processor(images=image, text=processor.apply_chat_template(messages))

# Generate
outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0]))
```

### Speech Recognition (Whisper)

```python
import easydel as ed
from transformers import AutoProcessor
import jax.numpy as jnp

# Load Whisper model
model_id = "openai/whisper-large-v3"

model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    dtype=jnp.float16,
    param_dtype=jnp.float16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
    ),
    partition_axis=ed.PartitionAxis(),
)

processor = AutoProcessor.from_pretrained(model_id)

# Load audio
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)

# Process
inputs = processor(audio, sampling_rate=sr, return_tensors="jax")

# Transcribe
outputs = model.generate(**inputs)
transcription = processor.decode(outputs[0])
print(transcription)
```

## Documentation

For comprehensive documentation, tutorials, and API reference:

### 📚 [EasyDeL Documentation](https://easydel.readthedocs.io/en/latest/)

#### Key Resources

- Installation Guide
- Training Tutorials (SFT, DPO, GRPO, etc.)
- eSurge Deployment Guide
- Model Architecture Details
- API Reference
- Advanced Topics (Sharding, MoE, Quantization)

## Community & Support

- **Discord**: [Join our Discord](https://discord.gg/FCAMNqnGtt) for discussions and support
- **Documentation**: [readthedocs.io](https://easydel.readthedocs.io/en/latest/)
- **GitHub Issues**: [Report bugs or request features](https://github.com/erfanzar/EasyDeL/issues)
- **Email**: <erfanzare810@gmail.com>

## Contributing

We welcome contributions! Whether it's:

- Adding new model architectures
- Improving documentation
- Fixing bugs
- Adding features
- Sharing examples

Please see our contributing guidelines in the repository.

## Citation

If you use EasyDeL in your research, please cite:

```bibtex
@misc{Zare Chavoshi_2023,
    title={EasyDeL: An open-source library for enhancing and streamlining the training process of machine learning models},
    url={https://github.com/erfanzar/EasyDeL},
    author={Zare Chavoshi, Erfan},
    year={2023}
}
```

## License

EasyDeL is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Flax NNX](https://flax.readthedocs.io/en/latest/nnx/index.html)
- Base idea of Model implementations inspired by [HuggingFace Transformers](https://github.com/huggingface/transformers)
- Trainer implementations are inspired from [TRL](https://github.com/huggingface/trl)
- Inference optimizations inspired by [vLLM](https://github.com/vllm-project/vllm)

---

- Made with <3 by the EasyDeL Author ; /
