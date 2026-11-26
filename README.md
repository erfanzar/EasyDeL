<!-- markdownlint-disable MD033 MD045 MD041 -->
<div align="center">
 <div style="margin-bottom: 36px;">
  <a href="https://github.com/erfanzar/EasyDeL">
   <img src="https://raw.githubusercontent.com/erfanzar/EasyDeL/main/images/easydel-logo-with-text.png" height="80" alt="EasyDeL">
  </a>
 </div>
 <div>
  <a href="https://discord.gg/FCAMNqnGtt"><img src="https://raw.githubusercontent.com/erfanzar/EasyDeL/main/images/discord-button.png" height="44" alt="Discord"></a>&nbsp;&nbsp;
  <a href="https://easydel.readthedocs.io/en/latest/"><img src="https://raw.githubusercontent.com/erfanzar/EasyDeL/main/images/documentation-button.png" height="44" alt="Documentation"></a>&nbsp;&nbsp;
  <a href="https://easydel.readthedocs.io/en/latest/install.html"><img src="https://raw.githubusercontent.com/erfanzar/EasyDeL/main/images/quick-start-button.png" height="44" alt="Quick Start"></a>
 </div>
</div>

---

# EasyDeL

EasyDeL is an open-source JAX framework for training, post-training, and serving large language, multimodal, and speech models on TPU pods and GPU clusters. It is built on [SpectraX](https://github.com/erfanzar/Spectrax) (no Flax dependency), uses [eJKernel](https://github.com/erfanzar/ejkernel) for TPU Pallas and GPU Triton/CUDA kernels, pairs Python model definitions with a native Rust host runtime ([surgecore](https://github.com/erfanzar/esurgecore)) for serving, and ships with [Eyvan](https://github.com/erfanzar/eyvan) for policy-aware TPU/GPU fleet scheduling, job execution, and experiment tracking.

## Contents

- [Workspace Stack](#workspace-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Inference with eSurge](#inference-with-esurge)
  - [Multi-Host OpenAI Serving](#multi-host-openai-serving)
  - [Unified Config with eLargeModel](#unified-config-with-elargemodel)
  - [Fine-Tuning and RL Training](#fine-tuning-and-rl-training)
- [Inference Engine (eSurge)](#inference-engine-esurge)
- [Trainers and RL Environments](#trainers-and-rl-environments)
- [Supported Models](#supported-models)
- [Parallelism, Kernels, and Quantization](#parallelism-kernels-and-quantization)
- [Customizing Models](#customizing-models)
- [Documentation and Community](#documentation-and-community)
- [Citation and License](#citation-and-license)

## Workspace Stack

This repository is a `uv` workspace monorepo. Installing `easydel` automatically installs all six companion packages (`spectrax-lib`, `ejkernel`, `eformer`, `eray`, `surgecore`, and `eyvan`). Each sibling library is developed here, kept independent of `easydel` (enforced by `lint-imports`), mirrored to its own repository, and released independently on PyPI. See [WORKSPACE.md](https://github.com/erfanzar/EasyDeL/blob/main/WORKSPACE.md) for layout and development details.

| Package | Path | PyPI | Role |
| --- | --- | --- | --- |
| **EasyDeL** | `libs/easydel` | `easydel` | Model zoo, trainers, RL environments, data pipeline, eSurge, `eLargeModel` |
| **SpectraX** | `libs/spectrax` | `spectrax-lib` | `spx.Module` object model, transforms, MPMD/SPMD pipeline runtime (9+ schedules) |
| **eJKernel** | `libs/ejkernel` | `ejkernel` | Pallas (TPU/GPU), Triton, CUDA, and XLA kernels (attention, MLA, KDA, GDN, MoE) |
| **eFormer** | `libs/eformer` | `eformer` | `escale` sharding meshes, optimizers, TensorStore checkpointing, implicit quantized arrays |
| **eRay** | `libs/eray` | `eray` | Ray pod/slice orchestration, `eray tpu job` / `sync` CLI, GCE helpers |
| **surgecore** | `libs/surgecore` | `surgecore` | Rust (PyO3) host runtime for eSurge: step builder, KV-cache core, scheduler, detokenizer |
| **Eyvan** | `libs/eyvan` | `eyvan` | Policy-aware fleet capacity plane: scheduler, worker runtime, experiment tracking, web dashboard |
| **esurge-dashboard** | `libs/esurge-dashboard` | — | Standalone Rust/Axum live serving workbench and metrics dashboard |

## Installation

Python 3.11–3.13 is required.

```bash
# Base install (automatically installs spectrax-lib, ejkernel, eformer, eray, surgecore, and eyvan)
uv pip install easydel

# TPU (JAX TPU + Pallas kernels)
uv pip install "easydel[tpu]"

# NVIDIA GPU (JAX CUDA + Triton/CUDA kernels + PyTorch bridge)
uv pip install "easydel[cuda]"

# Optional extras
uv pip install "easydel[torch]"     # PyTorch checkpoint conversion
uv pip install "easydel[mm]"        # Multimodal / Pillow support
uv pip install "easydel[lm_eval]"   # lm-evaluation-harness integration
uv pip install "easydel[profile]"   # XProf / TensorBoard profiling
```

To develop from source inside the monorepo (requires a stable Rust toolchain on `PATH` to build `libs/surgecore`):

```bash
git clone https://github.com/erfanzar/EasyDeL.git
cd EasyDeL
uv sync --extra tpu   # or --extra cuda
```

## Quick Start

### Inference with eSurge

Models load directly from Hugging Face (including PyTorch checkpoints and `gs://` paths) with 6D mesh sharding `(pp, dp, fsdp, ep, tp, sp)`:

```python
import jax.numpy as jnp
from transformers import AutoTokenizer
import easydel as ed

model_id = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, 1, -1, 1),  # (pp, dp, fsdp, ep, tp, sp)
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3,
        attn_dtype=jnp.bfloat16,
    ),
)

engine = ed.eSurge(
    model=model,
    processor=tokenizer,
    runtime={"max_model_len": 8192, "max_num_seqs": 32},
    cache={"hbm_utilization": 0.85, "enable_prefix_caching": True},
)

for out in engine.stream(
    "Explain paged KV caches concisely.",
    sampling_params=ed.SamplingParams(max_tokens=256, temperature=0.7),
):
    print(out.delta_text, end="", flush=True)
```

### Multi-Host OpenAI Serving

For multi-host TPU slices and DP-replica deployments, `easydel.inference.esurge.serving` launches an OpenAI-compatible server (and `esurge-dashboard`) from a single typed YAML config:

```bash
# Validate config off-slice
ENABLE_DISTRIBUTED_INIT=0 python -m easydel.inference.esurge.serving check --config serve.yaml

# Sync tree and launch across every host of a TPU slice
eray tpu sync ~/EasyDeL ~/EasyDeL --exclude .git --exclude target
eray tpu job start --name esurge -- \
  ~/venv/bin/python -m easydel.inference.esurge.serving serve --config ~/serve.yaml
```

Single-host servers can also be started directly in Python via `ed.eSurgeApiServer({"qwen3-8b": engine}, require_api_key=True).run(host="0.0.0.0", port=8000)` and queried with the standard `openai` client (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, `/metrics`, `/health`). See [docs/esurge_serving.md](https://github.com/erfanzar/EasyDeL/blob/main/libs/easydel/docs/esurge_serving.md).

### Unified Config with `eLargeModel`

`eLargeModel` wraps model loading, sharding, quantization, data mixtures, eSurge serving, training, and `lm-eval` into one serializable config:

```python
import easydel as ed

elm = (
    ed.eLargeModel.from_pretrained("Qwen/Qwen3-8B")
    .set_dtype("bf16")
    .set_sharding(axis_dims=(1, 1, 1, 1, -1, 1), axis_names=("pp", "dp", "fsdp", "ep", "tp", "sp"))
    .set_esurge(max_model_len=8192, max_num_seqs=32, hbm_utilization=0.85)
)

esurge = elm.build_esurge()
elm.to_json("run_config.json")
```

### Fine-Tuning and RL Training

#### Supervised Fine-Tuning (SFT)

```python
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer
import easydel as ed

model_id = "Qwen/Qwen3-8B"
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
)

trainer = ed.SFTTrainer(
    model=model,
    arguments=ed.SFTConfig(
        max_length=2048,
        dataset_text_field="text",
        packing=True,
        total_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        save_directory="./checkpoints",
    ),
    train_dataset=load_dataset("timdettmers/openassistant-guanaco", split="train"),
    processing_class=AutoTokenizer.from_pretrained(model_id),
)
trainer.train()
```

#### Online RL (GRPO and Multi-Turn `AgentFlowTrainer`)

Online RL trainers generate rollouts through eSurge by default, support unified `async_mode=True` with deadline-aware admission, and compose with sandboxed environments via `AgentFlowTrainer`:

```python
import easydel as ed

trainer = ed.AgentFlowTrainer(
    model=model,
    arguments=ed.AgentFlowConfig(
        base_trainer_type="grpo",
        environment="code_judge",
        environment_backend="local",  # or "kiri" / "docker"
        max_turns=6,
        group_size=4,
        max_prompt_length=2048,
        max_completion_length=2048,
        learning_rate=1e-6,
        total_batch_size=16,
    ),
    train_dataset=tasks_dataset,
    processing_class=tokenizer,
)
trainer.train()
```

## Inference Engine (eSurge)

- **Native Rust host core (`libs/surgecore`):** Per-step input construction, paged KV-cache pool/prefix cache/recurrent-snapshot allocator, `EngineCore` scheduler, and batched in-process detokenization run in Rust with zero Python scheduler overhead on the decode loop.
- **Scheduling and compilation:** Continuous batching with whole-prompt prefill admission, bucket-aligned chunked prefill, async mixed prefill+decode steps, upfront full-bucket compilation of helper programs, opt-in `freeze_startup_heap` (`gc.freeze`), and exact preemption resume.
- **Speculative decoding:** Registered drafters (`mtp`, `eagle3`, `dspark`, `dflash`, `gemma4_assistant`, GLM-5-Next `nextn`), flat ragged KDA verification without candidate replay, replay-free compressed-window rollback, and jitted rejection sampling.
- **Multi-host and DP serving:** MPMD pipeline-parallel serving, ZMQ step-replication and request planes, independent `dp > 1` replica worlds behind a host-only DP front router (prefix affinity yielding to shortest queue), and `libs/esurge-dashboard`.
- **OpenAI API and parsers:** Streaming `/v1/chat/completions`, `/v1/completions`, and `/v1/responses`, sparse per-request sampler constraints (`logit_bias`, allowed/banned masks, `min_tokens`, `ignore_eos`), 25+ model-specific tool-call and reasoning parsers, digest-only header API-key auth with RBAC, and `vWhisper` for speech transcription.
- **Online RL integration:** Monotonic `weight_version` provenance across `keep_requests=True` weight swaps and stale-request eviction (`abort_requests_before_version`).

## Trainers and RL Environments

All trainers share one compiled-step foundation (`BaseTrainer`) supporting 6D SPMD sharding, SpectraX MPMD pipeline schedules, denominator-weighted gradient accumulation, host-side batch prefetching, LoRA, fused optimizers (`AdamW`, `Lion`, `RMSProp`, `Muon`), and TensorStore checkpointing.

| Category | Trainers |
| --- | --- |
| **Supervised & Embeddings** | `SFTTrainer`, `Trainer`, `EmbeddingTrainer` |
| **Preference Optimization** | `DPOTrainer`, `OnlineDPOTrainer`, `BEMADPOTrainer`, `ORPOTrainer`, `CPOTrainer`, `KTOTrainer`, `BCOTrainer`, `XPOTrainer`, `TPOTrainer`, `NashMDTrainer` |
| **Online RL & Policy Gradient** | `GRPOTrainer`, `AsyncGRPOTrainer`, `GRPOWithReplayBufferTrainer`, `GMPOTrainer`, `GDPOTrainer`, `GSPOTrainer`, `GSPOTokenTrainer`, `GFPOTrainer`, `DPPOTrainer`, `RLOOTrainer`, `PAPOTrainer`, `MiniLLMTrainer`, `SSDTrainer`, `ReinforcePlusPlusTrainer`, `PPOTrainer`, `VAPOTrainer`, `SAOTrainer`, `CompactionRLTrainer`, `RLVRTrainer`, `NeMoGymTrainer` |
| **Multi-Turn & Agentic RL** | `AgentFlowTrainer`, `AgenticMoshPitTrainer` |
| **Distillation** | `DistillationTrainer`, `GKDTrainer`, `SeqKDTrainer`, `OnPolicyDistillationTrainer`, `SDFTTrainer` (`SelfDistillationTrainer`), `SDPOTrainer`, `GOLDTrainer`, `SparseDistillationTrainer` |
| **Reward & Drafters** | `RewardTrainer`, `PRMTrainer`, `SpeculativeDecodingTrainer`, `RayDistributedTrainer` |

Key RL and distillation options include IcePop importance-sampling ratios, DAPO dynamic sampling, Kimi K2.5 Toggle, unscorable-group discard, fail-closed off-policy checks, prefix-shared GRPO forward (dense and hybrid `qwen3_next`), vocab-streamed OPD divergences with memory-neutral Bregman forward-KL and trust-region mode, unified `async_mode`, and opt-in versioned `WeightTransport` (TCP + JAX Transfer Server).

### Sandboxed RL Environments (`easydel.environments`)

- **Environments:** `SWERepoEnvironment` (parent-child isolated grading, `git archive` seeding), `TerminalEnvironment`, `CodeJudgeEnvironment`, `StatefulPythonEnvironment`, `SyntheticToolEnvironment`, and `HarnessEnvironment` + `HarnessTITOGateway` (Token-In-Token-Out stitching for Anthropic and OpenAI APIs).
- **Judges, rewards, and collectors:** `RubricJudge`, `PairwiseJudge`, `RewardShaper`, `AutoEnvironment`, `SyntheticTaskStream`, `DifficultyTracker`, and `AsyncEnvironmentRolloutCollector`.
- **Sandboxes and CLI:** `LocalSandboxSession` (with optional `bwrap` jail), `ContainerSandboxSession`, `KiriRuntime` + `SandboxPool` (`shm:N` / `pod-shm:N` RAM-backed pools), and `easydel setup kiri` / `easydel setup cpu-vm`.

See [docs/trainers/environments.md](https://github.com/erfanzar/EasyDeL/blob/main/libs/easydel/docs/trainers/environments.md), [docs/trainers/agent_flow.md](https://github.com/erfanzar/EasyDeL/blob/main/libs/easydel/docs/trainers/agent_flow.md), and [docs/trainers/rl_safety.md](https://github.com/erfanzar/EasyDeL/blob/main/libs/easydel/docs/trainers/rl_safety.md).

## Supported Models

EasyDeL includes **88 model families** under `libs/easydel/easydel/modules/` accessible through `AutoEasyDeLModelForCausalLM`, `AutoEasyDeLModelForImageTextToText`, `AutoEasyDeLModelForSeq2SeqLM`, `AutoEasyDeLModelForSequenceClassification`, `AutoEasyDeLModelForSpeechSeq2Seq`, `AutoEasyDeLModelForZeroShotImageClassification`, and `AutoEasyDeLVisionModel`:

| Category | Families |
| --- | --- |
| **Dense & MoE LLMs** | Llama, Llama 4, Qwen 2/2-MoE/3/3-MoE/3.5/3.5-MoE/4-Exp, DeepSeek V2/V3/V4/V4.1, GLM / GLM-4 / GLM-4-MoE / GLM-4-MoE-Lite / GLM-MoE-DSA / GLM-5-Next, Gemma 1/2/3/4, Mistral / Mistral 3 / Mistral 4 / Mixtral, Hunyuan V3 (`hy_v3`), Phi / Phi-3 / PhiMoE, Cohere / Cohere 2, OLMo 1/2/3, Grok-1, DBRX, Arctic, Exaone / Exaone 4, Falcon / Falcon-H1, InternLM2, MiniMax, MosaicMPT, OpenELM, SmolLM3, Seed-OSS, StableLM, GPT-2 / GPT-J / GPT-NeoX / GPT-OSS, OPT, RoBERTa, Xerxes / Xerxes 2 |
| **Linear Attention & SSM** | Qwen3-Next, Kimi-Linear, GLM-5-Next, Mamba, Mamba 2, Falcon-Mamba, RWKV |
| **Vision-Language & Omni** | Llama 4 Vision, Qwen2-VL, Qwen3-VL / Qwen3-VL-MoE, Qwen3-Omni-MoE, Qwen3.5, Qwen4-Exp, DeepSeek-V4.1, GLM-4V / GLM-4V-MoE / GLM-4.6V / GLM-5-Next, Gemma 3/4, Mistral 3 (Pixtral), MiniMax-M3-VL, Hunyuan-VL, InternVL, Kimi-VL, PaliGemma, Muse-Glimmer, AyaVision, LLaVA |
| **Vision, Speech & Diffusion** | CLIP, SigLIP, Pixtral, Whisper, GIDD |
| **Speculative Drafters** | Eagle 3 (`eagle3`), DSpark (`dspark`), DFlash (`dflash`), Gemma 4 Assistant (`gemma4_assistant`), MTP / `nextn` heads |

## Parallelism, Kernels, and Quantization

- **6D Sharding & Pipeline Parallelism:** Models shard across `(pp, dp, fsdp, ep, tp, sp)` via `eformer.escale` and `EasyShardingMixin`. Setting `pp > 1` switches to SpectraX's MPMD pipeline runtime (GPipe, 1F1B, Eager 1F1B, ZeroBubble H1, Interleaved schedules, KimiK2, DualPipeV).
- **Attention & Custom Kernels (`ejkernel`):** `flash_attn2`, `ring`, `blocksparse` (Splash), `sdpa`, `ragged_page_attention_v2`/`v3`, `compressed_window_attention`, sparse MLA + `paged_pool_topk`, `ragged_kda` / `kda_chunk` / `ragged_gdn`, `state_space_v1`/`v2`, Sinkhorn-Knopp / mHC kernels, and A8 slab / channelwise grouped MoE matmuls.
- **Quantization & LoRA:** Decode pre-quantized Hugging Face checkpoints on load (FP8, AWQ, GPTQ, MXFP4, NVFP4, compressed-tensors, DeepSeek-V4), apply post-training quantization (`NF4`, `INT8`, `W8A8`) or module-path QAT, and attach/merge LoRA adapters with `model.apply_lora_to_layers(...)` / `model.unwrap_lora_to_layers()`.

## Customizing Models

Every model is a plain `spx.Module` (`EasyDeLBaseModule`) with explicit `GraphDef`/`State` separation:

```python
import jax.numpy as jnp
from spectrax import nn
import easydel as ed


class CustomBlock(ed.EasyDeLBaseModule):
    def __init__(self, config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, precision=None, *, rngs: nn.Rngs):
        super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x):
        return self.proj(x)
```

Runtime attention mechanisms can be switched via `model.update_module(attn_mechanism="flash_attn2")`, and layers can be replaced directly on the module tree before calling `save_pretrained` or `push_to_hub`.

## Documentation and Community

- **Documentation:** [easydel.readthedocs.io](https://easydel.readthedocs.io/en/latest/)
- **Workspace Guide:** [WORKSPACE.md](https://github.com/erfanzar/EasyDeL/blob/main/WORKSPACE.md)
- **Eyvan Fleet Plane:** [libs/eyvan/README.md](https://github.com/erfanzar/EasyDeL/blob/main/libs/eyvan/README.md)
- **Serving Guide:** [docs/esurge_serving.md](https://github.com/erfanzar/EasyDeL/blob/main/libs/easydel/docs/esurge_serving.md)
- **Environment Variables:** [docs/environment_variables.md](https://github.com/erfanzar/EasyDeL/blob/main/libs/easydel/docs/environment_variables.md)
- **RL Environments & AgentFlow:** [environments.md](https://github.com/erfanzar/EasyDeL/blob/main/libs/easydel/docs/trainers/environments.md), [agent_flow.md](https://github.com/erfanzar/EasyDeL/blob/main/libs/easydel/docs/trainers/agent_flow.md)
- **Discord:** [discord.gg/FCAMNqnGtt](https://discord.gg/FCAMNqnGtt)
- **Issues:** [github.com/erfanzar/EasyDeL/issues](https://github.com/erfanzar/EasyDeL/issues)
- **Contact:** <erfanzare810@gmail.com>

## Citation and License

EasyDeL is released under the Apache License 2.0 (see [LICENSE](https://github.com/erfanzar/EasyDeL/blob/main/LICENSE)). Note that `spectrax-lib` and `eyvan` are licensed under AGPL-3.0-or-later.

```bibtex
@misc{Zare Chavoshi_2023,
    title={EasyDeL: An open-source library for enhancing and streamlining the training process of machine learning models},
    url={https://github.com/erfanzar/EasyDeL},
    author={Zare Chavoshi, Erfan},
    year={2023}
}
```

## Acknowledgments

- Built on [JAX](https://github.com/google/jax) and [SpectraX](https://github.com/erfanzar/Spectrax)
- Model APIs inspired by [HuggingFace Transformers](https://github.com/huggingface/transformers)
- Trainer APIs inspired by [TRL](https://github.com/huggingface/trl)
- Serving architecture inspired by [vLLM](https://github.com/vllm-project/vllm)

---

- Made with <3 by the EasyDeL Author ; /
