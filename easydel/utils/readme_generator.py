# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for generating model cards / README files.

This module is used in a few different contexts:
- HuggingFace Hub model card generation via `EasyBridgeMixin`
- Training run summary README generation via `EASYDEL_TRAINER_README_TEMPLATE`
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

try:
    from eformer.loggings import get_logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    def get_logger(name: str | None = None) -> logging.Logger:
        return logging.getLogger(name or __name__)


try:
    from eformer.paths import ePath
except ModuleNotFoundError:  # pragma: no cover
    ePath = Path  # type: ignore[assignment]

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

DEFAULT_MODEL_CARD_TEMPLATE_NAME = "README.md.jinja"

logger = get_logger(__name__)


def _collapse_extra_blank_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text)


def _get_default_easydel_version() -> str:
    """Best-effort EasyDeL version without importing the whole package."""
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("easydel")
    except Exception:
        pass

    try:
        import importlib.util

        spec = importlib.util.find_spec("easydel")
        if spec and spec.origin:
            init_text = Path(spec.origin).read_text(encoding="utf-8")
            match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", init_text, flags=re.MULTILINE)
            if match:
                return match.group(1)
    except Exception:
        pass

    try:
        init_path = Path(__file__).resolve().parents[1] / "__init__.py"
        if init_path.exists():
            init_text = init_path.read_text(encoding="utf-8")
            match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", init_text, flags=re.MULTILINE)
            if match:
                return match.group(1)
    except Exception:
        pass

    return "unknown"


def _slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[\s_]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value


def _normalize_identifier(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[\s\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value


_TASK_DISPLAY_BY_SLUG: dict[str, str] = {
    "causallm": "CausalLM",
    "causal-lm": "CausalLM",
    "causal-language-model": "CausalLM",
    "sequenceclassification": "SequenceClassification",
    "sequence-classification": "SequenceClassification",
    "image-text-to-text": "ImageTextToText",
    "sequence-to-sequence": "Seq2SeqLM",
    "speech-sequence-to-sequence": "SpeechSeq2Seq",
    "zero-shot-image-classification": "ZeroShotImageClassification",
    "diffusion-language-model": "DiffusionLM",
    "base-module": "Base",
    "vision-module": "Vision",
    "any-to-any": "AnyToAny",
    "auto-bind": "Auto",
}

_AUTO_CLASS_BY_TASK_DISPLAY: dict[str, str] = {
    "CausalLM": "AutoEasyDeLModelForCausalLM",
    "SequenceClassification": "AutoEasyDeLModelForSequenceClassification",
    "ImageTextToText": "AutoEasyDeLModelForImageTextToText",
    "Seq2SeqLM": "AutoEasyDeLModelForSeq2SeqLM",
    "SpeechSeq2Seq": "AutoEasyDeLModelForSpeechSeq2Seq",
    "ZeroShotImageClassification": "AutoEasyDeLModelForZeroShotImageClassification",
    "DiffusionLM": "AutoEasyDeLModelForDiffusionLM",
    "Base": "AutoEasyDeLModel",
    "Vision": "AutoEasyDeLVisionModel",
    "AnyToAny": "AutoEasyDeLAnyToAnyModel",
}

_PIPELINE_TAG_BY_TASK_DISPLAY: dict[str, str] = {
    "CausalLM": "text-generation",
    "SequenceClassification": "text-classification",
    "Seq2SeqLM": "text2text-generation",
    "SpeechSeq2Seq": "automatic-speech-recognition",
    "ImageTextToText": "image-to-text",
    "ZeroShotImageClassification": "zero-shot-image-classification",
}

_ATTN_ENUM_BY_MECHANISM_KEY: dict[str, str] = {
    "auto": "AUTO",
    "vanilla": "VANILLA",
    "flash": "FLASH_ATTN2",
    "flash_attn": "FLASH_ATTN2",
    "flash_attention": "FLASH_ATTN2",
    "flashattn": "FLASH_ATTN2",
    "flash_attn2": "FLASH_ATTN2",
    "flash_attention2": "FLASH_ATTN2",
    "flashattn2": "FLASH_ATTN2",
    "blocksparse": "SPLASH",
    "splash_attn": "SPLASH",
    "splash_attention": "SPLASH",
    "ring": "RING",
    "ring_attn": "RING",
    "ring_attention": "RING",
    "sdpa": "SDPA",
    "cudnn": "CUDNN",
    "blockwise": "BLOCKWISE",
    "cuda_flash_attn2": "CUDA_FLASH_ATTN2",
    "paged": "RAGGED_PAGE_ATTENTION_V3",
    "paged_attn": "RAGGED_PAGE_ATTENTION_V3",
    "paged_attention": "RAGGED_PAGE_ATTENTION_V3",
    "page_attention": "PAGED_ATTENTION",
    "ragged_page_attention_v2": "RAGGED_PAGE_ATTENTION_V2",
    "ragged_page_attention_v3": "RAGGED_PAGE_ATTENTION_V3",
    "autoregressive_decodeattn": "REGRESSIVE_DECODE",
    "autoregressive_decode": "REGRESSIVE_DECODE",
}


EASYDEL_TRAINER_README_TEMPLATE = """
{%- set task_key_raw = (model.model_task_str | string) %}
{%- set task_key = (task_key_raw | lower | replace('-', '') | replace('_', '')) %}
{%- set auto_class_map = {
    "causallm": "AutoEasyDeLModelForCausalLM",
    "sequenceclassification": "AutoEasyDeLModelForSequenceClassification",
    "imagetexttotext": "AutoEasyDeLModelForImageTextToText",
    "sequencetosequence": "AutoEasyDeLModelForSeq2SeqLM",
    "speechseq2seq": "AutoEasyDeLModelForSpeechSeq2Seq",
    "zeroshotimageclassification": "AutoEasyDeLModelForZeroShotImageClassification",
    "diffusionlm": "AutoEasyDeLModelForDiffusionLM",
} %}
{%- set auto_class = auto_class_map.get(task_key, "AutoEasyDeLModelForCausalLM") %}
{%- set pipeline_tag_map = {
    "causallm": "text-generation",
    "sequenceclassification": "text-classification",
    "imagetexttotext": "image-to-text",
    "sequencetosequence": "text2text-generation",
    "speechseq2seq": "automatic-speech-recognition",
    "zeroshotimageclassification": "zero-shot-image-classification",
} %}
{%- set pipeline_tag = pipeline_tag_map.get(task_key) %}
{%- set shields_style = "flat-square" %}
{%- set attn_enum_map = {
    "auto": "AUTO",
    "vanilla": "VANILLA",
    "flash": "FLASH_ATTN2", "flash_attn": "FLASH_ATTN2", "flash_attention": "FLASH_ATTN2", "flashattn": "FLASH_ATTN2",
    "flash_attn2": "FLASH_ATTN2", "flash_attention2": "FLASH_ATTN2", "flashattn2": "FLASH_ATTN2",
    "blocksparse": "SPLASH", "splash_attn": "SPLASH", "splash_attention": "SPLASH",
    "ring": "RING", "ring_attn": "RING", "ring_attention": "RING",
    "sdpa": "SDPA",
    "cudnn": "CUDNN",
    "blockwise": "BLOCKWISE",
    "cuda_flash_attn2": "CUDA_FLASH_ATTN2",
    "paged": "RAGGED_PAGE_ATTENTION_V3", "paged_attn": "RAGGED_PAGE_ATTENTION_V3", "paged_attention": "RAGGED_PAGE_ATTENTION_V3",
    "page_attention": "PAGED_ATTENTION",
    "ragged_page_attention_v3": "RAGGED_PAGE_ATTENTION_V3", "ragged_page_attention_v2": "RAGGED_PAGE_ATTENTION_V2",
    "autoregressive_decodeattn": "REGRESSIVE_DECODE", "autoregressive_decode": "REGRESSIVE_DECODE"
} %}
{%- set attn_enum = attn_enum_map.get(model.attn_mechanism_str.lower(), "AUTO") %}
{%- set dtype_lower = (model.dtype_str | string | lower) %}
{%- set param_dtype_lower = (model.param_dtype_str | string | lower) %}
{%- set jnp_dtype_for_template = (
    'bfloat16' if 'bfloat16' in dtype_lower else
    'float32' if 'float32' in dtype_lower else
    'float64' if 'float64' in dtype_lower else
    'float16'
) %}
{%- set jnp_param_dtype_for_template = (
    'bfloat16' if 'bfloat16' in param_dtype_lower else
    'float32' if 'float32' in param_dtype_lower else
    'float64' if 'float64' in param_dtype_lower else
    'float16'
) %}

---
library_name: easydel
{% if pipeline_tag %}
pipeline_tag: {{ pipeline_tag }}
{% endif %}
tags:
  - easydel
  - jax
  - training
  - "{{ model.architecture }}"
  - "{{ model.model_task_str }}"
  - "{{ model.attn_mechanism_str }}"
{% if model.device_info.platform != "UNKNOWN" %}
  - "{{ model.device_info.platform }}"
{% endif %}
---

<p align="center">
  <a href="https://github.com/erfanzar/EasyDeL">
    <img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/easydel-logo-with-text.png" height="80" alt="EasyDeL" />
  </a>
</p>

<h1 align="center">Training Run: {{ model.name }}</h1>

<p align="center">
  <a href="https://github.com/erfanzar/EasyDeL">
    <img src="https://img.shields.io/static/v1?label=EasyDeL&message=v{{ easydel_version }}&color=blue&style={{ shields_style }}" alt="EasyDeL Version" />
  </a>
  <img src="https://img.shields.io/static/v1?label=Arch&message={{ model.architecture | urlencode }}&color=0A66C2&style={{ shields_style }}" alt="Model Architecture" />
  <img src="https://img.shields.io/static/v1?label=Task&message={{ model.model_task_str | urlencode }}&color=green&style={{ shields_style }}" alt="Task" />
  <img src="https://img.shields.io/static/v1?label=Attention&message={{ model.attn_mechanism_str | urlencode }}&color=8A2BE2&style={{ shields_style }}" alt="Attention Mechanism" />
</p>

## Run Summary

| Key | Value |
| --- | --- |
| Model | `{{ model.name }}` |
| Architecture | `{{ model.architecture }}` |
| Task | `{{ model.model_task_str }}` |
| Attention | `{{ model.attn_mechanism_str }}` (`AttentionMechanisms.{{ attn_enum }}`) |
| Platform | `{{ model.device_info.platform }}` |
| Devices (total/local) | `{{ model.device_info.device_count }}` / `{{ model.device_info.host_device_count }}` |
| Dtype / Param dtype | `{{ model.dtype_str }}` / `{{ model.param_dtype_str }}` |
| EasyDeL | `v{{ easydel_version }}` |

## Load This Checkpoint

```python
import easydel as ed
from jax import numpy as jnp, lax

repo_id = "user/model-id"  # TODO: set this to your local output directory or HF repo ID

dtype = jnp.{{ jnp_dtype_for_template }}
param_dtype = jnp.{{ jnp_param_dtype_for_template }}

model = ed.{{ auto_class }}.from_pretrained(
    repo_id,
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_dtype=dtype,
        attn_mechanism=ed.AttentionMechanisms.{{ attn_enum }},
    ),
    dtype=dtype,
    param_dtype=param_dtype,
    precision=lax.Precision("fastest"),
    auto_shard_model=True,
)
```

## Sharding Notes

EasyDeL typically shards across a 5D logical mesh: `("dp","fsdp","ep","tp","sp")`.

- The product of `sharding_axis_dims` should match your device count; you can use `-1` to infer the remaining dimension.
- `fsdp` is commonly the largest axis to reduce memory usage.
- For non-MoE models keep `ep=1`.

<details>
<summary>Example sharding configs</summary>

```text
8 devices: (1, 8, 1, 1, 1)  # pure FSDP
8 devices: (2, 4, 1, 1, 1)  # 2-way DP x 4-way FSDP
8 devices: (1, 4, 1, 2, 1)  # 4-way FSDP x 2-way TP
```
</details>

## Using via `eLargeModel` (ELM)

```python
from easydel import eLargeModel

elm = eLargeModel.from_pretrained(repo_id)
elm.set_dtype("bf16")
elm.set_sharding(axis_names=("dp", "fsdp", "ep", "tp", "sp"), axis_dims=(1, -1, 1, 1, 1))

model = elm.build_model()
# engine = elm.build_esurge()
```

## Hyperparameters

| Key | Value |
| --- | --- |
| Learning rate | `{{ arguments.learning_rate }}`{% if arguments.learning_rate_end is not none and arguments.learning_rate_end != arguments.learning_rate %} -> `{{ arguments.learning_rate_end }}`{% endif %} |
| Optimizer | `{{ arguments.optimizer }}` |
| Scheduler | `{{ arguments.scheduler }}` |
| Warmup steps | `{{ arguments.warmup_steps }}` |
| Weight decay | `{{ arguments.weight_decay }}` |
| Loss config | `{{ arguments.loss_config | string }}` |
| Epochs | `{{ arguments.num_train_epochs }}` |
| Total batch size | `{{ arguments.total_batch_size }}` |
| Max length | `{{ arguments.max_length }}` |
| Grad accumulation | `{{ arguments.gradient_accumulation_steps }}` |
| Gradient checkpointing | `{{ config.gradient_checkpointing }}` |
| Max training steps | `{{ arguments.max_training_steps if arguments.max_training_steps is not none else "Not Set" }}` |
| Max eval steps | `{{ arguments.max_evaluation_steps if arguments.max_evaluation_steps is not none else "Not Set" }}` |
| Training time limit | `{{ arguments.training_time_limit if arguments.training_time_limit is not none else "Not Set" }}` |

{% if model.partition_rules_str %}
<details>
<summary>Partition rules</summary>

```text
{{ model.partition_rules_str }}
```
</details>
{% endif %}

## Citation

```bibtex
@misc{Zare Chavoshi_2023,
    title={EasyDeL: An open-source library for enhancing and streamlining the training process of machine learning models},
    url={https://github.com/erfanzar/EasyDeL},
    author={Zare Chavoshi, Erfan},
    year={2023}
}
```

---
Generated by EasyDeL v{{ easydel_version }}.
"""


JINJA_TEMPLATE = """
{% set model_type = model.model_type_for_display %}
{% set task = model.task_display %}
{% set pipeline_tag = model.pipeline_tag %}
{% set auto_class = model.auto_class %}
{% set attn_enum = model.attn_enum %}
{% set shields_style = "flat-square" %}

---
library_name: easydel
{% if pipeline_tag %}
pipeline_tag: {{ pipeline_tag }}
{% endif %}
tags:
  - easydel
  - jax
  - "{{ model_type }}"
  - "{{ task }}"
  - "{{ model.attn_mechanism }}"
---

<p align="center">
  <img alt="EasyDeL" src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/easydel-logo-with-text.png" height="80">
</p>

<h1 align="center">{{ model.name }}</h1>

<div align="center">
  {{ model.description if model.description else "A model compatible with the EasyDeL JAX stack." }}
</div>


## Overview

{{ model.overview if model.overview else "This checkpoint is intended to be loaded with EasyDeL on JAX (CPU/GPU/TPU). It supports sharded loading with `auto_shard_model=True` and configurable precision via `dtype`, `param_dtype`, and `precision`." }}

## Quickstart

```python
import easydel as ed
from jax import numpy as jnp, lax

repo_id = "{{ model.repo_id }}"

dtype = jnp.bfloat16  # try jnp.float16 on many GPUs

model = ed.{{ auto_class }}.from_pretrained(
    repo_id,
    dtype=dtype,
    param_dtype=dtype,
    precision=lax.Precision("fastest"),
    sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
    sharding_axis_dims=(1, -1, 1, 1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_dtype=dtype,
        attn_mechanism=ed.AttentionMechanisms.{{ attn_enum }},
        fsdp_is_ep_bound=True,
        sp_is_ep_bound=True,
        moe_method=ed.MoEMethods.FUSED_MOE,
    ),
    auto_shard_model=True,
    partition_axis=ed.PartitionAxis(),
)
```

If the repository only provides PyTorch weights, pass `from_torch=True` to `from_pretrained(...)`.

## Sharding & Parallelism (Multi-Device)

EasyDeL can scale to multiple devices by creating a logical device mesh. Most EasyDeL loaders use a 5D mesh:

- `dp`: data parallel (replicated parameters, different batch shards)
- `fsdp`: parameter sharding (memory saver; often the biggest axis)
- `ep`: expert parallel (MoE; keep `1` for non-MoE models)
- `tp`: tensor parallel (splits large matmuls)
- `sp`: sequence parallel (splits sequence dimension)

Use `sharding_axis_names=("dp","fsdp","ep","tp","sp")` and choose `sharding_axis_dims` so that their product equals your device count.
You can use `-1` in `sharding_axis_dims` to let EasyDeL infer the remaining dimension.

<details>
<summary>Example sharding configs</summary>

```python
# 8 devices, pure FSDP
sharding_axis_dims = (1, 8, 1, 1, 1)

# 8 devices, 2-way DP x 4-way FSDP
sharding_axis_dims = (2, 4, 1, 1, 1)

# 8 devices, 4-way FSDP x 2-way TP
sharding_axis_dims = (1, 4, 1, 2, 1)
```
</details>

## Using via `eLargeModel` (ELM)

`eLargeModel` is a higher-level interface that wires together loading, sharding, training, and eSurge inference from a single config.

```python
from easydel import eLargeModel

repo_id = "{{ model.repo_id }}"

elm = eLargeModel.from_pretrained(repo_id)  # task is auto-detected
elm.set_dtype("bf16")
elm.set_sharding(axis_names=("dp", "fsdp", "ep", "tp", "sp"), axis_dims=(1, -1, 1, 1, 1))

model = elm.build_model()
# Optional: build an inference engine
# engine = elm.build_esurge()
```

<details>
<summary>ELM YAML config example</summary>

```yaml
model:
  name_or_path: "{{ model.repo_id }}"

loader:
  dtype: bf16
  param_dtype: bf16

sharding:
  axis_dims: [1, -1, 1, 1, 1]
  auto_shard_model: true
```
</details>

## Features

{% if model.features %}
**Model-specific:**
{% for feature in model.features %}
- {{ feature }}
{% endfor %}
{% endif %}

**EasyDeL:**
- JAX native implementation and sharded execution
- Configurable attention backends via `AttentionMechanisms.*`
- Precision control via `dtype`, `param_dtype`, and `precision`

## Installation

```bash
pip install easydel
```

## Links

- EasyDeL GitHub: https://github.com/erfanzar/EasyDeL
- Docs: https://easydel.readthedocs.io/en/latest/

## Supported Tasks

{% if model.supported_tasks %}
{% for supported_task in model.supported_tasks %}
- {{ supported_task }}
{% endfor %}
{% else %}
- {{ task }}
{% endif %}

## Limitations

{% if model.limitations %}
{% for limitation in model.limitations %}
- {{ limitation }}
{% endfor %}
{% else %}
- Refer to the original model card for training data, evaluation, and intended use.
{% endif %}

## License

EasyDeL is released under the Apache-2.0 license. The license for this model's weights may differ; please consult the original repository.

## Citation

```bibtex
@misc{Zare Chavoshi_2023,
    title={EasyDeL: An open-source library for enhancing and streamlining the training process of machine learning models},
    url={https://github.com/erfanzar/EasyDeL},
    author={Zare Chavoshi, Erfan},
    year={2023}
}
```
"""


@dataclass
class ModelInfo:
    """
    Model information container. Used to pass data to the Jinja template.
    """

    name: str = field(metadata={"help": "The name of the model."})
    type: str = field(metadata={"help": "The type of the model."})
    repo_id: str = field(metadata={"help": "The repository ID of the model."})
    description: str | None = field(default=None, metadata={"help": "A description of the model."})
    model_type: str | None = field(default=None, metadata={"help": "The model type."})
    model_task: str = field(
        default="CausalLM",
        metadata={"help": "The model task (e.g., CausalLM, SequenceClassification)."},
    )
    attn_mechanism: str = field(
        default="vanilla",
        metadata={"help": "The attention mechanism used (e.g., vanilla, flash_attn2)."},
    )
    features: list[str] | None = field(default=None, metadata={"help": "A list of features of the model."})
    supported_tasks: list[str] | None = field(default=None, metadata={"help": "A list of tasks supported by the model."})
    limitations: list[str] | None = field(default=None, metadata={"help": "A list of limitations of the model."})
    version: str = field(default_factory=_get_default_easydel_version, metadata={"help": "The version of EasyDeL."})
    overview: str | None = field(default=None, metadata={"help": "Custom overview text for the model."})

    def __post_init__(self):
        for attribute in ("model_type", "model_task", "attn_mechanism"):
            value = getattr(self, attribute, None)
            if hasattr(value, "value"):
                setattr(self, attribute, value.value)

    @property
    def model_type_for_display(self) -> str:
        return self.model_type or self.type

    @property
    def task_display(self) -> str:
        raw = str(self.model_task or "").strip() or "CausalLM"
        slug = _slugify(raw)
        return _TASK_DISPLAY_BY_SLUG.get(slug, raw)

    @property
    def pipeline_tag(self) -> str | None:
        return _PIPELINE_TAG_BY_TASK_DISPLAY.get(self.task_display)

    @property
    def auto_class(self) -> str:
        return _AUTO_CLASS_BY_TASK_DISPLAY.get(self.task_display, "AutoEasyDeLModel")

    @property
    def attn_enum(self) -> str:
        key = _normalize_identifier(self.attn_mechanism or "auto")
        return _ATTN_ENUM_BY_MECHANISM_KEY.get(key, "AUTO")

    @property
    def repo_is_local(self) -> bool:
        rid = str(self.repo_id or "")
        if not rid:
            return False
        if rid.startswith(("/", "./", "../")):
            return True
        if re.match(r"^[A-Za-z]:\\\\", rid):
            return True
        try:
            return Path(rid).exists()
        except Exception:
            return False

    @property
    def repo_badge_message(self) -> str:
        rid = str(self.repo_id or "")
        if self.repo_is_local:
            try:
                return Path(rid).name or rid
            except Exception:
                return rid
        return rid


class ReadmeGenerator:
    """Generate README files for EasyDeL models."""

    def __init__(self, template_dir: str | None = None, *, template_name: str | None = None):
        """
        Initialize the README generator.

        Args:
            template_dir: Optional directory to search for templates.
            template_name: Optional template filename to load from `template_dir`.
                If omitted, `README.md.jinja` will be tried before falling back
                to the built-in template string.
        """

        self.template_name = template_name
        search_paths = [template_dir] if template_dir and os.path.exists(template_dir) else []
        search_paths.append(os.path.dirname(__file__))
        self.env = Environment(
            loader=FileSystemLoader(search_paths),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        self.env.filters.setdefault("urlencode", lambda value: quote(str(value), safe=""))
        self._compiled_template = None

    def _get_template(self):
        if self._compiled_template is not None:
            return self._compiled_template

        candidate_names: list[str] = []
        if self.template_name:
            candidate_names.append(self.template_name)
        candidate_names.append(DEFAULT_MODEL_CARD_TEMPLATE_NAME)

        for name in candidate_names:
            try:
                self._compiled_template = self.env.get_template(name)
                return self._compiled_template
            except TemplateNotFound:
                continue

        self._compiled_template = self.env.from_string(JINJA_TEMPLATE)
        return self._compiled_template

    def generate_readme(
        self,
        model_info: ModelInfo,
        output_path: str | None = None,
    ) -> str:
        """
        Generate README content for a model.

        Args:
            model_info: Model information
            output_path: Optional path to save the README

        Returns:
            Generated README content
        """
        try:
            template = self._get_template()
            content = template.render(model=model_info)
            content = _collapse_extra_blank_lines(content).strip() + "\n"

            if output_path:
                output_path = ePath(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(content)
                logger.info(f"README saved to {output_path}")

            return content

        except Exception:
            logger.exception("Error generating README for %s", model_info.name)
            raise


# Example usage (for testing the generator script itself)
if __name__ == "__main__":
    # Create dummy model info for testing
    test_model_info = ModelInfo(
        name="TestModel-EasyDeL",
        type="TestLM",
        repo_id="test/test-model",
        model_task="CausalLM",
        attn_mechanism="flash_attn2",
        description="This is a test description for the example usage.",
        overview="This is a custom overview section for the test model.",
        supported_tasks=["Text Generation", "Summarization"],
        limitations=["May hallucinate.", "Requires significant compute."],
        features=["Feature A", "Feature B"],
    )

    # Generate the README
    generator = ReadmeGenerator()
    readme_content = generator.generate_readme(test_model_info, output_path="tmp-test-readme.md")
    print(f"Generated tmp-test-readme.md for model '{test_model_info.name}'")
    # print("\n--- README Content ---")
    # print(readme_content)
    # print("--- End README Content ---")
