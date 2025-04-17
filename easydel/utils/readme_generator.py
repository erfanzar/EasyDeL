# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
import os
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from easydel import __version__
from easydel.utils.helpers import get_logger

logger = get_logger(__name__)

JINJA_TEMPLATE = """
{% set auto_class_map = {
    "CausalLM": "AutoEasyDeLModelForCausalLM",
    "SequenceClassification": "AutoEasyDeLModelForSequenceClassification",
    "ImageTextToText": "AutoEasyDeLModelForImageTextToText",
    "Vision2Seq": "AutoEasyDeLModelForVision2Seq",
    "AudioToText": "AutoEasyDeLModelForAudioToText"
} %}
{% set auto_class = auto_class_map.get(model.model_task, "AutoEasyDeLModelForCausalLM") %}
{% set attn_enum_map = {
    "vanilla": "VANILLA",
    "flash": "FLASH",
    "flash_attn": "FLASH",
    "flash_attention": "FLASH",
    "flashattn": "FLASH",
    "flash_attn2": "FLASH_ATTN2",
    "flash_attention2": "FLASH_ATTN2",
    "flashattn2": "FLASH_ATTN2",
    "splash": "SPLASH",
    "splash_attn": "SPLASH",
    "splash_attention": "SPLASH",
    "ring": "RING",
    "ring_attn": "RING",
    "ring_attention": "RING",
    "paged": "PAGED",
    "paged_attn": "PAGED",
    "paged_attention": "PAGED",
    "mistral": "MISTRAL"
} %}
{% set attn_enum = attn_enum_map.get(model.attn_mechanism.lower(), "VANILLA") %}

---
tags:
- EasyDeL
- {{ model.type }}
- {{ model.model_task }}
- {{ model.attn_mechanism }}
- safetensors
- TPU
- GPU
- XLA
- Flax
---
<div align="center">
 <a href="https://github.com/erfanzar/EasyDeL">
 <img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/easydel-logo-with-text.png" height="80">
 </a>
 <br>
 [![EasyDeL](https://img.shields.io/badge/🤗_EasyDeL-{{ model.version }}-blue.svg)](https://github.com/erfanzar/EasyDeL)
 [![Model Type](https://img.shields.io/badge/Model_Type-{{ model.type }}-green.svg)](https://github.com/erfanzar/EasyDeL)
</div>

# {{ model.name }}

{{ model.description if model.description else "A model implemented using the EasyDeL framework, designed to deliver optimal performance for large-scale natural language processing tasks." }}

## Overview

This model is built using [EasyDeL](https://github.com/erfanzar/EasyDeL), an open-source framework designed to enhance and streamline the training and serving process of machine learning models, with a primary focus on Jax/Flax on TPU/GPU at scale.

{{ model.overview if model.overview else "EasyDeL provides an efficient, highly-optimized, and customizable machine learning model compatible with both GPU and TPU environments. Built with JAX, this model supports advanced features such as sharded model parallelism, making it suitable for distributed training and inference and customized kernels." }}

## Features Provided by EasyDeL

{% if model.features %}
**Model Specific Features:**
{% for feature in model.features %}
- {{ feature }}
{% endfor %}
{% endif %}

**EasyDeL Framework Features:**

- **Efficient Implementation**: Built with JAX/Flax for high-performance computation.
- **Modern Architecture**: Built on Flax NNX for better integration, modularity, and performance.
- **Multi-Device Support**: Optimized to run on TPU, GPU, and CPU environments.
- **Sharded Model Parallelism**: Supports model parallelism across multiple devices for scalability (using `auto_shard_model=True`).
- **Customizable Precision**: Allows specification of `dtype`, `param_dtype`, and `precision`.
- **Advanced Serving**: Includes `vInference` engine and OpenAI-compatible API server.
- **Optimized Kernels**: Integrates multiple attention mechanisms (like `{{ model.attn_mechanism }}`) and platform-specific optimizations.

## Installation

To use this model via EasyDeL, first install EasyDeL:

```bash
pip install easydel
```

## Usage

### Loading the Pre-trained Model

To load this pre-trained model with EasyDeL:

```python
from easydel import {{ auto_class }}, EasyDeLBaseConfigDict, AttentionMechanisms
from jax import numpy as jnp, lax

# Define max_length if needed for memory optimization
max_length = None

# Load model and parameters
# Set auto_shard_model=True to automatically distribute across devices
model = {{ auto_class }}.from_pretrained(
    "{{ model.repo_id }}",
    config_kwargs=EasyDeLBaseConfigDict(
        # use_scan_mlp=False, # Set to True to potentially reduce memory usage
        attn_dtype=jnp.float16, # Or jnp.bfloat16
        # freq_max_position_embeddings=max_length, # Set if using RoPE and need truncation
        # mask_max_position_embeddings=max_length, # Set if max length is defined
        attn_mechanism=AttentionMechanisms.{{ attn_enum }} # Matches the mechanism used by this model
    ),
    dtype=jnp.float16, # Or jnp.bfloat16 - Computation data type
    param_dtype=jnp.float16, # Or jnp.bfloat16 - Parameter data type
    precision=lax.Precision("fastest"), # Like "default", "fastest", "high", "highest"
    auto_shard_model=True, # Auto-shard across available devices
)
```

## Supported Tasks


{% if model.supported_tasks %}
This model, loaded via EasyDeL, is well-suited for the following tasks:
{% for task in model.supported_tasks %}
- **{{ task }}**
{% endfor %}
{% else %}
The primary task for this model is **{{ model.model_task }}**. Further specific supported tasks are not explicitly listed.
{% endif %}


## Limitations

{% if model.limitations %}
**Model Specific Limitations:**
{% for limitation in model.limitations %}
- {{ limitation }}
{% endfor %}
{% endif %}

**General Limitations:**

- **Hardware Dependency**: Performance can vary significantly based on the hardware (TPU/GPU) used.
- **JAX/Flax Setup Required**: The environment must support JAX/Flax for optimal use.
- **Experimental Features**: Some EasyDeL features (like custom kernels) may require additional configuration.

## License 📜

EasyDeL is released under the Apache v2 license. The license for this specific model might differ; please consult the original model repository or documentation.

```code
# Apache License 2.0 (referring to EasyDeL Framework)
# ... (Full license text usually included in the main repo) ...
```

## Citation

If you use EasyDeL in your research or work, please cite it:

```bibtex
@misc{Zare Chavoshi_2023,
    title={EasyDeL: An open-source library for enhancing and streamlining the training process of machine learning models},
    url={https://github.com/erfanzar/EasyDeL},
    author={Zare Chavoshi, Erfan},
    year={2023}
}
```

Please also consider citing the original paper or source for the **{{ model.name }}** model architecture if applicable.
"""


@dataclass
class ModelInfo:
	"""
	Model information container. Used to pass data to the Jinja template.
	"""

	name: str = field(metadata={"help": "The name of the model."})
	type: str = field(metadata={"help": "The type of the model."})
	repo_id: str = field(metadata={"help": "The repository ID of the model."})
	description: tp.Optional[str] = field(
		default=None, metadata={"help": "A description of the model."}
	)
	model_type: tp.Optional[str] = field(
		default=None, metadata={"help": "The model type."}
	)
	model_task: str = field(
		default="CausalLM",
		metadata={"help": "The model task (e.g., CausalLM, SequenceClassification)."},
	)
	attn_mechanism: str = field(
		default="vanilla",
		metadata={"help": "The attention mechanism used (e.g., vanilla, flash_attn2)."},
	)
	features: tp.Optional[tp.List[str]] = field(
		default=None, metadata={"help": "A list of features of the model."}
	)
	supported_tasks: tp.Optional[tp.List[str]] = field(
		default=None, metadata={"help": "A list of tasks supported by the model."}
	)
	limitations: tp.Optional[tp.List[str]] = field(
		default=None, metadata={"help": "A list of limitations of the model."}
	)
	version: str = field(
		default=__version__, metadata={"help": "The version of the model."}
	)
	overview: tp.Optional[str] = field(
		default=None, metadata={"help": "Custom overview text for the model."}
	)


class ReadmeGenerator:
	"""Generate README files for EasyDeL models."""

	def __init__(self, template_dir: tp.Optional[str] = None):
		"""
		Initialize the README generator.

		Args:
		    template_dir: tp.Optional custom template directory path
		"""

		# Setup Jinja environment
		if template_dir and os.path.exists(template_dir):
			self.env = Environment(
				loader=FileSystemLoader(template_dir),
				autoescape=select_autoescape(["html", "xml"]),
			)
		else:
			# Use default template
			self.env = Environment(
				loader=FileSystemLoader(os.path.dirname(__file__)),
				autoescape=select_autoescape(["html", "xml"]),
			)

	def generate_readme(
		self,
		model_info: ModelInfo,
		output_path: tp.Optional[str] = None,
	) -> str:
		"""
		Generate README content for a model.

		Args:
		    model_info: Model information
		    output_path: tp.Optional path to save the README

		Returns:
		    Generated README content
		"""
		try:
			template = self.env.from_string(JINJA_TEMPLATE)
			content = template.render(model=model_info).replace("\n\n\n", "\n").strip() + "\n"

			if output_path:
				output_path = Path(output_path)
				output_path.parent.mkdir(parents=True, exist_ok=True)
				with open(output_path, "w", encoding="utf-8") as f:
					f.write(content)
				logger.info(f"README saved to {output_path}")

			return content

		except Exception as e:
			logger.error(f"Error generating README: {str(e)}")
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
	readme_content = generator.generate_readme(
		test_model_info, output_path="tmp-test-readme.md"
	)
	print(f"Generated tmp-test-readme.md for model '{test_model_info.name}'")
	# print("\n--- README Content ---")
	# print(readme_content)
	# print("--- End README Content ---")
