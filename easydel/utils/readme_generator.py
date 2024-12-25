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
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from easydel import __version__
from easydel.etils.etils import get_logger

logger = get_logger(__name__)

JINJA_TEMPLATE = """
---
tags:
- EasyDeL
- {{model.type}}
- safetensors
- TPU
- GPU
- XLA
- Flax
---
# {{ model.name }}

[![EasyDeL](https://img.shields.io/badge/🤗_EasyDeL-{{ model.version }}-blue.svg)](https://github.com/erfanzar/EasyDeL)
[![Model Type](https://img.shields.io/badge/Model_Type-{{ model.type }}-green.svg)](https://github.com/erfanzar/EasyDeL)

{{ model.description if model.description else "A model implemented using the EasyDeL framework, designed to deliver optimal performance for large-scale natural language processing tasks." }}

## Overview

{{ model.overview if model.overview else "EasyDeL provides an efficient, highly-optimized, and customizable machine learning model compatible with both GPU and TPU environments. Built with JAX, this model supports advanced features such as sharded model parallelism, making it suitable for distributed training and inference and customized kernels." }}

## Features

{% if model.features %}
{% for feature in model.features %}
- {{ feature }}
{% endfor %}
{% else %}
- **Efficient Implementation**: Built with JAX/Flax for high-performance computation.
- **Multi-Device Support**: Optimized to run on TPU, GPU, and CPU environments for sharding model over 2^(1-1000+) of devices.
- **Sharded Model Parallelism**: Supports model parallelism across multiple devices for scalability.
- **Customizable Precision**: Allows specification of floating-point precision for performance optimization.
{% endif %}

## Installation

To install EasyDeL, simply run:

```bash
pip install easydel
```

## Usage

### Loading the Pre-trained Model

To load a pre-trained version of the model with EasyDeL:

```python
from easydel import AutoEasyDeLModelForCausalLM
from jax import numpy as jnp, lax

max_length = None # can be set to use lower memory for caching

# Load model and parameters
model = AutoEasyDeLModelForCausalLM.from_pretrained(
    "{{ model.repo_id }}",
    config_kwargs=ed.EasyDeLBaseConfigDict(
        use_scan_mlp=False,
        attn_dtype=jnp.float16,
        freq_max_position_embeddings=max_length,
        mask_max_position_embeddings=max_length,
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2
    ), 
    dtype=jnp.float16,
    param_dtype=jnp.float16,
    precision=lax.Precision("fastest"),
    auto_shard_model=True, 
)
```

## Supported Tasks

{% if model.supported_tasks %}
This model is well-suited for the following tasks:
{% for task in model.supported_tasks %}
- **{{ task }}**
{% endfor %}
{% else %}
[Need more information]
{% endif %}
 
## Limitations

{% if model.limitations %}
{% for limitation in model.limitations %}
- {{ limitation }}
{% endfor %}
{% else %}
- **Hardware Dependency**: Performance can vary significantly based on the hardware used.
- **JAX/Flax Setup Required**: The environment must support JAX/Flax for optimal use.
- **Experimental Features**: Some features (like custom kernel usage or ed-ops) may require additional configuration and tuning.
{% endif %}
"""


@dataclass
class ModelInfo:
	"""Model information container."""

	name: str
	type: str
	repo_id: str
	description: tp.Optional[str] = None
	model_type: tp.Optional[str] = None
	model_task: tp.Optional[str] = None
	features: tp.Optional[list] = None
	supported_tasks: tp.Optional[list] = None
	limitations: tp.Optional[list] = None
	version: str = __version__


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
		    template_name: Name of the template to use

		Returns:
		    Generated README content
		"""
		try:
			template = self.env.from_string(JINJA_TEMPLATE)
			content = template.render(model=model_info)

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


# Example usage
if __name__ == "__main__":
	model_info = ModelInfo(
		name="LLaMA-2-7B-EasyDeL",
		type="CausalLM",
		repo_id="erfanzar/LLaMA-2-7B-EasyDeL",
	)

	generator = ReadmeGenerator()
	readme = generator.generate_readme(model_info, "tmp-files/readme.md")
