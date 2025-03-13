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


from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.utils import get_logger

logger = get_logger(__name__)


@register_config("siglip_text_model")
class SiglipTextConfig(EasyDeLBaseConfig):
	r"""

	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read the
	documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 32000):
	        Vocabulary size of the Siglip text model. Defines the number of different tokens that can be represented by
	        the `inputs_ids` passed when calling [`SiglipModel`].
	    hidden_size (`int`, *optional*, defaults to 768):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 3072):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    num_hidden_layers (`int`, *optional*, defaults to 12):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 12):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    max_position_embeddings (`int`, *optional*, defaults to 64):
	        The maximum sequence length that this model might ever be used with. Typically set this to something large
	        just in case (e.g., 512 or 1024 or 2048).
	    hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
	        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
	        `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
	    layer_norm_eps (`float`, *optional*, defaults to 1e-06):
	        The epsilon used by the layer normalization layers.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    pad_token_id (`int`, *optional*, defaults to 1):
	        The id of the padding token in the vocabulary.
	    bos_token_id (`int`, *optional*, defaults to 49406):
	        The id of the beginning-of-sequence token in the vocabulary.
	    eos_token_id (`int`, *optional*, defaults to 49407):
	        The id of the end-of-sequence token in the vocabulary.
	    projection_size (`int`, *optional*, defaults to `hidden_size`):
	        The size of the projection head.

	Example:

	```python
	>>> from transformers import SiglipTextConfig, SiglipTextModel

	>>> # Initializing a SiglipTextConfig with google/siglip-base-patch16-224 style configuration
	>>> configuration = SiglipTextConfig()

	>>> # Initializing a SiglipTextModel (with random weights) from the google/siglip-base-patch16-224 style configuration
	>>> model = SiglipTextModel(configuration)

	>>> # Accessing the model configuration
	>>> configuration = model.config
	```"""

	model_type = "siglip_text_model"
	base_config_key = "text_config"

	def __init__(
		self,
		vocab_size=32000,
		hidden_size=768,
		intermediate_size=3072,
		num_hidden_layers=12,
		num_attention_heads=12,
		max_position_embeddings=64,
		hidden_act="gelu_pytorch_tanh",
		layer_norm_eps=1e-6,
		attention_dropout=0.0,
		pad_token_id=1,
		bos_token_id=49406,
		eos_token_id=49407,
		projection_size=None,
		**kwargs,
	):
		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			**kwargs,
		)

		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.max_position_embeddings = max_position_embeddings
		self.layer_norm_eps = layer_norm_eps
		self.hidden_act = hidden_act
		self.attention_dropout = attention_dropout
		self.projection_size = (
			projection_size if projection_size is not None else hidden_size
		)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.
		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			(
				"encoder/layers/.*/self_attn/(q_proj|k_proj|v_proj)/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			(
				"encoder/layers/.*/self_attn/out_proj/kernel",
				PartitionSpec("tp", ("fsdp", "sp")),
			),
			("embeddings/token_embedding/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
			("embeddings/position_embedding/embedding", PartitionSpec(None, ("fsdp", "sp"))),
			("encoder/layers/.*/mlp/fc1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("encoder/layers/.*/mlp/fc2/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("encoder/layers/.*/layer_norm[12]/(bias|scale)", PartitionSpec(None)),
			("final_layer_norm/(bias|scale)", PartitionSpec(None)),
			("head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("logit_(bias|scale)", PartitionSpec(None)),
			(".*", PartitionSpec(None)),
		)


@register_config("siglip_vision_model")
class SiglipVisionConfig(EasyDeLBaseConfig):
	r"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read the
	documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    hidden_size (`int`, *optional*, defaults to 768):
	        Dimensionality of the encoder layers and the pooler layer.
	    intermediate_size (`int`, *optional*, defaults to 3072):
	        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
	    num_hidden_layers (`int`, *optional*, defaults to 12):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 12):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_channels (`int`, *optional*, defaults to 3):
	        Number of channels in the input images.
	    image_size (`int`, *optional*, defaults to 224):
	        The size (resolution) of each image.
	    patch_size (`int`, *optional*, defaults to 16):
	        The size (resolution) of each patch.
	    hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
	        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
	        `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
	    layer_norm_eps (`float`, *optional*, defaults to 1e-06):
	        The epsilon used by the layer normalization layers.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	```"""

	model_type = "siglip_vision_model"
	base_config_key = "vision_config"

	def __init__(
		self,
		hidden_size=768,
		intermediate_size=3072,
		num_hidden_layers=12,
		num_attention_heads=12,
		num_channels=3,
		image_size=224,
		patch_size=16,
		hidden_act="gelu_pytorch_tanh",
		layer_norm_eps=1e-6,
		attention_dropout=0.0,
		**kwargs,
	):
		super().__init__(**kwargs)

		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.num_channels = num_channels
		self.patch_size = patch_size
		self.image_size = image_size
		self.attention_dropout = attention_dropout
		self.layer_norm_eps = layer_norm_eps
		self.hidden_act = hidden_act

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.
		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			(
				"embeddings/patch_embedding/kernel",
				PartitionSpec(None, None, None, ("fsdp", "sp")),
			),
			(
				"encoder/layers/.*/self_attn/(q_proj|k_proj|v_proj)/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			(
				"encoder/layers/.*/self_attn/out_proj/kernel",
				PartitionSpec("tp", ("fsdp", "sp")),
			),
			("embeddings/position_embedding/embedding", PartitionSpec(None, ("fsdp", "sp"))),
			("encoder/layers/.*/mlp/fc1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("encoder/layers/.*/mlp/fc2/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("encoder/layers/.*/layer_norm[12]/(bias|scale)", PartitionSpec(None)),
			("post_layernorm/(bias|scale)", PartitionSpec(None)),
			("head/attention/in_proj_weight", PartitionSpec(("fsdp", "sp"), "tp")),
			("head/attention/out_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("head/mlp/fc1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("head/mlp/fc2/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("head/layernorm/(bias|scale)", PartitionSpec(None)),
			("head/probe", PartitionSpec(None, None, None)),
			("logit_(bias|scale)", PartitionSpec(None)),
			(".*", PartitionSpec(None)),
		)


@register_config("siglip")
class SiglipConfig(EasyDeLBaseConfig):
	r"""
	[`SiglipConfig`] is the configuration class to store the configuration of a [`SiglipModel`]. It is used to
	instantiate a Siglip model according to the specified arguments, defining the text model and vision model configs.
	Instantiating a configuration with the defaults will yield a similar configuration to that of the Siglip
	[google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read the
	documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    text_config (`dict`, *optional*):
	        Dictionary of configuration options used to initialize [`SiglipTextConfig`].
	    vision_config (`dict`, *optional*):
	        Dictionary of configuration options used to initialize [`SiglipVisionConfig`].
	    kwargs (*optional*):
	        Dictionary of keyword arguments.

	"""

	model_type = "siglip"
	sub_configs = {"text_config": SiglipTextConfig, "vision_config": SiglipVisionConfig}

	def __init__(self, text_config=None, vision_config=None, **kwargs):
		super().__init__(**kwargs)

		if text_config is None:
			text_config = {}
			logger.info(
				"`text_config` is `None`. Initializing the `SiglipTextConfig` with default values."
			)

		if vision_config is None:
			vision_config = {}
			logger.info(
				"`vision_config` is `None`. initializing the `SiglipVisionConfig` with default values."
			)

		self.text_config = SiglipTextConfig(**text_config)
		self.vision_config = SiglipVisionConfig(**vision_config)
		self.text_config.read_basics_from_config(self)
		self.vision_config.read_basics_from_config(self)
		self.initializer_factor = 1.0

	@classmethod
	def from_text_vision_configs(
		cls,
		text_config: SiglipTextConfig,
		vision_config: SiglipVisionConfig,
		**kwargs,
	):
		r"""
		Instantiate a [`SiglipConfig`] (or a derived class) from siglip text model configuration and siglip vision
		model configuration.

		Returns:
		    [`SiglipConfig`]: An instance of a configuration object
		"""

		return cls(
			text_config=text_config.to_dict(),
			vision_config=vision_config.to_dict(),
			**kwargs,
		)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.
		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			(
				"text_model/embeddings/token_embedding/embedding",
				PartitionSpec("tp", ("fsdp", "sp")),
			),
			(
				"text_model/embeddings/position_embedding/embedding",
				PartitionSpec(None, ("fsdp", "sp")),
			),
			(
				"text_model/encoder/layers/.*/self_attn/(q_proj|k_proj|v_proj)/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			(
				"text_model/encoder/layers/.*/self_attn/out_proj/kernel",
				PartitionSpec("tp", ("fsdp", "sp")),
			),
			(
				"text_model/encoder/layers/.*/mlp/fc1/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			(
				"text_model/encoder/layers/.*/mlp/fc2/kernel",
				PartitionSpec("tp", ("fsdp", "sp")),
			),
			("text_model/encoder/layers/.*/layer_norm[12]/(bias|scale)", PartitionSpec(None)),
			("text_model/final_layer_norm/(bias|scale)", PartitionSpec(None)),
			("text_model/head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			(
				"vision_model/embeddings/patch_embedding/kernel",
				PartitionSpec(None, None, None, ("fsdp", "sp")),
			),
			(
				"vision_model/embeddings/position_embedding/embedding",
				PartitionSpec(None, ("fsdp", "sp")),
			),
			(
				"vision_model/encoder/layers/.*/self_attn/(q_proj|k_proj|v_proj)/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			(
				"vision_model/encoder/layers/.*/self_attn/out_proj/kernel",
				PartitionSpec("tp", ("fsdp", "sp")),
			),
			(
				"vision_model/encoder/layers/.*/mlp/fc1/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			(
				"vision_model/encoder/layers/.*/mlp/fc2/kernel",
				PartitionSpec("tp", ("fsdp", "sp")),
			),
			(
				"vision_model/encoder/layers/.*/layer_norm[12]/(bias|scale)",
				PartitionSpec(None),
			),
			("vision_model/post_layernorm/(bias|scale)", PartitionSpec(None)),
			(
				"vision_model/head/attention/in_proj_weight",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			(
				"vision_model/head/attention/out_proj/kernel",
				PartitionSpec("tp", ("fsdp", "sp")),
			),
			("vision_model/head/mlp/fc1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("vision_model/head/mlp/fc2/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("vision_model/head/layernorm/(bias|scale)", PartitionSpec(None)),
			("vision_model/head/probe", PartitionSpec(None, None, None)),
			("logit_(bias|scale)", PartitionSpec(None)),
			(".*", PartitionSpec(None)),
		)


__all__ = ["SiglipConfig", "SiglipTextConfig", "SiglipVisionConfig"]
