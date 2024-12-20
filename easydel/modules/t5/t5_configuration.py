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

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.modules.factory import register_config
from easydel.modules.modeling_utils import EasyDeLBaseConfig


@register_config("t5")
class T5Config(EasyDeLBaseConfig):
	"""
	Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
	the documentation from [`EasyDeLBaseConfig`] for more information.

	Args:
	    vocab_size (`int`, *optional*, defaults to 32128):
	        Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed when calling [`~easydel.modules.T5Model`] or [`~easydel.modules.TFT5Model`].
	    d_model (`int`, *optional*, defaults to 512):
	        Size of the encoder layers and the pooler layer.
	    d_kv (`int`, *optional*, defaults to 64):
	        Size of the keys and values in the attention.
	    d_ff (`int`, *optional*, defaults to 2048):
	        Size of the intermediate feed forward layer.
	    num_layers (`int`, *optional*, defaults to 6):
	        Number of hidden layers in the Transformer encoder.
	    num_decoder_layers (`int`, *optional*):
	        Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not
	        specified.
	    num_heads (`int`, *optional*, defaults to 8):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
	        The number of buckets to use for each attention layer.
	    relative_attention_max_distance (`int`, *optional*, defaults to 128):
	        The maximum distance between two tokens for them to be considered related.
	    dropout_rate (`float`, *optional*, defaults to 0.1):
	        The ratio for all dropout layers.
	    layer_norm_epsilon (`float`, *optional*, defaults to 1e-6):
	        The epsilon used by the layer normalization layers.
	    initializer_factor (`float`, *optional*, defaults to 1.0):
	        A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
	        testing).
	    feed_forward_proj (`str`, *optional*, defaults to `"relu"`):
	        Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. T5v1.1 uses the
	        `"gated-gelu"` feed forward projection. Original T5 uses `"relu"`.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models).
	    gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
	        What to save during gradient checkpointing. Choose one of `"nothing_saveable"`, `"first_half_saveable"`,
	        `"full_saveable"`.
	"""

	model_type: str = "t5"
	keys_to_ignore_at_inference = ["past_key_values"]
	attribute_map = {
		"hidden_size": "d_model",
		"num_attention_heads": "num_heads",
		"num_hidden_layers": "num_layers",
	}

	def __init__(
		self,
		vocab_size=32128,
		d_model=512,
		d_kv=64,
		d_ff=2048,
		num_layers=6,
		num_decoder_layers=None,
		num_heads=8,
		relative_attention_num_buckets=32,
		relative_attention_max_distance=128,
		dropout_rate=0.1,
		layer_norm_epsilon=1e-6,
		initializer_factor=1.0,
		feed_forward_proj="relu",
		is_encoder_decoder=True,
		use_cache=True,
		pad_token_id=0,
		eos_token_id=1,
		gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.d_model = d_model
		self.d_kv = d_kv
		self.d_ff = d_ff
		self.num_layers = num_layers
		self.gradient_checkpointing = gradient_checkpointing
		self.num_decoder_layers = (
			num_decoder_layers if num_decoder_layers is not None else self.num_layers
		)  # default = symmetry
		self.num_heads = num_heads
		self.relative_attention_num_buckets = relative_attention_num_buckets
		self.relative_attention_max_distance = relative_attention_max_distance
		self.dropout_rate = dropout_rate
		self.layer_norm_epsilon = layer_norm_epsilon
		self.initializer_factor = initializer_factor
		self.feed_forward_proj = feed_forward_proj
		self.use_cache = use_cache

		act_info = self.feed_forward_proj.split("-")
		self.dense_act_fn = act_info[-1]
		self.is_gated_act = act_info[0] == "gated"

		if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
			raise ValueError(
				f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
				"Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
				"'gated-gelu' or 'relu'"
			)

		# for backwards compatibility
		if feed_forward_proj == "gated-gelu":
			self.dense_act_fn = "gelu_new"

		super().__init__(
			pad_token_id=pad_token_id,
			eos_token_id=eos_token_id,
			is_encoder_decoder=is_encoder_decoder,
			**kwargs,
		)

	def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
		"""
		Get the partition rules for the model.

		Args:
		    fully_sharded_data_parallel (`bool`, *optional*, defaults to `True`):
		        Whether to use fully sharded data parallelism.

		Returns:
		    `Tuple[Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			(
				("wi_0/kernel", PartitionSpec("fsdp")),
				("wi_1/kernel", PartitionSpec("fsdp")),
				("wi/kernel", PartitionSpec("fsdp", "dp")),
				("wo/kernel", PartitionSpec("fsdp", "dp")),
				("SelfAttention/(q|k|v|o)/kernel", PartitionSpec("fsdp")),
				("EncDecAttention/(q|k|v|o)/kernel", PartitionSpec("fsdp")),
				(".*", PartitionSpec(None)),
			)
			if not fully_sharded_data_parallel
			else ((".*", PartitionSpec(("fsdp", "sp"))))
		)
