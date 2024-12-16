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

import typing as tp

import chex
from flax import nnx as nn
from jax.sharding import Mesh

from easydel.etils.etils import EasyDeLQuantizationMethods
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.loss_utils import LossConfig
from easydel.infra.modeling_outputs import (
	FlaxCausalLMOutput,
	FlaxSequenceClassifierOutput,
	MoeCausalLMOutput,
	MoeModelOutput,
)
from easydel.layers.caching.transformer_cache import TransformerCache

PartitionLike = tp.Optional[
	tp.Union[tp.Mapping[str, tp.Callable], tp.Mapping[tuple, tp.Callable]]
]
_CP = tp.Type[EasyDeLBaseConfig]
_T = tp.TypeVar("_T")


def return_type_adjuster(
	original_return_type: tp.Type[_T],
) -> tp.Callable[[tp.Callable[..., nn.Module]], tp.Callable[..., _T]]:
	def decorator(func: tp.Callable[..., nn.Module]) -> tp.Callable[..., _T]:
		def wrapper(*args: tp.Any, **kwargs: tp.Any) -> _T:
			return tp.cast(_T, func(*args, **kwargs))

		return wrapper

	return decorator


def get_module_repr(module: nn.Module) -> str:
	"""Get a string representation of module parameters."""
	module_name = type(module).__name__

	if isinstance(module, nn.Linear):
		in_features = (
			module.kernel.shape[0]
			if hasattr(module, "kernel")
			else module.kernel_init.__wrapped__.__code__.co_argcount - 1
		)
		out_features = (
			module.features if hasattr(module, "features") else module.kernel.shape[1]
		)
		use_bias = module.use_bias if hasattr(module, "use_bias") else False
		return (
			f"Linear(in_features={in_features}, out_features={out_features}, bias={use_bias})"
		)

	elif isinstance(module, nn.Dropout):
		rate = module.rate if hasattr(module, "rate") else 0.0
		return f"Dropout(p={rate})"

	elif isinstance(module, nn.Embed):
		if hasattr(module, "embedding"):
			num_embeddings, embedding_dim = module.embedding.shape
			return f"Embedding({num_embeddings}, {embedding_dim})"
		return "Embedding(...)"

	elif hasattr(module, "eps"):
		shape_str = ""
		if hasattr(module, "kernel"):
			shape_str = str(tuple(module.kernel.shape))
		elif hasattr(module, "scale"):
			shape_str = str(tuple(module.scale.shape))
		return f"{module_name}({shape_str}, eps={module.eps})"

	return module_name


def prettify_nnx(
	module: nn.Module,
	indent: str = "",
	depth: int = 0,
	max_depth: int = None,
	module_param=None,
) -> str:
	"""
	Recursively formats the structure of a Flax NNX module, mimicking PyTorch's module printing.
	"""
	if max_depth is not None and depth > max_depth:
		return ""

	output = []
	module_repr = get_module_repr(module)

	current_line = f"{indent}{module_repr}"

	children = list(module.iter_children())

	if module_param is not None:
		params_children = {key: param for key, param in module_param.items()}
	else:
		params_children = {}

	if children or any(
		isinstance(value, list) and all(isinstance(item, nn.Module) for item in value)
		for value in module.__dict__.values()
	):
		output.append(current_line + "(")
		new_indent = indent + "  "

		for i, (key, child) in enumerate(children):
			child_param = params_children.get(key, None)
			child_str = prettify_nnx(
				child,
				new_indent,
				depth + 1,
				max_depth,
				child_param,
			).lstrip()
			output.append(f"{new_indent}({key}): {child_str}")

		for key, value in module.__dict__.items():
			if isinstance(value, list) and all(isinstance(item, nn.Module) for item in value):
				output.append(f"{new_indent}({key}): ModuleList(")

				if value:
					first_item = value[0]
					item_param = params_children.get(key, [None])[0] if params_children else None

					if len(value) > 1:
						child_str = prettify_nnx(
							first_item,
							new_indent + "  ",
							depth + 1,
							max_depth,
							item_param,
						).lstrip()
						output.append(
							f"{new_indent}  (0-{len(value)-1}): {len(value)} x {child_str}"
						)
					else:
						child_str = prettify_nnx(
							first_item,
							new_indent + "  ",
							depth + 1,
							max_depth,
							item_param,
						).lstrip()
						output.append(f"{new_indent}  {child_str}")

				output.append(f"{new_indent})")

		output.append(f"{indent})")
	else:
		output.append(current_line)

	return "\n".join(output)


class BaseModuleProtocol(nn.Module):
	"""
	Protocol defining the common interface for EasyDeL modules.
	"""

	config_class: tp.Type[EasyDeLBaseConfig]
	config: EasyDeLBaseConfig
	base_model_prefix: str
	_model_task: tp.Optional[str] = None
	_model_type: tp.Optional[str] = None

	@tp.overload
	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxCausalLMOutput, tp.Tuple]:
		"""
		Forward pass for Causal Language Models (e.g., GPT).

		Args:
		    input_ids: Optional array of token IDs.
		    inputs_embeds: Optional array of input embeddings. Use this if you've pre-computed embeddings
		                   and want to bypass the embedding layer.
		    attention_mask: Optional array indicating which tokens should be attended to.
		    position_ids: Optional array specifying token positions.
		    segment_ids: Optional array indicating segment IDs (used in models like BERT).
		    past_key_values: Optional cache containing key and value tensors from previous model passes.
		                     Useful for faster inference.
		    output_attentions: Optional flag to return attention weights from each layer.
		    output_hidden_states: Optional flag to return hidden states from each layer.
		    return_dict: If True, returns a dictionary containing model outputs. Otherwise, return a tuple.

		Returns:
		    A FlaxCausalLMOutput if return_dict is True, or a tuple containing
		    model outputs. See return type for more details.
		"""

	@tp.overload
	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxSequenceClassifierOutput, tp.Tuple]:
		"""
		Forward pass for Sequence Classification Models (e.g., BERT for sentiment analysis).

		Args:
		    input_ids: Optional array of token IDs.
		    inputs_embeds: Optional array of input embeddings.
		    attention_mask: Optional array indicating which tokens should be attended to.
		    position_ids: Optional array specifying token positions.
		    segment_ids: Optional array indicating segment IDs.
		    output_attentions: Optional flag to return attention weights from each layer.
		    output_hidden_states: Optional flag to return hidden states from each layer.
		    return_dict: If True, returns a dictionary containing model outputs. Otherwise, return a tuple.

		Returns:
		   A FlaxSequenceClassifierOutput if return_dict is True, or a tuple containing
		    model outputs. See return type for more details.
		"""

	@tp.overload
	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[MoeModelOutput, tp.Tuple]:
		"""
		Forward pass for Mixture-of-Experts (MoE) Models.

		Args:
		    input_ids: Optional array of token IDs.
		    inputs_embeds: Optional array of input embeddings.
		    attention_mask: Optional array indicating which tokens should be attended to.
		    position_ids: Optional array specifying token positions.
		    segment_ids: Optional array indicating segment IDs.
		    output_attentions: Optional flag to return attention weights from each layer.
		    output_hidden_states: Optional flag to return hidden states from each layer.
		    output_router_logits: Optional flag to return the router logits,
		         which are used to determine which experts to use for each token.
		    past_key_values: Optional cache containing key and value tensors from previous model passes.
		    return_dict: If True, returns a dictionary containing model outputs. Otherwise, return a tuple.


		Returns:
		    A MoeModelOutput if return_dict is True, or a tuple containing
		    model outputs. See return type for more details.
		"""

	@tp.overload
	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[MoeCausalLMOutput, tp.Tuple]:
		"""
		Forward pass for Mixture-of-Experts (MoE) Causal Language Models.

		Args:
		    input_ids: Optional array of token IDs.
		    inputs_embeds: Optional array of input embeddings.
		    attention_mask: Optional array indicating which tokens should be attended to.
		    position_ids: Optional array specifying token positions.
		    segment_ids: Optional array indicating segment IDs.
		    output_attentions: Optional flag to return attention weights from each layer.
		    output_hidden_states: Optional flag to return hidden states from each layer.
		    output_router_logits: Optional flag to return the router logits,
		         which are used to determine which experts to use for each token.
		    past_key_values: Optional cache containing key and value tensors from previous model passes.
		    return_dict: If True, returns a dictionary containing model outputs. Otherwise, return a tuple.


		Returns:
		   A MoeCausalLMOutput if return_dict is True, or a tuple containing
		    model outputs. See return type for more details.
		"""

	@tp.overload
	def compute_loss(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		labels: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
		loss_config: tp.Optional[LossConfig] = None,
		loss_kwargs: tp.Optional[tp.Dict] = None,
	) -> tp.Union[FlaxCausalLMOutput, tp.Tuple]:
		"""
		Computes the loss for Causal Language Models.

		Args:
		    input_ids: Optional array of token IDs.
		    labels: Optional array of target token IDs.
		    inputs_embeds: Optional array of input embeddings.
		    attention_mask: Optional array indicating which tokens should be attended to.
		    position_ids: Optional array specifying token positions.
		    segment_ids: Optional array indicating segment IDs.
		    past_key_values: Optional cache containing key and value tensors from previous model passes.
		    output_attentions: Optional flag to return attention weights from each layer.
		    output_hidden_states: Optional flag to return hidden states from each layer.
		    return_dict: If True, returns a dictionary containing model outputs along with loss. Otherwise, return a tuple.

		Returns:
		    A FlaxCausalLMOutput if return_dict is True, or a tuple containing
		    model outputs including the loss. See return type for more details.
		"""

	@tp.overload
	def compute_loss(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		labels: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
		loss_config: tp.Optional[LossConfig] = None,
		loss_kwargs: tp.Optional[tp.Dict] = None,
	) -> tp.Union[FlaxSequenceClassifierOutput, tp.Tuple]:
		"""
		Computes the loss for Sequence Classification Models.

		Args:
		    input_ids: Optional array of token IDs.
		    labels: Optional array of target classification labels.
		    inputs_embeds: Optional array of input embeddings.
		    attention_mask: Optional array indicating which tokens should be attended to.
		    position_ids: Optional array specifying token positions.
		    segment_ids: Optional array indicating segment IDs.
		    output_attentions: Optional flag to return attention weights from each layer.
		    output_hidden_states: Optional flag to return hidden states from each layer.
		    return_dict: If True, returns a dictionary containing model outputs along with loss. Otherwise, return a tuple.

		Returns:
		    A FlaxSequenceClassifierOutput if return_dict is True, or a tuple containing
		    model outputs including the loss. See return type for more details.
		"""

	@tp.overload
	def compute_loss(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		labels: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
		loss_config: tp.Optional[LossConfig] = None,
		loss_kwargs: tp.Optional[tp.Dict] = None,
	) -> tp.Union[MoeModelOutput, tp.Tuple]:
		"""
		Computes the loss for Mixture-of-Experts (MoE) Models.

		Args:
		    input_ids: Optional array of token IDs.
		    labels: Optional array of target token IDs or labels for the specific task.
		    inputs_embeds: Optional array of input embeddings.
		    attention_mask: Optional array indicating which tokens should be attended to.
		    position_ids: Optional array specifying token positions.
		    segment_ids: Optional array indicating segment IDs.
		    output_attentions: Optional flag to return attention weights from each layer.
		    output_hidden_states: Optional flag to return hidden states from each layer.
		    output_router_logits: Optional flag to return the router logits.
		    past_key_values: Optional cache containing key and value tensors from previous model passes.
		    return_dict: If True, returns a dictionary containing model outputs along with loss. Otherwise, return a tuple.

		Returns:
		     A MoeModelOutput if return_dict is True, or a tuple containing
		    model outputs including the loss. See return type for more details.
		"""

	@tp.overload
	def compute_loss(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		labels: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
		loss_config: tp.Optional[LossConfig] = None,
		loss_kwargs: tp.Optional[tp.Dict] = None,
	) -> tp.Union[MoeCausalLMOutput, tp.Tuple]:
		"""
		Computes the loss for Mixture-of-Experts (MoE) Causal Language Models.

		Args:
		    input_ids: Optional array of token IDs.
		    labels: Optional array of target token IDs.
		    inputs_embeds: Optional array of input embeddings.
		    attention_mask: Optional array indicating which tokens should be attended to.
		    position_ids: Optional array specifying token positions.
		    segment_ids: Optional array indicating segment IDs.
		    output_attentions: Optional flag to return attention weights from each layer.
		    output_hidden_states: Optional flag to return hidden states from each layer.
		    output_router_logits: Optional flag to return the router logits.
		    past_key_values: Optional cache containing key and value tensors from previous model passes.
		    return_dict: If True, returns a dictionary containing model outputs along with loss. Otherwise, return a tuple.

		Returns:
		    A MoeCausalLMOutput if return_dict is True, or a tuple containing
		    model outputs including the loss. See return type for more details.
		"""

	def half(self):
		"""Converts Model paramters to float16."""

	def float(self):
		"""Converts Model paramters to float32."""

	def _reformat_dtype(self, dtype):
		"""Converts Model paramters to given data type."""

	def _get_mesh(self, mesh: tp.Optional[Mesh] = None) -> Mesh:
		"""Retrieves the mesh, either from the provided argument or the config."""

	def _get_partition_rules(self, partition_rules: PartitionLike) -> PartitionLike:
		"""Retrieves the partition rules from input or the config"""

	def _apply_sharding_fns(self, sharding_fns: tp.Mapping[str, tp.Callable]):
		"""Applies sharding functions to the model's state."""

	def shard_model(
		self,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[Mesh] = None,
	):
		"""Shards the model's parameters using the specified partitioning rules and mesh.

		Args:
		    partition_rules (PartitionLike, optional): Partitioning rules for sharding.
		    mesh (jax.sharding.Mesh, optional): The mesh to shard across.

		Returns:
		    nn.Module: The sharded model.
		"""

	def gather_model(
		self,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[Mesh] = None,
	):
		"""Gathers the model's parameters based on the specified partitioning rules and mesh.

		Args:
		    partition_rules (PartitionLike, optional): Partitioning rules for gathering.
		    mesh (jax.sharding.Mesh, optional): The mesh to gather from.

		Returns:
		    nn.Module: The gathered model.
		"""

	def quantize(
		self,
		method: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.A8BIT,
		block_size: int = 128,
		quantization_pattern: tp.Optional[str] = None,
	):
		"""Quantizes the model's linear layers.

		Args:
		    method (EasyDeLQuantizationMethods, optional): The quantization method to use.
		    block_size (int, optional): The block size for quantization.
		    quantization_pattern (str, optional): The quantization pattern to use.

		Returns:
		    nn.Module: The quantized model.
		"""

	def __repr__(self):
		return "EasyDeL-" + prettify_nnx(self)

	__str__ = __repr__
