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
from __future__ import annotations

import typing as tp
from abc import ABCMeta, abstractmethod
from mimetypes import common_types

import chex
from flax import nnx as nn
from jax.sharding import Mesh

from easydel.layers.linear import ParallelLinear

from ..base_config import EasyDeLBaseConfig
from ..etils import EasyDeLQuantizationMethods
from ..loss_utils import LossConfig, LossMetrics
from ..modeling_outputs import (
    CausalLMOutput,
    CLIPOutput,
    CLIPTextModelOutput,
    ImageClassifierOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)

PartitionLike = tp.Optional[tp.Mapping[str, tp.Callable] | tp.Mapping[tuple, tp.Callable]]  # noqa
_CP = type[EasyDeLBaseConfig]
_T = tp.TypeVar("_T")
Self = tp.TypeVar("Self")

if tp.TYPE_CHECKING:
    from transformers import PreTrainedModel

    from easydel.infra.base_state import EasyDeLState
    from easydel.layers.caching import PagesCache, PagesMetadata, TransformerCache, TransformerMetadata


def return_type_adjuster(
    original_return_type: type[_T],
) -> tp.Callable[[tp.Callable[..., nn.Module]], tp.Callable[..., _T]]:
    def decorator(func: tp.Callable[..., nn.Module]) -> tp.Callable[..., _T]:
        def wrapper(*args: tp.Any, **kwargs: tp.Any) -> _T:
            return tp.cast(_T, func(*args, **kwargs))

        return wrapper

    return decorator


def get_module_repr(module: nn.Module) -> str:
    """Get a string representation of module parameters."""
    module_name = type(module).__name__

    if isinstance(module, ParallelLinear):
        in_features = (
            (module.kernel.shape[0] if hasattr(module.kernel, "shape") else "Null")
            if hasattr(module, "kernel")
            else module.kernel_init.__wrapped__.__code__.co_argcount - 1
        )
        out_features = (
            module.features
            if hasattr(module, "features")
            else (module.kernel.shape[-1] if hasattr(module.kernel, "shape") else "Null")
        )
        use_bias = module.use_bias if hasattr(module, "use_bias") else False
        return f"Linear(in_features={in_features}, out_features={out_features}, bias={use_bias})"

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
    max_depth: int | None = None,
    module_param=None,
) -> str:
    """Format the structure of a Flax NNX module for display.

    Recursively creates a human-readable representation of a module's
    structure, similar to PyTorch's module printing.

    Args:
        module: The module to format.
        indent: Current indentation string.
        depth: Current recursion depth.
        max_depth: Maximum depth to recurse.
        module_param: Optional parameter dictionary.

    Returns:
        Formatted string representation of the module hierarchy.

    Example:
        >>> print(prettify_nnx(my_model, max_depth=2))
        MyModel(
          (encoder): Encoder(
            (layers): ModuleList(...)
          )
          (decoder): Decoder(...)
        )
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

        for _i, (key, child) in enumerate(children):
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
                        output.append(f"{new_indent}  (0-{len(value) - 1}): {len(value)} x {child_str}")
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


class BaseModuleProtocol(metaclass=ABCMeta):
    """
    Protocol defining the common interface for EasyDeL modules.
    """

    config_class: type[EasyDeLBaseConfig]
    config: EasyDeLBaseConfig
    base_model_prefix: str
    _model_task: str | None = None
    _model_type: str | None = None

    @tp.overload
    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
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


        Returns:
            A CausalLMOutput. See return type for more details.
        """

    @tp.overload
    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
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


        Returns:
           A SequenceClassifierOutput. See return type for more details.
        """

    @tp.overload
    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeModelOutput:
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



        Returns:
            A MoeModelOutput. See return type for more details.
        """

    @tp.overload
    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
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



        Returns:
           A MoeCausalLMOutput. See return type for more details.
        """

    @tp.overload
    def __call__(
        self,
        pixel_values: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> ImageClassifierOutput:
        """Process image inputs through the CLIP vision encoder.

        Args:
            pixel_values: Optional array of shape (batch_size, num_channels, height, width)
                containing the pixel values of the images to encode.
            output_attentions: Optional bool indicating whether to return attention weights.
            output_hidden_states: Optional bool indicating whether to return all hidden states.

        Returns:
            ImageClassifierOutput containing the model outputs.
        """
        ...

    @tp.overload
    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> CLIPTextModelOutput:
        """Process text inputs through the CLIP text encoder.

        Args:
            input_ids: Array of shape (batch_size, sequence_length) containing the input
                token ids.
            attention_mask: Array of shape (batch_size, sequence_length) containing the
                attention mask for padding tokens.
            position_ids: Array of shape (batch_size, sequence_length) containing position
                indices for tokens.
            output_attentions: Wheth
            def __call__(er to return attention weights. Defaults to False.
            output_hidden_states: Whether to return all hidden states. Defaults to False.

        Returns:
            Either a CLIPTextModelOutput containing the model outputs.
        """
        ...

    @tp.overload
    def __call__(
        self,
        input_ids: chex.Array | None = None,
        pixel_values: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        output_attentions=None,
        output_hidden_states=None,
    ) -> CLIPOutput:
        """Process both text and image inputs through the full CLIP model.

        This method handles the full CLIP model forward pass, encoding both text and image
        inputs and computing their similarity.

        Args:
            input_ids: Optional array of shape (batch_size, sequence_length) containing the
                input token ids for text.
            pixel_values: Optional array of shape (batch_size, num_channels, height, width)
                containing the pixel values of the images.
            attention_mask: Optional array of shape (batch_size, sequence_length) containing
                the attention mask for text padding tokens.
            position_ids: Optional array of shape (batch_size, sequence_length) containing
                position indices for text tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.


        Returns:
            CLIPOutput containing the model outputs (including text embeddings,
            image embeddings, and their similarity).
        """
        ...

    @tp.overload
    def compute_loss(
        self,
        input_ids: chex.Array | None = None,
        labels: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
    ) -> tuple[CausalLMOutput, LossMetrics]:
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


        Returns:
            A CausalLMOutput and a tuple containing model outputs including the loss.
            See return type for more details.
        """

    @tp.overload
    def compute_loss(
        self,
        input_ids: chex.Array | None = None,
        labels: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
    ) -> tuple[SequenceClassifierOutput, LossMetrics]:
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


        Returns:
            A SequenceClassifierOutput and a tuple containing model outputs including the loss.
            See return type for more details.
        """

    @tp.overload
    def compute_loss(
        self,
        input_ids: chex.Array | None = None,
        labels: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
    ) -> tuple[MoeModelOutput, LossMetrics]:
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


        Returns:
            A MoeModelOutput and a tuple containing model outputs including the loss.
            See return type for more details.
        """

    @tp.overload
    def compute_loss(
        self,
        input_ids: chex.Array | None = None,
        labels: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
    ) -> tuple[MoeCausalLMOutput, LossMetrics]:
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


        Returns:
            A MoeCausalLMOutput and a tuple containing model outputs including the loss.
            See return type for more details.
        """

    @tp.overload
    def compute_loss(
        self,
        *,
        labels: chex.Array | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
        **batch,
    ) -> tuple[tp.Any, LossMetrics]:
        """basic `compute_loss` call"""

    @property
    @abstractmethod
    def graphdef(self) -> nn.GraphDef: ...

    @property
    @abstractmethod
    def graphstate(self) -> nn.GraphState: ...

    @property
    @abstractmethod
    def graphother(self) -> nn.GraphState: ...

    @abstractmethod
    def to_dtype(self: Self, dtype) -> Self:
        """Converts Model paramters to given dtype"""

    @abstractmethod
    def half(self, change_runtime_dtype: bool = True):
        """Converts Model paramters to float16."""

    @abstractmethod
    def float(self, change_runtime_dtype: bool = True):
        """Converts Model paramters to float32."""

    @abstractmethod
    def _reformat_dtype(self, dtype):
        """Converts Model paramters to given data type."""

    @abstractmethod
    def _get_mesh(self, mesh: Mesh | None = None) -> Mesh:
        """Retrieves the mesh, either from the provided argument or the config."""

    @abstractmethod
    def _get_partition_rules(self, partition_rules: PartitionLike) -> PartitionLike:
        """Retrieves the partition rules from input or the config"""

    @abstractmethod
    def _apply_sharding_fns(self, sharding_fns: tp.Mapping[str, tp.Callable]):
        """Applies sharding functions to the model's state."""

    @abstractmethod
    def shard_model(
        self,
        partition_rules: PartitionLike = None,
        mesh: Mesh | None = None,
    ):
        """Shards the model's parameters using the specified partitioning rules and mesh.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules for sharding.
            mesh (jax.sharding.Mesh, optional): The mesh to shard across.

        Returns:
            nn.Module: The sharded model.
        """

    @abstractmethod
    def gather_model(
        self,
        partition_rules: PartitionLike = None,
        mesh: Mesh | None = None,
    ):
        """Gathers the model's parameters based on the specified partitioning rules and mesh.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules for gathering.
            mesh (jax.sharding.Mesh, optional): The mesh to gather from.

        Returns:
            nn.Module: The gathered model.
        """

    @property
    @abstractmethod
    def _shard_fns(self):
        """property shard functions for model state and parameters."""

    @property
    @abstractmethod
    def _gather_fns(self):
        """property gather functions for model state and parameters."""

    @abstractmethod
    def quantize(
        self: Self,
        method: EasyDeLQuantizationMethods,
        skip_modules: list[str] | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Quantizes the model's linear layers.

        Args:
            method (EasyDeLQuantizationMethods, optional): The quantization method to use.
            skip_modules (list[str] | None, optional): List of module names to skip.
            verbose (bool, optional): Whether to print verbose output.
            **kwargs: Additional keyword arguments.

        Returns:
            nn.Module: The quantized model.
        """

    @abstractmethod
    def to_state(self) -> EasyDeLState:
        """converts current model to a EasyDeLState"""

    @abstractmethod
    def to_torch(self) -> PreTrainedModel:
        """converts current model to a huggingface torch model"""
        ...

    @abstractmethod
    def prepare_inputs_for_call(self, **kwargs):
        """update inputs for calling model"""
        ...

    @abstractmethod
    def get_static_arguments(self) -> tuple:
        """return static arguments kwargs for `jax.jit` / `ejit`"""
        ...

    @classmethod
    @abstractmethod
    def lazy_init(cls: type[Self], *args, **kwargs) -> Self:
        """initialize the base class with nnx.eval_shape carefully"""
        ...

    @abstractmethod
    def apply_lora_to_layers(
        self: Self,
        lora_rank: int,
        lora_pattern: str | None = None,
        verbose: bool = False,
        rngs: nn.Rngs | None = None,
    ) -> Self:
        """Apply LoRA (Low-Rank Adaptation) to specified linear layers within a model."""
        ...

    @abstractmethod
    def merge_lora_params(self: Self, pytree: dict) -> Self:
        """
        Merge LoRA (Low-Rank Adaptation) parameters into the base model parameters.
        """
        ...

    @abstractmethod
    def split_lora_params(self: Self) -> dict:
        """
        Split LoRA (Low-Rank Adaptation) parameters from the base model parameters.
        """
        ...

    @abstractmethod
    def unwrap_lora_to_layers(self, verbose: bool = False):
        """UnWrap LoRA (Low-Rank Adaptation) from specified linear layers within a model."""
        ...

    @property
    @abstractmethod
    def transform_fn(self) -> tp.Callable:
        """generate transform function for converting torch to easydel module."""
        ...

    @property
    @abstractmethod
    def pure_transform_fn(self) -> tp.Callable:
        """generates a pure transform function for converting torch to easydel module."""
        ...

    @property
    @abstractmethod
    def params_sharding(self) -> dict:
        """return the sharding of the model parameters"""
        ...

    @abstractmethod
    def merge_params(self, tree):
        """merge state to the current model"""
        ...

    @abstractmethod
    def split_params(self):
        """split the model parameters"""
        ...

    @abstractmethod
    def split_params_dict(
        self,
        params_dict: dict,
    ) -> dict:
        """Splits the model parameters from a dictionary into separate state components."""

    @abstractmethod
    def merge_params_dict(self, params_dict: dict):
        """
        Merges the model parameters from a dictionary into the current model.
        """

    @abstractmethod
    def _flop(self, *args, **kwargs) -> float | None:
        """Calculates the FLOP (Floating Point Operations) from JaxPr."""
        ...

    def __str__(self):
        return printify_nnx(self)

    def __repr__(self):
        return printify_nnx(self)


def printify_nnx(model):
    try:
        return "EasyDeL-" + prettify_nnx(model)
    except AttributeError:
        return "EasyDeL-Partitions"
