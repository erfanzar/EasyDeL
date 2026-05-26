# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Utility functions and helpers for EasyDeL infrastructure.

Provides common utilities used throughout the EasyDeL framework, including
activation functions, dtype handling, module manipulation, and various
helper functions for model operations.

Constants:
    ACT2FN: Dictionary mapping activation names to functions
    ROPE_TYPES: Supported RoPE (Rotary Position Embedding) types

Functions:
    quick_gelu: Quick GELU activation function
    canonicalize_dtype: Canonicalize dtype for JAX arrays
    get_activation: Get activation function by name
    quantize_linear: Apply quantization to linear layers
    replace_dot: Replace JAX dot operations

Key Features:
    - Activation function registry
    - Data type canonicalization
    - Module quantization utilities
    - Sharding constraint helpers
    - Memory optimization tools

Example:
    >>> from easydel.infra.utils import ACT2FN, canonicalize_dtype
    >>> # Get activation function
    >>> activation = ACT2FN["gelu"]
    >>> # Canonicalize dtype
    >>> dtype = canonicalize_dtype(array, dtype=jnp.float32)
"""

from __future__ import annotations

import inspect
import re
import types
import typing as tp
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import lru_cache, partial

import jax
import jax.extend
import jax.tree_util
import numpy as np
import spectrax as spx
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, PRNGKeyArray
from spectrax import nn, with_sharding_constraint
from tqdm.auto import tqdm

from easydel.layers import EasyQuantizer, ParallelLinear, QuantizationConfig, eLoRA
from easydel.utils.compiling_utils import hash_fn
from easydel.utils.traversals import flatten_dict, unflatten_dict

from .errors import EasyDeLBlockWiseFFNError
from .etils import AVAILABLE_SPARSE_MODULE_TYPES, GRADIENT_CHECKPOINT_TARGETS, EasyDeLGradientCheckPointers

warnings.filterwarnings("ignore", message="Primitive dynamic_update_slice was not handled by class")
logger = get_logger(__name__)

_ATTN_CHECKPOINT_NAME_PATTERN = re.compile(r"^attn_")
_LMHEAD_CHECKPOINT_NAME_PATTERN = re.compile(r"^lm_head_")
_MLP_CHECKPOINT_NAME_PATTERN = re.compile(r"^mlp_")


def _select_checkpoint_names_by_regex(
    *,
    include_patterns: Sequence[re.Pattern[str]] | None = None,
    exclude_patterns: Sequence[re.Pattern[str]] | None = None,
) -> list[str]:
    """Filter the registry of checkpoint target names by include/exclude regex.

    Walks the canonical ``GRADIENT_CHECKPOINT_TARGETS`` list, keeping the
    names whose strings match any *include_patterns* (when provided) and do
    not match any *exclude_patterns*. Used by the family-based gradient
    checkpointing policies (e.g. ``mlp_notsaveable``) to derive their
    ``save_only_these_names`` argument lists.

    Args:
        include_patterns: Optional sequence of compiled regex patterns; if
            provided, only target names matched by at least one pattern are
            kept.
        exclude_patterns: Optional sequence of compiled regex patterns; any
            target name matched by at least one pattern is dropped.

    Returns:
        list[str]: Filtered checkpoint target names.

    Raises:
        ValueError: If filtering leaves the empty set.
    """
    names = [str(name) for name in GRADIENT_CHECKPOINT_TARGETS]
    if include_patterns:
        names = [name for name in names if any(pattern.search(name) for pattern in include_patterns)]
    if exclude_patterns:
        names = [name for name in names if not any(pattern.search(name) for pattern in exclude_patterns)]
    if not names:
        raise ValueError("Regex-based checkpoint target selection resolved to an empty set.")
    return names


def quick_gelu(x):
    """Quick GELU activation: ``x * sigmoid(1.702 * x)``.

    Cheaper approximation of the true GELU. The constant ``1.702`` was
    introduced in the original BERT codebase and is the variant exposed by
    HuggingFace under the name ``"quick_gelu"``; CLIP and several CLIP-derived
    vision encoders depend on this exact form.

    Args:
        x: Input array (any shape, any floating dtype).

    Returns:
        Array of the same shape and dtype as ``x``.
    """
    return x * jax.nn.sigmoid(1.702 * x)


# Registry of activation functions by name.
#
# Maps activation function names to their implementations.
# Supports common activations used in neural networks.
ACT2FN = {
    "gelu": partial(jax.nn.gelu, approximate=False),
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "gelu_new": partial(jax.nn.gelu, approximate=True),
    "gelu_pytorch_tanh": partial(jax.nn.gelu, approximate=True),
    "tanh": jax.nn.tanh,
    "sigmoid": jax.nn.sigmoid,
    "leaky_relu": partial(jax.nn.leaky_relu, negative_slope=0.01),
    "glu": jax.nn.glu,
    "elu": jax.nn.elu,
    "softmax": jax.nn.softmax,
    "quick_gelu": quick_gelu,
}

ROPE_TYPES = tp.Literal["none", "linear", "dynamic", "yarn", "su", "llama3", "longrope"] | None


with_sharding_constraint = with_sharding_constraint


def canonicalize_dtype(
    *args,
    dtype: jax.numpy.dtype | None = None,
    inexact: bool = True,
) -> jax.numpy.dtype:
    """Canonicalize an optional dtype to a definitive JAX dtype.

    Infers or validates the dtype for JAX operations. When ``dtype`` is
    ``None`` the result is the common ``jnp.result_type`` of the supplied
    ``*args`` (with ``None`` arguments ignored); otherwise the supplied
    ``dtype`` is returned (after the inexact check, if enabled).

    Args:
        *args: JAX-array-compatible values used for dtype inference when
            ``dtype`` is ``None``. ``None`` values are skipped.
        dtype: Optional dtype override. When provided, dtype inference is
            disabled and this value is returned (subject to the inexact
            check).
        inexact: When ``True``, the output dtype must be a subdtype of
            ``jnp.inexact`` (real or complex floating point). This is useful
            when the caller will apply operations that don't make sense on
            integer dtypes (e.g. taking a mean). If inference yields an
            integer dtype, it is promoted with ``jnp.float32``.

    Returns:
        jax.numpy.dtype: The dtype that ``*args`` should be cast to.

    Raises:
        ValueError: If ``inexact`` is ``True`` and the supplied ``dtype``
            is not an inexact subdtype.
    """
    if dtype is None:
        args_filtered = [jax.numpy.asarray(x) for x in args if x is not None]
        dtype = jax.numpy.result_type(*args_filtered)
        if inexact and not jax.numpy.issubdtype(dtype, jax.numpy.inexact):
            dtype = jax.numpy.promote_types(jax.numpy.float32, dtype)
    if inexact and not jax.numpy.issubdtype(dtype, jax.numpy.inexact):
        raise ValueError(f"Dtype must be inexact: {dtype}")
    return dtype


def get_gradient_checkpoint_policy(
    name: str | EasyDeLGradientCheckPointers,
    save_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> tp.Callable:
    """Get a gradient checkpointing policy by name or create a custom one.

    Retrieves a JAX gradient checkpointing policy function that determines
    which intermediate values to save during forward pass for use in backward pass.
    This is used to trade compute for memory in gradient calculations.

    Args:
        name: Name of the checkpointing policy or EasyDeLGradientCheckPointers enum.
            Supported values:
            - 'everything_saveable': Save all intermediate values
            - 'nothing_saveable': Save no intermediate values (maximum recomputation)
            - 'dots_saveable': Save dot product results
            - 'checkpoint_dots': Checkpoint dot operations
            - 'dots_with_no_batch_dims_saveable': Save dots without batch dimensions
            - 'checkpoint_dots_with_no_batch_dims': Checkpoint dots without batch dims
            - 'save_anything_except_these_names': Save all except specified names
            - 'save_any_names_but_these': Save any names except specified
            - 'save_only_these_names': Save only specified names
            - 'mlp_notsaveable': Save all known checkpoint names except MLP-family names
            - 'attn_notsaveable': Save all known checkpoint names except attention-family names
            - 'mlp_attn_notsaveable': Save all known checkpoint names except MLP and attention names
            - 'lmhead_notsaveable': Save all known checkpoint names except LM-head names
            - 'attn_lmhead_notsaveable': Save all known checkpoint names except attention and LM-head names
            - 'mlp_lmhead_notsaveable': Save all known checkpoint names except MLP and LM-head names
            - 'save_from_both_policies': Combine two policies
        save_names: List of checkpoint names to save (used with 'save_only_these_names')
        exclude_names: List of checkpoint names to exclude (used with 'save_anything_except_these_names')

    Returns:
        The corresponding JAX checkpoint policy function.

    Raises:
        KeyError: If the policy name is not recognized.
        ValueError: If save_names or exclude_names are not provided when required.

    Example:
        >>> # Basic policy
        >>> policy = get_gradient_checkpoint_policy('dots_saveable')
        >>>
        >>> # Custom policy saving only specific checkpoints
        >>> policy = get_gradient_checkpoint_policy(
        ...     'save_only_these_names',
        ...     save_names=['attn_output', 'mlp_output']
        ... )
    """
    if isinstance(name, EasyDeLGradientCheckPointers):
        name = name.value

    if name == "save_only_these_names":
        if save_names is None:
            raise ValueError("save_names must be provided when using 'save_only_these_names' policy")
        return jax.checkpoint_policies.save_only_these_names(*save_names)

    elif name == "mlp_notsaveable":
        save_names = _select_checkpoint_names_by_regex(exclude_patterns=[_MLP_CHECKPOINT_NAME_PATTERN])
        return jax.checkpoint_policies.save_only_these_names(*save_names)

    elif name == "attn_notsaveable":
        save_names = _select_checkpoint_names_by_regex(exclude_patterns=[_ATTN_CHECKPOINT_NAME_PATTERN])
        return jax.checkpoint_policies.save_only_these_names(*save_names)

    elif name == "mlp_attn_notsaveable":
        save_names = _select_checkpoint_names_by_regex(
            exclude_patterns=[_MLP_CHECKPOINT_NAME_PATTERN, _ATTN_CHECKPOINT_NAME_PATTERN]
        )
        return jax.checkpoint_policies.save_only_these_names(*save_names)

    elif name == "lmhead_notsaveable":
        save_names = _select_checkpoint_names_by_regex(exclude_patterns=[_LMHEAD_CHECKPOINT_NAME_PATTERN])
        return jax.checkpoint_policies.save_only_these_names(*save_names)

    elif name == "attn_lmhead_notsaveable":
        save_names = _select_checkpoint_names_by_regex(
            exclude_patterns=[_ATTN_CHECKPOINT_NAME_PATTERN, _LMHEAD_CHECKPOINT_NAME_PATTERN]
        )
        return jax.checkpoint_policies.save_only_these_names(*save_names)

    elif name == "mlp_lmhead_notsaveable":
        save_names = _select_checkpoint_names_by_regex(
            exclude_patterns=[_MLP_CHECKPOINT_NAME_PATTERN, _LMHEAD_CHECKPOINT_NAME_PATTERN]
        )
        return jax.checkpoint_policies.save_only_these_names(*save_names)

    elif name in ["save_anything_except_these_names", "save_any_names_but_these"]:
        if exclude_names is None:
            raise ValueError("exclude_names must be provided when using exclude-based policies")
        return jax.checkpoint_policies.save_any_names_but_these(*exclude_names)

    gradients = dict(
        everything_saveable=jax.checkpoint_policies.everything_saveable,
        nothing_saveable=jax.checkpoint_policies.nothing_saveable,
        dots_saveable=jax.checkpoint_policies.dots_saveable,
        checkpoint_dots=jax.checkpoint_policies.checkpoint_dots,
        dots_with_no_batch_dims_saveable=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        checkpoint_dots_with_no_batch_dims=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
        save_from_both_policies=jax.checkpoint_policies.save_from_both_policies,
    )

    if name not in gradients:
        raise KeyError(f"Unknown checkpoint policy: {name}")

    return gradients[name]


def create_transformer_checkpoint_policy(
    save_attention: bool = True,
    save_mlp: bool = True,
    save_residuals: bool = True,
    save_layer_outputs: bool = False,
    save_embeddings: bool = False,
    custom_names: list[str] | None = None,
) -> tp.Callable:
    """Create a checkpoint policy optimized for transformer models.

    Creates a custom checkpoint policy that selectively saves transformer
    components based on the checkpoint_name calls we've added to all models.

    Args:
        save_attention: Whether to save attention outputs (attn_query, attn_key, attn_value, attn_output)
        save_mlp: Whether to save MLP outputs (mlp_gate, mlp_up, mlp_down, mlp_output)
        save_residuals: Whether to save residual connections
        save_layer_outputs: Whether to save layer outputs
        save_embeddings: Whether to save embeddings and model outputs
        custom_names: Additional checkpoint names to save

    Returns:
        JAX checkpoint policy function

    Example:
        >>> # Save only critical transformer components
        >>> policy = create_transformer_checkpoint_policy(
        ...     save_attention=True,
        ...     save_mlp=False,  # Recompute MLP
        ...     save_residuals=True
        ... )
        >>> model = auto_remat(model, policy=policy)
    """
    names_to_save = []

    if save_attention:
        names_to_save.extend(["attn_query", "attn_key", "attn_value", "attn_output"])

    if save_mlp:
        names_to_save.extend(["mlp_gate", "mlp_up", "mlp_down", "mlp_output"])

    if save_residuals:
        names_to_save.extend(["residual"])

    if save_layer_outputs:
        names_to_save.extend(["layer_output"])

    if save_embeddings:
        names_to_save.extend(["embeddings", "model_output", "lm_head_output"])

    if custom_names:
        names_to_save.extend(custom_names)

    if not names_to_save:
        return jax.checkpoint_policies.nothing_saveable

    return jax.checkpoint_policies.save_only_these_names(*names_to_save)


def add_start_docstrings(*docstr):
    """Decorator factory that prepends docstring fragments to a function's ``__doc__``.

    Useful for sharing leading documentation (argument boilerplate, model
    description blocks, copyright notes) across multiple related functions
    without duplication. The decorator returned by ``add_start_docstrings``
    concatenates *docstr* in order and prepends the result to the existing
    function docstring (if any).

    Args:
        *docstr: Variable number of docstring fragments to prepend to the
            decorated function's ``__doc__``.

    Returns:
        Callable: A decorator that mutates the decorated function's
        ``__doc__`` in-place and returns the same function.
    """

    def docstring_decorator(fn):
        """Concatenate the captured docstring snippets onto ``fn.__doc__``.

        Args:
            fn: The function whose docstring will be augmented.

        Returns:
            Callable: The same function with its ``__doc__`` updated in-place.
        """
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def block_wise_ffn(remat_ffn: tp.Callable, inputs: jax.Array, chunk_size: int) -> jax.Array:
    """Apply a feed-forward network block-wise to reduce memory usage.

    Implements the block-wise feed-forward approach from the near-infinite
    context length paper. This technique processes the FFN in chunks along
    the sequence dimension to reduce peak memory usage during training.

    Args:
        remat_ffn: The feed-forward network function to apply. Should be
            rematerialized (checkpointed) for memory efficiency.
        inputs: Input tensor with shape (batch_size, sequence_length, hidden_dim).
        chunk_size: Size of chunks to process. Sequence length must be
            divisible by chunk_size.

    Returns:
        Output tensor with same shape as inputs.

    Raises:
        EasyDeLBlockWiseFFNError: If inputs have wrong shape or chunk_size
            doesn't divide sequence length evenly.

    Note:
        - For generation (sequence_length=1), applies FFN directly without chunking
        - For training, processes sequence in chunks to reduce memory
        - Requires sequence_length to be divisible by chunk_size

    Example:
        >>> ffn = lambda x: mlp(x)  # Your FFN function
        >>> chunked_output = block_wise_ffn(ffn, inputs, chunk_size=256)
    """
    generating = inputs.shape[1] == 1
    try:
        if generating:
            return remat_ffn(inputs)
        else:
            inputs_chunked = rearrange(inputs, "b (c n) d -> c b n d", n=chunk_size)
            _, outputs = jax.lax.scan(
                f=lambda carry, x: (carry, remat_ffn(x)),
                init=jnp.array(0, dtype=jnp.int32),
                xs=inputs_chunked,
                unroll=True,
            )
            return rearrange(outputs, "c b n d -> b (c n) d")
    except Exception as e:
        raise EasyDeLBlockWiseFFNError(
            "You Are using BlockWise FFN from near-infinite-context length paper and you might be passing "
            "input arguments in wrong way in case that you don't want to use this just pass "
            "`use_scan_mlp=False` in "
            "model config or in config_kwargs in AutoEasyDeLModelFor... or change `scan_mlp_chunk_size` "
            f"in configs for more information read Docs.\nOriginal Error\n{e}"
        ) from e


def is_flatten(pytree: dict):
    """Return whether *pytree* uses flat tuple-path keys instead of nested dicts.

    EasyDeL routinely round-trips parameter trees through
    :func:`easydel.utils.traversals.flatten_dict`, which keys each leaf by
    its full path tuple (e.g. ``("model", "layers", 0, "weight")``). This
    helper peeks at the first key to tell whether *pytree* is already in
    that flattened form so callers can avoid double-flattening.

    Args:
        pytree: A possibly-nested or possibly-flat parameter dict.

    Returns:
        bool: ``True`` if the first key is a tuple (flat form), ``False``
        otherwise (nested form).
    """
    mpl = next(iter(pytree.keys()))
    return isinstance(mpl, tuple)


def quantize_linear_layers(
    model: spx.Module,
    /,
    *,
    quantization_config: QuantizationConfig | None = None,
    verbose: bool = True,
) -> spx.Module:
    """Quantize matching ``ParallelLinear`` layers in *model* in place.

    Thin wrapper around :class:`EasyQuantizer` that swaps the weight
    parameters of selected linear layers for quantized counterparts
    (NF4, INT8, MXFP8, etc.) according to *quantization_config*. When
    *quantization_config* is ``None`` the model is returned unchanged so
    callers can opt out by passing through a config value of ``None``.

    Args:
        model: EasyDeL model whose linear layers should be quantized.
        quantization_config: Configuration selecting the quantization
            dtype, group size, and layer-name regex. ``None`` short-circuits
            and returns *model* unchanged.
        verbose: When ``True`` render a tqdm progress bar while iterating
            over candidate layers.

    Returns:
        spx.Module: The same *model* instance with quantized weights
        installed on the matching layers.
    """
    if quantization_config is None:
        return model

    quantizer = EasyQuantizer(quantization_config=quantization_config)
    return quantizer.apply_quantization(model, verbose=verbose)


def apply_lora_to_layers(
    model: spx.Module,
    /,
    *,
    lora_rank: int,
    lora_pattern: str | None = None,
    verbose: bool = True,
    rngs: spx.Rngs | None = None,
) -> spx.Module:
    """Wrap matching ``ParallelLinear`` modules with EasyDeL's LoRA adapter.

    Args:
        model: The EasyDeL model to modify.
        lora_rank: Rank of the low-rank adapter matrices. Must be positive.
        lora_pattern: A regular expression pattern to match the names of
            modules to which LoRA should be applied. Defaults to ``".*"`` so
            every ``ParallelLinear`` encountered is wrapped.
        verbose: Whether to display a progress bar while traversing modules.
        rngs: Random source used to initialize the LoRA adapter weights. When
            omitted, ``spx.Rngs(0)`` is used.

    Returns:
        The input model after matching layers have been replaced in-place by
        :class:`eLoRA` wrappers.

    Notes:
        The wrapper used here is EasyDeL's ``eLoRA`` instead of SpecTrax's raw
        ``nn.LoRA`` so the adapted layers continue to support EasyDeL-specific
        call conventions such as keyword forwarding and ``native_forward``.
    """
    from easydel.utils.traversals import get_module_from_path, iter_module_search, set_module_from_path

    if not (lora_rank > 0):
        raise ValueError("lora_rank should be a positive value and higher than `0`.")
    if lora_pattern is None:
        lora_pattern = ".*"
    if rngs is None:
        rngs = spx.Rngs(0)
    pattern = re.compile(lora_pattern)

    with tqdm(
        total=len([p[0] for p in iter_module_search(model, ParallelLinear)]),
        desc="Applying LoRA",
        disable=not verbose,
    ) as pbar:
        for path, _ in iter_module_search(model, ParallelLinear):
            if pattern.search(".".join([str(p) for p in path])):
                base_module: ParallelLinear = get_module_from_path(model=model, path=path)
                set_module_from_path(
                    model=model,
                    path=path,
                    new_value=eLoRA(
                        base_module.in_features,
                        lora_rank,
                        base_module.out_features,
                        base_module=base_module,
                        rngs=rngs,
                        dtype=base_module.dtype,
                    ),
                )
            pbar.update(1)

    return model


def split_lora_parameters(model: spx.Module) -> tp.Any:
    """Extract the ``lora_a`` / ``lora_b`` adapter weights from every LoRA layer.

    Walks *model* looking for ``nn.LoRA`` wrappers and harvests their two
    adapter parameter slots into a nested dict keyed by the module path. This
    is the inverse of :func:`merge_lora_parameters` and is used by checkpoint
    pipelines that want to store the adapter weights separately from the
    frozen base model.

    Args:
        model: EasyDeL model containing LoRA-wrapped linear layers.

    Returns:
        Any: A nested dict where each leaf is ``{"lora_a": ..., "lora_b": ...}``
        for one LoRA module, keyed by the module's traversal path.
    """
    from easydel.utils.traversals import get_module_from_path, iter_module_search

    od = {}
    with tqdm(
        total=len([p[0] for p in iter_module_search(model, nn.LoRA)]),
        desc="Split LoRA Params",
    ) as pbar:
        for path, _ in iter_module_search(model, nn.LoRA):
            base_module: nn.LoRA = get_module_from_path(model=model, path=path)
            od.update({path: {"lora_a": base_module.lora_a, "lora_b": base_module.lora_b}})
            pbar.update(1)
    return unflatten_dict(od)


def merge_lora_parameters(model: spx.Module, lora_tree: dict) -> spx.Module:
    """Restore LoRA adapter weights from *lora_tree* back onto *model*.

    Inverse of :func:`split_lora_parameters`: walks every ``nn.LoRA`` wrapper
    in *model* and overwrites its ``lora_a`` and ``lora_b`` slots with the
    values stored under the corresponding path in *lora_tree*. Accepts both
    nested and pre-flattened lora trees.

    Args:
        model: EasyDeL model containing LoRA-wrapped layers whose adapters
            should be restored.
        lora_tree: Mapping from module path tuples to dicts of
            ``{"lora_a": ..., "lora_b": ...}``. May be in nested or
            ``flatten_dict`` form.

    Returns:
        spx.Module: The same *model* with LoRA adapter values updated
        in-place.
    """
    from easydel.utils.traversals import get_module_from_path, iter_module_search

    if not is_flatten(lora_tree):
        lora_tree = flatten_dict(lora_tree)
    with tqdm(
        total=len([p[0] for p in iter_module_search(model, nn.LoRA)]),
        desc="Merge LoRA Params",
    ) as pbar:
        for path, _ in iter_module_search(model, nn.LoRA):
            base_module: nn.LoRA = get_module_from_path(model=model, path=path)
            base_module.lora_b = lora_tree[(*path, "lora_b")]
            base_module.lora_a = lora_tree[(*path, "lora_a")]
            pbar.update(1)
    return model


def unwrap_lora_to_layers(
    model: spx.Module,
    /,
    *,
    verbose: bool = True,
) -> spx.Module:
    """Merge LoRA adapters into their base linear layers and drop the wrappers.

    Walks *model* finding every ``nn.LoRA`` wrapper, computes the merged
    weight ``W + A @ B`` (using ``float32`` matmul precision so the merge
    is numerically faithful regardless of the storage dtype), writes the
    result back onto the underlying linear's ``weight``, and replaces the
    wrapper in the module tree with its unwrapped ``base_module``.

    Args:
        model: EasyDeL model with LoRA-wrapped linear layers.
        verbose: When ``True`` render a tqdm progress bar while iterating
            over LoRA wrappers.

    Returns:
        spx.Module: The same *model* with all LoRA wrappers replaced by
        their merged base linear modules.
    """
    from easydel.utils.traversals import get_module_from_path, iter_module_search, set_module_from_path

    lora_paths = [p[0] for p in iter_module_search(model, nn.LoRA)]
    with tqdm(
        total=len(lora_paths),
        desc="Unwrapping LoRA Layers",
        disable=not verbose,
    ) as pbar:
        for path, _ in iter_module_search(model, nn.LoRA):
            base_module: nn.LoRA = get_module_from_path(model=model, path=path)
            with jax.default_matmul_precision("float32"):
                base_module.base_module.weight.value = (
                    base_module.base_module.weight.value + base_module.lora_a.value @ base_module.lora_b.value
                )
            del base_module.lora_a, base_module.lora_b
            set_module_from_path(
                model=model,
                path=path,
                new_value=base_module.base_module,
            )
            pbar.update(1)

    return model


def apply_sparsity_to_params(
    params: dict[str, tp.Any] | tp.Any,
    sparsify_module: AVAILABLE_SPARSE_MODULE_TYPES = "bcoo",
    verbose: bool = True,
) -> dict[str, tp.Any] | tp.Any:
    """Convert dense weight matrices in *params* into a JAX sparse format.

    Walks the parameter pytree and rewrites every 2D/3D ``weight`` leaf into
    the requested ``jax.experimental.sparse`` container.

    Args:
        params: Parameter pytree (flat or nested dict).
        sparsify_module: Sparse format name (``"bcoo"``, ``"bcsr"``, ``"coo"``
            or ``"csr"``).
        verbose: Whether to render a tqdm progress bar.

    Returns:
        dict | Any: A pytree mirroring *params*' nesting with sparsified
        weights.

    Raises:
        ValueError: If ``sparsify_module`` is not recognized.
    """
    flatten = is_flatten(params)
    if not flatten:
        params = flatten_dict(params)
    from jax.experimental import sparse

    sparser = {
        "bcoo": sparse.BCOO,
        "bcsr": sparse.BCSR,
        "coo": sparse.COO,
        "csr": sparse.CSR,
    }.get(sparsify_module, None)
    if sparser is None:
        raise ValueError(f"unknown type of sparser {sparsify_module}")

    def _path_to_str(path):
        """Render a JAX pytree key path as a dotted string.

        Args:
            path: Iterable of pytree key objects (``DictKey``,
                ``GetAttrKey``, ``SequenceKey``, etc.).

        Returns:
            str: A dotted joining of human-readable key names.
        """
        path_keys = []
        for key in path:
            if hasattr(key, "key"):
                path_keys.append(str(key.key))
            elif hasattr(key, "name"):
                path_keys.append(str(key.name))
            elif hasattr(key, "idx"):
                path_keys.append(str(key.idx))
            else:
                path_keys.append(str(key))
        return ".".join(path_keys)

    def filter_params(path, array):
        """Sparsify only ``weight`` leaves with rank in ``{2, 3}``.

        Args:
            path: Pytree key path identifying *array*.
            array: A parameter leaf.

        Returns:
            Any: The sparsified array, or the original leaf when not
            eligible.
        """
        layer_name = _path_to_str(path)
        if layer_name.endswith("weight") and 4 > array.ndim > 1:
            array = sparser.fromdense(array)
        return array

    total_params = len(jax.tree_util.tree_leaves(params))
    with tqdm(
        total=total_params,
        desc=f"{sparsify_module.capitalize()}",
        disable=not verbose,
    ) as pbar:

        def _with_progress(path, array):
            """Sparsify one leaf and advance the surrounding progress bar.

            Args:
                path: Pytree key path for the leaf.
                array: The leaf value.

            Returns:
                Any: The (possibly sparsified) array.
            """
            pbar.set_postfix_str(_path_to_str(path))
            result = filter_params(path, array)
            pbar.update(1)
            return result

        params = jax.tree_util.tree_map_with_path(_with_progress, params)

    if not flatten:
        params = unflatten_dict(params)
    return params


def extract_static_parameters(module):
    """Resolve ``static_argnums`` for a module's ``forward`` / ``__call__``.

    Inspects *module* for a ``forward`` (preferred) or ``__call__`` method
    and scans its signature for parameter names that EasyDeL treats as
    JIT-static (``causal_mask``, ``frequencies``, ``output_attentions``,
    ``output_hidden_states``, ``output_router_logits``, ``mode``). The
    returned tuple of positional indices is suitable for passing to
    :func:`jax.jit` as ``static_argnums``.

    Args:
        module: An EasyDeL module class or instance whose call signature
            should be analyzed.

    Returns:
        tuple[int, ...] | None: Tuple of positional argument indices that
        should be treated as static, an empty tuple when the signature
        can't be inspected, or ``None`` if neither ``forward`` nor
        ``__call__`` is a Python function.
    """

    # Predefined list of parameters to check for static status
    target_params = [
        "causal_mask",
        "frequencies",
        "output_attentions",
        "output_hidden_states",
        "output_router_logits",
        "mode",
    ]
    obj = None
    for candidate_name in ("forward", "__call__"):
        candidate = getattr(module, candidate_name, None)
        if isinstance(candidate, (types.FunctionType, types.MethodType)):
            obj = candidate
            break
    if obj is not None:
        static_args = ()
        try:
            # Avoid inspect.unwrap() here; decorated callables can have cyclic
            # __wrapped__ chains in some runtimes (seen with remat wrappers).
            signature = inspect.signature(obj, follow_wrapped=False)
        except (TypeError, ValueError):
            return static_args
        for idx, (param_name, _param) in enumerate(signature.parameters.items()):
            if param_name in target_params:
                static_args += (idx,)
        return static_args
    return None


M = tp.TypeVar("M", bound=spx.Module)


@tp.overload
def auto_remat(  # pyright: ignore[reportOverlappingOverload]
    module: type[M],
    /,
    *,
    policy: EasyDeLGradientCheckPointers | str | tp.Callable = EasyDeLGradientCheckPointers.NONE,
    prevent_cse: bool = True,
    save_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> type[M]:
    """Single-module overload: see :func:`auto_remat` for full documentation."""
    ...


@tp.overload
def auto_remat(
    module1: type[M],
    module2: type[M],
    /,
    *,
    policy: EasyDeLGradientCheckPointers | str | tp.Callable = EasyDeLGradientCheckPointers.NONE,
    prevent_cse: bool = True,
    save_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> tuple[type[M], type[M]]:
    """Two-module overload: see :func:`auto_remat` for full documentation."""
    ...


@tp.overload
def auto_remat(
    *modules: type[M],
    policy: EasyDeLGradientCheckPointers | str | tp.Callable = EasyDeLGradientCheckPointers.NONE,
    prevent_cse: bool = True,
    save_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> tuple[type[M], ...]:
    """Variadic overload: see :func:`auto_remat` for full documentation."""
    ...


def auto_remat(
    *modules: type[M],
    policy: EasyDeLGradientCheckPointers | str | tp.Callable = EasyDeLGradientCheckPointers.NONE,
    prevent_cse: bool = True,
    save_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> type[M] | tuple[type[M], ...]:
    """Apply gradient checkpointing (rematerialization) to module(s).

    Wraps module ``forward`` methods with JAX's remat (rematerialization) to trade
    compute for memory during training. Supports fine-grained control via
    checkpoint_name annotations added to models.

    Args:
        *modules: One or more module classes to wrap with remat.
        policy: Checkpointing policy. Can be:
            - EasyDeLGradientCheckPointers enum value
            - String policy name (e.g., 'dots_saveable', 'nothing_saveable')
            - Custom callable policy (e.g., from create_transformer_checkpoint_policy)
            - 'save_only_these_names': Use with save_names param
            - 'save_anything_except_these_names': Use with exclude_names param
        prevent_cse: If True, prevents common subexpression elimination.
        save_names: List of checkpoint names to save (for 'save_only_these_names').
            Works with checkpoint_name calls in models.
        exclude_names: List of checkpoint names to exclude from saving.

    Returns:
        Single module or tuple of modules with remat applied.

    Examples:
        >>> # Basic usage with predefined policy
        >>> AttentionModule = auto_remat(AttentionModule, policy='dots_saveable')
        >>>
        >>> # Multiple modules
        >>> AttentionModule, MLPModule = auto_remat(
        ...     AttentionModule, MLPModule,
        ...     policy='nothing_saveable'
        ... )
        >>>
        >>> # Custom policy saving only specific checkpoints
        >>> model = auto_remat(
        ...     model,
        ...     policy='save_only_these_names',
        ...     save_names=['attn_output', 'mlp_output', 'residual']
        ... )
        >>>
        >>> # Using transformer-optimized policy
        >>> policy = create_transformer_checkpoint_policy(
        ...     save_attention=True,
        ...     save_mlp=False  # Recompute MLP to save memory
        ... )
        >>> model = auto_remat(model, policy=policy)
    """
    if policy == EasyDeLGradientCheckPointers.NONE or policy in ["", "none"]:
        if len(modules) == 1:
            return modules[0]
        return modules
    if isinstance(policy, (str, EasyDeLGradientCheckPointers)):
        policy = get_gradient_checkpoint_policy(policy, save_names, exclude_names)
    elif not callable(policy):
        raise ValueError(f"Invalid policy type: {type(policy)}")

    outs = ()
    for module in modules:
        assert issubclass(module, spx.Module)
        if getattr(module.forward, "_easydel_auto_remat_wrapped", False):
            outs += (module,)
            continue

        static_argnums = extract_static_parameters(module=module)
        if static_argnums is None:
            static_argnums = ()

        rematted_forward = spx.remat(
            fn=module.forward,
            prevent_cse=prevent_cse,
            policy=policy,
            mutable=["rng"],
        )
        setattr(rematted_forward, "_easydel_auto_remat_wrapped", True)  # noqa
        module.forward = rematted_forward

        outs += (module,)

    if len(outs) == 1:
        return outs[0]
    return outs


# Main FLOP counting function
def count_flop_jaxpr(jaxpr) -> int:
    """Estimate the FLOP count of a JAX ``Jaxpr`` by traversing its equations.

    Dispatches each equation's primitive through a hand-written cost table
    (``primitive_flops``) that knows how to estimate FLOPs for the common
    JAX primitives — element-wise binary/unary ops, ``dot_general``,
    convolutions, reductions, the fused attention primitive, and so on. Higher
    cost-order operations like ``log``, ``exp``, ``sqrt``, ``erf_inv`` and
    activation surrogates are charged a fixed multiplicative cost per element.
    Unknown primitives emit a warning and contribute zero FLOPs. Subjaxprs
    (e.g. inside ``scan`` / ``cond`` / ``custom_vjp_call_jaxpr`` /
    ``remat2``) are visited recursively.

    Args:
        jaxpr: A ``jax.core.Jaxpr`` (typically obtained from
            ``jax.make_jaxpr(fn)(*args).jaxpr``).

    Returns:
        int: Aggregate FLOP estimate. Memory ops (reshape, broadcast,
        gather, scatter metadata, sharding constraints, etc.) contribute
        zero.
    """

    def get_shape_size(shape) -> int:
        """Calculate total size of an array shape."""
        return int(np.prod(shape)) if shape else 1

    def compute_binary_op_flops(eqn) -> int:
        """Generic FLOP counter for binary operations with broadcasting."""
        shape0 = eqn.invars[0].aval.shape
        shape1 = eqn.invars[1].aval.shape
        output_shape = np.broadcast_shapes(shape0, shape1)
        return get_shape_size(output_shape)

    def compute_unary_op_flops(eqn) -> int:
        """FLOP counter for unary operations."""
        shape = eqn.invars[0].aval.shape
        return get_shape_size(shape)

    def compute_dot_general_flops(eqn) -> int:
        """Compute FLOPs for dot_general operation."""
        shapes = [var.aval.shape for var in eqn.invars]
        if len(shapes) != 2:
            return 0

        dimension_numbers = eqn.params.get("dimension_numbers", None)
        if not dimension_numbers:
            return 0

        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

        # Calculate sizes for contracting dimensions
        contracting_size = np.prod([shapes[0][d] for d in lhs_contract])

        # Calculate output shape size
        batch_size = np.prod([shapes[0][d] for d in lhs_batch])
        lhs_remaining = [d for i, d in enumerate(shapes[0]) if i not in lhs_contract and i not in lhs_batch]
        rhs_remaining = [d for i, d in enumerate(shapes[1]) if i not in rhs_contract and i not in rhs_batch]
        out_size = batch_size * np.prod(lhs_remaining) * np.prod(rhs_remaining)

        # Each output element requires 2*contracting_size - 1 operations
        return int(out_size * (2 * contracting_size - 1))

    def compute_conv_flops(eqn) -> int:
        """Compute FLOPs for convolution operation."""
        lhs_shape = eqn.invars[0].aval.shape
        rhs_shape = eqn.invars[1].aval.shape

        dimension_numbers = eqn.params.get("dimension_numbers", None)
        if not dimension_numbers:
            return 0

        lhs_spec, rhs_spec, _out_spec = dimension_numbers

        batch_size = lhs_shape[lhs_spec.index("N")]
        in_channels = lhs_shape[lhs_spec.index("C")]
        out_channels = rhs_shape[rhs_spec.index("O")]

        spatial_size = 1
        kernel_size = 1
        for d in range(len(lhs_spec) - 2):
            spatial_size *= lhs_shape[lhs_spec.index(str(d))]
            kernel_size *= rhs_shape[rhs_spec.index(str(d))]

        ops_per_point = 2 * kernel_size * in_channels - 1
        total_points = batch_size * spatial_size * out_channels

        return ops_per_point * total_points

    def compute_reduce_flops(eqn) -> int:
        """Compute FLOPs for reduction operations."""
        shape = eqn.invars[0].aval.shape
        reduced_axes = eqn.params.get("axes", ())

        if not reduced_axes:
            return 0

        reduced_size = np.prod([shape[ax] for ax in reduced_axes])
        remaining_shape = [s for i, s in enumerate(shape) if i not in reduced_axes]
        remaining_size = np.prod(remaining_shape) if remaining_shape else 1

        return int(remaining_size * (reduced_size - 1))

    def compute_attention_flops(eqn) -> int:
        """Compute FLOPs for attention operation."""
        q_shape = eqn.invars[0].aval.shape
        k_shape = eqn.invars[1].aval.shape

        batch, q_len, num_heads, head_dim = q_shape
        _, kv_len, _, _ = k_shape

        qk_flops = batch * num_heads * q_len * kv_len * (2 * head_dim - 1)
        softmax_flops = batch * num_heads * q_len * (kv_len + (kv_len - 1) + 1)
        av_flops = batch * num_heads * q_len * head_dim * (2 * kv_len - 1)

        return qk_flops + softmax_flops + av_flops

    def count_scan_flops(eqn) -> int:
        """Count FLOPs in a scan operation."""
        scan_jaxpr = eqn.params.get("jaxpr", None)
        if scan_jaxpr:
            body_flops = count_flop_jaxpr(scan_jaxpr)
            length = eqn.invars[0].aval.shape[0]
            return body_flops * length
        return 0

    def count_cond_flops(eqn) -> int:
        """Count FLOPs in a conditional operation."""
        true_jaxpr = eqn.params.get("true_jaxpr", None)
        false_jaxpr = eqn.params.get("false_jaxpr", None)

        total_flops = 0
        if true_jaxpr:
            total_flops += count_flop_jaxpr(true_jaxpr)
        if false_jaxpr:
            total_flops += count_flop_jaxpr(false_jaxpr)
        return total_flops // 2

    def get_scatter_flops(eqn) -> int:
        """Count FLOPs in a scatter operation."""
        updates_shape = eqn.invars[2].aval.shape
        return get_shape_size(updates_shape)

    def compute_select_n_flops(eqn) -> int:
        """Compute FLOPs for select_n operation."""
        pred_shape = eqn.invars[0].aval.shape
        return get_shape_size(pred_shape)

    def compute_cumsum_flops(eqn) -> int:
        """Compute FLOPs for cumulative sum."""
        shape = eqn.invars[0].aval.shape
        axis = eqn.params.get("axis", 0)
        # Each element adds to the previous sum
        return get_shape_size(shape) - shape[axis]

    def compute_max_flops(eqn) -> int:
        """Compute FLOPs for max operation."""
        if len(eqn.invars) == 2:
            # Binary max
            return compute_binary_op_flops(eqn)
        # Unary max
        return compute_unary_op_flops(eqn)

    def compute_pow_flops(eqn) -> int:
        """Compute FLOPs for power operation."""
        if len(eqn.invars) == 2:
            shape0 = eqn.invars[0].aval.shape
            shape1 = eqn.invars[1].aval.shape
            output_shape = np.broadcast_shapes(shape0, shape1)
            return 8 * get_shape_size(output_shape)  # Power is expensive
        return 8 * get_shape_size(eqn.invars[0].aval.shape)

    def compute_integer_pow_flops(eqn) -> int:
        """Compute FLOPs for integer power."""
        shape = eqn.invars[0].aval.shape
        power = eqn.params.get("y", 2)
        return (power - 1) * get_shape_size(shape)

    def compute_and_flops(eqn) -> int:
        """Compute FLOPs for logical and operation."""
        return compute_binary_op_flops(eqn)

    def count_custom_vjp_flops(eqn) -> int:
        """Count FLOPs in custom VJP operation."""
        fwd_jaxpr = eqn.params.get("fun_jaxpr", None)
        if fwd_jaxpr:
            return count_flop_jaxpr(fwd_jaxpr)
        return 0

    def compute_sqrt_flops(eqn) -> int:
        """Compute FLOPs for square root operation."""
        # Square root is typically more expensive than basic operations
        return 4 * compute_unary_op_flops(eqn)

    def compute_argmax_flops(eqn) -> int:
        """Compute FLOPs for argmax operation."""
        shape = eqn.invars[0].aval.shape
        axis = eqn.params.get("axes", (0,))[0]
        # For each output element, we need to compare n-1 elements where n is the size of the reduction axis
        remaining_size = get_shape_size(shape) // shape[axis]
        return remaining_size * (shape[axis] - 1)

    def compute_min_flops(eqn) -> int:
        """Compute FLOPs for min operation."""
        if len(eqn.invars) == 2:
            # Binary min
            return compute_binary_op_flops(eqn)
        # Unary min
        return compute_unary_op_flops(eqn)

    def compute_rem_flops(eqn) -> int:
        """Compute FLOPs for remainder operation."""
        # Remainder typically involves division and multiplication
        return 2 * compute_binary_op_flops(eqn)

    def compute_square_flops(eqn) -> int:
        """Compute FLOPs for square operation (x * x)."""
        # Square is a single multiplication of a number by itself
        shape = eqn.invars[0].aval.shape
        return get_shape_size(shape)

    def compute_triangular_solve_flops(eqn) -> int:
        """Compute FLOPs for triangular solve operation."""
        # For a triangular solve with a matrix of size n x n,
        # each row/column requires n^2/2 multiply-adds
        matrix_shape = eqn.invars[0].aval.shape
        n = matrix_shape[-1]  # Size of the last dimension
        batch_dims = matrix_shape[:-2]
        batch_size = np.prod(batch_dims) if batch_dims else 1
        return batch_size * n * (n + 1) * (2 * n + 1) // 6

    def compute_erf_inv_flops(eqn) -> int:
        """Compute FLOPs for inverse error function."""
        # erf_inv is computationally expensive, typically implemented
        # as a series expansion or numerical approximation
        return 15 * compute_unary_op_flops(eqn)

    def compute_or_flops(eqn) -> int:
        """Compute FLOPs for logical or operation."""
        return compute_binary_op_flops(eqn)

    def compute_shift_right_logical_flops(eqn) -> int:
        """Compute FLOPs for logical right shift."""
        return compute_binary_op_flops(eqn)

    # Dictionary mapping primitives to their FLOP counting functions
    primitive_flops: dict[str, tp.Callable] = {
        # Binary operations
        "mul": compute_binary_op_flops,
        "add": compute_binary_op_flops,
        "sub": compute_binary_op_flops,
        "div": compute_binary_op_flops,
        "gt": compute_binary_op_flops,
        "lt": compute_binary_op_flops,
        "ge": compute_binary_op_flops,
        "le": compute_binary_op_flops,
        "ne": compute_binary_op_flops,
        "eq": compute_binary_op_flops,
        # Unary operations
        "neg": compute_unary_op_flops,
        "sin": lambda eqn: 5 * compute_unary_op_flops(eqn),
        "cos": lambda eqn: 5 * compute_unary_op_flops(eqn),
        "exp": lambda eqn: 4 * compute_unary_op_flops(eqn),
        "log": lambda eqn: 6 * compute_unary_op_flops(eqn),
        "log1p": lambda eqn: 6 * compute_unary_op_flops(eqn),
        "tanh": lambda eqn: 7 * compute_unary_op_flops(eqn),
        "rsqrt": lambda eqn: 6 * compute_unary_op_flops(eqn),
        # Linear algebra
        "dot_general": compute_dot_general_flops,
        "conv_general_dilated": compute_conv_flops,
        # Reduction operations
        "reduce_sum": compute_reduce_flops,
        "reduce_max": compute_reduce_flops,
        "reduce_min": compute_reduce_flops,
        # Special operations
        "scatter-add": get_scatter_flops,
        "scan": count_scan_flops,
        "cond": count_cond_flops,
        # Memory operations (0 FLOPs)
        "broadcast_in_dim": lambda eqn: 0,
        "reshape": lambda eqn: 0,
        "transpose": lambda eqn: 0,
        "slice": lambda eqn: 0,
        "gather": lambda eqn: 0,
        "concatenate": lambda eqn: 0,
        "convert_element_type": lambda eqn: 0,
        "dynamic_slice": lambda eqn: 0,
        "pad": lambda eqn: 0,
        # Parallel/Sharding operations (0 FLOPs)
        "pjit": lambda eqn: 0,
        "shard_map": lambda eqn: 0,
        "sharding_constraint": lambda eqn: 0,
        # Other operations
        "dot_product_attention_fwd_wrapper": compute_attention_flops,
        "select_n": compute_select_n_flops,
        "cumsum": compute_cumsum_flops,
        "max": compute_max_flops,
        "iota": lambda eqn: 0,  # Memory operation, no FLOPs
        "pow": compute_pow_flops,
        "integer_pow": compute_integer_pow_flops,
        "and": compute_and_flops,
        "random_fold_in": lambda eqn: 0,  # Random number generation, no FLOPs
        "custom_vjp_call_jaxpr": count_custom_vjp_flops,
        "logistic": lambda eqn: 4 * compute_unary_op_flops(eqn),  # sigmoid function
        # No-op operations (0 FLOPs)
        "stop_gradient": lambda eqn: 0,  # Just passes through the value
        "squeeze": lambda eqn: 0,  # Reshapes data, no computation
        "copy": lambda eqn: 0,  # Memory operation only
        "split": lambda eqn: 0,
        "remat2": lambda eqn: 0,
        "random_seed": lambda eqn: 0,
        "random_unwrap": lambda eqn: 0,
        "random_wrap": lambda eqn: 0,
        "random_split": lambda eqn: 0,
        "random_bits": lambda eqn: 0,
        # Bitwise and type conversion operations
        "shift_right_logical": compute_shift_right_logical_flops,
        "or": compute_or_flops,
        "bitcast_convert_type": lambda eqn: 0,  # Type conversion, no computation
        # Mathematical operations
        "abs": compute_unary_op_flops,  # Single comparison/selection per element
        "erf_inv": compute_erf_inv_flops,  # Inverse error function
        "triangular_solve": compute_triangular_solve_flops,
        # Computation operations
        "square": compute_square_flops,
        "sqrt": compute_sqrt_flops,
        "argmax": compute_argmax_flops,
        "add_any": compute_binary_op_flops,  # Similar to regular add
        "min": compute_min_flops,
        "rem": compute_rem_flops,
    }

    flops = 0

    def visit_jaxpr(jaxpr):
        """Walk a jaxpr and accumulate FLOP estimates into the outer scope.

        Args:
            jaxpr: The jaxpr to traverse.

        Returns:
            None. ``flops`` from the enclosing scope is updated in-place.
        """
        nonlocal flops
        for eqn in jaxpr.eqns:
            primitive_name = eqn.primitive.name
            if primitive_name in primitive_flops:
                flops += primitive_flops[primitive_name](eqn)
            else:
                warnings.warn(f"Unhandled primitive {primitive_name}", stacklevel=1)

            # Recursively visit subjaxprs
            for subjaxpr in jax.core.jaxprs_in_params(eqn.params):
                visit_jaxpr(subjaxpr)

    visit_jaxpr(jaxpr)
    return flops


class TraceResult:
    """Container for XLA executable trace results with cost analysis.

    Wraps an XLA executable and provides lazy access to its cost analysis,
    including FLOP counts and other performance metrics.

    Attributes:
        _executable: The underlying XLA executable.
        _cached_cost: Cached cost analysis result.

    Properties:
        cost_analysis: Returns the cost analysis dict (cached after first access).
        flops: Returns the FLOP count from cost analysis.
    """

    def __init__(self, executable):
        """Wrap an XLA executable and prepare lazy cost-analysis caching.

        Args:
            executable: The compiled XLA executable.

        Returns:
            None.
        """
        self._executable = executable
        self._cached_cost = None

    @property
    @lru_cache(maxsize=1)  # noqa
    def cost_analysis(self):
        """Return the executable's cost-analysis dict (cached on first access).

        Returns:
            dict: XLA cost-analysis output (FLOPs, bytes accessed, etc.).
        """
        return self._executable.cost_analysis()

    @property
    def flops(self):
        """Return the ``flops`` entry from the cached cost analysis.

        Returns:
            int | float: FLOP count reported by XLA.
        """
        return self.cost_analysis["flops"]


class FunctionTracer:
    """Tracer for capturing new XLA executables during compilation.

    Used to track which functions are compiled during a trace operation.
    Captures the difference between executables before and after tracing.

    Attributes:
        new_executables: List of TraceResult objects for newly compiled functions.
        _before: Set of executables that existed before tracing started.

    Example:
        >>> with trace_functions() as tracer:
        ...     result = jitted_function(x)
        >>> print(f"Compiled {len(tracer.new_executables)} functions")
        >>> print(f"Total FLOPs: {sum(t.flops for t in tracer.new_executables)}")
    """

    def __init__(self):
        """Initialize an empty tracer with no captured executables.

        Returns:
            None.
        """
        self.new_executables: list[TraceResult] = []
        self._before: set = set()

    def __getitem__(self, idx):
        """Return the *idx*-th captured :class:`TraceResult`.

        Args:
            idx: Sequence index into ``self.new_executables``.

        Returns:
            TraceResult: The captured trace at that index.
        """
        return self.new_executables[idx]


class CompilationTracker:
    """Tracks XLA compilation and FLOP counts across function calls.

    Monitors the compilation of XLA executables and accumulates their
    FLOP counts. Useful for profiling and understanding computational
    costs of JAX programs.

    Attributes:
        first_time: Whether this is the first compilation trace.
        cached_flops: Total accumulated FLOPs from all compiled functions.
        functions: List of compiled XLA executables.

    Properties:
        online_flops: Current total FLOPs from all tracked functions.

    Methods:
        trace_compilation: Context manager for tracing compilation.

    Example:
        >>> tracker = CompilationTracker()
        >>> with tracker.trace_compilation():
        ...     result = model(inputs)
        >>> print(f"Total FLOPs: {tracker.cached_flops}")
    """

    def __init__(self):
        """Initialize the tracker in a not-yet-traced state.

        Returns:
            None.
        """
        self.first_time = True
        self.cached_flops = 0
        self.functions = None

    @property
    def online_flops(self):
        """Sum FLOPs across the currently tracked executables.

        Returns:
            int | float: Aggregate FLOP count, or ``0`` if nothing has been
            traced yet or cost analysis fails for every executable.
        """
        if self.functions is None:
            return 0
        cached_flops = 0
        for cm in self.functions:
            try:
                cached_flops += cm.cost_analysis()["flops"]
            except Exception:
                ...
        return cached_flops

    @contextmanager
    def trace_compilation(self):
        """Capture executables compiled inside the ``with`` block.

        On the first invocation, records the set of live executables before
        and after the block, accumulates ``cost_analysis()`` FLOPs into
        ``cached_flops``, and stores the new executables on ``functions``.
        Subsequent invocations are no-ops.

        Yields:
            None: Use inside ``with tracker.trace_compilation():`` to record
            compilation that happens during the body.
        """
        if self.first_time:
            before = set(jax.extend.backend.get_backend().live_executables())
            yield
            after = set(jax.extend.backend.get_backend().live_executables())
            new = after - before
            if new:
                cmpf = list(new)
                self.functions = cmpf
                for cm in cmpf:
                    try:
                        self.cached_flops += cm.cost_analysis()["flops"]
                    except Exception:
                        ...
            self.first_time = False
        else:
            yield


class ActivationType(StrEnum):
    """Enumeration of activation functions recognized by FLOP estimators.

    Attributes:
        GELU: Standard Gaussian Error Linear Unit.
        RELU: Rectified Linear Unit.
        SILU: Sigmoid Linear Unit (a.k.a. Swish).
        SWISH: Synonym for ``SILU``.
        GELU_NEW: GELU approximation using tanh (HF "new" variant).
        GELU_PYTORCH_TANH: PyTorch tanh-based GELU.
        TANH: Hyperbolic tangent.
        SIGMOID: Logistic sigmoid.
        LEAKY_RELU: ReLU with non-zero negative slope.
        GLU: Gated Linear Unit.
        ELU: Exponential Linear Unit.
        SOFTMAX: Softmax (classification head activation).
        QUICK_GELU: Inexpensive GELU approximation ``x * sigmoid(1.702 * x)``.
    """

    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"
    SWISH = "swish"
    GELU_NEW = "gelu_new"
    GELU_PYTORCH_TANH = "gelu_pytorch_tanh"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    GLU = "glu"
    ELU = "elu"
    SOFTMAX = "softmax"
    QUICK_GELU = "quick_gelu"


def flop_activation(activation_type: ActivationType, dim: int) -> float:
    """Estimate FLOPs for applying an activation across ``dim`` elements.

    Uses a fixed per-element cost table that approximates how many
    floating-point operations a given activation costs on common
    accelerators (e.g. ReLU = 1, GELU = 8, TANH = 5). The total is the
    per-element cost multiplied by ``dim``.

    Args:
        activation_type: Activation kind from :class:`ActivationType`.
            Unknown values fall back to a per-element cost of ``1``.
        dim: Number of elements the activation is applied to.

    Returns:
        float: Estimated total FLOP count.
    """

    # FLOPs per element for different activation functions
    flops_per_element = {
        ActivationType.GELU: 8,  # Approximation with several operations
        ActivationType.GELU_NEW: 8,  # Approximation with tanh
        ActivationType.GELU_PYTORCH_TANH: 8,  # Similar to GELU_NEW
        ActivationType.RELU: 1,  # Just a max operation
        ActivationType.SILU: 4,  # x * sigmoid(x) - sigmoid + multiplication
        ActivationType.SWISH: 4,  # Same as SILU
        ActivationType.TANH: 5,  # Approximation of tanh
        ActivationType.SIGMOID: 4,  # Approximation of sigmoid
        ActivationType.LEAKY_RELU: 2,  # Comparison + multiplication for negative slope
        ActivationType.GLU: 5,  # Gated operation - sigmoid + multiplication
        ActivationType.ELU: 2,  # Comparison + exp for negative values
        ActivationType.SOFTMAX: 5,  # Similar cost as sigmoid + normalization
        ActivationType.QUICK_GELU: 2,  # Simple approximation x * sigmoid(1.702 * x)
    }
    return flops_per_element.get(activation_type, 1) * dim


class AttnMaskType(StrEnum):
    """Coarse attention-mask categories used by the eSurge scheduler.

    Attributes:
        FULL: Full causal/non-causal attention.
        SLIDING: Sliding-window attention.
        CHUNK: Chunked / blockwise attention.
        LINEAR: Linear-attention layer (treated as full for cache grouping).
    """

    FULL = "ATTN_MASK_FULL"
    SLIDING = "ATTN_MASK_SLIDING"
    CHUNK = "ATTN_MASK_CHUNK"
    LINEAR = "ATTN_MASK_LINEAR"

    @classmethod
    def from_hf(
        cls,
        hf_type: tp.Literal[
            "sliding_attention",
            "full_attention",
            "chunk_attention",
            "chunked_attention",
            "linear_attention",
            "kda_linear_attention",
            "hybrid",
            "parallel_hybrid",
        ],
    ):
        """Map a HuggingFace ``layer_types`` string to an :class:`AttnMaskType`.

        Args:
            hf_type: HF layer-type identifier.

        Returns:
            AttnMaskType: The corresponding eSurge mask category.

        Raises:
            ValueError: If *hf_type* is not recognized.
        """
        if hf_type == "sliding_attention":
            return AttnMaskType.SLIDING
        elif hf_type in ("full_attention", "linear_attention", "kda_linear_attention", "hybrid", "parallel_hybrid"):
            # eSurge cache grouping is page-table based; linear attention layers
            # and parallel hybrid layers (attention+SSM) are treated as
            # full-attention groups for scheduler compatibility.
            return AttnMaskType.FULL
        elif hf_type in ["chunk_attention", "chunked_attention"]:
            return AttnMaskType.CHUNK
        else:
            raise ValueError(f"`hf_type` {hf_type} is not available")


@auto_pytree
class AttnMaskDetail:
    """Details for attention mask configuration.

    Specifies the type and parameters of attention masking to use.
    Registered as a JAX pytree for use in JAX transformations.

    Attributes:
        mask_type: Type of attention mask (FULL, SLIDING, or CHUNK).
        size: Size parameter for the mask (e.g., window size for sliding).
        offset: Optional offset for mask positioning.
        chunks: Optional number of chunks for chunk attention.
        bricks: Optional number of bricks for hierarchical attention.

    Example:
        >>> mask_detail = AttnMaskDetail(
        ...     mask_type=AttnMaskType.SLIDING,
        ...     size=512,
        ...     offset=0
        ... )
    """

    mask_type: AttnMaskType
    size: int
    offset: int | None = None
    chunks: int | None = None
    bricks: int | None = None


from easydel.infra.factory import TaskType  # noqa: E402


@dataclass
class FlopCalcConfig:
    """Configuration for calculating FLOPs in transformer models.

    Comprehensive configuration that captures all parameters needed to
    calculate the theoretical FLOP count for various transformer architectures
    including encoder-decoder, MoE, and vision transformers.

    Attributes:
        hidden_dim: Hidden dimension of the model.
        intermediate_dim: Dimension of FFN intermediate layer.
        num_layers: Number of decoder (or encoder-only) layers.
        num_heads: Number of attention heads.
        kv_heads: Number of key-value heads (for GQA/MQA).
        head_dim: Dimension of each attention head.
        seq_len: Sequence length for decoder or encoder-only models.
        enc_num_layers: Number of encoder layers (for seq2seq).
        enc_seq_len: Encoder sequence length (for seq2seq).
        glu: Whether using GLU activation in FFN.
        num_experts: Number of MoE experts.
        num_shared_experts: Number of shared experts in MoE.
        num_experts_per_tok: Experts activated per token.
        activation_type: Type of activation function.
        task: Model task type (affects head computation).
        vocab_size: Vocabulary size for LM head.
        num_labels: Number of labels for classification.
        vision_hidden_dim: Hidden dim for vision transformer.
        vision_intermediate_dim: FFN dim for vision transformer.
        vision_num_layers: Number of vision transformer layers.
        vision_num_heads: Number of vision attention heads.
        vision_seq_len: Vision sequence length (patches).
        include_loss: Whether to include loss computation in FLOPs.

    Example:
        >>> config = FlopCalcConfig(
        ...     hidden_dim=768,
        ...     intermediate_dim=3072,
        ...     num_layers=12,
        ...     num_heads=12,
        ...     kv_heads=12,
        ...     head_dim=64,
        ...     seq_len=1024,
        ...     task=TaskType.CAUSAL_LM,
        ...     vocab_size=50000
        ... )
        >>> flops = flops_per_token(config)
    """

    # Core transformer body: for decoder-only and encoder-only models
    hidden_dim: int
    intermediate_dim: int
    num_layers: int  # number of decoder (or encoder-only) layers
    num_heads: int
    kv_heads: int
    head_dim: int
    seq_len: int  # decoder (or encoder-only) sequence length

    # Optional encoder for seq2seq / encoder-decoder
    enc_num_layers: int = 0
    enc_seq_len: int = 0

    # MoE / GLU
    glu: bool = False
    num_experts: int = 1
    num_shared_experts: int = 0
    num_experts_per_tok: int = 1
    moe_intermediate_dim: int | None = None
    shared_expert_intermediate_dim: int = 0
    num_moe_layers: int | None = None
    layer_types: Sequence[str] | None = None
    sliding_window: int | None = None

    # Task specifics
    activation_type: ActivationType = ActivationType.GELU
    task: TaskType = TaskType.AUTO_BIND
    vocab_size: int = 0
    num_labels: int = 0

    # Vision tower (patch transformer)
    vision_hidden_dim: int = 0
    vision_intermediate_dim: int = 0
    vision_num_layers: int = 0
    vision_num_heads: int = 0
    vision_seq_len: int | None = 0
    vision_head_dim: int | None = None
    vision_activation_type: ActivationType = ActivationType.GELU

    include_loss: bool = False


def flop_layernorm(hidden_dim: int) -> float:
    """Estimate FLOPs for a single LayerNorm/RMSNorm over ``hidden_dim``.

    Args:
        hidden_dim: Hidden dimension being normalized.

    Returns:
        float: FLOP estimate (~8 * hidden_dim).
    """
    return 8 * hidden_dim


def flop_attention(
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int | None,
    seq_len: int,
) -> float:
    """Estimate per-token FLOPs for a multi-head self-attention block.

    Args:
        hidden_dim: Model hidden size.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (for GQA/MQA).
        head_dim: Per-head dim, or ``None`` to derive ``hidden_dim //
            num_heads``.
        seq_len: Effective sequence length.

    Returns:
        float: FLOP estimate including QKV projection, attention scores,
        masking, weighted sum, and output projection.
    """
    if head_dim is None:
        head_dim = hidden_dim // num_heads
    qkv_proj = 2 * hidden_dim * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
    dense_proj = 2 * hidden_dim * hidden_dim
    key_query_logits = 2 * seq_len**2 * num_heads * head_dim
    mask = 3 * seq_len * seq_len * num_heads
    mask_value = 2 * seq_len * seq_len * head_dim * num_heads
    seq_flops = key_query_logits + mask + mask_value
    attn = seq_flops / seq_len
    return qkv_proj + dense_proj + attn


def flop_linear_attention(
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int | None,
    seq_len: int,
) -> float:
    """Estimate per-token FLOPs for linear-recurrent attention / state-space blocks.

    Models attention variants whose cost is linear in ``seq_len`` rather
    than quadratic — linear attention, Mamba-style SSMs, gated delta nets,
    etc. Includes QKV projections, the output dense projection, and a
    recurrent-state update term that scales with ``seq_len``.

    Args:
        hidden_dim: Model hidden size.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (for GQA/MQA).
        head_dim: Per-head dim, or ``None`` to derive ``hidden_dim //
            num_heads``.
        seq_len: Effective sequence length.

    Returns:
        float: FLOP estimate for the block.
    """
    if head_dim is None:
        head_dim = hidden_dim // num_heads
    qkv_proj = 2 * hidden_dim * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
    dense_proj = 2 * hidden_dim * hidden_dim
    recurrent_state = 4 * seq_len * num_heads * head_dim
    return qkv_proj + dense_proj + recurrent_state


def flop_cross_attention(
    hidden_dim: int,
    num_heads: int,
    enc_seq_len: int,
    dec_seq_len: int,
) -> float:
    """Estimate FLOPs for an encoder-decoder cross-attention block.

    Args:
        hidden_dim: Model hidden size.
        num_heads: Number of attention heads.
        enc_seq_len: Encoder sequence length (keys/values).
        dec_seq_len: Decoder sequence length (queries).

    Returns:
        float: Total cross-attention FLOP estimate.
    """
    head_dim = hidden_dim // num_heads
    proj = 2 * hidden_dim * hidden_dim
    scores = 2 * head_dim * enc_seq_len * dec_seq_len * num_heads
    softmax = 5 * enc_seq_len * dec_seq_len * num_heads
    wsum = 2 * head_dim * enc_seq_len * dec_seq_len * num_heads
    out_proj = 2 * hidden_dim * hidden_dim
    return proj + scores + softmax + wsum + out_proj


def flop_dense_mlp(
    cfg: FlopCalcConfig,
    hidden_dim: int,
    intermediate_dim: int,
) -> float:
    """Estimate FLOPs for a dense FFN block (up + activation + down, optional gate).

    GLU-style MLPs (``cfg.glu=True``) include a gate projection in addition
    to the up/down projections, so the matmul cost picks up an extra
    ``hidden_dim * intermediate_dim`` term.

    Args:
        cfg: Full FLOP-calc configuration (provides ``glu`` flag and
            activation type).
        hidden_dim: Hidden dimension entering the MLP.
        intermediate_dim: FFN intermediate dimension.

    Returns:
        float: Estimated FLOPs for the block.
    """
    factor = 3 if cfg.glu else 2
    matmuls = 2 * factor * hidden_dim * intermediate_dim
    activation_flops = flop_activation(cfg.activation_type, intermediate_dim)
    return matmuls + activation_flops


def flop_moe_mlp(
    cfg: FlopCalcConfig,
    hidden_dim: int,
    intermediate_dim: int,
) -> float:
    """Estimate FLOPs for a Mixture-of-Experts FFN block.

    Sums the active-expert FFN cost (each per-token activation hits
    ``num_experts_per_tok`` experts), the optional shared-expert cost
    (executed for every token, ``num_shared_experts`` times), and the
    router projection cost (``2 * hidden_dim * num_experts`` when MoE is
    enabled).

    Args:
        cfg: Full FLOP-calc configuration with MoE topology fields.
        hidden_dim: Hidden dimension entering the MLP.
        intermediate_dim: Default FFN intermediate dim used when
            ``cfg.moe_intermediate_dim`` and
            ``cfg.shared_expert_intermediate_dim`` are not set.

    Returns:
        float: Estimated FLOPs for the block.
    """
    expert_dim = cfg.moe_intermediate_dim or intermediate_dim
    active_expert_cost = flop_dense_mlp(cfg, hidden_dim, expert_dim) * max(cfg.num_experts_per_tok, 1)

    shared_dim = cfg.shared_expert_intermediate_dim or intermediate_dim
    shared_expert_cost = flop_dense_mlp(cfg, hidden_dim, shared_dim) * max(cfg.num_shared_experts, 0)

    router = 2 * hidden_dim * cfg.num_experts if cfg.num_experts > 1 else 0
    return active_expert_cost + shared_expert_cost + router


def _num_moe_layers(cfg: FlopCalcConfig, layers: int) -> int:
    """Return the number of MoE FFN layers contained in a trunk of ``layers`` blocks.

    Falls back to ``0`` when MoE is disabled (``num_experts <= 1``). When
    ``cfg.num_moe_layers`` is set explicitly, the result is clamped to
    ``[0, layers]``; otherwise every block is considered MoE.

    Args:
        cfg: FLOP-calc configuration carrying MoE topology fields.
        layers: Total number of transformer blocks in the trunk.

    Returns:
        int: Number of MoE-bearing layers (the remainder is dense FFN).
    """
    if cfg.num_experts <= 1:
        return 0
    if cfg.num_moe_layers is not None:
        return max(0, min(cfg.num_moe_layers, layers))
    return layers


def _attention_flops_for_layer(
    cfg: FlopCalcConfig,
    hidden_dim: int,
    seq_len: int,
    layer_type: str | None,
) -> float:
    """Estimate attention FLOPs for one layer, honoring its layer-type marker.

    Dispatches between :func:`flop_attention` (standard self-attention) and
    :func:`flop_linear_attention` (linear / Mamba / gated-delta blocks),
    and shortens ``seq_len`` to ``cfg.sliding_window`` for sliding-window
    layers so the per-token cost reflects the bounded receptive field.

    Args:
        cfg: Full FLOP-calc configuration.
        hidden_dim: Model hidden dimension.
        seq_len: Effective sequence length before sliding-window clamping.
        layer_type: Optional HuggingFace-style layer-type marker
            (e.g. ``"sliding_attention"``, ``"linear_attention"``); ``None``
            and unrecognized strings fall through to dense attention.

    Returns:
        float: Per-layer attention FLOPs.
    """
    layer_type = (layer_type or "").lower()
    if "sliding" in layer_type and cfg.sliding_window:
        seq_len = min(seq_len, cfg.sliding_window)
    if "linear" in layer_type or "mamba" in layer_type or "gated_delta" in layer_type:
        return flop_linear_attention(hidden_dim, cfg.num_heads, cfg.kv_heads, cfg.head_dim, seq_len)
    return flop_attention(hidden_dim, cfg.num_heads, cfg.kv_heads, cfg.head_dim, seq_len)


def flop_mlp(
    cfg: FlopCalcConfig,
    hidden_dim: int,
    intermediate_dim: int,
) -> float:
    """Estimate FLOPs for an MLP / MoE FFN block.

    Args:
        cfg: Model FLOP configuration (provides activation, MoE topology).
        hidden_dim: Hidden dimension.
        intermediate_dim: FFN intermediate dimension.

    Returns:
        float: FLOPs including up/down/gate projections, activation, and
        router cost when MoE is active.
    """
    if cfg.num_experts > 1:
        return flop_moe_mlp(cfg, hidden_dim, intermediate_dim)
    return flop_dense_mlp(cfg, hidden_dim, intermediate_dim)


def flop_lm_head(hidden_dim: int, vocab_size: int) -> float:
    """Estimate FLOPs for the LM head projection.

    Args:
        hidden_dim: Hidden dimension.
        vocab_size: Vocabulary size.

    Returns:
        float: ``2 * hidden_dim * vocab_size + 5 * vocab_size``.
    """
    return 2 * hidden_dim * vocab_size + 5 * vocab_size


def flop_cls_head(hidden_dim: int, num_labels: int) -> float:
    """Estimate FLOPs for a classification head.

    Args:
        hidden_dim: Hidden dimension.
        num_labels: Number of output classes.

    Returns:
        float: ``2 * hidden_dim * num_labels + 5 * num_labels``.
    """
    return 2 * hidden_dim * num_labels + 5 * num_labels


def flop_loss(num_classes: int) -> float:
    """Estimate FLOPs for the cross-entropy loss over ``num_classes``.

    Args:
        num_classes: Number of output classes.

    Returns:
        float: ``3 * num_classes + 2``.
    """
    return 3 * num_classes + 2


def flop_transformer_body(
    layers: int,
    seq_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    cfg: FlopCalcConfig,
) -> float:
    """Estimate FLOPs for the transformer trunk (attention + MLP + norms).

    Args:
        layers: Number of transformer blocks.
        seq_len: Sequence length.
        hidden_dim: Hidden dimension.
        intermediate_dim: FFN intermediate dimension.
        cfg: Full FLOP-calc configuration.

    Returns:
        float: Total FLOPs across all layers.
    """
    if layers <= 0 or seq_len <= 0 or hidden_dim <= 0 or intermediate_dim <= 0 or cfg.num_heads <= 0:
        return 0.0

    ln = 2 * flop_layernorm(hidden_dim)
    layer_types = list(cfg.layer_types or ())
    if len(layer_types) < layers:
        layer_types.extend([None] * (layers - len(layer_types)))

    att = sum(
        _attention_flops_for_layer(cfg, hidden_dim, seq_len, layer_types[layer_idx]) for layer_idx in range(layers)
    )
    moe_layers = _num_moe_layers(cfg, layers)
    dense_layers = layers - moe_layers
    dense_mlp = flop_dense_mlp(cfg, hidden_dim, intermediate_dim)
    moe_mlp = flop_moe_mlp(cfg, hidden_dim, intermediate_dim) if moe_layers else 0.0
    return layers * ln + att + dense_layers * dense_mlp + moe_layers * moe_mlp


def flop_seq2seq(cfg: FlopCalcConfig) -> float:
    """Estimate FLOPs for an encoder-decoder seq2seq pass.

    Args:
        cfg: Full FLOP-calc configuration.

    Returns:
        float: Encoder + decoder FLOPs (decoder includes self-attention,
        cross-attention, MLP, and norms).
    """
    enc = flop_transformer_body(
        cfg.enc_num_layers,
        cfg.enc_seq_len,
        cfg.hidden_dim,
        cfg.intermediate_dim,
        cfg,
    )
    ln = 3 * flop_layernorm(cfg.hidden_dim)
    self_att = flop_attention(
        cfg.hidden_dim,
        cfg.num_heads,
        cfg.kv_heads,
        cfg.head_dim,
        cfg.seq_len,
    )
    cross_att = flop_cross_attention(
        cfg.hidden_dim,
        cfg.num_heads,
        cfg.enc_seq_len,
        cfg.seq_len,
    )
    mlp = flop_mlp(cfg, cfg.hidden_dim, cfg.intermediate_dim)
    dec = cfg.num_layers * (ln + self_att + cross_att + mlp)
    return enc + dec


def flop_vision_tower(cfg: FlopCalcConfig) -> float:
    """Estimate FLOPs for the vision tower trunk.

    Args:
        cfg: Full FLOP-calc configuration (uses ``vision_*`` fields).

    Returns:
        float: FLOPs for the vision encoder.
    """
    if (
        cfg.vision_num_layers <= 0
        or cfg.vision_hidden_dim <= 0
        or cfg.vision_intermediate_dim <= 0
        or cfg.vision_num_heads <= 0
        or cfg.vision_seq_len is None
        or cfg.vision_seq_len <= 0
    ):
        return 0.0

    vision_head_dim = cfg.vision_head_dim
    if vision_head_dim is None:
        vision_head_dim = cfg.vision_hidden_dim // cfg.vision_num_heads

    vision_cfg = replace(
        cfg,
        num_heads=cfg.vision_num_heads,
        kv_heads=cfg.vision_num_heads,
        head_dim=vision_head_dim,
        activation_type=cfg.vision_activation_type,
        glu=False,
        num_experts=1,
        num_shared_experts=0,
        num_experts_per_tok=1,
        moe_intermediate_dim=None,
        shared_expert_intermediate_dim=0,
        num_moe_layers=0,
        layer_types=None,
        sliding_window=None,
    )

    return flop_transformer_body(
        cfg.vision_num_layers,
        cfg.vision_seq_len,
        cfg.vision_hidden_dim,
        cfg.vision_intermediate_dim,
        vision_cfg,
    )


def flops_per_token(cfg: FlopCalcConfig) -> float:
    """Estimate task-specific FLOPs per token for the configured model.

    Dispatches by ``cfg.task`` to compute the appropriate trunk + head + loss
    cost (causal LM, classification, seq2seq, vision-language, etc.).

    Args:
        cfg: Full FLOP-calc configuration.

    Returns:
        float: Estimated FLOPs per token for one forward pass.

    Raises:
        NotImplementedError: If ``cfg.task`` is not recognized.
    """
    body_cost = 0
    head_cost = 0
    loss_cost = 0

    if cfg.task in {
        TaskType.CAUSAL_LM,
        TaskType.DIFFUSION_LM,
    }:
        body_cost = flop_transformer_body(
            cfg.num_layers,
            cfg.seq_len,
            cfg.hidden_dim,
            cfg.intermediate_dim,
            cfg,
        )
        head_cost = flop_lm_head(cfg.hidden_dim, cfg.vocab_size)
        loss_cost = flop_loss(cfg.vocab_size) if cfg.include_loss else 0

    elif cfg.task in {
        TaskType.SEQUENCE_CLASSIFICATION,
        TaskType.IMAGE_CLASSIFICATION,
        TaskType.AUDIO_CLASSIFICATION,
    }:
        body_cost = flop_transformer_body(
            cfg.num_layers,
            cfg.seq_len,
            cfg.hidden_dim,
            cfg.intermediate_dim,
            cfg,
        )
        head_cost = flop_cls_head(cfg.hidden_dim, cfg.num_labels)
        loss_cost = flop_loss(cfg.num_labels) if cfg.include_loss else 0

    elif cfg.task in {
        TaskType.SEQUENCE_TO_SEQUENCE,
        TaskType.SPEECH_SEQUENCE_TO_SEQUENCE,
    }:
        body_cost = flop_seq2seq(cfg)
        head_cost = flop_lm_head(cfg.hidden_dim, cfg.vocab_size)
        loss_cost = flop_loss(cfg.vocab_size) if cfg.include_loss else 0

    elif cfg.task == TaskType.VISION_LM:
        body_cost = flop_vision_tower(cfg)

    elif cfg.task == TaskType.IMAGE_TEXT_TO_TEXT:
        vision = flop_vision_tower(cfg)
        text = flop_transformer_body(
            cfg.num_layers,
            cfg.seq_len,
            cfg.hidden_dim,
            cfg.intermediate_dim,
            cfg,
        )

        body_cost = vision + text
        head_cost = flop_lm_head(cfg.hidden_dim, cfg.vocab_size)
        loss_cost = flop_loss(cfg.vocab_size) if cfg.include_loss else 0

    elif cfg.task == TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION:
        body_cost = flop_vision_tower(cfg)
        head_cost = flop_cls_head(cfg.hidden_dim, cfg.num_labels)

    elif cfg.task in {TaskType.BASE_MODULE, TaskType.BASE_VISION, TaskType.AUTO_BIND}:
        body_cost = flop_transformer_body(
            cfg.num_layers,
            cfg.seq_len,
            cfg.hidden_dim,
            cfg.intermediate_dim,
            cfg,
        )

    else:
        raise NotImplementedError(f"Unsupported task: {cfg.task}")

    return body_cost + head_cost + loss_cost


@contextmanager
def trace_functions():
    """Context manager that captures XLA executables compiled during the body.

    Records the set of live executables before entering the ``with`` block
    and, on exit, populates the yielded :class:`FunctionTracer` with a
    :class:`TraceResult` for each new executable.

    Yields:
        FunctionTracer: A tracer that exposes ``new_executables`` and
        therefore ``flops`` after the block exits.
    """
    tracer = FunctionTracer()
    tracer._before = set(jax.extend.backend.get_backend().live_executables())

    try:
        yield tracer
    finally:
        after = set(jax.extend.backend.get_backend().live_executables())
        new = after - tracer._before
        tracer.new_executables = [TraceResult(exe) for exe in new]


class ModuleCaches(spx.Buffer):
    """Cache container for module-level cached values.

    Uses spectrax.Buffer with kind="cache" to provide caching functionality
    for EasyDeL modules, particularly for caching computed values like
    frequencies, masks, and other reusable tensors.
    """

    def __init__(self, value, **kwargs):
        """Initialize a cache buffer holding *value*.

        Args:
            value: Initial cached value (any pytree).
            **kwargs: Forwarded to :class:`spx.Buffer`.

        Returns:
            None.
        """
        super().__init__(value, kind="cache", **kwargs)


class OverWriteWithGradient(spx.Parameter):
    """Parameter type that allows gradient overwrites.

    Special parameter container that permits gradients to directly
    overwrite the parameter values during optimization, useful for
    certain advanced optimization techniques.
    """


class hashable_dict(dict):
    """Dict subclass that participates in static-hash JIT cache keys.

    Provides a deterministic ``__hash__`` (delegated to :func:`hash_fn`) so
    instances can be passed through ``jax.jit`` ``static_argnums`` /
    ``static_argnames`` and used as keys in static-config caches.
    """

    __hash__ = hash_fn


class ArrayParam(spx.Parameter):
    """Parameterized array with serializable initialization.

    A parameter container that stores initialization metadata (method name
    and kwargs) as strings/dicts instead of functions, making it pickleable
    and serializable. This is particularly useful for checkpointing and
    distributed training.

    Use ``ArrayParam.bound(...)`` for any parameter that should round-trip
    through checkpoints — the init metadata travels with the parameter so
    it can be re-materialized lazily. Reach for raw ``spx.Parameter`` only
    when the value is computed at construction time and you do not need
    init-method introspection (e.g. all-zero biases, hand-rolled state in
    linear-attention/SSM modules).

    Attributes:
        shape: The shape of the parameter array.
        dtype: The data type of the parameter array.
        init_method: Name of the JAX initializer (e.g., "normal", "zeros", "ones").
        init_kwargs: Optional kwargs passed to the initializer.
    """

    shape: Sequence[int]
    dtype: DTypeLike
    init_method: str = "normal"
    init_kwargs: hashable_dict | None = None

    def __init__(
        self,
        value: Array,
        *,
        shape: Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
        init_method: str = "normal",
        init_kwargs: hashable_dict | None = None,
        sharding: spx.Sharding | tuple[object | None, ...] | None = None,
        axis_names: tuple[object | None, ...] | None = None,
        **kwargs,
    ):
        """Construct a serializable, lazily-initializable parameter.

        Args:
            value: Concrete parameter value (or an abstract shape stand-in).
            shape: Optional shape override (defaults to ``value.shape``).
            dtype: Optional dtype override (defaults to ``value.dtype``).
            init_method: Name of the JAX initializer (``"normal"``,
                ``"zeros"``, ``"ones"``, etc.).
            init_kwargs: Optional :class:`hashable_dict` of kwargs forwarded
                to the initializer.
            sharding: Optional explicit sharding metadata; either a
                :class:`spx.Sharding` or a tuple of mesh axis names.
            axis_names: Optional explicit per-dim axis names (an alternative
                way to express ``sharding``).
            **kwargs: Forwarded to :class:`spx.Parameter`.

        Returns:
            None.
        """
        if init_kwargs is None:
            init_kwargs = hashable_dict()
        elif not isinstance(init_kwargs, hashable_dict):
            init_kwargs = hashable_dict(init_kwargs)
        kwargs.pop("use_ref", None)
        meta = kwargs.pop("metadata", {}) or {}
        meta.update(
            {
                "shape": tuple(shape) if shape is not None else tuple(value.shape),
                "dtype": jnp.dtype(dtype) if dtype is not None else value.dtype,
                "init_method": init_method,
                "init_kwargs": init_kwargs,
            }
        )
        super().__init__(value, metadata=meta, sharding=sharding, axis_names=axis_names, **kwargs)

    @classmethod
    def bound(
        cls,
        shape: Sequence[int],
        dtype: DTypeLike,
        init_method: str,
        init_kwargs: hashable_dict | None = None,
        *,
        key: PRNGKeyArray | None = None,
        value: Array | None = None,
        use_ref: bool | None = None,
        sharding: spx.Sharding | tuple[object | None, ...] | None = None,
        axis_names: tuple[object | None, ...] | None = None,
        **metadata,
    ):
        """Create an ArrayParam with initialized value.

        Args:
            shape: Shape of the parameter array.
            dtype: Data type for the parameter.
            init_method: Name of JAX initializer (e.g., "normal", "zeros", "kaiming_uniform").
            init_kwargs: Optional keyword arguments for the initializer.
            key: PRNG key for random initialization. Required if value is None.
            value: Pre-computed value. If provided, skips initialization.
            use_ref: Whether to use reference semantics.
            **metadata: Additional metadata to store with the parameter.

        Returns:
            ArrayParam: An initialized ArrayParam instance.
        """
        if init_kwargs is None:
            init_kwargs = {}
        init_kwargs = hashable_dict(init_kwargs)
        # Some JAX initializers (zeros, ones) are direct functions that take (key, shape, dtype),
        # while others (normal, uniform, etc.) are factory functions that return an initializer.
        # We need to handle both cases.
        direct_initializers = {"zeros", "ones"}
        if init_method in direct_initializers:
            init_fn = getattr(jax.nn.initializers, init_method)
        else:
            init_fn = getattr(jax.nn.initializers, init_method, jax.nn.initializers.normal)(**init_kwargs)
        if value is None:
            value = init_fn(key, shape, dtype)
        return cls(
            value=value,
            shape=shape,
            dtype=dtype,
            init_method=init_method,
            init_kwargs=init_kwargs,
            use_ref=use_ref,
            sharding=sharding,
            axis_names=axis_names,
            **metadata,
        )

    def resure(self, key: PRNGKeyArray, shard_fn: tp.Callable[[Array], Array] | None = None) -> None:
        """Reinitialize the parameter value with a new random key.

        Regenerates the parameter value using the stored initialization method
        and optional sharding function. Useful for resetting parameters or
        applying sharding after initialization.

        Args:
            key: PRNG key for random initialization.
            shard_fn: Optional function to apply sharding to the reinitialized value.
        """
        init_kwargs = self.init_kwargs
        if init_kwargs is None:
            init_kwargs = {}
        # Some JAX initializers (zeros, ones) are direct functions that take (key, shape, dtype),
        # while others (normal, uniform, etc.) are factory functions that return an initializer.
        direct_initializers = {"zeros", "ones"}
        if self.init_method in direct_initializers:
            init_fn = getattr(jax.nn.initializers, self.init_method)
        else:
            init_fn = getattr(jax.nn.initializers, self.init_method, jax.nn.initializers.normal)(**init_kwargs)
        val = init_fn(key, self.shape, self.dtype)

        if shard_fn is not None:
            val = shard_fn(val)

        self.value = val
        self.raw_value = val


if tp.TYPE_CHECKING:
    from transformers import BaseImageProcessor, FeatureExtractionMixin, PreTrainedTokenizerBase, ProcessorMixin

    ProcessingClassType = PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
else:
    ProcessingClassType = tp.Any


# Canonical implementations live in `easydel.infra.sharding`. These re-exports
# preserve the historical `easydel.infra.utils` import paths.
from .sharding import sanitize_partition_spec_for_shape, sanitize_partition_specs_for_shape_tree  # noqa: E402,F401


def device_put_or_shard_abstract(leaf: tp.Any, sharding: tp.Any) -> tp.Any:
    """Place concrete leaves, or attach sharding to abstract shape leaves.

    ``jax.ShapeDtypeStruct`` is metadata used by lazy/abstract loading paths.
    It is not a valid value for :func:`jax.device_put`, but it can carry the
    same sharding annotation directly so later materialization knows the
    intended placement.
    """
    if isinstance(leaf, jax.ShapeDtypeStruct):
        kwargs = {
            "sharding": sharding,
            "weak_type": getattr(leaf, "weak_type", False),
        }
        if hasattr(leaf, "manual_axis_type"):
            kwargs["manual_axis_type"] = leaf.manual_axis_type
        if hasattr(leaf, "is_ref"):
            kwargs["is_ref"] = leaf.is_ref
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, **kwargs)
    return jax.device_put(leaf, sharding)


def materialize_meta_leaves(tree: tp.Any, *, seed: int = 0) -> tp.Any:
    """Replace ShapeDtypeStruct placeholder leaves with concrete values."""
    import spectrax as spx

    def _materialize_rng_leaf(shape: tuple[int, ...], dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
        nonlocal rng
        if not shape:
            return jnp.array(0, dtype=dtype), rng
        key_words = int(shape[-1])
        if key_words < 2:
            return jnp.zeros(shape, dtype=dtype), rng
        prefix = shape[:-1]
        count = int(np.prod(prefix, dtype=np.int64)) if prefix else 1
        keys = jax.random.split(rng, count + 1)
        rng = keys[0]
        raw_keys = jax.random.key_data(keys[1:]).reshape(count, -1)
        if key_words <= raw_keys.shape[-1]:
            packed = raw_keys[:, :key_words]
        else:
            padding = jnp.zeros((count, key_words - raw_keys.shape[-1]), dtype=raw_keys.dtype)
            packed = jnp.concatenate([raw_keys, padding], axis=-1)
        return packed.reshape(shape).astype(dtype), rng

    if isinstance(tree, spx.State):
        changed = False
        rng = jax.random.PRNGKey(seed)

        new_data: dict[str, dict[str, tp.Any]] = {}
        for collection, path, value in tree.items():
            if isinstance(value, jax.ShapeDtypeStruct):
                if collection == "rng":
                    value, rng = _materialize_rng_leaf(tuple(value.shape), value.dtype)
                else:
                    value = jnp.zeros(value.shape, dtype=value.dtype)
                changed = True
            new_data.setdefault(collection, {})[path] = value

        if not changed:
            return tree
        return spx.State(new_data)

    # Fallback for regular dict trees (legacy path).
    flat_tree = flatten_dict(tree)
    changed = False
    rng = jax.random.PRNGKey(seed)

    for leaf in flat_tree.values():
        value = getattr(leaf, "value", None)
        if not isinstance(value, jax.ShapeDtypeStruct):
            continue
        # Spectrax does not use RngCount / RngKey objects; rng leaves are
        # plain uint32 arrays in the ``rng`` collection.  If we encounter
        # a ShapeDtypeStruct in a generic dict tree, zero-fill it.
        leaf.value = jnp.zeros(value.shape, dtype=value.dtype)
        changed = True

    if not changed:
        return tree
    return unflatten_dict(flat_tree)


def jax_path_to_string(path: tuple[tp.Any, ...], sep: str = "/") -> str:
    """Render a JAX tree-key path as a forward-slash-separated string."""
    parts: list[str] = []
    for key in path:
        if isinstance(key, tuple | list):
            parts.append(jax_path_to_string(key, sep=sep))
        elif isinstance(key, jax.tree_util.SequenceKey):
            parts.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            parts.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            parts.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            parts.append(str(key.key))
        else:
            parts.append(str(key))
    return sep.join(parts)
