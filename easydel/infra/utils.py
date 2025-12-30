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
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, partial

import jax
import jax.extend
import jax.tree_util
import numpy as np
from eformer.escale import with_sharding_constraint
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from einops import rearrange
from flax import nnx as nn
from jaxtyping import Array, DTypeLike, PRNGKeyArray
from tqdm.auto import tqdm

from easydel.layers.linear import ParallelLinear
from easydel.layers.quantization import EasyDeLQuantizationConfig, EasyQuantizer
from easydel.utils.compiling_utils import hash_fn
from easydel.utils.traversals import flatten_dict, unflatten_dict

from .errors import EasyDeLBlockWiseFFNError
from .etils import AVAILABLE_SPARSE_MODULE_TYPES, EasyDeLGradientCheckPointers

warnings.filterwarnings(
    "ignore",
    message="Primitive dynamic_update_slice was not handled by class",
)
logger = get_logger(__name__)


def quick_gelu(x):
    """Quick GELU activation function.

    A faster approximation of GELU using sigmoid.

    Args:
        x: Input array.

    Returns:
        Activated array.
    """
    return x * jax.nn.sigmoid(1.702 * x)


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "gelu_pytorch_tanh": partial(nn.gelu, approximate=True),
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "leaky_relu": partial(nn.leaky_relu, negative_slope=0.01),
    "glu": nn.glu,
    "elu": nn.elu,
    "softmax": nn.softmax,
    "quick_gelu": quick_gelu,
}
"""Registry of activation functions by name.

Maps activation function names to their implementations.
Supports common activations used in neural networks.
"""

ROPE_TYPES = tp.Optional[tp.Literal["none", "linear", "dynamic", "yarn", "su", "llama3", "longrope"]]  # noqa


with_sharding_constraint = with_sharding_constraint


def canonicalize_dtype(
    *args,
    dtype: jax.numpy.dtype | None = None,
    inexact: bool = True,
) -> jax.numpy.dtype:
    """Canonicalize an optional dtype to the definitive dtype.

    Infers or validates the dtype for JAX operations. If dtype is None,
    infers from input arguments. Otherwise validates and returns the
    specified dtype.

    Args:
        *args: JAX array compatible values (None values ignored).
        dtype: Optional dtype override. If specified, arguments are
            cast to this dtype and inference is disabled.
      inexact: When True, the output dtype must be a subdtype
      of `jnp.inexact`. Inexact dtypes are real or complex floating points. This
      is useful when you want to apply operations that don'position_ids work directly on
      integers like taking a mean for example.
    Returns:
      The dtype that *args should be cast to.
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
    """The add_start_docstrings function is a decorator that adds the docstrings to the beginning of a function.
    The add_start_docstrings function takes in an arbitrary number of strings and returns a decorator.
    The returned decorator takes in one argument, fn, which is assumed to be a function. The docstring
    for fn is set equal to the concatenation of all the strings passed into add_start_docstrings
    plus (if it exists) the original docstring for fn.

    Args:
        *docstr: Pass in a variable number of arguments to the function

    Returns:
        A decorator that adds the docstrings to the function
    """

    def docstring_decorator(fn):
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
            return rearrange(
                jax.lax.scan(
                    f=lambda carry, idx: (carry.at[:, idx].set(remat_ffn(carry[:, idx])), None),
                    init=rearrange(inputs, "b (c n) d -> b c n d", c=chunk_size),
                    xs=jax.numpy.arange(chunk_size),
                    length=chunk_size,
                    unroll=True,
                )[0],
                "b c n d -> b (c n) d",
            )
    except Exception as e:
        raise EasyDeLBlockWiseFFNError(
            "You Are using BlockWise FFN from near-infinite-context length paper and you might be passing "
            "input arguments in wrong way in case that you don'position_ids want to use this just pass "
            "`use_scan_mlp=False` in "
            "model config or in config_kwargs in AutoEasyDeLModelFor... or change `scan_mlp_chunk_size` "
            f"in configs for more information read Docs.\nOriginal Error\n{e}"
        ) from e


def is_flatten(pytree: dict):
    """The is_flatten function checks if the pytree is flattened.
        If it is, then the first key in the dictionary will be a tuple of (mpl, mpl_id).
        Otherwise, it will be an integer representing mpl_id.

    Args:
        pytree: dict: Pass the pytree to the function

    Returns:
        True if the pytree is a flattened tree, and false otherwise
    """
    mpl = next(iter(pytree.keys()))
    return True if isinstance(mpl, tuple) else False


def quantize_linear_layers(
    model: nn.Module,
    /,
    *,
    quantization_config: EasyDeLQuantizationConfig | None = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Quantize parameters to requested precision, excluding specified layers.

    Args:
        model: The model to quantize.
        quantization_config: Quantization config specifying dtype, block_size, and pattern.
        verbose: Whether to use tqdm for logging.

    Returns:
        Quantized parameters in the same structure as the input.
    """
    if quantization_config is None:
        return model

    quantizer = EasyQuantizer(quantization_config=quantization_config)
    return quantizer.quantize_linears(model, verbose=verbose)


def apply_lora_to_layers(
    model: nn.Module,
    /,
    *,
    lora_rank: int,
    lora_pattern: str | None = None,
    verbose: bool = True,
    rngs: nn.Rngs | None = None,
) -> nn.Module:
    """
    Applies LoRA (Low-Rank Adaptation) to specified linear layers within a model.

    Args:
        model: The EasyDeL model to modify.
        lora_rank: The rank of the LoRA adapters.
        lora_pattern: A regular expression pattern to match the names of
                      modules to which LoRA should be applied. Defaults to ".*" (all linear layers).
        verbose: Whether to display a progress bar.
        rngs:  A `flax.nnx.Rngs` instance for random number generation. If None, initializes with a seed of 0.

    Returns:
        The modified model with LoRA applied to the specified layers.
    """
    from easydel.utils.traversals import get_module_from_path, iter_module_search, set_module_from_path

    if not (lora_rank > 0):
        raise ValueError("lora_rank should be a positive value and higher than `0`.")
    if lora_pattern is None:
        lora_pattern = ".*"
    if rngs is None:
        rngs = nn.Rngs(0)
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
                    new_value=nn.LoRA(
                        base_module=base_module,
                        rngs=rngs,
                        dtype=base_module.dtype,
                        param_dtype=base_module.param_dtype,
                        in_features=base_module.in_features,
                        lora_rank=lora_rank,
                        out_features=base_module.out_features,
                    ),
                )
            pbar.update(1)

    return model


def split_lora_params(model: nn.Module) -> nn.Module:
    """
    get LoRA (Low-Rank Adaptation) from layers within a model.

    Args:
        model: The EasyDeL model.
    Returns:
        LoRA Layer Weights.
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


def merge_lora_params(model: nn.Module, lora_tree: dict) -> nn.Module:
    """
    get LoRA (Low-Rank Adaptation) from layers within a model.

    Args:
        model: The EasyDeL model.
    Returns:
        LoRA Layer Weights.
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
    model: nn.Module,
    /,
    *,
    verbose: bool = True,
) -> nn.Module:
    """
    UnWrap LoRA (Low-Rank Adaptation) from specified linear layers within a model.
    """
    from easydel.utils.traversals import get_module_from_path, iter_module_search, set_module_from_path

    with tqdm(
        total=len([p[0] for p in iter_module_search(model, ParallelLinear)]),
        desc="Unwarping LoRA Layers",
        disable=not verbose,
    ) as pbar:
        for path, _ in iter_module_search(model, nn.LoRA):
            base_module: nn.LoRA = get_module_from_path(model=model, path=path)
            with jax.default_matmul_precision("float32"):
                base_module.base_module.kernel.value = (
                    base_module.base_module.kernel.value + base_module.lora_a.value @ base_module.lora_b.value
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
    assert sparser is not None, f"unkown type of sparser {sparsify_module}"

    def filter_params(path, array):
        layer_name = ".".join(path[0].key)
        if layer_name.endswith("kernel") and 4 > array.ndim > 1:
            array = sparser.fromdense(array)
        return array

    total_params = len(jax.tree_util.tree_leaves(params))
    with tqdm(
        total=total_params,
        desc=f"{sparsify_module.capitalize()}",
        disable=not verbose,
    ) as pbar:

        def _with_progress(path, array):
            pbar.set_postfix_str(".".join(path[0].key))
            result = filter_params(path, array)
            pbar.update(1)
            return result

        params = jax.tree_util.tree_map_with_path(_with_progress, params)

    if not flatten:
        params = unflatten_dict(params)
    return params


def extract_static_parameters(module):
    """
    Extract static_argnums for specified parameters across functions in a module.

    Args:
        module (types.ModuleType): The module to inspect

    Returns:
        dict: A dictionary mapping function names to their static parameter indices
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
    obj = getattr(module, "__call__", None)  # noqa
    if isinstance(obj, types.FunctionType | types.MethodType):
        static_args = ()
        signature = inspect.signature(obj)
        for idx, (param_name, _param) in enumerate(signature.parameters.items()):
            if param_name in target_params:
                static_args += (idx,)
        return static_args
    return None


M = tp.TypeVar("M", bound=nn.Module)


@tp.overload
def auto_remat(
    module: type[M],
    /,
    *,
    policy: EasyDeLGradientCheckPointers | str | tp.Callable = EasyDeLGradientCheckPointers.NONE,
    prevent_cse: bool = True,
    save_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> type[M]: ...


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
) -> tuple[type[M], type[M]]: ...


@tp.overload
def auto_remat(
    *modules: type[M],
    policy: EasyDeLGradientCheckPointers | str | tp.Callable = EasyDeLGradientCheckPointers.NONE,
    prevent_cse: bool = True,
    save_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> tuple[type[M], ...]: ...


def auto_remat(
    *modules: type[M],
    policy: EasyDeLGradientCheckPointers | str | tp.Callable = EasyDeLGradientCheckPointers.NONE,
    prevent_cse: bool = True,
    save_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> type[M] | tuple[type[M], ...]:
    """Apply gradient checkpointing (rematerialization) to module(s).

    Wraps module __call__ methods with JAX's remat (rematerialization) to trade
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
    if isinstance(policy, str | EasyDeLGradientCheckPointers):
        policy = get_gradient_checkpoint_policy(policy, save_names, exclude_names)
    elif not callable(policy):
        raise ValueError(f"Invalid policy type: {type(policy)}")

    outs = ()
    for module in modules:
        assert issubclass(module, nn.Module)
        static_argnums = extract_static_parameters(module=module)
        if static_argnums is None:
            static_argnums = ()

        module.__call__ = nn.remat(
            f=module.__call__,
            prevent_cse=prevent_cse,
            static_argnums=static_argnums,
            policy=policy,
        )

        outs += (module,)

    if len(outs) == 1:
        return outs[0]
    return outs


# Main FLOP counting function
def count_flop_jaxpr(jaxpr) -> int:
    """Count flops in a Jaxpr."""

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
        return out_size * (2 * contracting_size - 1)

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

        return remaining_size * (reduced_size - 1)

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
        self._executable = executable
        self._cached_cost = None

    @property
    @lru_cache(maxsize=1)  # noqa
    def cost_analysis(self):
        return self._executable.cost_analysis()

    @property
    def flops(self):
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
        self.new_executables: list[TraceResult] = []
        self._before: set = set()

    def __getitem__(self, idx):
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
        self.first_time = True
        self.cached_flops = 0
        self.functions = None

    @property
    def online_flops(self):
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


class ActivationType(str, Enum):
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
    """Calculate FLOPs for different activation functions."""

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


class AttnMaskType(str, Enum):
    FULL = "ATTN_MASK_FULL"
    SLIDING = "ATTN_MASK_SLIDING"
    CHUNK = "ATTN_MASK_CHUNK"

    @classmethod
    def from_hf(cls, hf_type: tp.Literal["sliding_attention", "full_attention", "chunk_attention", "chunked_attention"]):
        if hf_type == "sliding_attention":
            return AttnMaskType.SLIDING
        elif hf_type == "full_attention":
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


class TaskType(str, Enum):
    CAUSAL_LM = "causal-language-model"
    VISION_LM = "vision-language-model"
    DIFFUSION_LM = "diffusion-language-model"
    IMAGE_TEXT_TO_TEXT = "image-text-to-text"
    BASE_MODULE = "base-module"
    BASE_VISION = "vision-module"
    SEQUENCE_TO_SEQUENCE = "sequence-to-sequence"
    SPEECH_SEQUENCE_TO_SEQUENCE = "speech-sequence-to-sequence"
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    SEQUENCE_CLASSIFICATION = "sequence-classification"
    AUDIO_CLASSIFICATION = "audio-classification"
    IMAGE_CLASSIFICATION = "image-classification"
    ANY_TO_ANY = "any-to-any"
    AUTO_BIND = "auto-bind"


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
    vision_seq_len: int = 0

    include_loss: bool = False


def flop_layernorm(hidden_dim: int) -> float:
    return 8 * hidden_dim


def flop_attention(
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
) -> float:
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


def flop_cross_attention(
    hidden_dim: int,
    num_heads: int,
    enc_seq_len: int,
    dec_seq_len: int,
) -> float:
    head_dim = hidden_dim // num_heads
    proj = 2 * hidden_dim * hidden_dim
    scores = 2 * head_dim * enc_seq_len * dec_seq_len * num_heads
    softmax = 5 * enc_seq_len * dec_seq_len * num_heads
    wsum = 2 * head_dim * enc_seq_len * dec_seq_len * num_heads
    out_proj = 2 * hidden_dim * hidden_dim
    return proj + scores + softmax + wsum + out_proj


def flop_mlp(
    cfg: FlopCalcConfig,
    hidden_dim: int,
    intermediate_dim: int,
) -> float:
    factor = 3 if cfg.glu else 2
    base = factor * hidden_dim * intermediate_dim
    total_ffn = base * (cfg.num_experts_per_tok + cfg.num_shared_experts)
    activation_flops = flop_activation(
        cfg.activation_type,
        intermediate_dim * (cfg.num_experts_per_tok + cfg.num_shared_experts),
    )

    router = 2 * hidden_dim * cfg.num_experts if cfg.num_experts > 1 else 0
    return 2 * total_ffn + activation_flops + router


def flop_lm_head(hidden_dim: int, vocab_size: int) -> float:
    return 2 * hidden_dim * vocab_size + 5 * vocab_size


def flop_cls_head(hidden_dim: int, num_labels: int) -> float:
    return 2 * hidden_dim * num_labels + 5 * num_labels


def flop_loss(num_classes: int) -> float:
    return 3 * num_classes + 2


def flop_transformer_body(
    layers: int,
    seq_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    cfg: FlopCalcConfig,
) -> float:
    ln = 2 * flop_layernorm(hidden_dim)
    att = flop_attention(
        hidden_dim,
        cfg.num_heads,
        cfg.kv_heads,
        cfg.head_dim,
        seq_len,
    )
    mlp = flop_mlp(cfg, hidden_dim, intermediate_dim)
    return layers * (ln + att + mlp)


def flop_seq2seq(cfg: FlopCalcConfig) -> float:
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
    return flop_transformer_body(
        cfg.vision_num_layers,
        cfg.vision_seq_len,
        cfg.vision_hidden_dim,
        cfg.vision_intermediate_dim,
        cfg,
    )


def flops_per_token(cfg: FlopCalcConfig) -> float:
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
        try:
            vision = flop_vision_tower(cfg)
            text = flop_seq2seq(cfg)
        except ZeroDivisionError:
            vision = 0
            text = 0

        clm_head = flop_transformer_body(
            cfg.num_layers,
            cfg.seq_len,
            cfg.hidden_dim,
            cfg.intermediate_dim,
            cfg,
        )

        body_cost = vision + text + clm_head
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
    tracer = FunctionTracer()
    tracer._before = set(jax.extend.backend.get_backend().live_executables())

    try:
        yield tracer
    finally:
        after = set(jax.extend.backend.get_backend().live_executables())
        new = after - tracer._before
        tracer.new_executables = [TraceResult(exe) for exe in new]


class ModuleCaches(nn.Cache):
    """Cache container for module-level cached values.

    Extends flax.nnx.Cache to provide caching functionality for
    EasyDeL modules, particularly for caching computed values like
    frequencies, masks, and other reusable tensors.
    """


class OverWriteWithGradient(nn.Param):
    """Parameter type that allows gradient overwrites.

    Special parameter container that permits gradients to directly
    overwrite the parameter values during optimization, useful for
    certain advanced optimization techniques.
    """


class hashable_dict(dict):
    __hash__ = hash_fn


class ArrayParam(nn.Param):
    """Parameterized array with serializable initialization.

    A parameter container that stores initialization metadata (method name
    and kwargs) as strings/dicts instead of functions, making it pickleable
    and serializable. This is particularly useful for checkpointing and
    distributed training.

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
            shape=shape,
            dtype=dtype,
            init_method=init_method,
            init_kwargs=init_kwargs,
            value=value,
            use_ref=use_ref,
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

    ProcessingClassType = tp.Optional[  # noqa
        tp.Union[  # noqa
            PreTrainedTokenizerBase,
            BaseImageProcessor,
            FeatureExtractionMixin,
            ProcessorMixin,
        ]
    ]
else:
    ProcessingClassType = tp.Any
