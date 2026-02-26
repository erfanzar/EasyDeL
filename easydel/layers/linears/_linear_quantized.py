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

"""Quantized linear layers for memory-efficient neural networks.

This module provides quantized versions of linear layers that reduce memory
footprint while maintaining model quality. It supports multiple quantization
formats including INT8, NF4, MXFP4, MXFP8, and NVFP8.

Classes:
    ParallelLinearQuantized: Base quantized linear layer with parallel support.
    RowParallelLinearQuantized: Row-parallel variant for distributed training.
    ColumnParallelLinearQuantized: Column-parallel variant for distributed training.

Key Features:
    - On-the-fly dequantization during forward pass
    - Support for multiple quantization formats (INT8, NF4, MXFP4, MXFP8, NVFP8)
    - Conversion to/from non-quantized layers
    - Runtime quantization of activations
    - Integration with parallel linear layers

Example:
    >>> from easydel.layers.linears import ColumnParallelLinearQuantized
    >>> from easydel.layers.quantization import QuantizationConfig, QuantizationType
    >>>
    >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
    >>> layer = ColumnParallelLinearQuantized(
    ...     in_features=768,
    ...     out_features=3072,
    ...     config=config,
    ...     rngs=rngs
    ... )
    >>> output = layer(input_tensor)
"""

from __future__ import annotations

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from eformer.common_types import ColumnWise, Replicated, RowWise
from ejkernel.modules.operations import (  # pyright: ignore[reportMissingTypeStubs]
    quantized_matmul as ej_quantized_matmul,
)
from ejkernel.quantization import dequantize as ej_dequantize  # pyright: ignore[reportMissingTypeStubs]
from ejkernel.quantization import prepack_quantized_weights  # pyright: ignore[reportMissingTypeStubs]
from flax import nnx as nn
from flax.nnx import rnglib
from flax.nnx.nn import initializers
from flax.typing import Dtype, Initializer, PrecisionLike
from jax import shard_map

from easydel.layers._sharding import resolve_safe_sharding
from easydel.layers.quantization._configs import QuantizationType, resolve_ejkernel_quant_params
from easydel.layers.quantization._quants import quantize

from ._linear import ColumnParallelLinear, RowParallelLinear

if tp.TYPE_CHECKING:
    from easydel.layers.quantization._configs import QuantizationConfig


Array = jax.Array
Axis = int
Size = int

default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()

_QMM_NON_AFFINE_MODES = frozenset({"nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"})
_QMM_DEFAULT_POLICY_TABLE: dict[str, tp.Any] = {
    "tpu": {
        "affine": {
            "small": {"fuse": False, "platform": "xla"},
            "large": {
                "fuse": True,
                "platform": "auto",
                "allow_dense_fallback": False,
                "strict_fuse": True,
                "tpu_path": "predecode",
            },
            "default": {
                "fuse": False,
                "platform": "xla",
            },
        },
        "non_affine": {
            "small": {
                "fuse": True,
                "platform": "auto",
                "allow_dense_fallback": True,
                "strict_fuse": False,
                "tpu_path": "predecode",
            },
            "large": {
                "fuse": True,
                "platform": "auto",
                "allow_dense_fallback": True,
                "strict_fuse": False,
                "tpu_path": "predecode",
            },
            "default": {
                "fuse": True,
                "platform": "auto",
                "allow_dense_fallback": True,
                "strict_fuse": False,
                "tpu_path": "predecode",
            },
        },
        "default": {
            "default": {"fuse": False, "platform": "xla"},
        },
    },
    "default": {
        "default": {
            "default": {"fuse": True, "platform": "auto"},
        },
    },
}


def _policy_mode_key(mode: str | None) -> str:
    mode_n = str(mode).strip().lower() if mode is not None else ""
    if mode_n == "affine":
        return "affine"
    if mode_n in _QMM_NON_AFFINE_MODES:
        return "non_affine"
    return "default"


def _policy_size_key(m_tokens: int | None, threshold: int | None) -> str:
    if m_tokens is None or threshold is None:
        return "default"
    return "small" if int(m_tokens) <= int(threshold) else "large"


def _lookup_qmm_policy_entry(
    *,
    backend: str,
    mode: str | None,
    m_tokens: int | None,
    tpu_small_threshold: int | None,
    policy_table: dict[str, tp.Any] | None,
) -> dict[str, tp.Any]:
    table = policy_table if policy_table is not None else _QMM_DEFAULT_POLICY_TABLE
    backend_table = table.get(backend, table.get("default", {}))
    mode_table = backend_table.get(_policy_mode_key(mode), backend_table.get("default", {}))
    size_table = mode_table.get(_policy_size_key(m_tokens, tpu_small_threshold), mode_table.get("default", {}))
    return dict(size_table) if isinstance(size_table, dict) else {}


def _mesh_matches(lhs: jax.sharding.Mesh, rhs: jax.sharding.Mesh) -> bool:
    if lhs.axis_names != rhs.axis_names:
        return False
    if lhs.devices.shape != rhs.devices.shape:
        return False

    lhs_fingerprint = tuple(
        (
            getattr(device, "process_index", None),
            getattr(device, "id", None),
            getattr(device, "platform", None),
            getattr(device, "device_kind", None),
        )
        for device in lhs.devices.flat
    )
    rhs_fingerprint = tuple(
        (
            getattr(device, "process_index", None),
            getattr(device, "id", None),
            getattr(device, "platform", None),
            getattr(device, "device_kind", None),
        )
        for device in rhs.devices.flat
    )
    return lhs_fingerprint == rhs_fingerprint


def _pick_mesh_from_arrays(*arrays: jax.Array | None) -> jax.sharding.Mesh | jax.sharding.AbstractMesh | None:
    for array in arrays:
        if array is None:
            continue
        sharding = getattr(array, "sharding", None)
        if isinstance(sharding, jax.sharding.NamedSharding):
            return sharding.mesh
    try:
        from eformer.escale import get_incontext_mesh

        mesh = get_incontext_mesh(raise_error=False)
        if getattr(mesh, "empty", True):
            return None
        return mesh
    except Exception:
        return None


def _spec_for_mesh(array: jax.Array | None, mesh: jax.sharding.Mesh) -> jax.sharding.PartitionSpec:
    if array is None:
        return jax.sharding.PartitionSpec()
    sharding = getattr(array, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding) and _mesh_matches(sharding.mesh, mesh):
        return sharding.spec
    return jax.sharding.PartitionSpec()


def _spec_is_sharded(spec: jax.sharding.PartitionSpec) -> bool:
    return any(axis is not None for axis in tuple(spec))


def _spec_matches_kernel_parallel_layout(
    kernel_spec: jax.sharding.PartitionSpec,
    aux_spec: jax.sharding.PartitionSpec,
    direction: tp.Literal["row", "column"] | None,
) -> bool:
    """Return whether aux tensor sharding is compatible with kernel sharding.

    Quantized kernels, scales, and affine zero-points are packed with matching
    `(in_features, out_features_*)` semantics. When the kernel is sharded, aux
    tensors must use the same partitioning so local ejkernel calls see aligned
    shard-local shapes.
    """
    if direction not in {"row", "column"}:
        return False

    kernel_tuple = tuple(kernel_spec)
    aux_tuple = tuple(aux_spec)
    kernel_is_sharded = _spec_is_sharded(kernel_spec)
    aux_is_sharded = _spec_is_sharded(aux_spec)

    if not kernel_is_sharded:
        # Keep replicated aux params on the non-distributed path.
        return True
    if not aux_is_sharded:
        return False

    # Require exact parallel layout when entering shard_map.
    return kernel_tuple == aux_tuple


def _axis_names(axis_spec: tp.Any) -> tuple[str, ...]:
    if axis_spec is None:
        return ()
    if isinstance(axis_spec, str):
        return (axis_spec,)
    if isinstance(axis_spec, (list, tuple)):
        return tuple(axis for axis in axis_spec if isinstance(axis, str))
    return ()


def _extract_tp_axis_name(
    kernel_spec: jax.sharding.PartitionSpec,
    direction: tp.Literal["row", "column"] | None,
    mesh: jax.sharding.Mesh,
) -> str | None:
    if direction not in {"row", "column"}:
        return None
    dim = 0 if direction == "row" else 1
    if len(kernel_spec) <= dim:
        return None
    axis = kernel_spec[dim]
    if axis is None:
        return None
    candidates = _axis_names(axis)

    # Prefer canonical tensor axis name when available.
    for candidate in candidates:
        if candidate == "tp" and candidate in mesh.axis_names and mesh.shape[candidate] > 1:
            return candidate
    for candidate in candidates:
        if candidate in mesh.axis_names and mesh.shape[candidate] > 1:
            return candidate
    return None


def _pick_tensor_axis_name(mesh: jax.sharding.Mesh) -> str | None:
    """Pick a multi-device tensor axis from mesh, preferring canonical names."""

    for axis_name in ("tp", "tensor"):
        if axis_name in mesh.axis_names and mesh.shape.get(axis_name, 1) > 1:
            return axis_name
    for axis_name in mesh.axis_names:
        if mesh.shape.get(axis_name, 1) > 1:
            return axis_name
    return None


def _quantized_linear_sharding_fn(
    *,
    direction: tp.Literal["row", "column"] | None,
    param_name: str,
    mode: str,
    group_size: int,
    needs_biases: bool,
) -> tp.Any | None:
    """Return sharding dynamic-axes for a quantized linear parameter.

    Notes:
        - `prepack_quantized_weights(..., transpose=False)` stores packed tensors in
          `(in_features, out_features_*)` layout for all supported modes.
        - `group_size` only changes the grouped output extent (`out_features_*`), not
          the sharded axis semantics.
    """
    if direction is None:
        return None

    # Validate supported grouped quantization modes early.
    if mode not in {"affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}:
        return None
    if group_size <= 0:
        raise ValueError(f"`group_size` must be > 0, got {group_size}.")

    if param_name == "bias":
        return Replicated

    if param_name == "quant_biases" and not needs_biases:
        # Non-affine modes do not materialize per-group zero/bias tensors.
        return None

    if param_name in {"quant_kernel", "quant_scales", "quant_biases"}:
        return RowWise if direction == "row" else ColumnWise

    return None


def _quantized_linear_craft_spec(
    *,
    direction: tp.Literal["row", "column"] | None,
    use_bias: bool,
    mode: str,
    group_size: int,
    needs_biases: bool,
) -> dict[str, tp.Any]:
    """Craft dynamic sharding specs for quantized linear parameters."""
    specs: dict[str, tp.Any] = {}

    for param_name in ("quant_kernel", "quant_scales", "quant_biases"):
        spec = _quantized_linear_sharding_fn(
            direction=direction,
            param_name=param_name,
            mode=mode,
            group_size=group_size,
            needs_biases=needs_biases,
        )
        if spec is not None:
            specs[param_name] = spec

    if use_bias:
        spec = _quantized_linear_sharding_fn(
            direction=direction,
            param_name="bias",
            mode=mode,
            group_size=group_size,
            needs_biases=needs_biases,
        )
        if spec is not None:
            specs["bias"] = spec

    return specs


def _reconcile_input_k_dim(
    local_inputs: Array,
    local_kernel: Array,
    direction: tp.Literal["row", "column"] | None,
    tp_axis_name: str | None,
    *,
    allow_column_all_gather: bool = False,
) -> Array:
    """Align the K dimension of local inputs with the local kernel shard.

    Inside ``shard_map`` the local input slice may not match the local kernel's
    K extent.  This helper reconciles the mismatch:

    * **column-parallel** - kernels keep full K and shard N. Reconstructing K
      via ``all_gather`` can blow device memory, so it is opt-in.
    * **row-parallel** - kernels shard K, so replicated inputs are sliced to the
      local K chunk via ``dynamic_slice_in_dim``.
    """
    if local_inputs.shape[-1] == local_kernel.shape[0]:
        return local_inputs

    if tp_axis_name is None:
        raise ValueError(f"{direction}-parallel quantized matmul requires a TP axis to reconcile local K sizes.")

    if direction == "column":
        if not allow_column_all_gather:
            raise ValueError(
                "Column-parallel quantized matmul local K mismatch detected. "
                "Implicit input all_gather is disabled by default to avoid TPU OOM. "
                "Set `qmm_allow_input_all_gather=True` on the layer to enable it explicitly."
            )
        return jax.lax.all_gather(local_inputs, tp_axis_name, axis=-1, tiled=True)

    # Row-parallel: slice the local K chunk from replicated inputs.
    if local_inputs.shape[-1] % local_kernel.shape[0] != 0:
        raise ValueError(
            "Row-parallel quantized matmul got incompatible local K sizes: "
            f"input={local_inputs.shape[-1]} kernel={local_kernel.shape[0]}."
        )
    shard_index = jax.lax.axis_index(tp_axis_name)
    start = shard_index * local_kernel.shape[0]
    return jax.lax.dynamic_slice_in_dim(local_inputs, start, local_kernel.shape[0], axis=-1)


class ParallelLinearQuantized(nn.Module):
    """A quantized linear transformation layer with parallel execution support.

    This layer stores weights in a quantized format to reduce memory usage,
    and dequantizes them on-the-fly during forward passes. It supports multiple
    quantization schemes including INT8, NF4, and microscaling formats.

    The layer can be converted to/from non-quantized ParallelLinear layers,
    allowing for flexible model deployment strategies where you train in
    full precision and deploy with quantization.

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether the layer includes a bias term.
        dtype: Data type for computation (default inferred from inputs).
        param_dtype: Data type for non-quantized parameters (bias).
        precision: JAX precision setting for matrix multiplication.
        kernel_init: Initializer for kernel weights before quantization.
        bias_init: Initializer for bias term.
        config: Quantization configuration specifying format and parameters.
        rngs: Random number generators for initialization.
        quant_kernel: Quantized kernel weights.
        quant_scales: Per-block scaling factors for dequantization.
        quant_biases: Per-block biases for affine quantization (if applicable).
        bias: Optional bias parameter (not quantized).
        _direction: Parallelism direction ("row", "column", or None).

    Example:
        >>> from easydel.layers.linears import ParallelLinearQuantized
        >>> from easydel.layers.quantization import QuantizationConfig, QuantizationType
        >>>
        >>> # Create INT8 quantized layer
        >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
        >>> layer = ParallelLinearQuantized(
        ...     in_features=768,
        ...     out_features=3072,
        ...     config=config,
        ...     rngs=nn.Rngs(0)
        ... )
        >>>
        >>> # Forward pass with automatic dequantization
        >>> output = layer(jnp.ones((32, 768)))
    """

    _direction: tp.Literal["row", "column"] | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        config: QuantizationConfig,
        qmm_platform: tp.Literal["triton", "pallas", "cuda", "cute", "xla", "auto"] | None = None,
        qmm_use_best_config: bool | None = None,
        qmm_fuse: bool | None = None,
        qmm_strict_fuse: bool | None = None,
        qmm_allow_dense_fallback: bool | None = None,
        qmm_tpu_path: tp.Literal["hybrid", "packed", "predecode"] | None = None,
        qmm_tpu_auto_xla_max_m: int | None = 1024,
        qmm_policy_table: dict[str, tp.Any] | None = None,
        qmm_allow_input_all_gather: bool = False,
        rngs: rnglib.Rngs,
    ):
        """Initialize a quantized parallel linear layer.

        Creates a linear layer with quantized weights. The kernel is initialized
        using the provided initializer, then immediately quantized according to
        the configuration.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            use_bias: If True, adds a learnable bias to the output.
                Defaults to True.
            dtype: Data type for computation. If None, uses input dtype.
                Defaults to None.
            param_dtype: Data type for parameters (bias). Defaults to float32.
            precision: JAX precision for matrix multiplication. Can be None,
                'default', 'high', 'highest', or specific precision combinations.
                Defaults to None.
            kernel_init: Initializer function for the weight matrix.
                Defaults to lecun_normal().
            bias_init: Initializer function for the bias vector.
                Defaults to zeros.
            config: Quantization configuration specifying the quantization
                type (INT8, NF4, etc.) and related parameters like group_size.
            qmm_platform: Optional explicit platform override for ejkernel
                quantized_matmul calls.
            qmm_use_best_config: Optional ejkernel tuned-config toggle for
                quantized_matmul calls. Defaults to True when omitted.
            qmm_fuse: Optional fuse policy forwarded to ejkernel quantized_matmul.
                If None on TPU, defaults to False unless explicit fused controls
                are provided (e.g. qmm_tpu_path/qmm_strict_fuse/qmm_platform="pallas").
            qmm_strict_fuse: Optional strict fused-kernel policy forwarded to
                ejkernel quantized_matmul.
            qmm_allow_dense_fallback: Optional policy forwarded to ejkernel to
                permit/disallow dense dequantize+matmul fallback.
            qmm_tpu_path: Optional TPU path override ("hybrid", "packed",
                "predecode") forwarded per-call to ejkernel fused execution.
            qmm_tpu_auto_xla_max_m: TPU auto-policy threshold on flattened token
                count ``m``. When ``qmm_platform="auto"`` and no explicit fused
                controls are provided, calls with ``m <= qmm_tpu_auto_xla_max_m``
                use unfused XLA; larger ``m`` uses fused TPU path. Set to None
                to disable threshold-based switching.
            qmm_policy_table: Optional policy lookup table for runtime
                quantized matmul controls keyed by backend/mode/size bucket.
                When None, uses built-in defaults.
            qmm_allow_input_all_gather: Whether column-parallel execution may
                all_gather local input K slices when they mismatch the local
                kernel shard. Defaults to False to avoid OOM.
            rngs: Flax random number generators for initialization.

        Raises:
            ValueError: If config.dtype is not a supported quantization type.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.config = config
        self.qmm_platform = qmm_platform
        self.qmm_use_best_config = qmm_use_best_config
        self.qmm_fuse = qmm_fuse
        self.qmm_strict_fuse = qmm_strict_fuse
        self.qmm_allow_dense_fallback = qmm_allow_dense_fallback
        self.qmm_tpu_path = qmm_tpu_path
        self.qmm_tpu_auto_xla_max_m = qmm_tpu_auto_xla_max_m
        self.qmm_policy_table = qmm_policy_table
        self.qmm_allow_input_all_gather = bool(qmm_allow_input_all_gather)
        self.rngs = rngs

        kernel = kernel_init(rngs.params(), (in_features, out_features), param_dtype)
        quant_kernel, quant_scales, quant_biases = self._quantize_array(kernel)

        self.quant_kernel = nn.Param(quant_kernel)
        self.quant_scales = nn.Param(quant_scales)
        self.quant_biases = nn.Param(quant_biases)

        if use_bias:
            self.bias = nn.Param(bias_init(rngs.params(), (out_features,), param_dtype))

    def _qmm_runtime_kwargs(
        self,
        backend: str,
        *,
        m_tokens: int | None = None,
        quant_mode: str | None = None,
    ) -> dict[str, tp.Any]:
        """Resolve per-call ejkernel quantized_matmul controls.

        Control values are resolved from a policy lookup table keyed by
        backend, quantization mode, and size bucket, then explicit layer-level
        overrides are applied on top.
        """
        policy = _lookup_qmm_policy_entry(
            backend=backend,
            mode=quant_mode,
            m_tokens=m_tokens,
            tpu_small_threshold=self.qmm_tpu_auto_xla_max_m,
            policy_table=self.qmm_policy_table,
        )

        tpu_path = self.qmm_tpu_path if self.qmm_tpu_path is not None else policy.get("tpu_path")
        if tpu_path is not None:
            tpu_path = str(tpu_path).strip().lower()
            if tpu_path not in {"hybrid", "packed", "predecode"}:
                raise ValueError(f"qmm_tpu_path must be one of {{'hybrid','packed','predecode'}}, got {tpu_path!r}.")

        platform_requested = self.qmm_platform
        fuse_policy = policy.get("fuse")
        explicit_fused_controls = backend == "tpu" and (
            self.qmm_tpu_path is not None
            or self.qmm_strict_fuse is not None
            or self.qmm_allow_dense_fallback is not None
            or platform_requested == "pallas"
        )
        if self.qmm_fuse is not None:
            fuse = self.qmm_fuse
        elif explicit_fused_controls:
            fuse = True
        else:
            fuse = fuse_policy
        if fuse is None:
            fuse = backend != "tpu"
        fuse = bool(fuse)

        platform_policy = policy.get("platform")
        if platform_requested in {None, "auto"}:
            platform = platform_policy if platform_policy is not None else ("auto" if backend != "tpu" else "xla")
        else:
            platform = platform_requested
        if backend == "tpu" and fuse and platform_requested is None and platform == "auto":
            # Keep explicit fused TPU requests deterministic.
            platform = "pallas"
        if backend == "tpu" and fuse and self.qmm_tpu_path is not None and platform_requested in {None, "auto"}:
            # Explicit TPU path override implies TPU fused dispatch.
            platform = "pallas"
        if backend == "tpu" and platform == "auto" and not fuse:
            platform = "xla"

        use_best_config = self.qmm_use_best_config
        if use_best_config is None:
            use_best_config = True

        kwargs: dict[str, tp.Any] = {"fuse": fuse, "platform": platform, "use_best_config": bool(use_best_config)}

        allow_dense_fallback = (
            self.qmm_allow_dense_fallback
            if self.qmm_allow_dense_fallback is not None
            else policy.get("allow_dense_fallback")
        )
        strict_fuse = self.qmm_strict_fuse if self.qmm_strict_fuse is not None else policy.get("strict_fuse")

        if not fuse:
            if strict_fuse is not None:
                raise ValueError("qmm_strict_fuse requires qmm_fuse=True.")
            if allow_dense_fallback is not None:
                raise ValueError("qmm_allow_dense_fallback requires qmm_fuse=True.")
            if backend == "tpu" and tpu_path is not None:
                raise ValueError("qmm_tpu_path requires qmm_fuse=True.")
            return kwargs

        if allow_dense_fallback is not None:
            allow_dense_fallback = bool(allow_dense_fallback)
            kwargs["allow_dense_fallback"] = allow_dense_fallback

        if strict_fuse is not None:
            strict_fuse = bool(strict_fuse)
            if allow_dense_fallback and strict_fuse:
                raise ValueError("qmm_allow_dense_fallback=True is incompatible with qmm_strict_fuse=True.")
            kwargs["strict_fuse"] = strict_fuse
        elif allow_dense_fallback is not None:
            kwargs["strict_fuse"] = not allow_dense_fallback

        if backend == "tpu":
            kwargs["tpu_path"] = tpu_path if tpu_path is not None else "predecode"

        return kwargs

    def _resolve_ejkernel_params(self) -> tuple[str, int, int, bool]:
        """Resolve ejkernel quantization parameters from config."""
        return resolve_ejkernel_quant_params(self.config)

    def _quantize_array(self, array: jax.Array):
        """Quantize an array according to the configured quantization type.

        Applies the appropriate quantization function based on self.config.dtype
        to convert a full-precision array to its quantized representation.

        Args:
            array: Full-precision array to quantize. Typically the kernel
                weights with shape (in_features, out_features).

        Returns:
            A tuple of (quantized_array, scale_factors, bias_factors):
                - quantized_array: The quantized weight values
                - scale_factors: Per-block scaling factors for dequantization
                - bias_factors: Per-block biases for affine quantization
                  (None for non-affine modes)

        Raises:
            ValueError: If the configured quantization dtype is not supported.
        """
        mode, group_size, bits, needs_biases = self._resolve_ejkernel_params()
        if needs_biases:
            wq, scales, biases = prepack_quantized_weights(
                array,
                group_size=group_size,
                bits=bits,
                mode=mode,
                transpose=False,
            )
            return wq, scales, biases

        wq, scales = prepack_quantized_weights(
            array,
            group_size=group_size,
            bits=bits,
            mode=mode,
            transpose=False,
        )
        return wq, scales, None

    def _quantize_runtime(self, array: jax.Array):
        """Quantize activations at runtime if configured.

        Some quantization configurations support quantizing input activations
        in addition to weights. This method applies that quantization if
        config.runtime_dtype is set.

        Args:
            array: Input activation array to potentially quantize.

        Returns:
            Quantized array if runtime_dtype is configured, otherwise
            returns the input unchanged.
        """
        if self.config.runtime_dtype is not None:
            return quantize(
                array=array,
                dtype=self.config.runtime_dtype,
                group_size=self.config.group_size,
                simulate=False,
            )
        return array

    def _dequantize_array(self, wq: jax.Array, scale: jax.Array, bias: jax.Array | None):
        """Dequantize weights back to full precision for computation.

        Converts quantized weights to full precision using the stored
        scaling factors. The dequantization method depends on the
        quantization type.

        Args:
            wq: Quantized weight array.
            scale: Per-block scaling factors.
            bias: Per-block bias values (affine mode) or None.

        Returns:
            Dequantized weight array in full precision (param_dtype).

        Raises:
            ValueError: If the configured quantization dtype is not supported
                for dequantization.
        """
        mode, group_size, bits, _ = self._resolve_ejkernel_params()
        array = ej_dequantize(
            wq,
            scale,
            bias,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        return array.astype(self.param_dtype)

    def from_quantized(self, rngs: rnglib.Rngs | None = None) -> RowParallelLinear | ColumnParallelLinear:
        """Convert this quantized module back to a regular Linear module.

        Creates a non-quantized linear layer with the same configuration
        and dequantized weights. Useful for debugging, fine-tuning, or
        deployment scenarios where memory is not constrained.

        Args:
            rngs: Random number generators for the new module. If None,
                creates a default Rngs with seed 0.

        Returns:
            A RowParallelLinear or ColumnParallelLinear instance with
            dequantized weights, depending on self._direction.

        Raises:
            ValueError: If _direction is not "row" or "column".

        Example:
            >>> quantized_layer = ColumnParallelLinearQuantized(...)
            >>> regular_layer = quantized_layer.from_quantized()
            >>> # regular_layer is now a ColumnParallelLinear
        """
        if rngs is None:
            rngs = nn.Rngs(0)
        if self._direction == "row":
            linear_class = RowParallelLinear
        elif self._direction == "column":
            linear_class = ColumnParallelLinear
        else:
            raise ValueError(
                "unknown direction detected Try To use module with Known "
                "direction in ur impls to stop getting such errors."
            )
        linear = nn.eval_shape(
            lambda r: linear_class(
                in_features=self.in_features,
                out_features=self.out_features,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                rngs=r,
            ),
            rngs,
        )

        kernel_value = getattr(self.quant_kernel, "value", self.quant_kernel)
        scale_value = getattr(self.quant_scales, "value", self.quant_scales)
        bias_value = getattr(self.quant_biases, "value", self.quant_biases)
        dequantized_kernel = self._dequantize_array(kernel_value, scale_value, bias_value)
        linear.kernel = nn.Param(dequantized_kernel)

        if self.use_bias:
            linear.bias = nn.Param(self.bias.value)

        return linear

    def restage(self, kernel: jax.Array, bias: jax.Array | None):
        """Update the layer's weights by quantizing new kernel values.

        Replaces the current quantized weights with newly quantized versions
        of the provided kernel. This is useful when loading pre-trained
        weights or updating weights during training.

        Args:
            kernel: New kernel weights to quantize and store.
                Can be a jax.Array or nn.Param.
            bias: New bias values to store. Can be a jax.Array, nn.Param,
                or None. Ignored if use_bias is False.

        Returns:
            self: The updated layer instance (for method chaining).

        Example:
            >>> layer = ColumnParallelLinearQuantized(...)
            >>> new_weights = jnp.ones((768, 3072))
            >>> layer.restage(new_weights, None)
        """
        kernel_value = getattr(kernel, "value", kernel)
        bias_value = getattr(bias, "value", bias)
        wq, scale, quant_bias = self._quantize_array(kernel_value)
        self.quant_kernel.value = wq
        self.quant_scales.value = scale
        self.quant_biases.value = quant_bias
        if bias_value is not None and self.use_bias:
            self.bias.value = bias_value
        return self

    def _resolve_shard_specs(
        self,
        mesh: jax.sharding.Mesh,
        inputs_2d: Array,
        kernel_value: Array,
        scale_value: Array,
        bias_value: Array | None,
        is_tpu: bool,
    ) -> tuple | None:
        """Resolve partition specs for distributed quantized matmul.

        Returns ``(input_spec, kernel_spec, scale_spec, bias_spec,
        output_spec, tp_axis_name)`` when sharded execution is viable,
        or ``None`` to signal the caller should fall back to a direct matmul.
        """
        input_spec = _spec_for_mesh(inputs_2d, mesh)
        kernel_spec = _spec_for_mesh(kernel_value, mesh)
        scale_spec = _spec_for_mesh(scale_value, mesh)
        bias_spec = _spec_for_mesh(bias_value, mesh)

        tp_axis_name = None
        use_forced_layout = False

        # On TPU, force a shard_map route even when NamedSharding metadata is
        # absent during lowered compilation.
        if is_tpu and self._direction in {"row", "column"}:
            tp_axis_name = _pick_tensor_axis_name(mesh)
            if tp_axis_name is not None:
                use_forced_layout = True
                input_spec = jax.sharding.PartitionSpec(None, None)
                kernel_spec = (
                    jax.sharding.PartitionSpec(tp_axis_name, None)
                    if self._direction == "row"
                    else jax.sharding.PartitionSpec(None, tp_axis_name)
                )
                scale_spec = kernel_spec
                if bias_value is not None:
                    bias_rank = int(getattr(bias_value, "ndim", 0))
                    kernel_axes = tuple(kernel_spec)
                    if bias_rank > 0 and bias_rank <= len(kernel_axes):
                        bias_spec = jax.sharding.PartitionSpec(*kernel_axes[-bias_rank:])
                    elif bias_rank > 0:
                        bias_spec = jax.sharding.PartitionSpec(*((None,) * bias_rank))
                    else:
                        bias_spec = jax.sharding.PartitionSpec()
                else:
                    bias_spec = jax.sharding.PartitionSpec()

        if not use_forced_layout:
            if not any(_spec_is_sharded(s) for s in (input_spec, kernel_spec, scale_spec, bias_spec)):
                return None
            if self._direction not in {"row", "column"}:
                return None
            if not _spec_matches_kernel_parallel_layout(kernel_spec, scale_spec, self._direction):
                return None
            if bias_value is not None and not _spec_matches_kernel_parallel_layout(
                kernel_spec, bias_spec, self._direction
            ):
                return None

            tp_axis_name = _extract_tp_axis_name(kernel_spec, self._direction, mesh)

            # Reject non-TP axes on the K dimension.
            kernel_k_axes = _axis_names(kernel_spec[0] if len(kernel_spec) > 0 else None)
            input_k_axes = _axis_names(input_spec[-1] if len(input_spec) > 0 else None)
            if any(a != tp_axis_name for a in (*kernel_k_axes, *input_k_axes)):
                return None

            # Row-parallel reduction assumes batch is local; reject TP on batch.
            if self._direction == "row" and tp_axis_name is not None:
                input_batch_axes = _axis_names(input_spec[0] if len(input_spec) > 0 else None)
                if tp_axis_name in input_batch_axes:
                    return None

        output_spec = jax.sharding.PartitionSpec(
            input_spec[0] if len(input_spec) > 0 else None,
            kernel_spec[1] if len(kernel_spec) > 1 else None,
        )

        return input_spec, kernel_spec, scale_spec, bias_spec, output_spec, tp_axis_name

    def _distributed_quantized_matmul(
        self,
        inputs_2d: Array,
        kernel_value: Array,
        scale_value: Array,
        bias_value: Array | None,
        *,
        group_size: int,
        bits: int,
        mode: str,
    ) -> Array:
        """Run quantized matmul under shard_map with explicit TP communication.

        Quantized CUDA/Pallas kernels are invoked on local shards only; cross-shard
        semantics are restored with collectives:
            - row parallel: reduce partial outputs with ``psum`` on TP
            - column parallel: gather TP-sharded inputs when needed
        """
        backend = jax.default_backend()
        qmm_kwargs = self._qmm_runtime_kwargs(backend, m_tokens=int(inputs_2d.shape[0]), quant_mode=mode)

        def _direct_matmul() -> Array:
            return ej_quantized_matmul(
                inputs_2d,
                kernel_value,
                scale_value,
                bias_value,
                transpose=False,
                group_size=group_size,
                bits=bits,
                mode=mode,
                **qmm_kwargs,
            )

        mesh = _pick_mesh_from_arrays(inputs_2d, kernel_value, scale_value, bias_value)
        if mesh is None:
            return _direct_matmul()

        resolved = self._resolve_shard_specs(mesh, inputs_2d, kernel_value, scale_value, bias_value, backend == "tpu")
        if resolved is None:
            return _direct_matmul()

        input_spec, kernel_spec, scale_spec, bias_spec, output_spec, tp_axis_name = resolved
        direction = self._direction

        def _local_matmul(local_inputs, local_kernel, local_scales, local_biases=None):
            local_inputs = _reconcile_input_k_dim(
                local_inputs,
                local_kernel,
                direction,
                tp_axis_name,
                allow_column_all_gather=self.qmm_allow_input_all_gather,
            )
            out = ej_quantized_matmul(
                local_inputs,
                local_kernel,
                local_scales,
                local_biases,
                transpose=False,
                group_size=group_size,
                bits=bits,
                mode=mode,
                **qmm_kwargs,
            )
            if tp_axis_name is not None and direction == "row":
                out = jax.lax.psum(out, tp_axis_name)
            return out

        shard_kw = dict(mesh=mesh, out_specs=output_spec, check_vma=False)

        if bias_value is None:

            @partial(shard_map, in_specs=(input_spec, kernel_spec, scale_spec), **shard_kw)
            def _mapped(local_inputs, local_kernel, local_scales):
                return _local_matmul(local_inputs, local_kernel, local_scales)

            return _mapped(inputs_2d, kernel_value, scale_value)

        @partial(shard_map, in_specs=(input_spec, kernel_spec, scale_spec, bias_spec), **shard_kw)
        def _mapped(local_inputs, local_kernel, local_scales, local_biases):
            return _local_matmul(local_inputs, local_kernel, local_scales, local_biases)

        return _mapped(inputs_2d, kernel_value, scale_value, bias_value)

    @jax.named_scope("easydel-linear-quantized-call")
    def __call__(self, inputs: Array, w: Array | None = None) -> Array:
        """Apply the quantized linear transformation to inputs.

        Dequantizes weights on-the-fly and performs matrix multiplication
        with the input. When enabled, uses ejkernel's fused quantized
        matmul kernels for weight dequantization and multiplication.

        Args:
            inputs: Input array of shape (..., in_features).
            w: Optional pre-dequantized weights to use instead of the
                stored quantized weights. Useful for debugging or when
                weights have been modified externally.

        Returns:
            Output array of shape (..., out_features) after linear
            transformation and optional bias addition.

        Note:
            The computation path varies by configuration:
            - If w is provided: uses standard matmul with the provided weights.
            - Otherwise: uses ejkernel quantized_matmul.
        """
        if self.dtype is not None:
            inputs = inputs.astype(self.dtype)

        if w is not None:
            kernel = w.astype(self.dtype) if self.dtype is not None else w
            subscript = "...ik,...kj->...ij" if inputs.ndim > 1 else "...k,...kj->...j"
            out = jnp.einsum(subscript, inputs, kernel, precision=self.precision, optimize=True)
        else:
            kernel_value = getattr(self.quant_kernel, "value", self.quant_kernel)
            scale_value = getattr(self.quant_scales, "value", self.quant_scales)
            bias_value = getattr(self.quant_biases, "value", self.quant_biases)

            mode, group_size, bits, needs_biases = self._resolve_ejkernel_params()
            if needs_biases and bias_value is None:
                raise ValueError("Affine quantization requires quant_biases; re-quantize the module weights.")

            out = self._distributed_quantized_matmul(
                inputs.reshape((-1, inputs.shape[-1])),
                kernel_value,
                scale_value,
                bias_value,
                group_size=group_size,
                bits=bits,
                mode=mode,
            ).reshape((*inputs.shape[:-1], self.out_features))

            if self.dtype is not None:
                out = out.astype(self.dtype)

        bias_value: jax.Array | None = self.bias.value if self.use_bias else None
        if self.use_bias and bias_value is not None:
            if self.dtype is not None:
                bias_value = bias_value.astype(self.dtype)
            out = out + jnp.reshape(bias_value, (1,) * (out.ndim - 1) + (-1,))

        return out

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, tp.Any]:
        """Return dynamic partition specs for quantized parameters."""
        if self._direction is None:
            return {}
        mode, group_size, _bits, needs_biases = self._resolve_ejkernel_params()
        specs = _quantized_linear_craft_spec(
            direction=self._direction,
            use_bias=self.use_bias,
            mode=mode,
            group_size=group_size,
            needs_biases=needs_biases,
        )
        if partition_manager is None:
            return specs

        mesh = _kwargs.get("mesh")

        def _shape_of(name: str) -> tuple[int, ...] | None:
            if not hasattr(self, name):
                return None
            value = getattr(getattr(self, name), "value", getattr(self, name))
            if value is None or not hasattr(value, "shape"):
                return None
            return tuple(value.shape)

        safe_specs: dict[str, tp.Any] = {}
        for name, axes in specs.items():
            shape = _shape_of(name)
            if shape is None:
                safe_specs[name] = axes
                continue
            safe_specs[name] = resolve_safe_sharding(
                axes=axes,
                shape=shape,
                partition_manager=partition_manager,
                mesh=mesh,
            )
        return safe_specs

    @property
    def wqdtype(self) -> QuantizationType:
        """Get the weight quantization data type.

        Returns:
            The QuantizationType enum value indicating the quantization
            format used for weights (INT8, NF4, MXFP4, etc.).
        """
        return QuantizationType(self.config.dtype)

    def __repr__(self):
        """Return a string representation of the quantized layer.

        Returns:
            A string showing the class name, in_features, out_features,
            and the quantization data type.
        """
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"wqdtype={self.wqdtype}"
            ")"
        )

    def __str__(self):
        """Return a string representation of the quantized layer.

        Returns:
            Same as __repr__.
        """
        return self.__repr__()


class RowParallelLinearQuantized(ParallelLinearQuantized):
    """Row-parallel variant of quantized linear layer.

    This class specializes ParallelLinearQuantized for row-wise parallelism,
    where the input dimension is partitioned across devices. In row parallelism,
    each device holds a subset of input features and computes partial results
    that are then reduced across devices.

    Attributes:
        _direction: Fixed to "row" to indicate row-wise parallelism.

    Example:
        >>> layer = RowParallelLinearQuantized(
        ...     in_features=768,
        ...     out_features=3072,
        ...     config=config,
        ...     rngs=rngs
        ... )
    """

    _direction: tp.Literal["row"] = "row"


class ColumnParallelLinearQuantized(ParallelLinearQuantized):
    """Column-parallel variant of quantized linear layer.

    This class specializes ParallelLinearQuantized for column-wise parallelism,
    where the output dimension is partitioned across devices. In column parallelism,
    each device computes a subset of output features independently without
    requiring reduction.

    Attributes:
        _direction: Fixed to "column" to indicate column-wise parallelism.

    Example:
        >>> layer = ColumnParallelLinearQuantized(
        ...     in_features=768,
        ...     out_features=3072,
        ...     config=config,
        ...     rngs=rngs
        ... )
    """

    _direction: tp.Literal["column"] = "column"
