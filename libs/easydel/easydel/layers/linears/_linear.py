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

"""Tensor-parallel linear layers (column-, row-, and unsharded variants).

The classes here are the workhorse projections used inside attention QKV /
output projections and MLP gate/up/down matmuls throughout EasyDeL. They wrap
``spectrax.Parameter`` weights with explicit :class:`TensorLayout` shardings
so that the same module can run unsharded on a single device, FSDP-sharded
along the contraction axis, or tensor-parallel along the output axis without
the caller having to build different objects.

The matmul itself is written with ``jnp.einsum`` and respects ``self.dtype``
for compute and ``self.param_dtype`` for storage; both are decoupled so that
e.g. fp4-stored weights can still be matmul'd in bf16. Conversion to a
quantized clone of the same layer is provided by :meth:`to_quantized`,
preserving direction and the runtime distributed-matmul hook.

Layer summary:

* :class:`ParallelLinear` – base class, ``_direction = None`` (replicated),
  also the default for layers that are not explicitly tensor-parallel.
* :class:`RowParallelLinear` – input axis is sharded across the TP mesh axis;
  output is replicated and requires an all-reduce on the partial sums in the
  caller's mesh. Use as the *second* projection in an MLP.
* :class:`ColumnParallelLinear` – output axis is sharded across the TP mesh;
  no comm needed at the matmul itself, but downstream consumers must handle
  the sharded output (typically followed by a row-parallel layer that
  consumes the same TP partition).
"""

from __future__ import annotations

import collections.abc
import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from jax import lax
from jaxtyping import Array, Shaped
from spectrax.common_types import ColumnWise, Replicated, RowWise, SRowWise

from easydel.infra.sharding import TensorLayout, sharding_for_layout
from easydel.layers.quantization._configs import QuantizationConfig

if tp.TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig


class FusedProjectionLayout(tp.Protocol):
    """Minimal protocol carried by fused projection linears."""

    def split(self, x: Array, *, config: "EasyDeLBaseConfig | None" = None) -> tuple[Array, ...]: ...

    def reform_param(
        self,
        target_prefix: str,
        *,
        config: "EasyDeLBaseConfig | None" = None,
        include_bias: bool = False,
    ) -> dict: ...


def promote_dtype(values, *, dtype=None):
    """Cast a tuple of arrays to a shared dtype.

    Lightweight replacement for ``flax.linen.dtypes.promote_dtype`` that simply
    forwards everything through ``jnp.asarray`` when a target dtype is given.

    Args:
        values: Iterable of arrays (or array-like values) to be promoted.
        dtype: Target dtype to cast all values to. If ``None``, the inputs are
            returned unchanged.

    Returns:
        Tuple of arrays cast to ``dtype`` if specified, otherwise the original
        ``values`` argument unchanged.
    """
    if dtype is None:
        return values
    return tuple(jnp.asarray(v, dtype=dtype) for v in values)


if tp.TYPE_CHECKING:
    from ._linear_quantized import ColumnParallelLinearQuantized, RowParallelLinearQuantized

Dtype = jnp.dtype
Initializer = jax.nn.initializers.Initializer
PrecisionLike = lax.PrecisionLike
Shape = collections.abc.Sequence[int]
AxisNames = str | collections.abc.Sequence[str] | tuple[str, ...]

# Default initializers
default_kernel_init = jax.nn.initializers.lecun_normal()
default_bias_init = jax.nn.initializers.zeros


class ParallelLinear(spx.Module):
    """Base linear layer ``y = scale * (x @ W) + b`` with explicit sharding hooks.

    This is the workhorse linear projection used throughout EasyDeL — attention
    QKV/output, MLP gate/up/down, classification heads, etc. The behaviour
    mirrors a stock ``nn.Linear`` (matmul plus optional bias) but with three
    extras geared for very large transformer training:

    * **Sharding-aware kernel placement.** The weight ``Parameter`` is created
      with a :func:`sharding_for_layout` annotation derived from the subclass'
      ``_direction``: ``RowWise`` for :class:`RowParallelLinear`, ``ColumnWise``
      for :class:`ColumnParallelLinear`, ``Replicated`` (via ``None``) for the
      base class. Bias placement follows the *output* axis of the kernel — see
      :meth:`__init__` for the SRowWise/Replicated choice.
    * **Optional output scaling.** ``scale`` may be a constant float, the
      string ``"fan_in"`` (multiplies the output by ``in_features ** -0.5``,
      muP-style residual rescaling) or ``"fan_out"`` (``out_features ** -0.5``).
      Resolved once in ``__init__`` into a closure ``_scale_operator`` so the
      forward pass has no Python branching.
    * **Quantized friend pattern.** :meth:`to_quantized` returns a
      :class:`RowParallelLinearQuantized` / :class:`ColumnParallelLinearQuantized`
      twin built with :func:`jax.eval_shape` (so no data is materialized on the
      conversion path) and the existing kernel/bias are restaged into it,
      enabling post-training quantization in-place on a live state tree.

    The compute path uses ``jnp.einsum`` with the appropriate subscript for
    1-D vs ND inputs and respects ``precision`` (forwarded to the einsum).

    Attributes:
        in_features (int): Size of the input feature axis.
        out_features (int | Sequence[int]): Size of the output feature axis.
            When passed as a sequence, the kernel is built with width
            ``sum(out_features)`` for fused QKV / gate-up projections that are
            split downstream by the optional fused projection layout.
        use_bias (bool): Whether the layer carries a learnable bias.
        dtype (Dtype | None): Compute dtype. ``None`` defers to the input.
        param_dtype (Dtype): Storage dtype for ``weight`` and ``bias``.
        precision (PrecisionLike): JAX precision flag forwarded to the einsum.
        kernel_init (Initializer): Weight initializer (default LeCun normal).
        bias_init (Initializer): Bias initializer (default zeros).
        weight (spx.Parameter[Array]): Kernel of shape
            ``(in_features, sum(out_features))``, sharding determined by
            ``_direction``.
        bias (spx.Parameter[Array] | None): Bias of shape ``(out_features,)``,
            ``None`` when ``use_bias=False``. Sharded ``SRowWise`` for
            column-parallel layers, ``Replicated`` otherwise.
        distributed_matmul (Any | None): Optional injectable matmul backend
            (e.g. an all-gather-fused matmul); ``None`` means "use the
            jit-compiler default", which is correct in almost every case.
        _direction (Literal["row", "column"] | None): Class-level marker
            consumed by :func:`sharding_for_layout` and by
            :meth:`_quantized_friend` to pick the right quantized clone.
    """

    _direction: tp.Literal["row", "column"] | None = None

    if tp.TYPE_CHECKING:

        def __call__(
            self,
            inputs: Shaped[Array, "... in_features"],
            w: Array | None = None,
        ) -> Shaped[Array, "... out_features"]: ...

    def __init__(
        self,
        in_features: int,
        out_features: int | collections.abc.Sequence[int],
        *,
        scale: float | tp.Literal["fan_in", "fan_out"] = 1.0,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: spx.Rngs | None = None,
        layout: FusedProjectionLayout | None = None,
        sharding_layout: TensorLayout | tp.Any | None = None,
        bias_sharding_layout: TensorLayout | tp.Any | None = None,
    ):
        """Initialize a parallel linear layer.

        Creates a linear transformation layer with configurable parameters
        and optional output scaling.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample. Can also be a sequence
                of integers for tensor-parallel merged outputs.
            scale: Output scaling factor. Can be:
                - A float value for direct scaling
                - "fan_in" for 1/sqrt(in_features) scaling
                - "fan_out" for 1/sqrt(out_features) scaling
                Defaults to 1.0 (no scaling).
            use_bias: If True, adds a learnable bias to the output.
                Defaults to True.
            dtype: Data type for computation. If None, uses input dtype.
                Defaults to None.
            param_dtype: Data type for storing parameters. Defaults to float32.
            precision: JAX precision for matrix multiplication. Can be None,
                'default', 'high', 'highest', or specific precision tuples.
                Defaults to None.
            kernel_init: Initializer function for the weight matrix.
                Defaults to lecun_normal().
            bias_init: Initializer function for the bias vector.
                Defaults to zeros.
            rngs: Random number generators for initialization. If None,
                creates a default Rngs with seed 0.
            layout: Optional fused projection layout used by checkpoint reform
                and activation split helpers.
            sharding_layout: Optional explicit weight layout. When omitted,
                the layer keeps the legacy row/column layout derived from
                ``_direction``.
            bias_sharding_layout: Optional explicit bias layout. When omitted,
                bias placement follows the legacy row/column rule.
        """
        rngs_computed: spx.Rngs
        if rngs is None:
            rngs_computed = spx.Rngs(0)
        else:
            rngs_computed = rngs

        out_features_sum: int
        if isinstance(out_features, collections.abc.Sequence):
            out_features_sum = sum(out_features)
        else:
            out_features_sum = out_features

        scale_computed: float
        scale_is_fan_in: bool = scale == "fan_in"
        scale_is_fan_out: bool = scale == "fan_out"
        if scale_is_fan_in:
            scale_computed = in_features**-0.5
        elif scale_is_fan_out:
            scale_computed = out_features_sum**-0.5
        else:
            scale_computed = scale

        needs_scaling: bool = scale_computed != 1.0
        if needs_scaling:

            def _scale_operator(x: Array) -> Array:
                scaled: Array = x * scale_computed
                return scaled

        else:

            def _scale_operator(x: Array) -> Array:
                return x

        self._scale_operator: tp.Callable[[Array], Array] = _scale_operator
        self.in_features: int = in_features
        self.out_features: int | collections.abc.Sequence[int] = out_features
        self.layout: FusedProjectionLayout | None = layout
        self.sharding_layout: TensorLayout | None = TensorLayout.from_any(sharding_layout)
        self.bias_sharding_layout: TensorLayout | None = TensorLayout.from_any(bias_sharding_layout)

        self.use_bias: bool = use_bias
        self.dtype: Dtype | None = dtype
        self.param_dtype: Dtype = param_dtype
        self.precision: PrecisionLike = precision
        self.kernel_init: Initializer = kernel_init
        self.bias_init: Initializer = bias_init

        weight_key: tp.Any = rngs_computed.parameters
        weight_shape: tuple[int, int] = (in_features, out_features_sum)
        weight_initialized: Array = kernel_init(weight_key, weight_shape, param_dtype)
        weight_layout = self.sharding_layout
        if weight_layout is None:
            if self._direction == "row":
                weight_layout = TensorLayout.from_any(RowWise)
            elif self._direction == "column":
                weight_layout = TensorLayout.from_any(ColumnWise)
        self.weight: spx.Parameter = spx.Parameter(weight_initialized, sharding=sharding_for_layout(weight_layout))

        if use_bias:
            bias_key: tp.Any = rngs_computed.parameters
            bias_shape: tuple[int] = (out_features_sum,)
            bias_initialized: Array = bias_init(bias_key, bias_shape, param_dtype)
            # Bias sharding must match the weight's output-axis sharding:
            #  * column-parallel weight is ([FSDP,SP], TP) — output (column)
            #    dim is TP-sharded, so the 1-D bias along that dim must be
            #    sharded by TP (`SRowWise`). Replicating it would add the
            #    full bias to each rank's partial output.
            #  * row-parallel weight is (TP, [FSDP,SP]) — output is the
            #    second axis, replicated across TP, so bias is replicated.
            #  * unspecified direction: replicated (safe default).
            if self.bias_sharding_layout is not None:
                bias_layout = self.bias_sharding_layout
            elif self._direction == "column":
                bias_layout = TensorLayout.from_any(SRowWise)
            else:
                bias_layout = TensorLayout.from_any(Replicated)
            self.bias: spx.Parameter | None = spx.Parameter(
                bias_initialized,
                sharding=sharding_for_layout(bias_layout),
            )
        else:
            self.bias = None
        self.distributed_matmul: tp.Any | None = None

    def split(self, outputs: Array, *, config: "EasyDeLBaseConfig | None" = None) -> tuple[Array, ...]:
        """Split outputs with the projection's fused layout."""
        if self.layout is None:
            raise ValueError("This linear layer does not carry a fused projection layout.")
        return self.layout.split(outputs, config=config)

    def build_reform_param(
        self,
        target_prefix: str,
        *,
        config: "EasyDeLBaseConfig | None" = None,
        include_bias: bool | None = None,
    ) -> dict:
        """Build checkpoint reform rules from the projection's fused layout."""
        if self.layout is None:
            raise ValueError("This linear layer does not carry a fused projection layout.")
        return self.layout.reform_param(
            target_prefix,
            config=config,
            include_bias=self.use_bias if include_bias is None else include_bias,
        )

    def forward(
        self, inputs: Shaped[Array, "... in_features"], w: Array | None = None
    ) -> Shaped[Array, "... out_features"]:
        """Apply the linear transformation using native JAX operations.

        Performs the matrix multiplication y = x @ W + b with proper dtype
        promotion and optional scaling.

        Args:
            inputs: The input array of shape (..., in_features). The batch
                dimensions can be arbitrary.
            w: Optional weight matrix to use instead of self.weight. This is
                useful for weight sharing or external weight injection.
                Defaults to None (uses self.weight).

        Returns:
            The transformed output array of shape (..., out_features).
            If scale is configured, the output is scaled accordingly.
        """
        w_is_none: bool = w is None
        kernel: Array
        if w_is_none:
            kernel = self.weight.value
        else:
            kernel = w
        if kernel is None:
            raise ValueError("ParallelLinear kernel is missing. This layer cannot run without kernel weights.")

        has_bias: bool = self.use_bias
        bias: Array | None
        if has_bias and self.bias is not None:
            bias = self.bias.value
        else:
            bias = None

        bias_is_not_none: bool = bias is not None
        inputs_promoted: Array
        kernel_promoted: Array
        bias_promoted: Array | None
        if bias_is_not_none:
            inputs_promoted, kernel_promoted, bias_promoted = promote_dtype((inputs, kernel, bias), dtype=self.dtype)
        else:
            inputs_promoted, kernel_promoted = promote_dtype((inputs, kernel), dtype=self.dtype)
            bias_promoted = None

        out_features = self.out_features
        if isinstance(out_features, collections.abc.Sequence):
            out_features = sum(out_features)
        with jax.named_scope(f"easydel/linear/{type(self).__name__}/in{self.in_features}/out{out_features}"):
            y: Shaped[Array, "... out_features"] = jnp.einsum(
                "...i,io->...o",
                inputs_promoted,
                kernel_promoted,
                precision=self.precision,
            )

        y_scaled: Shaped[Array, "... out_features"] = self._scale_operator(y)

        y_final: Shaped[Array, "... out_features"]
        if bias_promoted is not None:
            y_final = y_scaled + bias_promoted
        else:
            y_final = y_scaled

        return y_final

    def native_forward(
        self,
        inputs: Shaped[Array, "... in_features"],
        *,
        w: Array | None = None,
    ) -> Shaped[Array, "... out_features"]:
        """Trace-safe alias to :meth:`forward` used by LM-head projection helpers.

        ``make_lm_head_fn`` and the rematerialisation logic in the LM-head
        path need a stable, non-overridable function name to call so that the
        bypass remains trace-safe even when subclasses override ``forward``
        for fused execution. This alias forwards verbatim and is intentionally
        not decorated with ``@jax.named_scope`` so the LM-head profile
        attribution stays clean.

        Args:
            inputs: Input tensor of shape ``(..., in_features)``.
            w: Optional kernel override (e.g. a tied embedding matrix). When
                ``None`` the layer's own ``self.weight.value`` is used.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        return self.forward(inputs=inputs, w=w)

    def to_quantized(
        self,
        config: QuantizationConfig,
        **kwargs,
    ) -> ColumnParallelLinearQuantized | RowParallelLinearQuantized:
        """Convert this layer to a quantized version.

        Creates a quantized linear layer with the same configuration but
        weights stored in a compressed format according to the provided
        quantization configuration.

        Args:
            config: Quantization configuration specifying the quantization
                type (INT8, NF4, etc.) and related parameters.
            **kwargs: Optional runtime quantized-matmul controls forwarded
                to the quantized linear module (for example qmm platform/path
                overrides and tuned-config toggles).

        Returns:
            A RowParallelLinearQuantized or ColumnParallelLinearQuantized
            instance, depending on self._direction.

        Raises:
            ValueError: If _direction is not "row" or "column".

        Example:
            >>> from easydel.layers.quantization import QuantizationConfig, QuantizationType
            >>> layer = ColumnParallelLinear(768, 3072, rngs=spx.Rngs(0))
            >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
            >>> quantized_layer = layer.to_quantized(config)
        """
        firend = self._quantized_friend
        lazy_module = jax.eval_shape(
            lambda rngs: firend(
                in_features=self.in_features,
                out_features=self.out_features,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                config=config,
                **kwargs,
                rngs=rngs,
            ),
            spx.Rngs(0),
        )

        if isinstance(self.weight.value, jax.ShapeDtypeStruct):
            return lazy_module

        return lazy_module.restage(kernel=self.weight, bias=self.bias)

    @property
    def _quantized_friend(self) -> type[RowParallelLinearQuantized] | type[ColumnParallelLinearQuantized]:
        """Get the corresponding quantized layer class.

        Returns:
            The quantized layer class matching this layer's parallelism
            direction (RowParallelLinearQuantized or ColumnParallelLinearQuantized).

        Raises:
            ValueError: If _direction is not "row" or "column".
        """
        from ._linear_quantized import ColumnParallelLinearQuantized, RowParallelLinearQuantized

        if self._direction == "row":
            return RowParallelLinearQuantized
        elif self._direction == "column":
            return ColumnParallelLinearQuantized
        else:
            raise ValueError("unknown direction, with no friend!")


class RowParallelLinear(ParallelLinear):
    """:class:`ParallelLinear` with kernel sharded along the *contraction* axis.

    The weight matrix is partitioned ``(TP, [FSDP, SP])`` so that the input
    features axis (the one that gets contracted away in ``x @ W``) is split
    across the tensor-parallel mesh axis. Each TP rank holds the rows of ``W``
    corresponding to its slice of the input, computes a partial product, and
    the all-reduce required to sum the partial outputs is left to the
    surrounding mesh / shard_map / collective scheduler — this layer does
    *not* call ``lax.psum`` itself.

    Bias is replicated (the output axis is replicated under row-parallelism),
    so the same bias is added on every TP rank after the all-reduce.

    Conventional usage is the second projection in an MLP::

        h = ColumnParallelLinear(d, ff)(x)   # outputs sharded along TP
        h = activation(h)
        y = RowParallelLinear(ff, d)(h)      # contracts along TP, then all-reduce

    Sets ``_direction = "row"``; everything else is inherited from
    :class:`ParallelLinear`.
    """

    _direction: tp.Literal["row"] = "row"


class ColumnParallelLinear(ParallelLinear):
    """:class:`ParallelLinear` with kernel sharded along the *output* axis.

    The weight matrix is partitioned ``([FSDP, SP], TP)`` so that the output
    features axis is split across the tensor-parallel mesh axis. Each TP rank
    holds a column block of ``W`` and produces an independent slice of the
    output — no communication is needed at the matmul itself. The bias along
    the output axis is correspondingly sharded ``SRowWise`` (one block per TP
    rank); replicating it would double-count the bias.

    Conventional usage is the first projection in an MLP, paired with a
    downstream :class:`RowParallelLinear` that consumes the same TP partition::

        h = ColumnParallelLinear(d, ff)(x)   # outputs sharded along TP
        h = activation(h)
        y = RowParallelLinear(ff, d)(h)      # contracts along TP, then all-reduce

    Sets ``_direction = "column"``; everything else is inherited from
    :class:`ParallelLinear`.
    """

    _direction: tp.Literal["column"] = "column"
