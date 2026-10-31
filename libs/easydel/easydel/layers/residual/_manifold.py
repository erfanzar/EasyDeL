# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Manifold-constrained hyper-connections over explicit residual streams.

Unlike the low-rank, element-wise gates in :class:`GatedResidual`, mHC reads
streams with scalar gates and mixes residual streams with a Sinkhorn-projected
matrix. Both are residual families, not interchangeable parameterizations.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import jax
import spectrax as spx
from eformer import common_types
from ejkernel.modules import MHCFusedConfig, mhc_coefficients, mhc_collapse, mhc_expand, sinkhorn_knopp
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Float


class HyperStreamSharding(common_types.DynamicShardingAxes):
    """Logical layout of ``[batch, sequence, stream, hidden]`` residuals.

    Streams stay local; tensor parallelism applies to hidden features, not the
    small stream count. The resolver supplies training/decode and stage-local
    axis mappings, including custom physical mesh names.
    """

    axes: tp.ClassVar = [common_types.BATCH, common_types.QUERY_LENGTH, common_types.EMPTY, common_types.EMBED]
    mode: tp.ClassVar = 1


def _stream_layout(mesh_source: object | None, shape: tuple[int, ...]):
    """Resolve the current mesh and optional logical stream layout dynamically."""
    mesh = getattr(mesh_source, "mesh", mesh_source)
    mesh = getattr(mesh, "jax_mesh", mesh)
    resolver = getattr(mesh_source, "runtime_sharding_resolver", None)
    if mesh is None or resolver is None:
        return mesh, None
    spec = resolver.with_mesh(mesh).resolve(dynamic_axes=HyperStreamSharding, shape=shape)
    return mesh, spec


@dataclass(frozen=True)
class ManifoldHyperConnectionConfig:
    """Scalar settings shared by the manifold connection and learned head.

    Args:
        hidden_size: Width of each residual stream.
        hc_mult: Number of explicit residual streams.
        eps: Offset added to read gates and Sinkhorn normalization denominators.
        norm_eps: Epsilon of the unweighted, flattened-stream RMSNorm.
        iters: Number of Sinkhorn column normalizations (with intervening rows).
        initializer: Standard deviation of the normal projection initializer.
        use_fused_coefficients: Opt into packed TPU gate/Sinkhorn forward and
            backward kernels. First-order reverse AD only on the optimized path;
            leave disabled for higher derivatives. Does not change parameters.
    """

    hidden_size: int
    hc_mult: int = 4
    eps: float = 1e-6
    norm_eps: float = 1e-6
    iters: int = 20
    initializer: float = 0.02
    use_fused_coefficients: bool = False

    @classmethod
    def from_model_config(cls, config: tp.Any) -> ManifoldHyperConnectionConfig:
        """Read the standard ``hc_*`` model-config fields (DeepSeek-V4 / V4.1, GLM-5-Next).

        Maps ``hidden_size``, ``hc_mult``, ``hc_eps``, ``rms_norm_eps`` (flattened-stream norm),
        ``hc_sinkhorn_iters``, ``initializer_range`` and ``hc_use_fused_coefficients``.
        """
        return cls(
            hidden_size=config.hidden_size,
            hc_mult=config.hc_mult,
            eps=config.hc_eps,
            norm_eps=config.rms_norm_eps,
            iters=config.hc_sinkhorn_iters,
            initializer=config.initializer_range,
            use_fused_coefficients=config.hc_use_fused_coefficients,
        )


def _normalized_flat_streams(x: Array, eps: float) -> Array:
    """Flatten streams and apply unweighted RMSNorm entirely in fp32."""
    flat = x.reshape(*x.shape[:2], -1).astype(jnp.float32)
    return flat * jax.lax.rsqrt(jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + eps)


def _project_streams(x: Array, weight: Array, eps: float, precision, mesh, spec) -> Array:
    """Project feature-sharded streams without flattening the global activation.

    Only per-token RMS statistics and projection logits are reduced across the
    feature axes. Flattening happens *inside* each shard, with identically
    sliced weights, avoiding a redistribution of the ``hc * hidden`` stream.
    Parameter storage remains the original two-dimensional checkpoint layout.
    """
    feature = spec[3] if spec is not None else None
    axes = feature if isinstance(feature, tuple) else (() if feature is None else (feature,))
    axes = tuple(axis for axis in axes if mesh.shape[axis] > 1)
    # The RMS scale is one scalar per token, so it is applied to the [.., hc*(hc+2)] logits instead of the
    # [.., hc*D] streams: the sum of squares is an fp32 reduce and the projection reads the streams in their own
    # dtype with fp32 accumulation, so no fp32 copy of the stream (or of its normalized form) is materialized in
    # the forward or kept for the weight gradient.
    compute_dtype = jnp.result_type(x.dtype, weight.dtype)
    if not axes:
        flat = x.reshape(*x.shape[:2], -1)
        scale = jax.lax.rsqrt(jnp.mean(jnp.square(flat.astype(jnp.float32)), axis=-1, keepdims=True) + eps)
        logits = jnp.matmul(
            flat.astype(compute_dtype),
            weight.astype(compute_dtype).T,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        return logits * scale

    width = weight.shape[0]
    global_width = x.shape[2] * x.shape[3]

    def project(local_x, local_weight):
        flat = local_x.reshape(*local_x.shape[:2], -1)
        square_sum = jax.lax.psum(jnp.sum(jnp.square(flat.astype(jnp.float32)), axis=-1, keepdims=True), axes)
        scale = jax.lax.rsqrt(square_sum / global_width + eps)
        logits = jnp.matmul(
            flat.astype(compute_dtype),
            local_weight.reshape(width, -1).astype(compute_dtype).T,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        # ``scale`` is already identical on every feature shard, so it commutes with the psum.
        return jax.lax.psum(logits, axes) * scale

    return jax.shard_map(
        project,
        mesh=mesh,
        in_specs=(spec, Ps(None, None, feature)),
        out_specs=Ps(spec[0], spec[1], None),
        check_vma=False,
    )(x, weight.reshape(width, x.shape[2], x.shape[3]))


class ManifoldHyperConnection(spx.Module):
    """Learned read/write gates and a doubly-stochastic residual mixer.

    The direct ``fn``, ``base``, and ``scale`` parameters preserve checkpoint
    paths and initialization order; no projection submodule is inserted.
    """

    def __init__(
        self,
        config: ManifoldHyperConnectionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        mesh_source: object | None = None,
        fused_kernel: tp.Literal["none", "pallas"] = "none",
        fused_tile_rows: int = 256,
        fused_fn_pieces: int = 3,
    ):
        """Initialize the connection without depending on a model config type.

        Args:
            config: Scalar manifold settings.
            dtype: Activation dtype metadata; mixing math always uses fp32.
            param_dtype: Parameter storage dtype.
            precision: Precision of the learned projection matmul.
            rngs: Random streams, consumed in ``fn``, ``base``, ``scale`` order.
            mesh_source: Optional mesh or object with a dynamic ``mesh`` property.
                The property is read on every forward, not captured at init, so
                model-config stage-local mesh resolution remains intact. Both
                JAX meshes and wrappers exposing ``jax_mesh`` are supported.
            fused_kernel: ``"none"`` (default) keeps the XLA path below unchanged.
                ``"pallas"`` routes :meth:`forward` through ejkernel ``mhc_collapse``
                and :meth:`residual_write` through ``mhc_expand`` on their Pallas
                TPU implementations (one sweep over the streams per pass, analytic
                backward, the projection at fp32 accuracy regardless of
                ``precision``). :meth:`coefficients` is never fused. Opt-in only:
                on TPU v5 at ``[1, S, 4, 1024]`` bf16 the fused site measured
                1.6-1.7x slower forward+backward than XLA, faster only for
                forward-only use at long sequences. Requires a per-device hidden
                width that is a multiple of 128.
            fused_tile_rows: Pallas rows per grid step (positive multiple of 128).
            fused_fn_pieces: bf16 pieces an fp32 ``fn`` (and the fp32 logit
                cotangent) is split into for the MXU dots; 3 = fp32 accuracy.

        Raises:
            ValueError: Unknown ``fused_kernel``.
        """
        # infra.utils itself imports layers, so this dependency must stay lazy.
        from easydel.infra.utils import ArrayParam

        if fused_kernel not in ("none", "pallas"):
            raise ValueError(f"fused_kernel must be 'none' or 'pallas', got {fused_kernel!r}")
        self.fused_kernel = fused_kernel
        self.fused_tile_rows = int(fused_tile_rows)
        self.fused_fn_pieces = int(fused_fn_pieces)
        self.config = config
        self.mesh_source = mesh_source
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hc_mult = config.hc_mult
        self.hc_eps = config.eps
        self.rms_norm_eps = config.norm_eps
        self.hc_sinkhorn_iters = config.iters
        self.use_fused_coefficients = config.use_fused_coefficients
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = ArrayParam.bound(
            shape=(mix, self.hc_mult * config.hidden_size),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer},
            key=rngs.param,
        )
        self.base = ArrayParam.bound(shape=(mix,), dtype=param_dtype, init_method="zeros", key=rngs.param)
        self.scale = ArrayParam.bound(shape=(3,), dtype=param_dtype, init_method="ones", key=rngs.param)

    def forward(self, hidden_streams: Float[Array, "batch seq hc hidden"]) -> tuple[Array, Array, Array]:
        """Return ``(post, comb, collapsed)`` for streams ``[B, S, hc, D]``.

        Args:
            hidden_streams: Explicit residual streams.

        Returns:
            Write gates ``[B, S, hc]`` and mixer ``[B, S, hc, hc]`` in fp32,
            and the read-weighted sum ``[B, S, D]`` in the input dtype.
            The write helper applies the mixer transposed.
        """
        if self.fused_kernel == "pallas":
            return self._fused_forward(hidden_streams)
        pre, post, comb = self.coefficients(hidden_streams)
        return post, comb, manifold_collapse(pre, hidden_streams)

    def _fused_cfg(self) -> MHCFusedConfig:
        """Pallas dispatch configuration of the opt-in fused path."""
        return MHCFusedConfig(
            platform="pallas",
            backend="tpu",
            tile_rows=self.fused_tile_rows,
            fn_pieces=self.fused_fn_pieces,
            grad_pieces=self.fused_fn_pieces,
        )

    def _fused_specs(self, hidden_streams: Array):
        """``(jax_mesh, stream_spec)`` when the streams must run under ``shard_map``, else ``(None, None)``."""
        jax_mesh, stream_spec = _stream_layout(self.mesh_source, hidden_streams.shape)
        if jax_mesh is None or getattr(jax_mesh, "size", 1) <= 1:
            return None, None
        if stream_spec is None:
            stream_spec = Ps()
        stream_spec = Ps(*(tuple(stream_spec) + (None,) * (4 - len(stream_spec))))
        return jax_mesh, stream_spec

    def _fused_forward(self, hidden_streams: Array) -> tuple[Array, Array, Array]:
        """``(post, comb, collapsed)`` through ejkernel ``mhc_collapse`` (Pallas), sharded like the streams.

        The RMS denominator is taken from the streams (local width times the hidden-axis size), matching the
        unfused flattened-stream RMSNorm.
        """
        hc = self.hc_mult
        fn3 = self.fn.value.reshape(-1, hc, hidden_streams.shape[-1])
        kwargs = dict(sinkhorn_iters=self.hc_sinkhorn_iters, eps=self.hc_eps, rms_eps=self.rms_norm_eps)
        jax_mesh, spec = self._fused_specs(hidden_streams)
        if jax_mesh is None:
            return mhc_collapse(hidden_streams, fn3, self.base.value, self.scale.value, cfg=self._fused_cfg(), **kwargs)
        return mhc_collapse(
            hidden_streams,
            fn3,
            self.base.value,
            self.scale.value,
            cfg=self._fused_cfg(),
            mesh=jax_mesh,
            in_specs=(spec, Ps(None, None, spec[3]), Ps(), Ps()),
            out_specs=(Ps(spec[0], spec[1], None), Ps(spec[0], spec[1], None, None), Ps(spec[0], spec[1], spec[3])),
            **kwargs,
        )

    def residual_write(
        self,
        hidden_streams: Float[Array, "batch seq hc hidden"],
        sublayer_output: Float[Array, "batch seq hidden"],
        post: Float[Array, "batch seq hc"],
        comb: Float[Array, "batch seq hc hc"],
    ) -> Float[Array, "batch seq hc hidden"]:
        """Write a sub-layer output onto the streams with this connection's kernel choice.

        With ``fused_kernel="none"`` this is exactly :func:`manifold_residual_write` at this module's
        ``precision``; with ``"pallas"`` it runs ejkernel ``mhc_expand`` (Pallas TPU), sharded like the streams.

        Args:
            hidden_streams: Original residual streams ``[B, S, hc, D]``.
            sublayer_output: Sub-layer result ``[B, S, D]``.
            post: Write gates ``[B, S, hc]``.
            comb: Stream mixer ``[B, S, hc, hc]``, indexed ``[source, target]``.

        Returns:
            Updated streams ``[B, S, hc, D]`` in the stream dtype.
        """
        if self.fused_kernel != "pallas":
            return manifold_residual_write(hidden_streams, sublayer_output, post, comb, precision=self.precision)
        jax_mesh, spec = self._fused_specs(hidden_streams)
        if jax_mesh is None:
            return mhc_expand(hidden_streams, sublayer_output, post, comb, cfg=self._fused_cfg())
        token, matrix = Ps(spec[0], spec[1], None), Ps(spec[0], spec[1], None, None)
        return mhc_expand(
            hidden_streams,
            sublayer_output,
            post,
            comb,
            cfg=self._fused_cfg(),
            mesh=jax_mesh,
            in_specs=(spec, Ps(spec[0], spec[1], spec[3]), token, matrix),
            out_specs=spec,
        )

    def coefficients(self, hidden_streams: Float[Array, "batch seq hc hidden"]) -> tuple[Array, Array, Array]:
        """Return ``(pre, post, comb)`` in fp32 without collapsing the streams.

        DeepSeek-V4.1 applies each sublayer's ``pre`` one sublayer later (attention collapses with the previous
        FFN's ``pre``), so it needs the read gates separately from the collapse :meth:`forward` performs.

        Args:
            hidden_streams: Explicit residual streams ``[B, S, hc, D]``.

        Returns:
            Read gates ``[B, S, hc]``, write gates ``[B, S, hc]`` and the doubly-stochastic mixer
            ``[B, S, hc, hc]``, all fp32.
        """
        hc = self.hc_mult
        eps = self.hc_eps
        batch, seq = hidden_streams.shape[:2]
        jax_mesh, stream_spec = _stream_layout(self.mesh_source, hidden_streams.shape)
        mix_logits = _project_streams(
            hidden_streams, self.fn.value, self.rms_norm_eps, self.precision, jax_mesh, stream_spec
        )
        # Named so selective remat can keep the tiny [B, S, (hc+2)*hc] logits
        # and coefficients instead of replaying the projection and Sinkhorn.
        mix_logits = checkpoint_name(mix_logits, "mhc_logits")
        base = self.base.value.astype(jnp.float32)
        scale = self.scale.value.astype(jnp.float32)
        if self.use_fused_coefficients:

            def coefficients(logits, b, s):
                return mhc_coefficients(logits, b, s, hc_mult=hc, n_iters=self.hc_sinkhorn_iters, eps=eps)

            if jax_mesh is not None and getattr(jax_mesh, "size", 1) > 1 and jax.default_backend() == "tpu":
                # Same token-local coefficient layout as the unfused Sinkhorn.
                token_spec = Ps(stream_spec[0], stream_spec[1], None) if stream_spec is not None else Ps()
                matrix_spec = Ps(stream_spec[0], stream_spec[1], None, None) if stream_spec is not None else Ps()
                pre, post, comb = jax.shard_map(
                    coefficients,
                    mesh=jax_mesh,
                    in_specs=(token_spec, Ps(), Ps()),
                    out_specs=(token_spec, token_spec, matrix_spec),
                    check_vma=True,
                )(mix_logits, base, scale)
            else:
                pre, post, comb = coefficients(mix_logits, base, scale)
        else:
            # Apply the three scales in one vector operation. This also avoids the
            # three scalar-pad cotangents whose combined bf16 conversion mislowers
            # for small token-sharded shapes on JAX 0.11.2 / TPU.
            scale_indices = jnp.asarray([0] * hc + [1] * hc + [2] * (hc * hc))
            gated_logits = mix_logits * scale[scale_indices] + base
            pre = jax.nn.sigmoid(gated_logits[..., :hc]) + eps
            post = 2.0 * jax.nn.sigmoid(gated_logits[..., hc : 2 * hc])
            comb_logits = gated_logits[..., 2 * hc :].reshape(batch, seq, hc, hc)
            comb = jax.nn.softmax(comb_logits, axis=-1) + eps

            if jax_mesh is not None and getattr(jax_mesh, "size", 1) > 1 and jax.default_backend() == "tpu":
                # Mosaic needs a manual partition boundary. Each token's full small
                # matrix stays local, but token partitions must not be all-gathered
                # across batch/sequence axes merely to run Sinkhorn.
                coeff_spec = Ps(stream_spec[0], stream_spec[1], None, None) if stream_spec is not None else Ps()
                comb = jax.lax.with_sharding_constraint(comb, NamedSharding(jax_mesh, coeff_spec))
                comb = jax.shard_map(
                    lambda c: sinkhorn_knopp(c, self.hc_sinkhorn_iters, eps),
                    mesh=jax_mesh,
                    in_specs=(coeff_spec,),
                    out_specs=coeff_spec,
                    check_vma=False,
                )(comb)
            else:
                comb = sinkhorn_knopp(comb, self.hc_sinkhorn_iters, eps)

        return checkpoint_name((pre, post, comb), "mhc_coefficients")


def manifold_collapse(
    pre: Float[Array, "batch seq hc"], hidden_streams: Float[Array, "batch seq hc hidden"]
) -> Float[Array, "batch seq hidden"]:
    """Read-weighted stream sum ``sum_h pre[..., h] * streams[..., h, :]`` in fp32, cast to the stream dtype.

    Args:
        pre: Read gates ``[B, S, hc]`` (fp32).
        hidden_streams: Residual streams ``[B, S, hc, D]``.

    Returns:
        Collapsed sublayer input ``[B, S, D]`` in ``hidden_streams.dtype``.
    """
    collapsed = jnp.sum(pre[..., None] * hidden_streams.astype(jnp.float32), axis=2)
    return collapsed.astype(hidden_streams.dtype)


class ManifoldHyperHead(spx.Module):
    """Learned final stream collapse with direct ``hc_fn/base/scale`` leaves."""

    def __init__(
        self,
        config: ManifoldHyperConnectionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the learned head with checkpoint-compatible parameters.

        Args:
            config: Scalar manifold settings; ``iters`` is unused by the head.
            dtype: Activation dtype metadata; gating math always uses fp32.
            param_dtype: Parameter storage dtype.
            precision: Precision of the learned projection matmul.
            rngs: Random streams consumed in ``hc_fn/base/scale`` order.
        """
        from easydel.infra.utils import ArrayParam

        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hc_mult = config.hc_mult
        self.hc_eps = config.eps
        self.rms_norm_eps = config.norm_eps
        self.hc_fn = ArrayParam.bound(
            shape=(self.hc_mult, self.hc_mult * config.hidden_size),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer},
            key=rngs.param,
        )
        self.hc_base = ArrayParam.bound(shape=(self.hc_mult,), dtype=param_dtype, init_method="zeros", key=rngs.param)
        self.hc_scale = ArrayParam.bound(shape=(1,), dtype=param_dtype, init_method="ones", key=rngs.param)

    def forward(self, x: Float[Array, "batch seq hc hidden"]) -> Float[Array, "batch seq hidden"]:
        """Collapse explicit streams using learned scalar read gates.

        Args:
            x: Residual streams ``[B, S, hc, D]``.

        Returns:
            Weighted stream sum ``[B, S, D]`` in the input dtype.
        """
        flat = _normalized_flat_streams(x, self.rms_norm_eps)
        mixes = jnp.matmul(flat, self.hc_fn.value.astype(jnp.float32).T, precision=self.precision)
        pre = jax.nn.sigmoid(mixes * self.hc_scale.value.astype(jnp.float32) + self.hc_base.value.astype(jnp.float32))
        pre = pre + self.hc_eps
        return jnp.sum(pre[..., None] * x.astype(jnp.float32), axis=2).astype(x.dtype)


def mean_hyper_head(hidden_streams: Float[Array, "batch seq hc hidden"]) -> Float[Array, "batch seq hidden"]:
    """Collapse streams by their unweighted mean, without learned parameters.

    Args:
        hidden_streams: Residual streams ``[B, S, hc, D]``.

    Returns:
        Mean stream ``[B, S, D]`` in the input dtype, using JAX mean semantics.
    """
    return jnp.mean(hidden_streams, axis=2)


class MeanHyperHead(spx.Module):
    """Parameter-free module form of :func:`mean_hyper_head`."""

    def forward(self, hidden_streams: Float[Array, "batch seq hc hidden"]) -> Float[Array, "batch seq hidden"]:
        """Return the unweighted stream mean, preserving the input dtype."""
        return mean_hyper_head(hidden_streams)


def manifold_residual_write(
    hidden_streams: Float[Array, "batch seq hc hidden"],
    sublayer_output: Float[Array, "batch seq hidden"],
    post: Float[Array, "batch seq hc"],
    comb: Float[Array, "batch seq hc hc"],
    *,
    precision: jax.lax.PrecisionLike = None,
) -> Float[Array, "batch seq hc hidden"]:
    """Write a sub-layer output onto transposed-mixer residual streams.

    Args:
        hidden_streams: Original residual streams ``[B, S, hc, D]``.
        sublayer_output: Sub-layer result ``[B, S, D]``.
        post: Scalar write gates ``[B, S, hc]``.
        comb: Stream mixer ``[B, S, hc, hc]``, indexed as ``[source, target]``.
        precision: Einsum precision for the transposed stream mix.

    Returns:
        ``post * output + comb.T @ streams``. Gates and mixer are cast to the
        residual dtype *before* multiplication, matching low-precision residual
        arithmetic rather than computing in fp32 and casting the result.
    """
    dtype = hidden_streams.dtype
    return post.astype(dtype)[..., None] * sublayer_output[..., None, :] + jnp.einsum(
        "bsji,bsjd->bsid", comb.astype(dtype), hidden_streams, precision=precision
    )
