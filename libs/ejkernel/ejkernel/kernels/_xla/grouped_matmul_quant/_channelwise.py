"""Explicit CHANNELWISE grouped matmul with frozen integer weight codes."""

import jax
from jax import lax
from jax import numpy as jnp

from ..._registry import Backend, Platform, kernel_registry
from ..grouped_matmul._xla_impl_fwd import grouped_matmul as _grouped_matmul_impl


def _mask_rows(value, sizes):
    valid = jnp.arange(value.shape[0]) < jnp.sum(sizes)
    return jnp.where(valid[:, None], value, jnp.zeros((), value.dtype))


def _dot(lhs, codes, sizes, dtype, tiling):
    # TPU ragged-dot does not guarantee initialized padding in primal or
    # transpose buffers. Mask both sides so neither values nor cotangents leak.
    # Integer operands are nondifferentiable; avoid sub-byte selects on INT4.
    if jnp.issubdtype(lhs.dtype, jnp.floating):
        lhs = _mask_rows(lhs, sizes)
    precision = lax.Precision.HIGHEST if lhs.dtype == jnp.float32 else lax.Precision.DEFAULT
    out = _grouped_matmul_impl(lhs, codes, sizes, dtype, tiling, None, None, False, False, precision)
    return _mask_rows(out, sizes)


@kernel_registry.register("grouped_matmul_channelwise", Platform.XLA, Backend.ANY)
def grouped_matmul_channelwise(
    lhs,
    codes,
    scales,
    group_sizes,
    *,
    activation_bits=16,
    preferred_element_type=jnp.bfloat16,
    tiling=None,
):
    """Multiply sorted activation rows by CHANNELWISE represented weights.

    ``codes[E,K,N]`` must be signed int4/int8; ``scales[E,1,N]`` are
    floating point. Nonnegative ``group_sizes[E]`` must sum to at most M.
    Unassigned trailing rows are padding with zero outputs and cotangents.
    Bits 16 leaves activations unquantized (float32 accumulation); bits 4/8
    use symmetric rowwise absmax/qmax quantization, round-to-even and clip.
    A4 requires int4 weights. A8 exactly upcasts int4 weights to int8 so
    both integer dot operands have the same dtype and accumulate in int32.
    A4 also uses exact INT8 arithmetic on measured v5p prefill families to
    avoid the slow native INT4 ragged lowering. The four-bit grid and stored
    weight codes are unchanged; the XLA cast may materialize INT8 temporaries.

    AD contract: codes and group membership are frozen. The activation JVP
    uses a straight-through surrogate multiplying its tangent by represented
    weights, NOT the natural derivative of round/absmax. Scale derivatives
    use the actual forward quantized activations, including at zero rows.
    ``preferred_element_type`` controls output dtype; tiling is an XLA hint.
    """
    if activation_bits not in (4, 8, 16):
        raise ValueError("activation_bits must be 4, 8, or 16")
    if codes.dtype not in (jnp.dtype(jnp.int4), jnp.dtype(jnp.int8)):
        raise TypeError("codes must have signed int4 or int8 dtype")
    if activation_bits == 4 and codes.dtype != jnp.dtype(jnp.int4):
        raise ValueError("activation_bits=4 requires signed int4 weights")
    if lhs.ndim != 2 or codes.ndim != 3 or lhs.shape[1] != codes.shape[1]:
        raise ValueError("expected lhs[M,K] and codes[E,K,N]")
    if scales.shape != (codes.shape[0], 1, codes.shape[2]):
        raise ValueError("expected scales[E,1,N]")
    if group_sizes.shape != (codes.shape[0],) or not jnp.issubdtype(group_sizes.dtype, jnp.integer):
        raise ValueError("expected integer group_sizes[E]")
    if not jnp.issubdtype(lhs.dtype, jnp.floating) or not jnp.issubdtype(scales.dtype, jnp.floating):
        raise TypeError("lhs and scales must be floating point")
    if not jnp.issubdtype(jnp.dtype(preferred_element_type), jnp.floating):
        raise TypeError("preferred_element_type must be floating point")

    from ._scale_rows import expand_group_scales as rows

    # v5p's large INT4 ragged lowering is much slower than exact INT8
    # arithmetic on these measured families. This changes arithmetic width,
    # never the four-bit activation grid or the persistent weight storage.
    widen_int4 = (
        activation_bits == 4
        and 1280 <= lhs.shape[0] <= 81920
        and codes.shape in ((128, 2560, 1280), (128, 640, 2560))
        and jax.default_backend() == "tpu"
    )
    if widen_int4:
        from jax.experimental.pallas import tpu as pltpu

        widen_int4 = pltpu.get_tpu_info().chip_version.value == "v5p"

    def base(x, weights, sizes):
        if activation_bits == 16:
            return _dot(x, weights.astype(x.dtype), sizes, jnp.float32, tiling)
        from ..quantized_matmul._integer_quantization import quantize_rows

        q, act_scale = quantize_rows(_mask_rows(x, sizes), activation_bits)
        if widen_int4:
            q = q.astype(jnp.int8)
        acc = _dot(q, weights.astype(q.dtype), sizes, jnp.int32, tiling)
        return acc.astype(jnp.float32) * act_scale

    # Every array must be an explicit operand, including frozen integer arrays.
    # Capturing traced codes/group sizes in the JVP closure breaks AD of an
    # already-compiled call with a DynamicJaxprTracer constant-handler error.
    @jax.custom_jvp
    def apply(x, s, weights, sizes):
        scaled = base(x, weights, sizes) * rows(s.astype(jnp.float32), sizes, x.shape[0])
        # Mask after scaling: an unused NaN scale must not turn padding into NaN.
        return _mask_rows(scaled, sizes).astype(preferred_element_type)

    @apply.defjvp
    def apply_jvp(primals, tangents):
        x, s, weights, sizes = primals
        dx, ds, _, _ = tangents
        b = base(x, weights, sizes)
        sr = rows(s.astype(jnp.float32), sizes, x.shape[0])
        dx_dot = _dot(dx.astype(jnp.float32), weights.astype(jnp.float32), sizes, jnp.float32, tiling)
        tangent = dx_dot * sr + b * rows(ds.astype(jnp.float32), sizes, x.shape[0])
        return (
            _mask_rows(b * sr, sizes).astype(preferred_element_type),
            _mask_rows(tangent, sizes).astype(preferred_element_type),
        )

    return apply(lhs, scales, codes, group_sizes)


__all__ = ("grouped_matmul_channelwise",)
