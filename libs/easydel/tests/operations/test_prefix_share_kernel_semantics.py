"""Prefix-shared attention semantics of the registered dense attention operations.

A prefix-shared forward runs completion queries (length ``C``) against a key/value
sequence that is a left-padded prompt prefix (length ``P``) followed by the completion
itself (``kv_len = P + C``), so ``q_len != kv_len``. Visibility is expressed with
segment ids (``-1`` = padding) and causality with absolute positions
(``q_positions = P + arange(C)``, ``kv_positions = arange(P + C)``). Each case checks
the operation output and its q/k/v gradients against an independent dense-mask
reference in which a query sees exactly the keys with ``kv_pos <= q_pos`` whose
segment id equals its own.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.operations import OperationMetadata, OperationRegistry
from ejkernel.modules.operations.configs import FlashAttentionConfig
from ejkernel.types import MaskInfo
from jax import lax
from jax import numpy as jnp

HQ, HKV, D = 2, 1, 128
ATOL = 5e-3


def _inputs(batch: int, prefix: int, completion: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    kv_len = prefix + completion
    query = rng.standard_normal((batch, completion, HQ, D)).astype(np.float32)
    key = rng.standard_normal((batch, kv_len, HKV, D)).astype(np.float32)
    value = rng.standard_normal((batch, kv_len, HKV, D)).astype(np.float32)
    # Left-padded ragged prompts and right-padded ragged completions.
    prompt_valid = np.zeros((batch, prefix), dtype=bool)
    completion_valid = np.zeros((batch, completion), dtype=bool)
    for row in range(batch):
        prompt_len = max(1, prefix - (row * 3) % max(prefix // 2, 1))
        prompt_valid[row, prefix - prompt_len :] = True
        completion_len = max(1, completion - (row * 2) % max(completion // 2, 1))
        completion_valid[row, :completion_len] = True
    q_seg = np.where(completion_valid, 0, -1).astype(np.int32)
    kv_seg = np.concatenate([np.where(prompt_valid, 0, -1), q_seg], axis=1).astype(np.int32)
    q_pos = np.broadcast_to(prefix + np.arange(completion), (batch, completion)).astype(np.int32)
    kv_pos = np.broadcast_to(np.arange(kv_len), (batch, kv_len)).astype(np.int32)
    cot = rng.standard_normal((batch, completion, HQ, D)).astype(np.float32)
    return query, key, value, q_seg, kv_seg, q_pos, kv_pos, cot


def _dense_visibility(q_seg, kv_seg, q_pos, kv_pos):
    return (
        (q_seg[:, :, None] >= 0)
        & (kv_seg[:, None, :] >= 0)
        & (q_seg[:, :, None] == kv_seg[:, None, :])
        & (kv_pos[:, None, :] <= q_pos[:, :, None])
    )


def _reference(query, key, value, visible):
    key = jnp.repeat(key, HQ // HKV, axis=2)
    value = jnp.repeat(value, HQ // HKV, axis=2)
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, precision=jax.lax.Precision.HIGHEST) / np.sqrt(D)
    logits = jnp.where(visible[:, None], logits, -1e30)
    weights = jax.nn.softmax(logits, axis=-1)
    weights = jnp.where(visible[:, None], weights, 0.0)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, value, precision=jax.lax.Precision.HIGHEST)


# Test-local flash_attn2 pins. The operation forwards an explicit ``precision`` (default
# ``lax.Precision.DEFAULT``) to ejkernel, which overrides the ambient
# ``jax.default_matmul_precision("highest")`` used here. The XLA flash kernel then runs bf16
# matmul passes on TPU: max error ~6e-3 (out) / ~9e-3 (grads) at the small shape and ~2e-3 /
# ~3e-3 at the aligned shape, against ~4e-7 / ~7e-7 with HIGHEST (same masks, same inputs).
# The Pallas TPU kernel accepts only DEFAULT precision; its fp32 dots already reach ~5e-7, but
# it cannot run the small shape. Pinning the platform keeps the result independent of whichever
# backend the autotune cache happens to hold.
#   xla_highest  XLA flash kernel with an explicit HIGHEST precision (masking checked at full precision)
#   pallas       Pallas TPU kernel at its only supported precision (DEFAULT)
_FLASH_PINS = {
    None: (None, None),
    "xla_highest": (FlashAttentionConfig(platform="xla", backend="any"), lax.Precision.HIGHEST),
    "pallas": (FlashAttentionConfig(platform="pallas", backend="tpu"), None),
}


def _operation(name: str, pin: str | None = None):
    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, -1, 1, 1, 1))
    cfg, _ = _FLASH_PINS[pin]
    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=config,
        operation_configs=None if cfg is None else {name: cfg},
    )
    return OperationRegistry.create(name, metadata)


def _mask(representation: str, q_seg, kv_seg, q_pos, kv_pos) -> tuple[MaskInfo, bool]:
    if representation == "segments_positions":
        mask = MaskInfo.from_segments(
            q_segment_ids=jnp.asarray(q_seg),
            kv_segment_ids=jnp.asarray(kv_seg),
            q_positions=jnp.asarray(q_pos),
            kv_positions=jnp.asarray(kv_pos),
        )
        return mask, True
    visible = _dense_visibility(q_seg, kv_seg, q_pos, kv_pos)
    mask = MaskInfo(
        _attention_mask=jnp.asarray(visible[:, None]),
        q_positions=jnp.asarray(q_pos),
        kv_positions=jnp.asarray(kv_pos),
        causal_mask_baked_in=True,
    )
    return mask, True


def _check(name: str, representation: str, prefix: int, completion: int, pin: str | None = None):
    batch = jax.device_count()
    query, key, value, q_seg, kv_seg, q_pos, kv_pos, cot = _inputs(batch, prefix, completion)
    visible = _dense_visibility(q_seg, kv_seg, q_pos, kv_pos)
    q_valid = jnp.asarray(q_seg >= 0)[:, :, None, None].astype(jnp.float32)
    op = _operation(name, pin)
    _, precision = _FLASH_PINS[pin]
    call_kwargs = {} if precision is None else {"precision": precision}
    mask, causal = _mask(representation, q_seg, kv_seg, q_pos, kv_pos)

    def op_loss(q, k, v):
        out = op(q, k, v, mask_info=mask, causal=causal, **call_kwargs).attention_outputs
        return jnp.sum(out * cot * q_valid), out

    def ref_loss(q, k, v):
        out = _reference(q, k, v, jnp.asarray(visible))
        return jnp.sum(out * cot * q_valid), out

    with jax.default_matmul_precision("highest"):
        (_, out), grads = jax.value_and_grad(op_loss, argnums=(0, 1, 2), has_aux=True)(query, key, value)
        (_, ref), ref_grads = jax.value_and_grad(ref_loss, argnums=(0, 1, 2), has_aux=True)(query, key, value)
    rows = np.asarray(q_seg >= 0)
    errors = {"out": float(np.max(np.abs(np.asarray(out)[rows] - np.asarray(ref)[rows])))}
    for label, got, want in zip(("dq", "dk", "dv"), grads, ref_grads, strict=True):
        errors[label] = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    print(f"[prefix-share-probe] {name}[{pin}] {representation} P={prefix} C={completion} errors={errors}")
    assert all(np.isfinite(v) and v <= ATOL for v in errors.values()), errors


# Small shapes take every kernel's unaligned path (blocksparse falls back to vanilla);
# the 512-aligned shape drives the TPU block kernels themselves.
SMALL, ALIGNED = (12, 8), (512, 512)
_BLOCKSPARSE_DENSE = (
    "blocksparse with a stored dense mask keeps the pre-existing path: the kernel reduces the mask to segment ids "
    "and applies an index-aligned causal triangle, which is wrong for q_len != kv_len (dk ~156, dv ~93 on TPU)"
)
_FLASH_DENSE = (
    "flash_attn2 with a stored dense mask keeps the pre-existing path (only segments+positions without a dense "
    "mask derives causality from positions): the mask becomes segment ids and kv_pos <= q_pos is lost "
    "(output error ~4.1 on TPU)"
)
_RING_TOLERANCE = "ring with segments+positions: max error ~6e-3 (out) / ~9e-3 (grads) on TPU, beyond 5e-3"

# Observed on TPU (fp32, highest matmul precision), recorded per mechanism:
#   vanilla      supported for both representations and both shapes (errors <= 8e-7, incl. gradients)
#   sdpa         both representations supported (dense: causality is baked in; segments+positions: the
#                operation derives causality from positions)
#   flash_attn2  segments+positions masks correctly (folded into a bias): <= 1e-6 with the XLA kernel at
#                HIGHEST precision and with the Pallas kernel; see _FLASH_PINS for why the pins exist.
#                A stored dense mask keeps the old (wrong for q_len != kv_len) path; its aligned case
#                is not run because that old path previously did not finish on TPU.
#   blocksparse  small shape runs the vanilla fallback; at the aligned shape the operation shifts the TPU
#                kernel's causal mask by kv_len - q_len for segments+positions; a stored dense mask keeps
#                the old index-aligned (wrong) path
#   ring         segments+positions approximately right but outside tolerance; dense_mask did not finish
CASES = [
    pytest.param("vanilla", "segments_positions", *SMALL),
    pytest.param("vanilla", "segments_positions", *ALIGNED),
    pytest.param("vanilla", "dense_mask", *SMALL),
    pytest.param("vanilla", "dense_mask", *ALIGNED),
    pytest.param("sdpa", "dense_mask", *SMALL),
    pytest.param("sdpa", "dense_mask", *ALIGNED),
    pytest.param("sdpa", "segments_positions", *SMALL),
    pytest.param("sdpa", "segments_positions", *ALIGNED),
    pytest.param("flash_attn2", "segments_positions", *SMALL, "xla_highest"),
    pytest.param("flash_attn2", "segments_positions", *ALIGNED, "xla_highest"),
    pytest.param("flash_attn2", "dense_mask", *SMALL, marks=pytest.mark.xfail(strict=True, reason=_FLASH_DENSE)),
    pytest.param("blocksparse", "segments_positions", *SMALL),
    pytest.param("blocksparse", "segments_positions", *ALIGNED),
    pytest.param("blocksparse", "dense_mask", *ALIGNED, marks=pytest.mark.xfail(strict=True, reason=_BLOCKSPARSE_DENSE)),
    pytest.param("ring", "segments_positions", *SMALL, marks=pytest.mark.xfail(strict=True, reason=_RING_TOLERANCE)),
    pytest.param(
        "ring",
        "dense_mask",
        *SMALL,
        marks=pytest.mark.skip(reason="did not finish within 2 minutes on TPU; not characterised"),
    ),
    # Last so that a hang cannot hide the results above; finishes in well under 2 minutes on TPU now.
    pytest.param("flash_attn2", "segments_positions", *ALIGNED, "pallas"),
    pytest.param("flash_attn2", "segments_positions", *ALIGNED),
]
CASES = [c if len(c.values) == 5 else pytest.param(*c.values, None, marks=c.marks) for c in CASES]


@pytest.mark.parametrize("name,representation,prefix,completion,pin", CASES)
def test_prefix_shared_attention_matches_dense_reference(name, representation, prefix, completion, pin):
    _check(name, representation, prefix, completion, pin)
