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

"""Numeric attention-sink parity for the flash / ring / block-sparse adapters.

Sink logits reach the kernels through ``shard_map`` where the head axis is local
(``Hq / tp``). Sink layouts must therefore be unambiguous after sharding:

* 1-D ``(Hq,)`` is per-query-head, any other 1-D length is shared;
* per-KV-head sinks are 2-D ``(Hkv, S)``; a 2-D leading singleton is shared.

Each case runs the adapter on a ``tp=4`` mesh and on a single device and checks
both against an independent dense f32 reference that applies the sinks with the
canonical semantics. Shared lengths 2 and 4 deliberately equal ``Hkv`` and a
local head count (``Hq / tp = 2``).

These are computation tests: they need an accelerator (4 chips is enough).

Run:
    env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
      uv run pytest libs/easydel/tests/operations/kernels/test_sink_sharding_numeric.py
"""

import jax
import numpy as np
import pytest
from easydel.infra import EasyDeLBaseConfig
from easydel.operations._operation_impl import OperationMetadata
from easydel.operations._operation_meta import normalize_head_sinks
from easydel.operations.kernels.blocksparse_attention import BlockSparseAttn
from easydel.operations.kernels.flash_attention import FlashAttn
from easydel.operations.kernels.ring_attention import RingAttn
from jax import numpy as jnp
from jax import random as jr

HQ, HKV, SEQ, HEAD_DIM = 8, 2, 256, 64
TP = 4


def _sink_values(shape):
    """Sink logits that differ per row so a mis-mapped head changes the output."""
    n = int(np.prod(shape))
    return jnp.linspace(-2.0, 3.0, n, dtype=jnp.float32).reshape(shape)


def _per_head_sinks(sinks):
    """Canonical per-query-head ``(HQ, S)`` view of ``sinks`` (independent of production code)."""
    if sinks.ndim == 1:
        if sinks.shape[0] == HQ:
            return sinks[:, None]
        return jnp.broadcast_to(sinks[None, :], (HQ, sinks.shape[0]))
    if sinks.shape[0] == 1:
        return jnp.broadcast_to(sinks, (HQ, sinks.shape[1]))
    if sinks.shape[0] == HKV:
        return jnp.repeat(sinks, HQ // HKV, axis=0)
    assert sinks.shape[0] == HQ
    return sinks


def _reference(q, k, v, sinks, causal=True):
    """Dense f32 attention with sink logits appended to the softmax denominator."""
    qf = q.astype(jnp.float32).transpose(0, 2, 1, 3)
    kf = jnp.repeat(k.astype(jnp.float32).transpose(0, 2, 1, 3), HQ // HKV, axis=1)
    vf = jnp.repeat(v.astype(jnp.float32).transpose(0, 2, 1, 3), HQ // HKV, axis=1)
    logits = jnp.einsum("bhtd,bhkd->bhtk", qf * HEAD_DIM**-0.5, kf, precision=jax.lax.Precision.HIGHEST)
    if causal:
        logits = jnp.where(jnp.tril(jnp.ones((SEQ, SEQ), dtype=bool)), logits, -1e30)
    aux = _per_head_sinks(sinks)
    aux = jnp.broadcast_to(aux[None, :, None, :], (q.shape[0], HQ, SEQ, aux.shape[-1]))
    weights = jax.nn.softmax(jnp.concatenate([logits, aux], axis=-1), axis=-1)[..., :SEQ]
    return jnp.einsum("bhtk,bhkd->bhtd", weights, vf, precision=jax.lax.Precision.HIGHEST).transpose(0, 2, 1, 3)


def _run(adapter_cls, axis_dims, q, k, v, sinks):
    config = EasyDeLBaseConfig(sharding_axis_dims=axis_dims)
    metadata = OperationMetadata(runtime_dtype=jnp.bfloat16, base_config=config)
    op = adapter_cls(metadata)
    with metadata.mesh:
        out = op(query=q, key=k, value=v, softmax_aux=sinks, causal=True).attention_outputs
    return np.asarray(out, dtype=np.float32)


_ALL_LAYOUTS = {
    "per_head_1d": (HQ,),
    "shared_1d_len2": (2,),  # == Hkv and == local Hq under tp=4
    "shared_1d_len4": (4,),
    "shared_2d_singleton": (1, 3),
    # Shared row whose width equals Hq: must stay shared, never be read as one sink per head.
    "shared_2d_singleton_width_hq": (1, HQ),
    "per_head_2d": (HQ, 3),
    "per_kv_2d": (HKV, 3),
}


def _cases():
    # The Pallas TPU block-sparse kernel reduces every layout to per-head (Hq,) sinks
    # (logsumexp over the sink axis) before the splash kernel, so it is tested with all of them.
    for name, shape in _ALL_LAYOUTS.items():
        yield pytest.param(FlashAttn, shape, id=f"flash-{name}")
        yield pytest.param(RingAttn, shape, id=f"ring-{name}")
        yield pytest.param(BlockSparseAttn, shape, id=f"blocksparse-{name}")


@pytest.mark.skipif(len(jax.devices()) < TP, reason=f"requires at least {TP} devices")
@pytest.mark.parametrize(("adapter_cls", "sink_shape"), list(_cases()))
def test_sink_layouts_match_single_device_and_dense_reference(adapter_cls, sink_shape):
    kq, kk, kv = jr.split(jr.PRNGKey(0), 3)
    q = jr.normal(kq, (1, SEQ, HQ, HEAD_DIM), dtype=jnp.bfloat16)
    k = jr.normal(kk, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.bfloat16)
    v = jr.normal(kv, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.bfloat16)
    sinks = _sink_values(sink_shape)

    ref = np.asarray(_reference(q, k, v, sinks), dtype=np.float32)
    single = _run(adapter_cls, (1, 1, 1, 1, 1, 1), q, k, v, sinks)
    sharded = _run(adapter_cls, (1, 1, 1, 1, TP, 1), q, k, v, sinks)

    # bf16 kernels vs f32 reference: healthy error is ~1e-2; a mis-mapped sink is O(0.1-1).
    assert np.abs(single - ref).max() < 0.1, "single-device output disagrees with the dense sink reference"
    assert np.abs(sharded - single).max() < 0.1, "tp=4 output disagrees with the single-device run"


def test_single_head_blocksparse_prefill_stays_causal():
    """One query head must not be mistaken for one query token (r10 S07-2: causality was dropped)."""
    kq, kk, kv = jr.split(jr.PRNGKey(3), 3)
    q = jr.normal(kq, (1, SEQ, 1, HEAD_DIM), dtype=jnp.float32)
    k = jr.normal(kk, (1, SEQ, 1, HEAD_DIM), dtype=jnp.float32)
    v = jr.normal(kv, (1, SEQ, 1, HEAD_DIM), dtype=jnp.float32)

    def dense(causal):
        logits = jnp.einsum("btd,bkd->btk", q[:, :, 0] * HEAD_DIM**-0.5, k[:, :, 0], precision="highest")
        if causal:
            logits = jnp.where(jnp.tril(jnp.ones((SEQ, SEQ), dtype=bool)), logits, -1e30)
        weights = jax.nn.softmax(logits, axis=-1)
        return np.asarray(jnp.einsum("btk,bkd->btd", weights, v[:, :, 0], precision="highest"))[:, :, None, :]

    out = _run(BlockSparseAttn, (1, 1, 1, 1, 1, 1), q, k, v, None)
    causal_ref, full_ref = dense(True), dense(False)
    assert np.abs(causal_ref - full_ref).max() > 0.1  # negative control: causality matters here
    assert np.abs(out - causal_ref).max() < 0.1, "single-head block-sparse prefill is not causal"


@pytest.mark.parametrize("sink_shape", [(HQ,), (HKV, 3)], ids=["per_head_1d", "per_kv_2d"])
def test_decode_attention_applies_sinks(sink_shape):
    """AutoRegressiveDecodeAttn.__call__ used to drop ``softmax_aux``, so decode ignored sinks (r10 S07-4)."""
    from types import SimpleNamespace

    from easydel.operations.kernels.decode_attention import AutoRegressiveDecodeAttn

    kq, kk, kv = jr.split(jr.PRNGKey(4), 3)
    q = jr.normal(kq, (1, 1, HQ, HEAD_DIM), dtype=jnp.float32)
    k = jr.normal(kk, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.float32)
    v = jr.normal(kv, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.float32)
    # 256 keys give a logsumexp of ~6, so sinks in [5, 9] take ~25-95% of the mass: an ignored
    # sink moves the output far past the parity tolerance.
    sinks = jnp.linspace(5.0, 9.0, int(np.prod(sink_shape)), dtype=jnp.float32).reshape(sink_shape)

    def dense(aux):
        kf = jnp.repeat(k, HQ // HKV, axis=2)
        vf = jnp.repeat(v, HQ // HKV, axis=2)
        logits = jnp.einsum("bthd,bkhd->bhtk", q * HEAD_DIM**-0.5, kf, precision="highest")
        if aux is not None:
            rows = _per_head_sinks(aux)
            logits = jnp.concatenate([logits, jnp.broadcast_to(rows[None, :, None, :], (1, HQ, 1, rows.shape[1]))], -1)
        weights = jax.nn.softmax(logits, axis=-1)[..., :SEQ]
        return np.asarray(jnp.einsum("bhtk,bkhd->bthd", weights, vf, precision="highest"))

    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    metadata = OperationMetadata(runtime_dtype=jnp.float32, base_config=config)
    op = AutoRegressiveDecodeAttn(metadata)
    cache_metadata = SimpleNamespace(starts=jnp.zeros((1,), jnp.int32), indexes=jnp.full((1,), SEQ, jnp.int32))
    with metadata.mesh:
        out = op(query=q, key=k, value=v, cache_metadata=cache_metadata, softmax_aux=sinks).attention_outputs
    out = np.asarray(out, np.float32)
    ref, no_sink = dense(sinks), dense(None)
    assert np.abs(ref - no_sink).max() > 0.1  # negative control: these sinks matter
    assert np.abs(out - ref).max() < 2e-2, np.abs(out - ref).max()


def test_decode_attention_respects_partial_cache_bounds():
    """Decode must attend only ``[starts, indexes)``, never unwritten/stale slots (r11 S07-1)."""
    from types import SimpleNamespace

    from easydel.operations.kernels.decode_attention import AutoRegressiveDecodeAttn

    valid = 16
    kq, kk, kv = jr.split(jr.PRNGKey(5), 3)
    q = jr.normal(kq, (1, 1, HQ, HEAD_DIM), dtype=jnp.float32)
    k = jr.normal(kk, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.float32)
    v = jr.normal(kv, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.float32)
    # Stale slots past ``indexes`` hold large values: attending them moves the output by O(100).
    v = v.at[:, valid:].set(100.0)
    sinks = jnp.linspace(0.0, 1.0, HQ, dtype=jnp.float32)

    kf = jnp.repeat(k[:, :valid], HQ // HKV, axis=2)
    vf = jnp.repeat(v[:, :valid], HQ // HKV, axis=2)
    logits = jnp.einsum("bthd,bkhd->bhtk", q * HEAD_DIM**-0.5, kf, precision="highest")
    logits = jnp.concatenate([logits, jnp.broadcast_to(sinks[None, :, None, None], (1, HQ, 1, 1))], -1)
    weights = jax.nn.softmax(logits, axis=-1)[..., :valid]
    ref = np.asarray(jnp.einsum("bhtk,bkhd->bthd", weights, vf, precision="highest"))

    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    metadata = OperationMetadata(runtime_dtype=jnp.float32, base_config=config)
    op = AutoRegressiveDecodeAttn(metadata)
    cache_metadata = SimpleNamespace(starts=jnp.zeros((1,), jnp.int32), indexes=jnp.full((1,), valid, jnp.int32))
    with metadata.mesh:
        out = op(query=q, key=k, value=v, cache_metadata=cache_metadata, softmax_aux=sinks).attention_outputs
    out = np.asarray(out, np.float32)
    assert np.abs(out - ref).max() < 2e-2, np.abs(out - ref).max()


def _decode_window_reference(q, k, v, sinks, lo, hi):
    """Dense decode over keys ``[lo, hi)`` with one zero-value sink column per query head."""
    kf = jnp.repeat(k[:, lo:hi], HQ // HKV, axis=2)
    vf = jnp.repeat(v[:, lo:hi], HQ // HKV, axis=2)
    logits = jnp.einsum("bthd,bkhd->bhtk", q * HEAD_DIM**-0.5, kf, precision="highest")
    logits = jnp.concatenate([logits, jnp.broadcast_to(sinks[None, :, None, None], (1, HQ, 1, 1))], -1)
    weights = jax.nn.softmax(logits, axis=-1)[..., : hi - lo]
    return np.asarray(jnp.einsum("bhtk,bkhd->bthd", weights, vf, precision="highest"))


def _decode_inputs(seed):
    kq, kk, kv = jr.split(jr.PRNGKey(seed), 3)
    q = jr.normal(kq, (1, 1, HQ, HEAD_DIM), dtype=jnp.float32)
    k = jr.normal(kk, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.float32)
    v = jr.normal(kv, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.float32)
    return q, k, v, jnp.linspace(0.0, 1.0, HQ, dtype=jnp.float32)


def _run_decode(q, k, v, sinks, indexes, **kwargs):
    from types import SimpleNamespace

    from easydel.operations.kernels.decode_attention import AutoRegressiveDecodeAttn

    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    metadata = OperationMetadata(runtime_dtype=jnp.float32, base_config=config)
    op = AutoRegressiveDecodeAttn(metadata)
    cache_metadata = SimpleNamespace(starts=jnp.zeros((1,), jnp.int32), indexes=jnp.full((1,), indexes, jnp.int32))
    with metadata.mesh:
        out = op(query=q, key=k, value=v, cache_metadata=cache_metadata, softmax_aux=sinks, **kwargs)
    return np.asarray(out.attention_outputs, np.float32)


def test_decode_attention_bounds_survive_lazy_segment_mask():
    """A segment-id MaskInfo (no materialized mask) must not drop the cache bounds (r12 S07-1)."""
    from ejkernel.types import MaskInfo

    valid = 16
    q, k, v, sinks = _decode_inputs(6)
    v = v.at[:, valid:].set(100.0)  # stale slots: any leak moves the output by O(100)
    mask_info = MaskInfo.from_segments(
        q_segment_ids=jnp.zeros((1, 1), jnp.int32), kv_segment_ids=jnp.zeros((1, SEQ), jnp.int32)
    )
    out = _run_decode(q, k, v, sinks, valid, mask_info=mask_info)
    ref = _decode_window_reference(q, k, v, sinks, 0, valid)
    assert np.abs(out - ref).max() < 2e-2, np.abs(out - ref).max()


@pytest.mark.parametrize("window", [(3, 3), (3, 0), 4])
def test_decode_attention_sliding_window_at_cache_end_stays_exact(window):
    """A window slice clipped at the cache end must not expose older keys (r12 S07-2)."""
    q, k, v, sinks = _decode_inputs(7)
    left = window - 1 if isinstance(window, int) else window[0]
    q_pos = SEQ - 1
    # Slots just left of the window hold large values; the clipped slice includes them.
    v = v.at[:, q_pos - left - 3 : q_pos - left].set(100.0)
    out = _run_decode(q, k, v, sinks, SEQ, sliding_window=window)
    ref = _decode_window_reference(q, k, v, sinks, q_pos - left, SEQ)
    assert np.abs(out - ref).max() < 2e-2, (window, np.abs(out - ref).max())


@pytest.mark.parametrize(
    "sink_shape",
    [(HQ,), (2,), (1, 3), (HKV, 3), (HQ, 3)],
    ids=["per_head_1d", "shared_1d_len_hkv", "shared_2d_singleton", "per_kv_2d", "per_head_2d"],
)
def test_segmented_vanilla_attention_sink_layouts(sink_shape):
    """Packed (segment-id) vanilla attention must follow the canonical sink layout (r15 S07-1).

    A 1-D sink of length Hq is per query head; it used to be read as a shared sink set there.
    """
    from easydel.operations.kernels.vanilla_attention import VanillaAttn
    from ejkernel.types import MaskInfo

    kq, kk, kv = jr.split(jr.PRNGKey(8), 3)
    q = jr.normal(kq, (1, SEQ, HQ, HEAD_DIM), dtype=jnp.float32)
    k = jr.normal(kk, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.float32)
    v = jr.normal(kv, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.float32)
    sinks = _sink_values(sink_shape)
    # One packed document covering the whole row, so the dense causal reference applies unchanged.
    seg = jnp.zeros((1, SEQ), jnp.int32)
    mask_info = MaskInfo.from_segments(q_segment_ids=seg, kv_segment_ids=seg)

    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    metadata = OperationMetadata(runtime_dtype=jnp.float32, runtime_softmax_dtype=jnp.float32, base_config=config)
    op = VanillaAttn(metadata)
    assert op._can_use_segmented_attention(mask_info)  # this test must exercise the segmented path
    with metadata.mesh:
        out = op(query=q, key=k, value=v, mask_info=mask_info, softmax_aux=sinks, causal=True).attention_outputs
    out = np.asarray(out, np.float32)
    ref = np.asarray(_reference(q, k, v, sinks), np.float32)
    assert np.abs(out - ref).max() < 2e-2, np.abs(out - ref).max()


def test_reference_negative_control_per_head_vs_shared():
    """The dense reference must separate per-head sinks from a shared sink (guards against vacuous tests)."""
    kq, kk, kv = jr.split(jr.PRNGKey(1), 3)
    q = jr.normal(kq, (1, SEQ, HQ, HEAD_DIM), dtype=jnp.float32)
    k = jr.normal(kk, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.float32)
    v = jr.normal(kv, (1, SEQ, HKV, HEAD_DIM), dtype=jnp.float32)
    per_head = _sink_values((HQ,))
    shared = jnp.broadcast_to(per_head[:1], (HQ,))
    gap = np.abs(np.asarray(_reference(q, k, v, per_head)) - np.asarray(_reference(q, k, v, shared))).max()
    assert gap > 0.05, f"per-head vs shared sinks are not distinguishable by the reference (gap={gap})"


@pytest.mark.parametrize(
    ("shape", "expected_shape", "per_head"),
    [
        ((HQ,), (HQ,), True),  # per-head vector stays 1-D when per_head_2d=False
        ((3,), (3,), False),  # shared, no collision with any head count
        ((HKV,), (HQ, HKV), True),  # shared but equals Hkv: made explicit per-head
        ((1, 3), (3,), False),  # 2-D singleton is shared
        ((1, HQ), (HQ, HQ), True),  # shared row of width Hq: expanded, not flattened to per-head (HQ,)
        ((HKV, 3), (HQ, 3), True),  # per-KV rows expand to query heads
        ((HQ, 3), (HQ, 3), True),
    ],
)
def test_normalize_head_sinks_layouts(shape, expected_shape, per_head):
    """Layout normalization; the broadcast/repeat branches run jnp ops, so this needs an accelerator."""
    sinks = np.linspace(-2.0, 3.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    out, is_per_head = normalize_head_sinks(
        sinks, num_q_heads=HQ, num_kv_heads=HKV, per_head_2d=False, head_shard_factor=TP
    )
    assert tuple(out.shape) == expected_shape
    assert is_per_head is per_head


@pytest.mark.cpu_ok
def test_normalize_head_sinks_rejects_bad_leading_dim():
    with pytest.raises(ValueError, match="first dim"):
        normalize_head_sinks(np.zeros((3, 2), np.float32), num_q_heads=HQ, num_kv_heads=HKV, per_head_2d=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
