# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Pallas TPU blocksparse attention for queries continuing a key/value prefix (``q_len != kv_len``).

The kv sequence is a left-padded prefix (length ``P``) followed by the queries' own tokens (length
``C``); queries sit at absolute positions ``P + i``. With segment ids plus explicit positions (no
stored dense mask) the causal triangle must be aligned to the positions, not top-left by index.
Outputs and q/k/v gradients are compared with an independent dense fp32 reference at highest
matmul precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.blocksparse_attention import blocksparse_attention as pallas_kernel
from ejkernel.modules.operations import blocksparse_attention
from ejkernel.modules.operations.configs import BlockSparseAttentionConfig
from ejkernel.ops import BwdParams, FwdParams
from ejkernel.types import MaskInfo


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU blocksparse tests require a TPU backend")

B, HQ, HKV, D = 2, 4, 2, 128
TOL = 1e-4
_FWD = FwdParams(q_blocksize=128, kv_blocksize=128, num_warps=None, num_stages=None)
_BWD = BwdParams(q_blocksize=128, kv_blocksize=128, num_warps=None, num_stages=None)
CFG = {
    "pallas": BlockSparseAttentionConfig(fwd_params=_FWD, bwd_params=_BWD, platform="pallas", backend="tpu"),
    "xla": BlockSparseAttentionConfig(fwd_params=_FWD, bwd_params=_BWD, platform="xla", backend="any"),
}


def _inputs(prefix: int, completion: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    kv_len = prefix + completion
    q = rng.standard_normal((B, HQ, completion, D)).astype(np.float32)
    k = rng.standard_normal((B, HKV, kv_len, D)).astype(np.float32)
    v = rng.standard_normal((B, HKV, kv_len, D)).astype(np.float32)
    prompt_valid = np.zeros((B, prefix), dtype=bool)
    completion_valid = np.zeros((B, completion), dtype=bool)
    for row in range(B):
        prompt_len = max(1, prefix - (row * 37 + 5) % max(prefix // 2, 1))
        prompt_valid[row, prefix - prompt_len :] = True  # left padded
        completion_len = max(1, completion - (row * 23 + 3) % max(completion // 2, 1))
        completion_valid[row, :completion_len] = True  # right padded
    q_seg = np.where(completion_valid, 0, -1).astype(np.int32)
    kv_seg = np.concatenate([np.where(prompt_valid, 0, -1), q_seg], axis=1).astype(np.int32)
    q_pos = np.broadcast_to(prefix + np.arange(completion), (B, completion)).astype(np.int32)
    kv_pos = np.broadcast_to(np.arange(kv_len), (B, kv_len)).astype(np.int32)
    cot = rng.standard_normal((B, HQ, completion, D)).astype(np.float32)
    return q, k, v, q_seg, kv_seg, q_pos, kv_pos, cot


def _visibility(q_seg, kv_seg, q_pos, kv_pos, window):
    vis = (
        (q_seg[:, :, None] >= 0)
        & (kv_seg[:, None, :] >= 0)
        & (q_seg[:, :, None] == kv_seg[:, None, :])
        & (kv_pos[:, None, :] <= q_pos[:, :, None])
    )
    if window is not None:
        vis &= kv_pos[:, None, :] >= q_pos[:, :, None] - (window - 1)
    return vis


def _reference(q, k, v, visible):
    k = jnp.repeat(k, HQ // HKV, axis=1)
    v = jnp.repeat(v, HQ // HKV, axis=1)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k, precision=jax.lax.Precision.HIGHEST) * (D**-0.5)
    logits = jnp.where(visible[:, None], logits, -1e30)
    weights = jnp.where(visible[:, None], jax.nn.softmax(logits, axis=-1), 0.0)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v, precision=jax.lax.Precision.HIGHEST)


def _errors(fn, visible, data):
    q, k, v, q_seg, *_, cot = data
    q_valid = jnp.asarray(q_seg >= 0)[:, None, :, None].astype(jnp.float32)

    def op_loss(a, b, c):
        out = fn(a, b, c)
        return jnp.sum(out * cot * q_valid), out

    def ref_loss(a, b, c):
        out = _reference(a, b, c, visible)
        return jnp.sum(out * cot * q_valid), out

    with jax.default_matmul_precision("highest"):
        (_, out), grads = jax.value_and_grad(op_loss, argnums=(0, 1, 2), has_aux=True)(q, k, v)
        (_, ref), ref_grads = jax.value_and_grad(ref_loss, argnums=(0, 1, 2), has_aux=True)(q, k, v)
    rows = np.repeat(np.asarray(q_seg >= 0)[:, None, :], HQ, axis=1)
    errors = {"out": float(np.max(np.abs(np.asarray(out)[rows] - np.asarray(ref)[rows])))}
    for label, got, want in zip(("dq", "dk", "dv"), grads, ref_grads, strict=True):
        errors[label] = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    return errors


def _mask_info(q_seg, kv_seg, q_pos, kv_pos):
    return MaskInfo.from_segments(
        q_segment_ids=jnp.asarray(q_seg),
        kv_segment_ids=jnp.asarray(kv_seg),
        q_positions=jnp.asarray(q_pos),
        kv_positions=jnp.asarray(kv_pos),
    )


# (512, 512) is q_len=512 / kv_len=1024, which was wrong (out ~3, dq ~50) before the fix.
@pytest.mark.parametrize(
    "platform,prefix,completion,window",
    [
        ("pallas", 512, 512, None),
        ("pallas", 768, 256, None),
        ("pallas", 256, 128, None),
        ("pallas", 512, 512, 200),
        ("xla", 512, 512, None),
    ],
)
def test_prefix_positions_match_dense_reference(platform, prefix, completion, window):
    data = _inputs(prefix, completion)
    _, _, _, q_seg, kv_seg, q_pos, kv_pos, _ = data
    visible = jnp.asarray(_visibility(q_seg, kv_seg, q_pos, kv_pos, window))

    def fn(a, b, c):
        mask = _mask_info(q_seg, kv_seg, q_pos, kv_pos)
        return blocksparse_attention(a, b, c, mask_info=mask, causal=True, sliding_window=window, cfg=CFG[platform])

    errors = _errors(fn, visible, data)
    assert all(np.isfinite(e) and e <= TOL for e in errors.values()), errors


def test_positions_not_following_the_offset_poison_the_row():
    """Positions the bottom-right shift cannot express yield NaN for that batch row, never wrong numbers."""
    q, k, v, q_seg, kv_seg, q_pos, kv_pos, _ = _inputs(256, 128)
    q_pos = q_pos.copy()
    q_pos[1] += 7  # row 1 skips positions: kv_pos <= q_pos is no longer an index offset of kv_len - q_len
    out = blocksparse_attention(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        mask_info=_mask_info(q_seg, kv_seg, q_pos, kv_pos),
        causal=True,
        cfg=CFG["pallas"],
    )
    out = np.asarray(out)
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


@pytest.mark.parametrize("causal", [False, True])
def test_unequal_lengths_without_positions_are_unchanged(causal):
    """``q_len != kv_len`` with segment ids only keeps the kernel's default (index based) path."""
    q, k, v, q_seg, kv_seg, *_ = _inputs(256, 128, seed=5)
    q, k, v = (jnp.asarray(x) for x in (q, k, v))
    mask = MaskInfo.from_segments(jnp.asarray(q_seg), jnp.asarray(kv_seg))
    out = blocksparse_attention(q, k, v, mask_info=mask, causal=causal, cfg=CFG["pallas"])
    legacy = pallas_kernel(
        q, k, v, jnp.asarray(q_seg), jnp.asarray(kv_seg), causal=causal, fwd_params=_FWD, bwd_params=_BWD
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(legacy))


@pytest.mark.parametrize("causal", [False, True])
def test_cache_style_positions_with_dense_mask_are_unchanged(causal):
    """Positions plus a stored dense mask (what a KV cache attaches) keep the pre-existing path."""
    q, k, v, q_seg, kv_seg, q_pos, kv_pos, _ = _inputs(256, 128, seed=9)
    q, k, v = (jnp.asarray(x) for x in (q, k, v))
    dense = jnp.asarray(_visibility(q_seg, kv_seg, q_pos, kv_pos, None)[:, None])
    with_pos = MaskInfo(_attention_mask=dense, q_positions=jnp.asarray(q_pos), kv_positions=jnp.asarray(kv_pos))
    plain = MaskInfo(_attention_mask=dense)
    assert not with_pos.uses_position_causality()
    a = blocksparse_attention(q, k, v, mask_info=plain, causal=causal, cfg=CFG["pallas"])
    b = blocksparse_attention(q, k, v, mask_info=with_pos, causal=causal, cfg=CFG["pallas"])
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("causal", [False, True])
def test_equal_lengths_with_positions_are_unchanged(causal):
    """With ``q_len == kv_len`` explicit positions do not change the Pallas result."""
    q, k, v, *_ = _inputs(0, 256, seed=3)
    q, k, v = (jnp.asarray(x) for x in (q, k, v))
    seg = np.array([[0] * 128 + [1] * 120 + [-1] * 8] * B, dtype=np.int32)
    positions = jnp.broadcast_to(jnp.arange(256, dtype=jnp.int32), (B, 256))
    plain = MaskInfo.from_segments(jnp.asarray(seg))
    with_pos = MaskInfo.from_segments(jnp.asarray(seg), q_positions=positions, kv_positions=positions)
    a = blocksparse_attention(q, k, v, mask_info=plain, causal=causal, cfg=CFG["pallas"])
    b = blocksparse_attention(q, k, v, mask_info=with_pos, causal=causal, cfg=CFG["pallas"])
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
