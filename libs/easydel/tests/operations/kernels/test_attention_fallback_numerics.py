"""Public numerical regressions for attention fallback paths (accelerator only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.operations import OperationMetadata
from easydel.operations.kernels import paged_flash_attention as paged
from easydel.operations.kernels.decode_attention import AutoRegressiveDecodeAttn
from easydel.operations.kernels.flash_attention import FlashAttn
from easydel.operations.kernels.glm_moe_dsa_indexer import GlmMoeDsaIndexerOp
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as Ps


def metadata():
    return OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        _stored_mesh=Mesh(np.asarray(jax.devices()[:1]), ("tp",)),
    )


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("additive", [False, True])
def test_indexer_broadcast_key_mask_applies_to_every_query(rank, additive):
    mask = np.array([[False, True, True]])
    if additive:
        mask = np.where(mask, 0.0, -np.inf).astype(np.float32)
    if rank == 3:
        mask = mask[:, None, :]
    output = GlmMoeDsaIndexerOp(SimpleNamespace()).forward_native(
        jnp.ones((1, 3, 1, 1)),
        jnp.array([[[100.0], [1.0], [2.0]]]),
        jnp.ones((1, 3, 1)),
        jnp.arange(3)[None],
        qk_rope_head_dim=0,
        index_topk=1,
        softmax_scale=1.0,
        attention_mask=jnp.asarray(mask),
        precision=jax.lax.Precision.HIGHEST,
    )
    # Row zero has no valid causal key; later rows must not select masked key zero.
    np.testing.assert_array_equal(output.topk_indices[0, 1:, 0], [1, 2])
    np.testing.assert_array_equal(output.topk_mask[0, 1:], [[False, True, False], [False, False, True]])


@pytest.mark.parametrize("path", ["unequal_width", "multihost"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_flash_fallback_preserves_numerator_dropout_and_precision(monkeypatch, path, normalize, dropout):
    meta = metadata()
    if path == "multihost":
        monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
        monkeypatch.setattr(jax, "process_count", lambda: 2)
    value_dim = 1 if path == "unequal_width" else 2
    q = jnp.zeros((1, 3, 2, 2), jnp.float32)
    k = jnp.zeros((1, 4, 1, 2), jnp.float32)
    v = jnp.ones((1, 4, 1, value_dim), jnp.float32)
    output = FlashAttn(meta).forward_native(
        q,
        k,
        v,
        normalize_output=normalize,
        dropout_prob=dropout,
        dropout_seed=17,
        precision=jax.lax.Precision.HIGHEST,
    )
    weights = np.ones((1, 2, 3, 4), np.float32) / (4 if normalize else 1)
    if dropout:
        keep = np.asarray(jax.random.bernoulli(jax.random.PRNGKey(17), 1.0 - dropout, weights.shape))
        weights = weights * keep / (1.0 - dropout)
    expected = np.repeat(weights.sum(-1).transpose(0, 2, 1)[..., None], value_dim, axis=-1)
    np.testing.assert_array_equal(output.attention_outputs, expected)


@pytest.mark.parametrize("feature", ["mask", "bias", "both", "init_bias"])
@pytest.mark.parametrize("window", [None, 2])
def test_gpu_decode_preserves_visibility_and_bias(monkeypatch, feature, window):
    meta = metadata()
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    mask = np.array([[[[True, False, True, True]]]]) if feature in ("mask", "both") else None
    bias = np.array([[[[0.0, np.log(2.0), np.log(3.0), 0.0]]]], np.float32) if feature != "mask" else None
    values = np.array([2.0, 100.0, 8.0, 999.0], np.float32)
    output = AutoRegressiveDecodeAttn(meta).forward_native(
        jnp.zeros((1, 1, 1, 2)),
        jnp.zeros((1, 4, 1, 2)),
        jnp.asarray(values).reshape(1, 4, 1, 1),
        SimpleNamespace(starts=jnp.array([0]), indexes=jnp.array([3])),
        mask_info=None if mask is None else MaskInfo(_attention_mask=jnp.asarray(mask)),
        bias=None if bias is None or feature == "init_bias" else jnp.asarray(bias),
        init_bias=(lambda: jnp.asarray(bias)) if feature == "init_bias" else None,
        sliding_window=window,
        precision=jax.lax.Precision.HIGHEST,
    )
    valid = np.arange(4) < 3
    if window:
        valid &= np.arange(4) >= 2 - (window - 1)
    if mask is not None:
        valid &= mask.reshape(-1)
    weights = np.ones(4) if bias is None else np.exp(bias.reshape(-1))
    expected = np.sum(weights[valid] * values[valid]) / np.sum(weights[valid])
    np.testing.assert_allclose(output.attention_outputs, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("q_len,causal,window", [(1, False, None), (2, True, None), (1, False, 2)])
@pytest.mark.parametrize("caller_mask", [False, True])
def test_paged_attention_excludes_tail_and_padding_pages(monkeypatch, q_len, causal, window, caller_mask):
    meta = metadata()
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(paged, "with_sharding_constraint", lambda arr, **kwargs: arr)

    # Independent dense implementation of the kernel boundary. The adapter,
    # not this reference, must supply each request's live-length visibility.
    def reference(q, keys, values, bias, cuq, cuk, sinks, tables, *, mask_info, causal, sliding_window, **kwargs):
        dense_v = np.asarray(values)[np.asarray(tables)].reshape(2, 8, 1, 1)
        scores = np.zeros((2, 1, q_len, 8), np.float32)
        valid = np.asarray(mask_info.get_or_compute_attention_mask())
        weights = np.where(valid, np.exp(scores), 0.0)
        denom = weights.sum(-1, keepdims=True)
        weights = weights / np.where(denom > 0, denom, 1)
        return jnp.asarray(np.einsum("bhqk,bkhd->bqhd", weights, dense_v))

    monkeypatch.setattr(paged, "flash_attention", reference)
    # Each request has one partial page plus a padded table column with poisoned values.
    vals = np.array([2, 100, 100, 100, 4, 6, 8, 100, 999, 999, 999, 999], np.float32)
    cache = SimpleNamespace(
        key_cache=jnp.zeros((3, 4, 1, 1)),
        value_cache=jnp.asarray(vals).reshape(3, 4, 1, 1),
        metadata=SimpleNamespace(data_parallel_size=1),
    )
    lengths = [1, 3]
    mask = None
    if caller_mask:
        mask = np.ones((2, 1, q_len, 8), bool)
        mask[1, :, :, 1] = False
    output = paged.PagedFlashAttn(meta).forward_native(
        jnp.zeros((2, q_len, 1, 1)),
        None,
        None,
        cache,
        SimpleNamespace(pages_tables=jnp.array([[0, 2], [1, 2]]), context_lens=jnp.array(lengths)),
        mask_info=None if mask is None else MaskInfo(_attention_mask=jnp.asarray(mask)),
        causal=causal,
        sliding_window=window,
    )
    expected = np.zeros((2, q_len, 1, 1), np.float32)
    for b, length in enumerate(lengths):
        for qi in range(q_len):
            position = length - q_len + qi
            valid = [
                ki
                for ki in range(length)
                if position >= 0
                and (not causal or ki <= position)
                and (window is None or abs(ki - position) < window)
                and (mask is None or mask[b, 0, qi, ki])
            ]
            if valid:
                expected[b, qi, 0, 0] = np.mean(vals[b * 4 + np.asarray(valid)])
    np.testing.assert_allclose(output.attention_outputs, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "enabled,rows,pages,expected_local",
    [(False, 2, 4, False), (True, 3, 4, False), (True, 2, 3, False), (True, 2, 4, True)],
)
def test_paged_dp_pool_sharding_and_page_ids_share_one_decision(monkeypatch, enabled, rows, pages, expected_local):
    monkeypatch.setattr(paged.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(paged, "ENABLE_DP_LOCAL_PAGE_PATH", enabled)
    monkeypatch.setattr(paged, "with_sharding_constraint", lambda arr, **kwargs: arr)
    # Use the real semantic EMPTY sentinel rather than assuming its representation.
    from spectrax import common_types as ct

    resolver = SimpleNamespace(resolve=lambda axes, **kwargs: Ps(None if axes[0] == ct.EMPTY else "dp"))
    meta = SimpleNamespace(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        mesh=None,
        runtime_sharding_resolver=SimpleNamespace(with_mesh=lambda mesh: resolver),
        head_shard_factor=lambda *args: 1,
        get_shardings=lambda *args, **kwargs: SimpleNamespace(query=Ps("dp"), bias=Ps(), softmax_aux=Ps(), output=Ps()),
        get_operation_config=lambda name: None,
    )
    tables = jnp.asarray([[0], [2]] if rows == 2 else [[0], [1], [2]], jnp.int32)

    def reference(q, keys, values, bias, cuq, cuk, sinks, dispatched_tables, *, in_specs, **kwargs):
        # Return observable localized/global IDs and whether the pool is sharded.
        sharded = in_specs[1][0] is not None
        return jnp.stack((dispatched_tables[:, 0], jnp.full((rows,), sharded)), axis=-1)

    monkeypatch.setattr(paged, "flash_attention", reference)
    cache = SimpleNamespace(
        key_cache=jnp.zeros((pages, 4, 1, 1)),
        value_cache=jnp.zeros((pages, 4, 1, 1)),
        metadata=SimpleNamespace(data_parallel_size=2),
    )
    output = paged.PagedFlashAttn(meta).forward_native(
        jnp.zeros((rows, 1, 1, 1)),
        None,
        None,
        cache,
        SimpleNamespace(pages_tables=tables, context_lens=jnp.ones((rows,), jnp.int32)),
    )
    expected_ids = [0, 0] if expected_local else np.asarray(tables)[:, 0]
    np.testing.assert_array_equal(output.attention_outputs[:, 0], expected_ids)
    np.testing.assert_array_equal(output.attention_outputs[:, 1], expected_local)
