"""MLA block/config forwarding contracts without array computation."""

from types import SimpleNamespace

import pytest
from easydel.operations.kernels import multi_latent_ragged_page_attention as adapter
from ejkernel.modules.operations.configs import (
    MultiLatentRaggedPageAttentionConfig,
    MultiLatentRaggedPageAttentionV2Config,
)
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps
from spectrax import common_types as ct

pytestmark = pytest.mark.cpu_ok


class TensorDescriptor:
    """Shape/dtype descriptor that never allocates or computes arrays."""

    def __init__(self, shape, dtype=jnp.float32):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)

    def reshape(self, *shape):
        return TensorDescriptor(shape, self.dtype)


def run_adapter(monkeypatch, version, cfg, override=None, dtype=jnp.float32):
    """Run the real adapter with shape descriptors and a configuration-echo kernel."""
    op_class = adapter.MultiLatentRaggedPageAttn if version == 1 else adapter.MultiLatentRaggedPageAttnV2
    updated_pages = TensorDescriptor((2, 2, 1, 6), dtype)

    def kernel(*args, cfg, **kwargs):
        # The public output exposes the supplied configuration, not private helper calls.
        return cfg.to_dict(), updated_pages

    monkeypatch.setattr(op_class, "_kernel_function", staticmethod(kernel))
    monkeypatch.setattr(adapter.jax, "shard_map", lambda fn, **kwargs: fn)
    monkeypatch.setattr(adapter.jnp, "asarray", lambda value, dtype: value)
    monkeypatch.setattr(adapter.jnp, "array", lambda value, dtype: value)
    mesh = SimpleNamespace(shape={"tp": 2})
    resolver = SimpleNamespace(resolve=lambda axes, **kwargs: Ps(*("tp" if ax == ct.HEAD else None for ax in axes)))
    metadata = SimpleNamespace(
        mesh=mesh,
        runtime_sharding_resolver=SimpleNamespace(with_mesh=lambda mesh: resolver),
        get_operation_config=lambda name: cfg,
    )
    q = TensorDescriptor((2, 4, 4), dtype)
    qpe = TensorDescriptor((2, 4, 2), dtype)
    kv = TensorDescriptor((2, 4), dtype)
    kpe = TensorDescriptor((2, 2), dtype)
    cache = SimpleNamespace(kv_pages=TensorDescriptor((2, 2, 1, 6), dtype), metadata=None)
    scheduling = SimpleNamespace(
        request_distribution=(0, 1, 1),
        context_lens=TensorDescriptor((1,), jnp.int32),
        pages_tables=TensorDescriptor((1, 1), jnp.int32),
        query_start_loc=TensorDescriptor((2,), jnp.int32),
    )
    output = op_class(metadata).forward_core(q, qpe, kv, kpe, cache, scheduling, num_queries_per_block=override)
    assert output.cache_view.kv_pages is updated_pages
    assert output.attention_weights is None
    return output.attention_outputs


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(
    "version,blocks,override,expected_fp32,expected_bf16",
    [
        (1, 1, None, 1, 1),
        (2, (2, 32, 64), None, (2, 2, 2), (1, 1, 1)),
        (2, (3, 17, 33), None, (2, 2, 2), (1, 1, 1)),
        (2, [2, 32, 64], None, (2, 2, 2), (1, 1, 1)),
        (2, 32, None, 2, 1),
        (2, None, None, 2, 1),
        (2, (2, 32, 64), (7, 48, 96), (2, 2, 2), (1, 1, 1)),
        (2, (2, 32, 64), 48, 2, 1),
    ],
)
def test_mla_preserves_version_specific_block_semantics(
    monkeypatch, as_dict, dtype, version, blocks, override, expected_fp32, expected_bf16
):
    cfg_class = MultiLatentRaggedPageAttentionConfig if version == 1 else MultiLatentRaggedPageAttentionV2Config
    cfg = cfg_class(
        num_queries_per_block=blocks,
        num_kv_pages_per_block=2,
        chunk_prefill_size=16,
        vmem_limit_bytes=4096,
        platform="xla",
        backend="any",
        num_warps=8,
        num_stages=2,
    )
    config = cfg.to_dict() if as_dict else cfg
    # Exercise JSON-style lists in the mapping path too, before dataclass normalization.
    if as_dict:
        config["num_queries_per_block"] = blocks
    output = run_adapter(monkeypatch, version, config, override, dtype)
    # Four global heads / two TP ranks: two fp32 or one packed bf16 head.
    # A single-token decode tile must still satisfy the kernel's block divisibility.
    expected = expected_fp32 if dtype == jnp.float32 else expected_bf16
    assert output == {**cfg.to_dict(), "num_queries_per_block": expected}


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("dtype,expected", [(jnp.float32, 2), (jnp.bfloat16, 1)])
def test_mla_v1_still_resolves_packed_local_head_divisors(monkeypatch, as_dict, dtype, expected):
    cfg = MultiLatentRaggedPageAttentionConfig(num_queries_per_block=64)
    output = run_adapter(monkeypatch, 1, cfg.to_dict() if as_dict else cfg, dtype=dtype)
    assert output["num_queries_per_block"] == expected


@pytest.mark.parametrize("blocks", [(2, 32), (2, 32, 64, 128)])
def test_mla_v2_rejects_invalid_token_blocks(monkeypatch, blocks):
    with pytest.raises(ValueError, match="num_queries_per_block"):
        run_adapter(monkeypatch, 2, {"num_queries_per_block": blocks})


@pytest.mark.parametrize(
    "blocks,expected", [((2, 0, 64), (2, 1, 2)), ((2, -1, 64), (2, 1, 2)), (0, 1), (-1, 1), (1.5, 1)]
)
def test_mla_v2_normalizes_legacy_block_requests(monkeypatch, blocks, expected):
    output = run_adapter(monkeypatch, 2, {"num_queries_per_block": blocks})
    assert output["num_queries_per_block"] == expected
