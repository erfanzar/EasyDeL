"""Hardware-only regression coverage of MLA version/configuration behavior."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.operations import OperationMetadata
from easydel.operations.kernels.multi_latent_ragged_page_attention import (
    MultiLatentRaggedPageAttn,
    MultiLatentRaggedPageAttnV2,
)
from ejkernel.modules.operations.configs import (
    MultiLatentRaggedPageAttentionConfig,
    MultiLatentRaggedPageAttentionV2Config,
)
from jax import numpy as jnp
from jax.sharding import Mesh


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("as_dict", [False, True])
def test_mla_config_forward_outputs_and_cache_update(monkeypatch, version, as_dict):
    op_class = MultiLatentRaggedPageAttn if version == 1 else MultiLatentRaggedPageAttnV2
    cfg_class = MultiLatentRaggedPageAttentionConfig if version == 1 else MultiLatentRaggedPageAttentionV2Config
    cfg = cfg_class(
        chunk_prefill_size=16,
        num_kv_pages_per_block=2,
        num_queries_per_block=(2, 32, 64) if version == 2 else 2,
        vmem_limit_bytes=4096,
        platform="xla",
        num_warps=8,
        num_stages=2,
    )

    def reference_kernel(q, qpe, kv, kpe, pages, *args, cfg, **kwargs):
        # Encode config values in observable outputs instead of asserting private helper calls.
        block = cfg.num_queries_per_block
        block = sum(block) if isinstance(block, tuple) else block
        result = q * cfg.chunk_prefill_size + block + cfg.num_warps + cfg.num_stages
        return result, pages + cfg.vmem_limit_bytes

    monkeypatch.setattr(op_class, "_kernel_function", staticmethod(reference_kernel))
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tp",))
    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        _stored_mesh=mesh,
        operation_configs={op_class.get_impl_name(): cfg.to_dict() if as_dict else cfg},
    )
    q = jnp.ones((2, 4, 4), jnp.float32)
    qpe = jnp.zeros((2, 4, 2), jnp.float32)
    kv = jnp.zeros((2, 4), jnp.float32)
    kpe = jnp.zeros((2, 2), jnp.float32)
    pages = jnp.zeros((2, 2, 1, 6), jnp.float32)
    cache = SimpleNamespace(kv_pages=pages, metadata=SimpleNamespace(data_parallel_size=1))
    scheduling = SimpleNamespace(
        request_distribution=jnp.array([0, 1, 1], jnp.int32),
        context_lens=jnp.array([2], jnp.int32),
        pages_tables=jnp.array([[0]], jnp.int32),
        query_start_loc=jnp.array([0, 2], jnp.int32),
    )
    output = op_class(metadata).forward_core(q, qpe, kv, kpe, cache, scheduling)
    # Four fp32 heads: the fallback maps (2, 32, 64) to (2, 4, 4).
    np.testing.assert_allclose(output.attention_outputs, 16 + (10 if version == 2 else 2) + 8 + 2)
    np.testing.assert_array_equal(output.cache_view.kv_pages, pages + 4096)
    assert output.attention_weights is None


@pytest.mark.parametrize("op_class", [MultiLatentRaggedPageAttn, MultiLatentRaggedPageAttnV2])
@pytest.mark.cpu_ok
def test_mla_missing_positional_keys_rejected_without_array_computation(op_class):
    op = op_class(SimpleNamespace())
    with pytest.raises(ValueError, match="keys_pe"):
        op.forward_core(None, None, None, None, None, None)
