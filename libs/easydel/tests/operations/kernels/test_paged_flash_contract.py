"""Paged flash's GPU argument contract, without device arrays or GPU execution."""

from types import SimpleNamespace

import pytest
from easydel.operations.kernels import paged_flash_attention as adapter
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps

pytestmark = pytest.mark.cpu_ok


class TensorDescriptor:
    """Shape/dtype-only descriptor; no array allocation or computation."""

    def __init__(self, shape, dtype=jnp.float32):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)

    def astype(self, dtype):
        return TensorDescriptor(self.shape, dtype)

    def reshape(self, *shape):
        return TensorDescriptor(shape, self.dtype)

    def __getitem__(self, item):
        # Only symbolic shape flow is needed; no values are evaluated.
        return self

    def __sub__(self, other):
        return self

    def __add__(self, other):
        return self

    def __lt__(self, other):
        return self

    def __ge__(self, other):
        return self

    def __and__(self, other):
        return self


@pytest.mark.parametrize("softmax_dtype", [None, jnp.float32])
def test_paged_flash_satisfies_float32_logits_contract(monkeypatch, softmax_dtype):
    def flash(query, key, value, *args, logits_dtype, **kwargs):
        if logits_dtype != jnp.float32:
            raise ValueError("GPU flash kernels require float32 logits")
        return TensorDescriptor(query.shape, query.dtype)

    monkeypatch.setattr(adapter.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(adapter, "flash_attention", flash)
    monkeypatch.setattr(adapter, "unwrap_metadata", lambda metadata, name: metadata)
    monkeypatch.setattr(adapter, "with_sharding_constraint", lambda arr, **kwargs: arr)
    # Symbolically trace mask construction without allocating any JAX arrays.
    monkeypatch.setattr(adapter.jnp, "asarray", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(adapter.jnp, "arange", lambda size, **kwargs: TensorDescriptor((size,), jnp.int32))
    resolver = SimpleNamespace(resolve=lambda axes, **kwargs: Ps(*([None] * len(axes))))
    shardings = SimpleNamespace(query=Ps(), bias=Ps(), softmax_aux=Ps(), output=Ps())
    metadata = SimpleNamespace(
        runtime_dtype=jnp.bfloat16,
        runtime_softmax_dtype=softmax_dtype,
        mesh=None,
        runtime_sharding_resolver=SimpleNamespace(with_mesh=lambda mesh: resolver),
        head_shard_factor=lambda *args: 1,
        get_shardings=lambda *args, **kwargs: shardings,
        get_operation_config=lambda name: None,
    )
    query = TensorDescriptor((1, 2, 4, 8))
    cache = SimpleNamespace(
        key_cache=TensorDescriptor((2, 4, 2, 8)),
        value_cache=TensorDescriptor((2, 4, 2, 8)),
        metadata=SimpleNamespace(data_parallel_size=1),
    )
    output = adapter.PagedFlashAttn(metadata).forward_native(
        query,
        None,
        None,
        cache,
        SimpleNamespace(
            pages_tables=TensorDescriptor((1, 2), jnp.int32),
            context_lens=TensorDescriptor((1,), jnp.int32),
        ),
    )
    assert output.attention_outputs.shape == query.shape
    assert output.attention_outputs.dtype == jnp.bfloat16
    assert output.attention_weights is None
