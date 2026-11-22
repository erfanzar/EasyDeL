"""Abstract-only compressed cache contracts and distributed-ring safety checks."""

import jax
import numpy as np
import pytest
from easydel.caching import RaggedPagesMetadata
from easydel.caching.turboquant_ragged_page import TurboQuantRaggedPagesCacheConfig, TurboQuantRaggedPagesCacheView
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.quantization import TurboQuantConfig, TurboQuantConstants
from easydel.operations import OperationMetadata, OperationRegistry
from ejkernel.types import MaskInfo
from jax import numpy as jnp

from .test_paged_reference_contract import _metadata

pytestmark = pytest.mark.cpu_ok

PAGE_FIELDS = ("key_indices_pages", "key_signs_pages", "key_norms_pages", "value_indices_pages", "value_norms_pages")
PAGE_SHAPES = ((4, 2, 1, 4), (4, 2, 1, 1), (4, 2, 1, 2), (4, 2, 1, 4), (4, 2, 1))
PAGE_DTYPES = (jnp.uint8, jnp.uint8, jnp.bfloat16, jnp.uint8, jnp.bfloat16)


def _constants():
    rng = np.random.default_rng(561)
    return TurboQuantConstants(
        key_codebook=np.linspace(-0.7, 0.7, 8, dtype=np.float32),
        value_codebook=np.linspace(-0.8, 0.8, 16, dtype=np.float32),
        rotation_matrix=np.linalg.qr(rng.normal(size=(8, 8)))[0].astype(np.float32),
        qjl_projection=rng.normal(size=(8, 8)).astype(np.float32),
        qjl_dim=8,
        bits=4,
    )


def _compressed_cache(dummy, pages, constants):
    config = TurboQuantRaggedPagesCacheConfig(
        1, 4, 1, 8, 8, page_size=2, num_pages=4, version="v3", turboquant_config=TurboQuantConfig()
    )
    return TurboQuantRaggedPagesCacheView(
        metadata=config,
        layer_index=0,
        kv_pages=dummy,
        constants=constants,
        **dict(zip(PAGE_FIELDS, pages, strict=True)),
    )


@pytest.mark.parametrize("name", ["paged_attention_reference", "ragged_page_attention_v2", "ragged_page_attention_v3"])
@pytest.mark.parametrize("dropout,normalized", [(True, True), (False, False), (True, False)])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("update", [False, True])
def test_registered_compressed_cache_preserves_five_array_layout(name, dropout, normalized, batched, update):
    """Trace public registered calls with the real scalar dummy and compressed storage."""
    operation = OperationRegistry.create(name, _metadata())
    query_shape = (2, 2, 2, 8) if batched else (4, 2, 8)
    kv_shape = (2, 2, 1, 8) if batched else (4, 1, 8)
    inputs = [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in (query_shape, kv_shape, kv_shape)]
    inputs += [jax.ShapeDtypeStruct((), jnp.bfloat16)]
    inputs += [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in zip(PAGE_SHAPES, PAGE_DTYPES, strict=True)]
    inputs += [jax.ShapeDtypeStruct((2, 2), jnp.int32), jax.ShapeDtypeStruct((2,), jnp.int32)]
    constants = _constants()

    def forward(q, k, v, dummy, *pages_and_metadata):
        *pages, tables, contexts = pages_and_metadata
        view = _compressed_cache(dummy, pages, constants)
        runtime = RaggedPagesMetadata(tables, contexts, jnp.array([0, 2, 4], dtype=jnp.int32), contexts, page_size=2)
        result = operation(
            query=q,
            key=k if update else None,
            value=v if update else None,
            cache_view=view,
            cache_metadata=runtime,
            compute_dtype=jnp.float32,
            deterministic=not dropout,
            dropout_prob=0.25,
            dropout_rng=jax.random.key(23),
            normalize_output=normalized,
            mask_info=MaskInfo(
                _attention_mask=jnp.ones((2 if batched else 1, 1, 2 if batched else 4, 4), dtype=jnp.bool_),
                _q_segment_ids=jnp.ones((2 if batched else 1, 2 if batched else 4), dtype=jnp.int32),
                _kv_segment_ids=jnp.ones((2 if batched else 1, 4), dtype=jnp.int32),
            ),
            bias=jnp.zeros((2 if batched else 1, 2, 2 if batched else 4, 4), dtype=jnp.float32),
            softmax_aux=jnp.zeros((2, 1), dtype=jnp.float32),
            alibi_slopes=jnp.ones((2,), dtype=jnp.float32),
            chunk_size=4,
        )
        assert isinstance(result.cache_view, TurboQuantRaggedPagesCacheView)
        assert result.cache_view.kv_pages is dummy
        return result.attention_outputs, *(getattr(result.cache_view, field) for field in PAGE_FIELDS)

    program = jax.make_jaxpr(forward)(*inputs)
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [
        (query_shape, jnp.float32),
        *zip(PAGE_SHAPES, PAGE_DTYPES, strict=True),
    ]
    # Cache-only reads must return the exact original storage, not requantize
    # historical cells. Writes return five new states, not a dense replacement.
    if not update:
        assert list(program.jaxpr.outvars[1:]) == list(program.jaxpr.invars[4:9])


def _distributed_metadata():
    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 4))
    return OperationMetadata(
        base_config=config,
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        sequence_axis_name="sp",
    )


@pytest.mark.parametrize("name", ["ring", "ring_paged"])
@pytest.mark.parametrize("visibility", ["pairwise", "bias_with_segments", "bias_with_positions"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("gradient", [False, True])
def test_registered_distributed_ring_visibility_forward_and_gradient(name, visibility, causal, gradient):
    """Trace the real four-member ring with global bias and local segment/position metadata."""
    operation = OperationRegistry.create(name, _distributed_metadata())
    descriptors = [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in ((1, 16, 2, 8), (1, 24, 1, 8), (1, 24, 1, 8))]

    def forward(q, k, v):
        rows = jnp.arange(16)[None, None, :, None]
        cols = jnp.arange(24)[None, None, None, :]
        kwargs = {"bias": ((rows * cols + 3 * cols) % 11).astype(jnp.float32) / 8}
        if visibility == "pairwise":
            info = MaskInfo(_attention_mask=(rows + 2 * cols) % 5 != 0)
        else:
            qs = jnp.arange(16)[None, :] // 8
            ks = jnp.arange(24)[None, :] // 12
            info = MaskInfo(_q_segment_ids=qs, _kv_segment_ids=ks)
            if visibility == "bias_with_positions":
                info = info.replace(q_positions=jnp.arange(16)[None, :] % 8, kv_positions=jnp.arange(24)[None, :] % 12)
        output = operation(q, k, v, mask_info=info, causal=causal, **kwargs).attention_outputs
        return output.sum() if gradient else output

    program = jax.make_jaxpr(jax.grad(forward, argnums=(0, 1, 2)) if gradient else forward)(*descriptors)
    expected = descriptors if gradient else descriptors[:1]
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [(x.shape, x.dtype) for x in expected]


@pytest.mark.parametrize("name", ["ring", "ring_paged"])
@pytest.mark.parametrize("gradient", [False, True])
def test_paged_distributed_ring_generated_visibility_forward_and_gradient(name, gradient):
    """Generated request visibility and compressed page updates also trace through the ring/VJP."""
    operation = OperationRegistry.create(name, _distributed_metadata())
    descriptors = [jax.ShapeDtypeStruct((4, heads, 8), jnp.float32) for heads in (2, 1, 1)]
    descriptors += [jax.ShapeDtypeStruct((), jnp.bfloat16)]
    descriptors += [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in zip(PAGE_SHAPES, PAGE_DTYPES, strict=True)]

    def forward(q, k, v, dummy, *pages):
        runtime = RaggedPagesMetadata(
            jnp.array([[2, 0], [3, 1]], dtype=jnp.int32),
            jnp.array([3, 2], dtype=jnp.int32),
            jnp.array([0, 2, 4], dtype=jnp.int32),
            jnp.array([3, 2], dtype=jnp.int32),
            page_size=2,
        )
        output = operation(
            q,
            k,
            v,
            cache_view=_compressed_cache(dummy, pages, _constants()),
            cache_metadata=runtime,
        ).attention_outputs
        return output.sum() if gradient else output

    program = jax.make_jaxpr(jax.grad(forward, argnums=(0, 1, 2)) if gradient else forward)(*descriptors)
    expected = descriptors[:3] if gradient else descriptors[:1]
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [(x.shape, x.dtype) for x in expected]


@pytest.mark.parametrize("gradient", [False, True])
def test_single_shard_ring_pairwise_forward_and_gradient_remain_traceable(gradient):
    """Single-shard pairwise attention retains its public output and autodiff layouts."""
    operation = OperationRegistry.create("ring", _metadata())
    descriptors = [jax.ShapeDtypeStruct((1, 4, heads, 8), jnp.float32) for heads in (2, 1, 1)]

    def forward(q, k, v):
        result = operation(
            q,
            k,
            v,
            mask_info=MaskInfo(_attention_mask=jnp.ones((1, 1, 4, 4), dtype=jnp.bool_)),
        )
        return jnp.sum(result.attention_outputs) if gradient else result.attention_outputs

    program = jax.make_jaxpr(jax.grad(forward) if gradient else forward)(*descriptors)
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [((1, 4, 2, 8), jnp.float32)]
