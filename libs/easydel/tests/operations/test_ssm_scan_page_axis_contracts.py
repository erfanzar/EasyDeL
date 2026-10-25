"""Shared page and ring-adapter contracts without allocating device arrays."""

from types import SimpleNamespace

import jax
import pytest
from easydel.axis import ATTN_DP
from easydel.infra.etils import EasyDeLBackends
from easydel.operations import OperationRegistry
from easydel.operations.kernels import ragged_page_attention, ring_attention, unified_attention
from easydel.operations.kernels.ssm1 import packed_ssm1_fwd
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps
from spectrax import common_types

pytestmark = pytest.mark.cpu_ok


def _has_scan(program):
    """Inspect abstract nested programs, including ejkernel's jit boundaries."""
    if hasattr(program, "eqns"):
        return any(eqn.primitive.name == "scan" or _has_scan(eqn.params) for eqn in program.eqns)
    if hasattr(program, "jaxpr") and program.jaxpr is not program:
        return _has_scan(program.jaxpr)
    if isinstance(program, dict):
        return any(_has_scan(value) for value in program.values())
    if isinstance(program, tuple | list):
        return any(_has_scan(value) for value in program)
    return False


@pytest.mark.parametrize(
    "associative,mambapy", [(False, False), (True, False), (False, True), (True, True), (None, True)]
)
@pytest.mark.parametrize("segmented", [False, True])
def test_ssm1_scan_selection_and_output_layout_by_abstract_tracing(associative, mambapy, segmented):
    """Trace abstract descriptors only: no arrays, lowering, compilation or execution."""
    operation = OperationRegistry.create("ssm1", SimpleNamespace(backend=EasyDeLBackends.CPU))
    shapes = [(2, 5, 3), (3, 4), (2, 5, 4), (2, 5, 4), (3,), (2, 5, 3), (2, 3, 4), (2, 5, 3), (2, 5)]
    descriptors = [jax.ShapeDtypeStruct(shape, jnp.int32 if i == 8 else jnp.float32) for i, shape in enumerate(shapes)]

    def forward(x, a, b, c, d, dt, state, gate, segments):
        result = operation(
            x,
            a,
            b,
            c,
            d,
            dt,
            ssm_state=state,
            gate=gate,
            segment_ids=segments if segmented else None,
            use_associative_scan=associative,
            use_mambapy=mambapy,
        )
        return result.attention_outputs, result.ssm_state

    program = jax.make_jaxpr(forward)(*descriptors)
    assert [aval.shape for aval in program.out_avals] == [(2, 5, 3), (2, 3, 4)]
    # A sequential recurrence must appear in the public operation's traced
    # program only when explicitly selected. Parallel prefixes use affine composition.
    assert _has_scan(program) == (associative is False)


@pytest.mark.parametrize("associative", [False, True])
def test_packed_scan_selection_and_layout_by_abstract_tracing(associative):
    shapes = [(2, 5, 3), (3, 4), (2, 5, 4), (2, 5, 4), (3,), (2, 5, 3), (2, 5), (2, 3, 4)]
    descriptors = [jax.ShapeDtypeStruct(shape, jnp.bool_ if i == 6 else jnp.float32) for i, shape in enumerate(shapes)]
    program = jax.make_jaxpr(lambda *args: packed_ssm1_fwd(*args, use_associative_scan=associative))(*descriptors)
    assert [aval.shape for aval in program.out_avals] == [(2, 5, 3), (2, 3, 4)]
    assert any(eqn.primitive.name == "scan" for eqn in program.jaxpr.eqns) == (not associative)


@pytest.mark.parametrize("adapter", [ragged_page_attention, unified_attention])
@pytest.mark.parametrize("dp_size", [None, 0, 1, 2, 8])
def test_page_axis_compatibility_preserves_cache_policy(adapter, dp_size):
    metadata = SimpleNamespace() if dp_size is None else SimpleNamespace(data_parallel_size=dp_size)
    cache = SimpleNamespace(metadata=metadata)
    expected = ATTN_DP if dp_size is not None and dp_size > 1 else common_types.EMPTY
    assert adapter.dp_page_axis(cache) == expected
    assert adapter._dp_page_axis(cache) == expected


class HeadDescriptor:
    """Shape/dtype and head labels only; no numerical arrays or operations."""

    def __init__(self, shape, heads=None, dtype=jnp.float32):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.heads = tuple(range(shape[2])) if heads is None else heads

    def astype(self, dtype):
        return HeadDescriptor(self.shape, self.heads, dtype)


@pytest.mark.parametrize("kv_heads,shards", [(3, 2), (2, 4), (2, 2), (1, 4), (3, 1)])
def test_registered_ring_preserves_global_gqa_mapping(monkeypatch, kv_heads, shards):
    """Exercise real ring routing with a shape-only kernel and repeat primitive."""
    query_heads = 12

    def repeat_heads(value, repetitions, axis):
        assert axis == 2
        shape = (*value.shape[:2], value.shape[2] * repetitions, value.shape[3])
        return HeadDescriptor(shape, tuple(head for head in value.heads for _ in range(repetitions)), value.dtype)

    def kernel(query, key, value, *args, **kwargs):
        # Simulate blocked local GQA mapping using only Python labels.
        assert key.heads == value.heads
        assert key.shape[2] == 1 or key.shape[2] % shards == 0
        group_size = query.shape[2] // key.shape[2]
        return tuple(key.heads[head // group_size] for head in range(query.shape[2]))

    monkeypatch.setattr(ring_attention.jnp, "repeat", repeat_heads)
    monkeypatch.setattr(ring_attention, "ring_attention", kernel)
    monkeypatch.setattr(
        ring_attention.RingAttn,
        "create_stable_sharding",
        lambda self, state_ps, **kwargs: state_ps,
    )
    spec = Ps(None, None, "tp", None)
    metadata = SimpleNamespace(
        mesh=SimpleNamespace(shape={"tp": shards}),
        runtime_dtype=jnp.float32,
        sequence_axis_name="sp",
        get_shardings=lambda **kwargs: SimpleNamespace(query=spec, key=spec, value=spec, output=spec, softmax_aux=None),
        get_operation_config=lambda name: None,
    )
    operation = OperationRegistry.create("ring", metadata)
    output = operation.forward_native(
        HeadDescriptor((1, 3, query_heads, 8)),
        HeadDescriptor((1, 3, kv_heads, 8)),
        HeadDescriptor((1, 3, kv_heads, 8)),
    )
    assert output.attention_outputs == tuple(head // (query_heads // kv_heads) for head in range(query_heads))
    assert output.attention_weights is None
