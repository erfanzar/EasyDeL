"""Compressed-context dropout: abstract CPU traces and independent hardware parity."""

import jax
import numpy as np
import pytest
from easydel.caching import ParallelHybridCacheView, RaggedPagesMetadata
from easydel.caching.turboquant_ragged_page import TurboQuantRaggedPagesCacheConfig, TurboQuantRaggedPagesCacheView
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.quantization import TurboQuantConfig, TurboQuantConstants
from jax import numpy as jnp
from spectrax import common_types

PAGE_FIELDS = ("key_indices_pages", "key_signs_pages", "key_norms_pages", "value_indices_pages", "value_norms_pages")
SHAPES = ((4, 2, 1, 4), (4, 2, 1, 1), (4, 2, 1, 2), (4, 2, 1, 4), (4, 2, 1))
DTYPES = (np.uint8, np.uint8, jnp.bfloat16, np.uint8, jnp.bfloat16)


def _constants():
    rng = np.random.default_rng(174)
    rotation = np.linalg.qr(rng.normal(size=(8, 8)))[0].astype(np.float32)
    return TurboQuantConstants(
        key_codebook=np.linspace(-0.7, 0.7, 8, dtype=np.float32),
        value_codebook=np.linspace(-0.9, 0.9, 16, dtype=np.float32),
        rotation_matrix=rotation,
        qjl_projection=rng.normal(size=(8, 8)).astype(np.float32),
        qjl_dim=8,
        bits=4,
    )


def _config():
    return TurboQuantRaggedPagesCacheConfig(
        2, 8, 1, 8, 8, page_size=2, num_pages=4, version="v3", turboquant_config=TurboQuantConfig()
    )


def _performer():
    return FlexibleAttentionModule(
        EasyDeLBaseConfig(
            attn_mechanism="ragged_page_attention_v3",
            attn_dtype=np.float32,
            attn_softmax_dtype=np.float32,
            sharding_axis_dims=(1, 1, -1, 1, 1, 1),
            layer_types=["full_attention", "chunked_attention"],
            attention_chunk_size=3,
        ),
        softmax_scale=0.5,
    )


@pytest.mark.cpu_ok
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("rank_major", [False, True])
@pytest.mark.parametrize("read_only", [False, True])
def test_real_compressed_dropout_trace_preserves_five_arrays_and_scalar_dummy(hybrid, normalized, rank_major, read_only):
    """Trace genuine compressed views through registered operations without executing arrays."""
    performer, constants, config = _performer(), _constants(), _config()

    def forward(q, k, v, tables, contexts, starts, rng, dummy, *storage):
        component = TurboQuantRaggedPagesCacheView(
            metadata=config,
            layer_index=1,
            kv_pages=dummy,
            constants=constants,
            **dict(zip(PAGE_FIELDS, storage, strict=True)),
        )
        runtime = RaggedPagesMetadata(tables, contexts, starts, contexts, page_size=2, version="v3")
        # Exercise real layer-group selection before both the compressed writes
        # and transient read view. The root payload intentionally has no tables.
        root = RaggedPagesMetadata(None, None, None, None, cache_groups=(runtime,), layer_to_cache_group=(0, 0))
        view = ParallelHybridCacheView(transformer=component, recurrent=None, layer_index=0) if hybrid else component
        result = performer(
            q,
            None if read_only else k,
            None if read_only else v,
            mode=common_types.MODE_DECODE,
            cache_view=view,
            cache_metadata=root,
            deterministic=False,
            dropout_prob=0.25,
            dropout_rng=rng,
            normalize_output=normalized,
            logits_soft_cap=3.0,
            sliding_window=4,
            softmax_aux=np.zeros((2,), dtype=np.float32),
        )
        updated = result.cache_view.transformer if hybrid else result.cache_view
        assert isinstance(updated, TurboQuantRaggedPagesCacheView)
        assert updated.metadata is config
        assert updated.constants is constants
        return result.attention_outputs, updated.kv_pages, *(getattr(updated, field) for field in PAGE_FIELDS)

    shapes = ((8, 2, 8), (8, 1, 8), (8, 1, 8), (4, 2), (4,), (2, 3) if rank_major else (5,), (2,), ())
    descriptors = [
        jax.ShapeDtypeStruct(shape, np.float32 if i < 3 or i == 7 else np.uint32 if i == 6 else np.int32)
        for i, shape in enumerate(shapes)
    ]
    descriptors += [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in zip(SHAPES, DTYPES, strict=True)]
    traced = jax.make_jaxpr(forward)(*descriptors)
    assert [(a.shape, a.dtype) for a in traced.out_avals] == [
        ((8, 2, 8), np.dtype(np.float32)),
        ((), np.dtype(np.float32)),
        *((shape, np.dtype(dtype)) for shape, dtype in zip(SHAPES, DTYPES, strict=True)),
    ]


def _numpy_decode(ki, ks, kn, vi, vn, constants):
    """Independent packed-byte decoder matching the documented linear QJL estimator."""

    def indices(packed):
        return np.stack((packed & 15, packed >> 4), axis=-1).reshape(*packed.shape[:-1], -1)

    signs = np.where((ks[..., None] & (1 << np.arange(8))) != 0, 1.0, -1.0).reshape(*ks.shape[:-1], -1)
    centroids = constants.key_codebook[indices(ki)]
    rotated_keys = centroids * kn[..., :1].astype(np.float32)
    rotated_keys += (signs @ constants.qjl_projection) * kn[..., 1:].astype(np.float32) * (np.sqrt(np.pi / 2) / 8)
    keys = rotated_keys @ constants.rotation_matrix
    values = (constants.value_codebook[indices(vi)] @ constants.rotation_matrix) * vn[..., None].astype(np.float32)
    return keys, values


@pytest.fixture
def highest_matmul_precision():
    """Keep compression and independent fp32 reference math at matching precision."""
    with jax.default_matmul_precision("highest"):
        yield


@pytest.mark.usefixtures("highest_matmul_precision")
@pytest.mark.parametrize("normalized", [False, True])
def test_compressed_dropout_numerical_context_and_untouched_history(normalized):
    """Accelerator-only; an independent NumPy reference includes history and scheduled writes."""
    from ejkernel.quantization.turboquant import turboquant_compress_keys, turboquant_compress_values

    constants, config, performer = _constants(), _config(), _performer()
    rng = np.random.default_rng(2026)
    raw_k = jnp.asarray(rng.normal(size=(4, 2, 1, 8)), dtype=jnp.float32)
    raw_v = jnp.asarray(rng.normal(size=(4, 2, 1, 8)), dtype=jnp.float32)
    storage = (
        *turboquant_compress_keys(raw_k, constants.rotation_matrix, constants.key_codebook, constants.qjl_projection),
        *turboquant_compress_values(raw_v, constants.rotation_matrix, constants.value_codebook),
    )
    view = TurboQuantRaggedPagesCacheView(
        metadata=config,
        layer_index=1,
        kv_pages=jnp.zeros(()),
        constants=constants,
        **dict(zip(PAGE_FIELDS, storage, strict=True)),
    )
    # Request 0 has two historical tokens on page 2 and writes page 0.
    # Request 1 is idle. Request 2's invalid page must never mutate real storage.
    runtime = RaggedPagesMetadata(
        jnp.asarray([[2, 0], [1, -1], [-1, -1]]),
        jnp.asarray([3, 1, 1]),
        jnp.asarray([0, 1, 1, 2]),
        jnp.asarray([3, 1, 1]),
        page_size=2,
        version="v3",
    )
    query = jnp.asarray(rng.normal(size=(3, 2, 8)), dtype=jnp.float32)
    key = jnp.asarray(rng.normal(size=(3, 1, 8)), dtype=jnp.float32)
    value = jnp.asarray(rng.normal(size=(3, 1, 8)), dtype=jnp.float32)
    dropout_rng = jax.random.PRNGKey(83)
    kwargs = dict(
        mode=common_types.MODE_DECODE,
        cache_view=view,
        cache_metadata=runtime,
        deterministic=False,
        dropout_prob=0.25,
        dropout_rng=dropout_rng,
        normalize_output=normalized,
        logits_soft_cap=3.0,
        precision=jax.lax.Precision.HIGHEST,
    )
    result = performer(query, key, value, **kwargs)
    again = performer(query, key, value, **kwargs)
    np.testing.assert_array_equal(result.attention_outputs, again.attention_outputs)
    expected_storage = (
        *turboquant_compress_keys(key[:1], constants.rotation_matrix, constants.key_codebook, constants.qjl_projection),
        *turboquant_compress_values(value[:1], constants.rotation_matrix, constants.value_codebook),
    )
    for name, original, compressed in zip(PAGE_FIELDS, storage, expected_storage, strict=True):
        updated = np.asarray(getattr(result.cache_view, name))
        expected = np.asarray(original).copy()
        expected[0, 0] = np.asarray(compressed[0])
        np.testing.assert_array_equal(updated, expected)
    decoded_k, decoded_v = _numpy_decode(*(np.asarray(getattr(result.cache_view, n)) for n in PAGE_FIELDS), constants)
    historical_k = np.concatenate((decoded_k[2], decoded_k[0, :1]), axis=0)[:, 0]
    historical_v = np.concatenate((decoded_v[2], decoded_v[0, :1]), axis=0)[:, 0]
    logits = np.asarray(query[0]) @ historical_k.T * 0.5
    logits = 3 * np.tanh(logits / 3)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    if normalized:
        weights /= weights.sum(axis=-1, keepdims=True)
    # Without normalization, vanilla/Flash return the stabilized numerator.
    keep = np.asarray(jax.random.bernoulli(jax.random.fold_in(dropout_rng, 0), 0.75, (1, 2, 3, 4)))[0, :, 0, :3]
    expected = (weights * keep / 0.75) @ historical_v
    np.testing.assert_allclose(np.asarray(result.attention_outputs[0]), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(np.asarray(result.attention_outputs[1:]), 0)


class _HostPages(np.ndarray):
    """NumPy-only equivalent of JAX's functional dropped-OOB scatter."""

    @property
    def at(self):
        return _HostScatter(self)


class _HostScatter:
    def __init__(self, pages):
        self.pages = pages

    def __getitem__(self, indices):
        self.indices = indices
        return self

    def set(self, values, *, mode):
        assert mode == "drop"
        pages = self.pages.copy()
        destinations, offsets = self.indices
        for page, cell, value in zip(destinations, offsets, values, strict=True):
            if 0 <= page < pages.shape[0]:
                pages[page, cell] = value
        return pages


@pytest.mark.cpu_ok
@pytest.mark.parametrize("rank_major", [False, True])
@pytest.mark.parametrize("read_only", [False, True])
def test_host_compressed_transport_matches_independent_qjl_decoder_and_drops_invalid_writes(
    monkeypatch, rank_major, read_only
):
    """Actual compression/transport on NumPy; no JAX array computation occurs.

    Norm storage uses fp32 in this host substitute. Real bf16 layouts and the
    registered stochastic attention path are covered by the abstract traces.
    """
    from types import SimpleNamespace

    from easydel.operations import AttentionOutput, OperationRegistry
    from easydel.operations.kernels import ragged_page_attention as paged
    from ejkernel.quantization.turboquant import codebook, ops, packing

    operation = OperationRegistry.create("paged_attention_reference", _performer().metadata)

    def host_loop(start, end, body, carry):
        for row in range(start, end):
            carry = body(row, carry)
        return carry

    monkeypatch.setattr(
        paged,
        "jax",
        SimpleNamespace(
            lax=SimpleNamespace(
                fori_loop=host_loop, cond=lambda predicate, yes, no, value: (yes if predicate else no)(value)
            ),
            random=SimpleNamespace(fold_in=lambda key, row: (key, row)),
        ),
    )
    host_numpy = SimpleNamespace(**vars(np), bfloat16=np.float32)
    monkeypatch.setattr(paged, "jnp", host_numpy)
    monkeypatch.setattr(ops, "jnp", host_numpy)
    monkeypatch.setattr(packing, "jnp", host_numpy)
    monkeypatch.setattr(codebook, "jnp", host_numpy)
    monkeypatch.setattr(
        codebook,
        "jax",
        SimpleNamespace(nn=SimpleNamespace(one_hot=lambda indices, count, dtype: np.eye(count, dtype=dtype)[indices])),
    )
    constants, config = _constants(), _config()
    rng = np.random.default_rng(111)
    original_k = rng.normal(size=(4, 2, 1, 8)).astype(np.float32)
    original_v = rng.normal(size=(4, 2, 1, 8)).astype(np.float32)
    storage = (
        *ops.turboquant_compress_keys(
            original_k, constants.rotation_matrix, constants.key_codebook, constants.qjl_projection
        ),
        *ops.turboquant_compress_values(original_v, constants.rotation_matrix, constants.value_codebook),
    )
    view = TurboQuantRaggedPagesCacheView(
        metadata=config,
        layer_index=1,
        kv_pages=np.zeros((), dtype=np.float32),
        constants=constants,
        **{name: data.view(_HostPages) for name, data in zip(PAGE_FIELDS, storage, strict=True)},
    )
    tables = np.array([[2, 0], [1, -1], [1, -1], [-1, -1]], dtype=np.int32)
    contexts = np.array([3, 1, 1, 1], dtype=np.int32)
    starts = np.array([[0, 1, 1], [0, 1, 2]] if rank_major else [0, 1, 1, 2, 3], dtype=np.int32)
    runtime = RaggedPagesMetadata(tables, contexts, starts, contexts, page_size=2, version="v3")
    query = rng.normal(size=(8, 2, 8)).astype(np.float32)
    key = rng.normal(size=(8, 1, 8)).astype(np.float32)
    value = rng.normal(size=(8, 1, 8)).astype(np.float32)
    expected_storage = [np.array(array) for array in storage]
    if not read_only:
        compressed = (
            *ops.turboquant_compress_keys(
                key, constants.rotation_matrix, constants.key_codebook, constants.qjl_projection
            ),
            *ops.turboquant_compress_values(value, constants.rotation_matrix, constants.value_codebook),
        )
        for expected, data in zip(expected_storage, compressed, strict=True):
            expected[0, 0] = data[0]
            expected[1, 0] = data[4 if rank_major else 1]
    expected_keys, expected_values = _numpy_decode(*expected_storage, constants)

    def host_dense(**kwargs):
        assert kwargs["normalize_output"] is False
        assert kwargs["dropout_rng"][0] == 83 and kwargs["dropout_prob"] == 0.25
        assert kwargs["logits_soft_cap"] == 3.0
        row = kwargs["dropout_rng"][1]
        expected_k = np.zeros((4, 1, 8), dtype=np.float32)
        expected_v = np.zeros_like(expected_k)
        if row == 0:
            expected_k[:3] = np.concatenate((expected_keys[2], expected_keys[0, :1]), axis=0)
            expected_v[:3] = np.concatenate((expected_values[2], expected_values[0, :1]), axis=0)
        elif row == 2:
            expected_k[0] = expected_keys[1, 0]
            expected_v[0] = expected_values[1, 0]
        # Idle requests are not evaluated; invalid pages produce zero operands.
        assert row in (0, 2, 3)
        np.testing.assert_allclose(kwargs["key"][0], expected_k, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(kwargs["value"][0], expected_v, rtol=2e-6, atol=2e-6)
        visible = kwargs["mask_info"]._attention_mask
        assert int(visible.sum()) == {0: 3, 2: 1, 3: 0}[row]
        return AttentionOutput(attention_outputs=np.full_like(kwargs["query"], 19), attention_weights=None)

    monkeypatch.setattr(OperationRegistry, "create", lambda *args, **kwargs: host_dense)
    result = operation.forward_native(
        query=query,
        key=None if read_only else key,
        value=None if read_only else value,
        cache_view=view,
        cache_metadata=runtime,
        deterministic=False,
        normalize_output=False,
        dropout_rng=83,
        dropout_prob=0.25,
        chunk_size=3,
        sliding_window=4,
        logits_soft_cap=3.0,
    )
    expected_output = np.zeros_like(query)
    expected_output[0] = 19
    expected_output[4 if rank_major else 1] = 19
    np.testing.assert_array_equal(result.attention_outputs, expected_output)
    assert isinstance(result.cache_view, TurboQuantRaggedPagesCacheView)
    assert result.cache_view.kv_pages is view.kv_pages
    for name, expected in zip(PAGE_FIELDS, expected_storage, strict=True):
        np.testing.assert_array_equal(getattr(result.cache_view, name), expected)
    for name, original in zip(PAGE_FIELDS, storage, strict=True):
        np.testing.assert_array_equal(getattr(view, name), original)
