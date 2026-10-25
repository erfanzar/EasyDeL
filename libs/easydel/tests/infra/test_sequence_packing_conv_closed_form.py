"""Closed-form packed depthwise conv vs the token-serial scan reference; requires accelerator execution."""

import jax
import numpy as np
import pytest
from easydel.infra.sequence_packing import (
    _segmented_depthwise_causal_conv1d_scan,
    segmented_depthwise_causal_conv1d,
)
from jax import numpy as jnp

CHANNELS = 6

# Hand-written segment patterns covering the window-reset rules (-1 = pad).
PATTERNS = {
    "single_segment": [0, 0, 0, 0, 0, 0, 0, 0],
    "two_segments": [0, 0, 0, 1, 1, 1, 1, 1],
    "one_token_segments": [0, 1, 2, 3, 4, 5, 6, 7],
    "short_segments": [0, 0, 1, 2, 2, 3, 3, 3],
    "leading_pad": [-1, -1, 0, 0, 0, 1, 1, 1],
    "middle_pad_same_id": [0, 0, -1, 0, 0, 0, 0, 0],
    "trailing_pad": [0, 0, 0, 1, 1, -1, -1, -1],
    "all_pad": [-1, -1, -1, -1, -1, -1, -1, -1],
    "pad_everywhere": [-1, 0, -1, 0, 0, -1, 1, -1],
    "repeated_id_non_contiguous": [3, 3, 5, 3, 3, 3, 5, 5],
    "last_token_only": [-1, -1, -1, -1, -1, -1, -1, 2],
}


def _random_patterns(seed: int, count: int, seq_len: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        ids, current = [], 0
        for _ in range(seq_len):
            draw = rng.random()
            if draw < 0.15:
                ids.append(-1)
            else:
                if draw > 0.8:
                    current += 1
                ids.append(current)
        out.append(ids)
    return out


def _inputs(seq_len: int, kernel_size: int, batch: int, dtype=jnp.float32, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    k_x, k_w, k_b, k_s = jax.random.split(key, 4)
    x = jax.random.normal(k_x, (batch, seq_len, CHANNELS), dtype=jnp.float32).astype(dtype)
    kernel = jax.random.normal(k_w, (CHANNELS, kernel_size), dtype=jnp.float32).astype(dtype)
    bias = jax.random.normal(k_b, (CHANNELS,), dtype=jnp.float32).astype(dtype)
    state = jax.random.normal(k_s, (batch, CHANNELS, kernel_size), dtype=jnp.float32).astype(dtype)
    return x, kernel, bias, state


def _run_both(x, kernel, segment_ids, **kwargs):
    closed = jax.jit(lambda *a: segmented_depthwise_causal_conv1d(*a, **kwargs))(x, kernel, segment_ids)
    ref = jax.jit(lambda *a: _segmented_depthwise_causal_conv1d_scan(*a, **kwargs))(x, kernel, segment_ids)
    return closed, ref


def _assert_parity(closed, ref, *, atol=1e-5, rtol=1e-5):
    (out, state), (ref_out, ref_state) = closed, ref
    assert out.shape == ref_out.shape and out.dtype == ref_out.dtype
    assert state.shape == ref_state.shape and state.dtype == ref_state.dtype
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(ref_out, np.float32), atol=atol, rtol=rtol)
    # The final window is a copy of input/state values: it must match bit-for-bit.
    np.testing.assert_array_equal(np.asarray(state, np.float32), np.asarray(ref_state, np.float32))


STATE_MODES = ["none", "state", "state_same_id", "state_other_id", "state_forced_reset", "no_state_with_id"]


def _state_kwargs(mode: str, state, segment_ids):
    first_valid = [next((s for s in row if s >= 0), 0) for row in np.asarray(segment_ids).tolist()]
    if mode == "none":
        return {}
    if mode == "state":
        return {"initial_state": state}
    if mode == "state_same_id":
        return {"initial_state": state, "initial_segment_id": jnp.asarray(first_valid, jnp.int32)}
    if mode == "state_other_id":
        return {"initial_state": state, "initial_segment_id": jnp.asarray(first_valid, jnp.int32) + 7}
    if mode == "state_forced_reset":
        return {"initial_state": state, "initial_segment_id": -1}
    if mode == "no_state_with_id":
        return {"initial_segment_id": jnp.asarray(first_valid, jnp.int32)}
    raise AssertionError(mode)


@pytest.mark.parametrize("kernel_size", [1, 2, 4])
@pytest.mark.parametrize("state_mode", STATE_MODES)
def test_handwritten_patterns_match_scan(kernel_size, state_mode):
    patterns = list(PATTERNS.values())
    segment_ids = jnp.asarray(patterns, dtype=jnp.int32)
    x, kernel, bias, state = _inputs(segment_ids.shape[1], kernel_size, batch=len(patterns))
    kwargs = dict(bias=bias, activation=jax.nn.silu, **_state_kwargs(state_mode, state, segment_ids))
    _assert_parity(*_run_both(x, kernel, segment_ids, **kwargs))


@pytest.mark.parametrize("seq_len", [1, 2, 3, 5, 37])
@pytest.mark.parametrize("state_mode", STATE_MODES)
def test_random_patterns_match_scan(seq_len, state_mode):
    patterns = _random_patterns(seed=seq_len, count=16, seq_len=seq_len)
    segment_ids = jnp.asarray(patterns, dtype=jnp.int32)
    x, kernel, bias, state = _inputs(seq_len, 4, batch=len(patterns), seed=seq_len)
    kwargs = dict(bias=bias, **_state_kwargs(state_mode, state, segment_ids))
    _assert_parity(*_run_both(x, kernel, segment_ids, **kwargs))


@pytest.mark.parametrize("seq_len", [0, 2])
def test_padding_or_empty_chunk_matches_scan(seq_len):
    x, kernel, bias, state = _inputs(seq_len, 4, batch=2)
    segment_ids = jnp.full((2, seq_len), -1, dtype=jnp.int32)
    closed = segmented_depthwise_causal_conv1d(x, kernel, segment_ids, bias=bias, initial_state=state)
    ref = _segmented_depthwise_causal_conv1d_scan(x, kernel, segment_ids, bias=bias, initial_state=state)
    _assert_parity(closed, ref)
    np.testing.assert_array_equal(np.asarray(closed[1]), np.asarray(state))


def test_bf16_with_output_dtype_matches_scan():
    patterns = _random_patterns(seed=11, count=8, seq_len=64)
    segment_ids = jnp.asarray(patterns, dtype=jnp.int32)
    x, kernel, bias, state = _inputs(64, 4, batch=8, dtype=jnp.bfloat16, seed=3)
    kwargs = dict(bias=bias, initial_state=state, activation=jax.nn.silu, output_dtype=jnp.bfloat16)
    # bf16 outputs: one bf16 rounding of a <=4-term fp32 sum differs from the reference by at most ~2 ulp.
    _assert_parity(*_run_both(x, kernel, segment_ids, **kwargs), atol=2e-2, rtol=2e-2)


def test_conv_weight_layout_is_accepted():
    segment_ids = jnp.asarray([PATTERNS["two_segments"]], dtype=jnp.int32)
    x, kernel, _, _ = _inputs(8, 4, batch=1)
    weight = kernel.T[:, None, :]  # [kernel_size, 1, channels] conv weight layout
    _assert_parity(*_run_both(x, weight, segment_ids))


@pytest.mark.parametrize("state_mode", ["none", "state", "state_other_id"])
def test_gradients_match_scan(state_mode):
    patterns = _random_patterns(seed=5, count=4, seq_len=19)
    segment_ids = jnp.asarray(patterns, dtype=jnp.int32)
    x, kernel, bias, state = _inputs(19, 4, batch=4, seed=9)
    extra = _state_kwargs(state_mode, state, segment_ids)
    initial_segment_id = extra.get("initial_segment_id")
    cotangent = jax.random.normal(jax.random.PRNGKey(4), x.shape, dtype=jnp.float32)
    state_cotangent = jax.random.normal(jax.random.PRNGKey(6), state.shape, dtype=jnp.float32)

    def loss(fn, x, kernel, bias, state):
        out, final_state = fn(
            x,
            kernel,
            segment_ids,
            bias=bias,
            initial_state=state if "initial_state" in extra else None,
            initial_segment_id=initial_segment_id,
            activation=jax.nn.silu,
        )
        return jnp.sum(out * cotangent) + jnp.sum(final_state * state_cotangent)

    grads = jax.jit(jax.grad(lambda *a: loss(segmented_depthwise_causal_conv1d, *a), argnums=(0, 1, 2, 3)))(
        x, kernel, bias, state
    )
    ref_grads = jax.jit(jax.grad(lambda *a: loss(_segmented_depthwise_causal_conv1d_scan, *a), argnums=(0, 1, 2, 3)))(
        x, kernel, bias, state
    )
    for got, want in zip(grads, ref_grads, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-4, rtol=1e-4)


def _sub_jaxprs(value):
    if hasattr(value, "jaxpr") and hasattr(value, "consts"):  # ClosedJaxpr
        yield value.jaxpr
    elif hasattr(value, "eqns"):  # Jaxpr
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _sub_jaxprs(item)


def _primitive_names(jaxpr) -> set[str]:
    names = set()
    for eqn in jaxpr.eqns:
        names.add(eqn.primitive.name)
        for param in eqn.params.values():
            for sub in _sub_jaxprs(param):
                names |= _primitive_names(sub)
    return names


@pytest.mark.parametrize("with_state", [False, True])
def test_closed_form_has_no_serial_loop(with_state):
    x, kernel, bias, state = _inputs(64, 4, batch=2)
    segment_ids = jnp.zeros((2, 64), dtype=jnp.int32)

    def fn(x, kernel, bias, state):
        return segmented_depthwise_causal_conv1d(
            x, kernel, segment_ids, bias=bias, initial_state=state if with_state else None
        )

    names = _primitive_names(jax.make_jaxpr(fn)(x, kernel, bias, state).jaxpr)
    assert not names & {"scan", "while"}, names
    grad_names = _primitive_names(
        jax.make_jaxpr(jax.grad(lambda *a: jnp.sum(fn(*a)[0]), argnums=(0, 1)))(x, kernel, bias, state).jaxpr
    )
    assert not grad_names & {"scan", "while"}, grad_names
    # Sanity: the reference really is the serial loop this test guards against.
    ref_names = _primitive_names(
        jax.make_jaxpr(lambda *a: _segmented_depthwise_causal_conv1d_scan(*a[:2], segment_ids))(x, kernel).jaxpr
    )
    assert "scan" in ref_names
