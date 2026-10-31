"""CPU-safe integrated dispatch tests: NumPy stands in for array computation.

These tests execute the real MambaMixer -> SSM1Op branch selection, replacing
only numeric primitives and backend implementations. Unsegmented non-mambapy
inputs retain the registered kernel for both associative and sequential scans;
segmented inputs select the configured reset-aware reference recurrence. Each
backend/algorithm combination emits a distinct value to expose dropped selectors.
These tests do not execute JAX arrays or validate numerical parity.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.modules.mamba import modeling_mamba
from easydel.operations.kernels import ssm1


class _IdentityConv:
    weight = np.ones((2, 1, 1), dtype=np.float32)
    use_bias = False

    def __call__(self, inputs):
        return inputs


@pytest.mark.cpu_ok
@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize(
    ("use_mambapy", "use_associative_scan", "backend_value"),
    [(False, True, 7.0), (True, True, 11.0), (True, False, 13.0), (False, False, 17.0)],
)
def test_mamba_to_operation_backend_and_scan_selection(
    monkeypatch, use_mambapy, use_associative_scan, backend_value, segmented
):
    monkeypatch.setattr(modeling_mamba, "jnp", np)
    monkeypatch.setattr(ssm1, "jnp", np)
    monkeypatch.setattr(ssm1, "normalize_packed_segment_ids", lambda ids, *args, **kwargs: ids)
    monkeypatch.setattr(
        modeling_mamba,
        "segmented_depthwise_causal_conv1d",
        lambda x, *args, **kwargs: (x, np.zeros((1, 2, 1), dtype=x.dtype)),
    )
    monkeypatch.setattr(modeling_mamba, "checkpoint_name", lambda x, **kwargs: x)
    monkeypatch.setattr(modeling_mamba, "apply_logical_sharding", lambda x, **kwargs: x)
    monkeypatch.setattr(
        modeling_mamba, "jax", SimpleNamespace(nn=SimpleNamespace(softplus=lambda x: np.logaddexp(0, x)))
    )

    def optimized(hidden_states, *args, use_associative_scan, **kwargs):
        value = 7 if use_associative_scan else 17
        return np.full_like(hidden_states, value), np.zeros((1, 2, 1), dtype=np.float32), None

    def reference(hidden_states, *args, use_associative_scan, **kwargs):
        value = 11 if use_associative_scan else 13
        return np.full_like(hidden_states, value), np.zeros((1, 2, 1), dtype=np.float32)

    monkeypatch.setattr(ssm1, "state_space_v1", optimized)
    monkeypatch.setattr(ssm1, "_reference_ssm1_fwd", reference)
    mixer = SimpleNamespace(
        config=SimpleNamespace(
            use_associative_scan=use_associative_scan, use_mambapy=use_mambapy, runtime_sharding_resolver=None
        ),
        intermediate_size=2,
        ssm_state_size=1,
        time_step_rank=1,
        in_proj=lambda x: np.concatenate((x, x), axis=-1),
        conv1d=_IdentityConv(),
        act=lambda x: x,
        x_proj=lambda x: np.ones((*x.shape[:-1], 3), dtype=x.dtype),
        dt_proj=lambda x: np.broadcast_to(x, (*x.shape[:-1], 2)),
        A_log=SimpleNamespace(value=np.zeros((2, 1), dtype=np.float32)),
        D=SimpleNamespace(value=np.ones((2,), dtype=np.float32)),
        activation="silu",
        ssm_op=lambda **kwargs: ssm1.SSM1Op.forward_native(None, **kwargs),
        out_proj=lambda x: x,
    )
    output, cache = modeling_mamba.MambaMixer.forward(
        mixer,
        np.ones((1, 3, 2), dtype=np.float32),
        segment_ids=np.asarray([[0, 1, -1]]) if segmented else None,
    )
    if segmented:
        backend_value = 11.0 if use_associative_scan else 13.0
    np.testing.assert_array_equal(output, np.full((1, 3, 2), backend_value, dtype=np.float32))
    assert cache is None
