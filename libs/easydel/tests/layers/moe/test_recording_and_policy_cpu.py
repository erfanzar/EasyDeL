"""Pure Python/NumPy policy and recorder contract regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.layers.linears._linear_quantized import ParallelLinearQuantized
from easydel.layers.moe import ExpertLoadRecorder

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize("mark_steps", [False, True])
def test_recording_sums_genuine_calls_and_exports_the_same_counts(ep_size, mark_steps):
    recorder = ExpertLoadRecorder(ep_size=ep_size)
    recorder.add(np.array([1, 3]), 0, "prefill")
    if mark_steps:
        recorder.mark_step()
    recorder.add(np.array([5, 1]), 0, "prefill")
    np.testing.assert_array_equal(recorder.stack()[0], [6, 4])
    exported = recorder.to_arrays()["logical_count"]
    np.testing.assert_array_equal(exported.sum(axis=0)[0], [6, 4])
    assert exported.shape == (2 if mark_steps else 1, 1, 2)


def test_recording_deduplicates_only_explicit_call_replicas():
    recorder = ExpertLoadRecorder(ep_size=2)
    for call, counts in [("chunk0", [1, 3]), ("chunk1", [5, 1])]:
        for replica in [1, 0]:
            recorder.add(np.array(counts), 4, "prefill", call_id=call, replica_id=replica)
    recorder.mark_step()
    for replica in [0, 1]:
        recorder.add(np.array([2, 0]), 4, "decode", call_id="chunk0", replica_id=replica)
    np.testing.assert_array_equal(recorder.stack()[4], [8, 4])
    np.testing.assert_array_equal(recorder.to_arrays()["logical_count"][:, 0], [[6, 4], [2, 0]])
    np.testing.assert_array_equal(recorder.stack("prefill")[4], [6, 4])


def test_anonymous_layers_have_consistent_summary_and_export():
    recorder = ExpertLoadRecorder()
    recorder.add(np.array([1, 3]), None, "prefill")
    recorder.add(np.array([5, 1]), None, "prefill")
    assert recorder.summary()["n_layers"] == 2
    np.testing.assert_array_equal(recorder.to_arrays()["logical_count"][0], [[1, 3], [5, 1]])


def _policy(**overrides):
    defaults = dict(
        qmm_platform=None,
        qmm_fuse=False,
        qmm_strict_fuse=None,
        qmm_allow_dense_fallback=None,
        qmm_tpu_path=None,
        qmm_tpu_auto_xla_max_m=1024,
        qmm_policy_table=None,
        qmm_use_best_config=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize("tokens", [1, 1024, 1025, 8192])
@pytest.mark.parametrize("mode", ["affine", "nf4"])
def test_unfused_override_ignores_fused_policy_defaults(tokens, mode):
    # 'tpu' is only an input to pure Python policy lookup; no device is queried.
    actual = ParallelLinearQuantized._qmm_runtime_kwargs(_policy(), "tpu", m_tokens=tokens, quant_mode=mode)
    assert actual == {"fuse": False, "platform": "xla", "use_best_config": True}


@pytest.mark.parametrize("control", ["qmm_strict_fuse", "qmm_allow_dense_fallback", "qmm_tpu_path"])
def test_unfused_override_still_rejects_explicit_fused_controls(control):
    value = "packed" if control == "qmm_tpu_path" else True
    with pytest.raises(ValueError, match=f"{control} requires qmm_fuse=True"):
        ParallelLinearQuantized._qmm_runtime_kwargs(_policy(**{control: value}), "tpu", m_tokens=8192)
