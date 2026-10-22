"""Public execution contracts with real host metadata/wavefront orchestration and NumPy device stand-ins.

These tests do not run JAX numerics: only device transfers, compiled model/sampler functions and the device-only
logits-index adapter are replaced. BatchMetadataPreparer, PipelineMicrobatchExecutor and sampler compaction run
normally, so results cover metadata packing, microbatch splitting and manager state updates, not dispatch markers.
"""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners import execution_manager as manager_module
from easydel.inference.esurge.runners.execution_manager import ExecutionManager
from easydel.inference.esurge.runners.execution_types import (
    BackboneOutputs,
    BatchMetadata,
    MinimalDeviceState,
    ModelStepOutputs,
    StepFunctionInputs,
)
from easydel.inference.esurge.runners.executors import batch_preparer as preparer_module
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer
from easydel.inference.esurge.runners.executors.pipeline_microbatch import PipelineMicrobatchExecutor
from easydel.inference.esurge.runners.executors.sampler_executor import SamplerRuntime

pytestmark = pytest.mark.cpu_ok


def _host_put(value, sharding=None):
    """Snapshot mutable staging buffers as a real device transfer would, without touching a JAX device."""
    if isinstance(value, tuple):
        return tuple(_host_put(v) for v in value)
    return np.array(value, copy=True)


class _HostModelExecutor:
    """Deterministic compiled-executable stand-in: metadata visibly affects tokens and cache state."""

    pipeline_plan = SimpleNamespace(is_enabled=True, stage_meshes=(None, None))
    pipeline_runtime = object()
    supports_pipeline_model_step = False
    backbone_error = None

    def has_backbone(self, num_tokens, *, use_pipeline_runtime=True):
        if self.backbone_error:
            raise self.backbone_error
        return True

    def has_lm_head(self, padded_num_reqs):
        return True

    def last_pipeline_stats(self):
        return {}

    def _uses_mpmd_mesh(self):
        return True

    @staticmethod
    def _backbone(state, other, cache, metadata):
        hidden = metadata.input_ids_buf.astype(np.float32).reshape(-1, 1)
        if metadata.model_side_inputs:
            hidden += metadata.model_side_inputs["token_types"].reshape(-1, 1)
        if metadata.dp_recurrent_state_indices is not None:
            hidden += metadata.dp_recurrent_state_indices.reshape(-1, 1)
        if metadata.spec_recurrent_commit is not None:
            hidden += metadata.spec_recurrent_commit.sum(axis=0).reshape(-1, 1)
        return BackboneOutputs(kv_pages=cache + int(metadata.scheduled.sum()), hidden_states=hidden)

    def get_backbone(self, **kwargs):
        return self._backbone

    def execute_backbones_many(self, *, num_tokens, input_batches):
        outputs = []
        cache = input_batches[0][2]
        for state, other, _, metadata in input_batches:
            output = self._backbone(state, other, cache, metadata)
            outputs.append(output)
            cache = output.kv_pages
        return outputs

    @staticmethod
    def _logits(hidden):
        return -((hidden - np.arange(128, dtype=np.float32)) ** 2)

    def get_pipeline_lm_head(self, *, padded_num_reqs, part_rows):
        def project(state, other, hidden_parts, index_parts):
            rows = np.concatenate([hidden[indices] for hidden, indices in zip(hidden_parts, index_parts, strict=True)])
            return self._logits(rows)

        return project

    def get_compiled(self, **kwargs):
        def forward(state, other, cache, metadata):
            out = self._backbone(state, other, cache, metadata)
            return ModelStepOutputs(
                out.kv_pages, out.hidden_states, self._logits(out.hidden_states[metadata.logits_indices])
            )

        return forward

    def runtime_graph_args(self):
        return None, None


class _HostSamplerRuntime(SamplerRuntime):
    """Use the real host compaction algorithm, replacing only the compiled sampler boundary."""

    def __init__(self):
        self._sampler_min_input_pad = 1
        self._sampler_prefix_cpu_by_reqs = {}
        self._sampler_token_counts = None
        for name in (
            "gather_positions",
            "sampling_seeds",
            "scatter_positions",
            "window_row_indices",
            "scheduled",
            "seq_lens",
            "active_mask",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalties",
            "presence_penalties",
            "repetition_penalties",
        ):
            dtype = (
                bool
                if name == "active_mask"
                else np.float32
                if "penalties" in name or name in {"temperature", "top_p", "min_p"}
                else np.int32
            )
            setattr(self, f"_sampler_{name}_cpu", np.zeros(4, dtype=dtype))

    def sample_tokens(self, num_tokens, padded_num_reqs, *, logits, rng_key, **kwargs):
        return rng_key + 1, logits.argmax(axis=1).astype(np.int32), np.ones(padded_num_reqs, dtype=bool), None


def _manager(monkeypatch, *, slot_indexed=False, enable_spec=False):
    host_jax = SimpleNamespace(device_put=_host_put, Array=type("_DeviceArray", (), {}))
    monkeypatch.setattr(preparer_module, "jax", host_jax)
    monkeypatch.setattr(preparer_module, "host_payload_broadcast_needed", lambda host_payloads_replicated=None: False)
    monkeypatch.setattr(manager_module, "jax", host_jax)
    # This adapter only places fixed device indices; NumPy is sufficient for host control-flow coverage.
    monkeypatch.setattr(
        PipelineMicrobatchExecutor, "_get_pipeline_logits_indices", lambda self, count, like: np.arange(count)
    )
    metadata = SimpleNamespace(version="v3", data_parallel_size=1, max_num_pages_per_req=2, get_max_num_seqs=lambda: 4)
    manager = object.__new__(ExecutionManager)
    manager._verbose = manager._debug_mode = False
    manager._slot_indexed_state = slot_indexed
    manager._batch_preparer = BatchMetadataPreparer(
        metadata=metadata,
        empty_sharding=None,
        max_num_tokens=4,
        max_num_reqs=4,
        max_model_len=8,
        min_input_pad=1,
        slot_indexed_state=slot_indexed,
        enable_spec_recurrent_commit=enable_spec,
    )
    manager.model = SimpleNamespace(mesh=SimpleNamespace(is_mpmd=True))
    manager._model_executor = _HostModelExecutor()
    manager._sampler_executor = SimpleNamespace(cache_key=lambda **kwargs: tuple(kwargs.items()), has=lambda key: True)
    manager.sampler_runtime = _HostSamplerRuntime()
    manager._pp_microbatch = PipelineMicrobatchExecutor(manager)
    manager.pp_microbatch_count = manager.pp_microbatch_size = "auto"
    manager.min_input_pad = 1
    manager._model_num_tokens_paddings = (2, 4)
    manager.graphstate = manager.graphother = None
    manager._req_num_tokens_placeholder = np.zeros(4, dtype=np.int32)
    manager._empty_sharding = None
    manager.kv_pages = 0
    manager.rng_key = 0
    return manager


def _execute(manager, **overrides):
    zero = np.zeros(4, dtype=np.int32)
    kwargs = dict(
        num_tokens=4,
        scheduled_full_cpu=np.ones(4, dtype=np.int32),
        req_num_tokens_full_cpu=np.full(4, 2, dtype=np.int32),
        active_mask_full_cpu=np.ones(4, dtype=bool),
        window_row_indices_cpu=np.arange(4, dtype=np.int32),
        # The production runner always passes this allocated array, even for attention-only models.
        recurrent_slot_indices_cpu=np.array([3, 0, 2, 1], dtype=np.int32),
        input_ids_buf=None,
        position_ids_buf=None,
        padded_num_reqs=4,
        token_ids_cpu=np.array([[0, 11, 0], [0, 22, 0], [0, 33, 0], [0, 44, 0]], dtype=np.int32),
        num_computed_tokens_cpu=np.ones(4, dtype=np.int32),
        temperature_cpu=np.ones(4, dtype=np.float32),
        top_p_cpu=np.ones(4, dtype=np.float32),
        top_k_cpu=zero,
        min_p_cpu=zero,
        frequency_penalties_cpu=zero,
        presence_penalties_cpu=zero,
        repetition_penalties_cpu=np.ones(4, dtype=np.float32),
        page_table_cpu=np.ones((4, 2), dtype=np.int32),
        wait_for_outputs=False,
    )
    kwargs.update(overrides)
    return manager.execute(**kwargs)


@pytest.mark.parametrize("slots", [None, np.zeros(4, dtype=np.int32), np.array([3, 0, 2, 1], dtype=np.int32)])
def test_plain_decode_preserves_wavefront_outputs_with_runner_bookkeeping(monkeypatch, slots):
    manager = _manager(monkeypatch)
    tokens, valid, input_ids, positions, hidden, logits, metrics = _execute(manager, recurrent_slot_indices_cpu=slots)
    np.testing.assert_array_equal(tokens, [11, 22, 33, 44])
    np.testing.assert_array_equal(valid, [True] * 4)
    np.testing.assert_array_equal(logits.argmax(axis=1), tokens)
    np.testing.assert_array_equal(input_ids, [33, 44])
    np.testing.assert_array_equal(positions, [1, 1])
    np.testing.assert_array_equal(hidden[:, 0], [33, 44])
    assert manager.kv_pages == 4 and manager.rng_key == 1
    assert metrics["pp_microbatches"] == 2


@pytest.mark.parametrize("kind", ["side_inputs", "commit", "zero_commit", "slots"])
def test_unsupported_metadata_survives_full_window_execution(monkeypatch, kind):
    manager = _manager(monkeypatch, slot_indexed=kind == "slots", enable_spec=kind in {"commit", "zero_commit"})
    overrides = {}
    delta = np.zeros(4, dtype=np.int32)
    if kind == "side_inputs":
        delta = np.array([1, 2, 3, 4], dtype=np.int32)
        overrides["model_side_inputs_cpu"] = {"token_types": delta}
    elif kind in {"commit", "zero_commit"}:
        value = 1 if kind == "commit" else 0
        overrides["spec_recurrent_commit_cpu"] = np.full((2, 4), value, dtype=np.int32)
        delta.fill(2 * value)
    else:
        delta = np.array([3, 0, 2, 1], dtype=np.int32)
    tokens, valid, input_ids, positions, hidden, logits, metrics = _execute(manager, **overrides)
    expected = np.array([11, 22, 33, 44]) + delta
    np.testing.assert_array_equal(tokens, expected)
    np.testing.assert_array_equal(valid, [True] * 4)
    np.testing.assert_array_equal(hidden[:, 0], expected)
    np.testing.assert_array_equal(logits.argmax(axis=1), expected)
    np.testing.assert_array_equal(input_ids, [11, 22, 33, 44])
    np.testing.assert_array_equal(positions, [1] * 4)
    assert manager.kv_pages == 4 and manager.rng_key == 1
    assert "pp_microbatches" not in metrics


def test_public_prefill_propagates_backbone_lookup_errors(monkeypatch):
    manager = _manager(monkeypatch, slot_indexed=True)
    manager._model_executor.backbone_error = TypeError("internal bug")
    with pytest.raises(TypeError, match="internal bug"):
        _execute(manager, req_num_tokens_full_cpu=np.full(4, 8, dtype=np.int32))
    assert manager.kv_pages == 0 and manager.rng_key == 0


def test_public_prefill_selects_direct_backbone_for_multi_token_rows(monkeypatch):
    manager = _manager(monkeypatch, slot_indexed=True)

    def has_backbone(num_tokens, *, use_pipeline_runtime=True):
        # Only the direct variant exists. A lost flag produces a public missing-bucket failure.
        return not use_pipeline_runtime

    manager._model_executor.has_backbone = has_backbone
    tokens, valid, _, _, hidden, logits, metrics = _execute(
        manager,
        scheduled_full_cpu=np.array([2, 2, 0, 0], dtype=np.int32),
        active_mask_full_cpu=np.array([True, True, False, False]),
        req_num_tokens_full_cpu=np.full(4, 8, dtype=np.int32),
    )
    np.testing.assert_array_equal(tokens, [-1] * 4)
    np.testing.assert_array_equal(valid, [False] * 4)
    assert hidden.shape == (4, 1) and logits is None
    assert metrics["prefill_only"] == 1 and manager.kv_pages == 4


def test_minimal_device_state_preserves_fields_immutability_and_pytree_roundtrip():
    tokens = np.array([[1, 2, 0], [3, 4, 5]], dtype=np.int32)
    counts = np.array([2, 3], dtype=np.int32)
    state = MinimalDeviceState(token_ids=tokens, num_tokens=counts)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    assert len(leaves) == 2
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(restored.token_ids, tokens)
    np.testing.assert_array_equal(restored.num_tokens, counts)
    updated = replace(state, num_tokens=counts + 1)
    np.testing.assert_array_equal(updated.num_tokens, [3, 4])
    np.testing.assert_array_equal(state.num_tokens, [2, 3])
    with pytest.raises(FrozenInstanceError):
        state.num_tokens = counts + 1


def test_print_status_reports_shapes(capsys):
    metadata = BatchMetadata(
        packed_qsl_seqlens=np.zeros((2, 5), dtype=np.int32),
        packed_i32_padded=np.zeros((3, 4), dtype=np.int32),
        packed_f32_padded=np.zeros((6, 4), dtype=np.float32),
        packed_misc_i32=np.zeros((5,), dtype=np.int32),
        pages_tables=np.zeros((4, 2), dtype=np.int32),
        input_ids_buf=np.zeros((8,), dtype=np.int32),
        position_ids_buf=np.zeros((8,), dtype=np.int32),
    )
    view = SimpleNamespace(kv_pages=np.zeros((3, 2, 2, 2)))
    inputs = StepFunctionInputs(
        kv_pages=SimpleNamespace(views=[view]),
        scheduled_full=np.zeros((4,), dtype=np.int32),
        req_num_tokens_full=np.zeros((4,), dtype=np.int32),
        active_mask_full=np.zeros((4,), dtype=bool),
        rng_key=np.zeros((2,), dtype=np.uint32),
        batch_metadata=metadata,
    )
    inputs.print_status()
    out = capsys.readouterr().out
    assert "StepFunctionInputs Status" in out
    assert "scheduled_full:" in out and "(4,)" in out
