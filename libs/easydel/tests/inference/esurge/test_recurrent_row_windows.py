# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""eSurge row windows over plain-path recurrent pools: every request keeps its own state.

eSurge runs a step as several row windows when more requests are resident than one forward takes (``max_num_seqs``
above the cache's rows per forward, ``metadata.get_max_num_seqs()``). Each window numbers its rows from 0. The KDA layer
declares the eSurge slot contract (``runners/slot_contract.py``), so wherever a row can sit past its window-local index
(several row windows per step, sparse packing of zero-token rows) every request owns a stable pooled slot, shipped with
every step as ``cache_metadata.recurrent_state_indices``; single-window serving keeps the in-place identity layout and
ships nothing. Speculative candidate rows (laid out by the window's row count) keep the identity layout and make the
runner refuse a multi-window configuration at init.

The two-window cases cap the rows per forward at 2 (``get_max_num_seqs`` patched on the cache config classes; production
derives it as ``131072 // pages_per_request``) with 4 resident requests on a tiny GLM-5-Next. Computation tests: the
suite's conftest refuses them on a CPU backend.
"""

import gc

import easydel as ed
import easydel.layers.linear_attention._kda as kda_layer_module
import jax
import jax.numpy as jnp
import pytest
import spectrax as spx
from easydel.caching.ragged_page.cache import RaggedPagesCacheConfig
from easydel.caching.unified_attention.cache import UnifiedAttentionCacheConfig
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.runners.slot_contract import resolve_recurrent_slot_contract
from easydel.inference.sampling_params import SamplingParams

try:
    from ._common import make_core_for_runner, served_request
except ImportError:  # standalone `python test_x.py`
    from _common import make_core_for_runner, served_request

VOCAB = 128
MAX_TOKENS = 12
REQUESTS = [
    {"id": "r0", "prompt_token_ids": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]},
    {"id": "r1", "prompt_token_ids": [10, 11, 12, 13, 14]},
    {"id": "r2", "prompt_token_ids": [42, 7, 99, 5, 23, 55, 8, 13]},
    {"id": "r3", "prompt_token_ids": [21, 34, 2]},
]


def _make_tiny_glm5_next():
    """Three KDA layers and one DSA layer in f32 (the model of ``test_kda_spec_rollback.py``)."""
    config = ed.Glm5NextTextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=64,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense"] * 4,
        linear_head_dim=8,
        linear_num_heads=2,
        linear_conv_kernel_dim=4,
        hc_mult=2,
        max_position_embeddings=512,
        # Wider weights keep the greedy stream varied (a near-constant stream would hide state bugs).
        initializer_range=0.2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32
    config.kvdtype = jnp.float32
    with config.mesh:
        model = ed.Glm5NextForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model():
    return _make_tiny_glm5_next()


def _cap_rows_per_forward(monkeypatch: pytest.MonkeyPatch, rows: int) -> None:
    """Make the paged cache configs report ``rows`` request rows per forward."""
    for cls in (RaggedPagesCacheConfig, UnifiedAttentionCacheConfig):
        monkeypatch.setattr(cls, "get_max_num_seqs", lambda self, _rows=rows: _rows)


class _FlatDriverCalls:
    """Counts, at trace time, the flat KDA driver calls made with and without explicit pool slots."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.with_slots = 0
        self.without_slots = 0
        driver = kda_layer_module.kda_ragged_rows_forward

        def counted(*args, **kwargs):
            if kwargs.get("state_indices") is None:
                self.without_slots += 1
            else:
                self.with_slots += 1
            return driver(*args, **kwargs)

        monkeypatch.setattr(kda_layer_module, "kda_ragged_rows_forward", counted)

    def reset(self) -> None:
        self.with_slots = 0
        self.without_slots = 0


class _InertDrafter:
    """A drafter the runner must refuse before it ever drafts (speculative candidate rows + several row windows)."""

    num_draft_tokens = 3
    supports_return_full_log_probs = False
    supports_prefix_draft = False
    requires_target_kv_cache = False

    def reset(self, batch_size: int) -> None:
        del batch_size

    def reset_request(self, request_id: str | None = None) -> None:
        del request_id

    def draft(self, *args, **kwargs):
        raise AssertionError("the runner must refuse this configuration at init")


def _runner(model, *, async_scheduling: bool = False, drafter=None) -> eSurgeRunner:
    return eSurgeRunner(
        model=model,
        hbm_utilization=0.02,
        page_size=16,
        max_cache_tokens=2048,
        max_model_len=128,
        max_num_batched_tokens=64,
        min_input_pad=1,
        min_token_pad=16,
        max_num_seqs=len(REQUESTS),
        max_num_seq_buckets=[len(REQUESTS)],
        async_scheduling=async_scheduling,
        use_aot_forward=True,
        verbose=False,
        enable_overlap_execution=False,
        enable_sampler_metrics=False,
        drafter=drafter,
    )


def _close(runner: eSurgeRunner) -> None:
    runner.shutdown()
    runner.executor_manager.kv_pages = None
    gc.collect()


def _serve_greedy(model) -> tuple[int, dict[str, list[int]]]:
    """Serve every request greedily with all four resident at once; returns ``(rows per forward, tokens)``."""
    runner = _runner(model)
    try:
        runner.compile(max_num_batched_tokens=64)
        scheduler = make_core_for_runner(
            runner, max_num_batched_tokens=64, enable_prefix_caching=False, async_scheduling=False, num_draft_tokens=0
        )
        requests = []
        for spec in REQUESTS:
            request = served_request(
                request_id=spec["id"],
                prompt_token_ids=list(spec["prompt_token_ids"]),
                sampling_params=SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True),
                eos_token_id=None,
            )
            requests.append(request)
            scheduler.add_request(request)
        for _ in range(256):
            scheduler_output = scheduler.schedule()
            output = runner.execute_model(scheduler_output)
            scheduler.commit(scheduler_output, output)
            if all(request.is_finished() for request in requests):
                break
        else:
            raise AssertionError("generation did not finish within 256 steps")
        tokens = {request.request_id: [int(t) for t in request.output_token_ids] for request in requests}
        return int(runner.num_reqs_max_model_len), tokens
    finally:
        _close(runner)


def test_single_window_serving_ships_no_pool_slots(tiny_model, monkeypatch):
    """One window and no sparse row packing (async scheduling): the KDA layers keep the in-place identity layout.

    First in the file, so this runner traces its programs in a fresh process (the counters count traces).
    """
    calls = _FlatDriverCalls(monkeypatch)
    runner = _runner(tiny_model, async_scheduling=True)
    try:
        assert runner.num_reqs_max_model_len == runner.max_num_reqs
        runner.compile(max_num_batched_tokens=64)
    finally:
        _close(runner)
    assert calls.without_slots > 0, "no compiled program traced the flat KDA driver"
    assert calls.with_slots == 0, "single-window serving shipped recurrent_state_indices"


def test_two_row_windows_serve_like_one_window(tiny_model, monkeypatch):
    """Four resident requests in two 2-row windows per step produce the one-window greedy streams token for token."""
    calls = _FlatDriverCalls(monkeypatch)
    one_rows, one_window = _serve_greedy(tiny_model)
    assert one_rows == len(REQUESTS)
    assert all(len(tokens) == MAX_TOKENS and len(set(tokens)) > 2 for tokens in one_window.values()), one_window

    _cap_rows_per_forward(monkeypatch, 2)
    calls.reset()
    two_rows, two_windows = _serve_greedy(tiny_model)
    assert two_rows == 2, "the step did not run as two row windows"
    assert calls.with_slots > 0 and calls.without_slots == 0, "the KDA layers did not receive the rows' pool slots"
    assert two_windows == one_window


def test_speculative_candidate_rows_refuse_several_row_windows(tiny_model, monkeypatch):
    """Candidate rows are laid out by the window's row count, so a multi-window speculative config fails at init."""
    _cap_rows_per_forward(monkeypatch, 2)
    with pytest.raises(ValueError, match=r"max_num_seqs=4 exceeds rows per forward=2 .*speculative decoding"):
        _runner(tiny_model, drafter=_InertDrafter())


def test_kda_layers_declare_the_slot_contract(tiny_model):
    """Every KDA layer declares the slot contract; one undeclared recurrent layer keeps the identity layout."""
    assert resolve_recurrent_slot_contract(tiny_model, num_recurrent_layers=3) == "slots"
    assert resolve_recurrent_slot_contract(tiny_model, num_recurrent_layers=4) == "none"
