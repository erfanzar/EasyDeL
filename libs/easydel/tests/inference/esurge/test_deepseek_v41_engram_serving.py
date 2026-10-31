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

"""eSurge serving of DeepSeek-V4.1 WITH engram layers (computation test).

Same harness as ``test_deepseek_v41_serving.py`` (real ``eSurgeRunner`` + ``Scheduler``, tiny fp32 / HIGHEST model) but
the config carries engram layers (the engram fields of ``tests/modules/_deepseek_v41_reference.case_config_kwargs
("engram")``). The model raises without ``engram_hash_ids``, so this only passes when eSurge computes the n-gram hash
ids of every scheduled token (prefill chunks and decode tokens) and feeds them to the model. Every engine token must
equal the argmax of a teacher-forced stateless forward fed host-computed hash ids (``EngramHasher.hash`` over the whole
sequence). ``index_topk`` / the candidate budget cover every reachable entry (non-truncating selection), as in the text
serving test.
"""

from __future__ import annotations

import gc
import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.layers.engram import EngramHasher
from easydel.modules.deepseek_v41 import DeepseekV41Config, DeepseekV41ForCausalLM

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

VOCAB = 256
MAX_NEW_TOKENS = 16
_REFERENCE_PAD_LEN = 64
TEXT_PROMPTS = (
    [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
    [10, 11, 12, 13, 14],
)
_ENGRAM_KEYS = (
    "engram_layer_ids",
    "engram_vocab_size",
    "engram_n_heads",
    "engram_head_dim",
    "engram_max_ngram_size",
    "engram_compressed_vocab_size",
    "engram_pad_token_id",
)


def _make_tiny_v41_with_engram(**overrides) -> DeepseekV41ForCausalLM:
    engram = {k: v for k, v in ref.case_config_kwargs("engram").items() if k in _ENGRAM_KEYS}
    assert engram["engram_compressed_vocab_size"] == VOCAB
    cfg = DeepseekV41Config(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=6,
        num_nextn_predict_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=16,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        compress_ratios=[0, 2, 2, 1, 1, 1, 0],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 5],
        candidate_source_layer_id=3,
        candidate_topk_blocks=64,
        candidate_block_size=2,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=128,
        hc_mult=2,
        hc_sinkhorn_iters=5,
        sliding_window=8,
        rms_norm_eps=1e-6,
        dspark_target_layer_ids=[3, 4, 5],
        dspark_noise_token_id=VOCAB - 1,
        image_token_id=VOCAB - 2,
        max_position_embeddings=512,
        initializer_range=0.2,
        **engram,
        **overrides,
    )
    cfg.moe_force_xla_gmm = True
    cfg.fsdp_is_ep_bound = False
    cfg.attn_dtype = jnp.float32
    cfg.kvdtype = jnp.float32
    model = DeepseekV41ForCausalLM(
        config=cfg,
        rngs=spx.Rngs(0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    model.eval()
    return model


class _ServingHarness:
    def __init__(self, model, *, long_prefill_token_threshold: int | None = None, max_model_len: int = 128):
        self.runner = eSurgeRunner(
            model=model,
            hbm_utilization=0.02,
            page_size=16,
            max_cache_tokens=2048,
            max_model_len=max_model_len,
            max_num_batched_tokens=64,
            min_input_pad=1,
            min_token_pad=16,
            max_num_seqs=2,
            max_num_seq_buckets=[2],
            async_scheduling=False,
            use_aot_forward=False,
            verbose=False,
            enable_overlap_execution=False,
            enable_sampler_metrics=False,
        )
        self.runner.compile(max_num_batched_tokens=64)
        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=64,
            enable_prefix_caching=True,  # the compressed-window guard must force this off
            async_scheduling=False,
            long_prefill_token_threshold=long_prefill_token_threshold,
        )
        self._req_counter = 0

    def generate_greedy(self, request_specs: list[dict]) -> list[list[int]]:
        requests = []
        for spec in request_specs:
            req = EngineRequest(
                request_id=f"req-{self._req_counter}",
                prompt_token_ids=list(spec["prompt_token_ids"]),
                sampling_params=SamplingParams(
                    max_tokens=int(spec.get("max_tokens", MAX_NEW_TOKENS)), temperature=0.0, ignore_eos=True
                ),
                eos_token_id=None,
            )
            self._req_counter += 1
            requests.append(req)
            self.scheduler.add_request(req)
        for _ in range(64 * len(requests)):
            scheduler_output = self.scheduler.schedule()
            output = self.runner.execute_model(scheduler_output)
            self.scheduler.update_from_output(scheduler_output, output)
            if all(r.is_finished() for r in requests):
                break
        else:
            raise AssertionError("generation did not finish within the step budget")
        outputs = [list(r.output_token_ids) for r in requests]
        for spec, toks in zip(request_specs, outputs, strict=True):
            assert len(toks) == int(spec.get("max_tokens", MAX_NEW_TOKENS))
        return outputs

    def close(self) -> None:
        self.runner.shutdown()
        self.runner.executor_manager.kv_pages = None
        self.runner = None
        self.scheduler = None
        gc.collect()


def _stateless_logits(model, ids: list[int], *, hash_ids: np.ndarray | None = None) -> np.ndarray:
    """Logits ``[PAD_LEN, vocab]`` of the stateless forward with host-computed engram hash ids."""
    from easydel.utils.inference_mode import set_inference_mode

    padded = list(ids) + [0] * (_REFERENCE_PAD_LEN - len(ids))
    arr = np.asarray([padded], dtype=np.int64)
    if hash_ids is None:
        hash_ids = EngramHasher.from_config(model.config, np.arange(VOCAB, dtype=np.int64)).hash(arr)
    with model.mesh, set_inference_mode():
        out = model(
            input_ids=jnp.asarray(arr, dtype=jnp.int32),
            position_ids=jnp.arange(_REFERENCE_PAD_LEN, dtype=jnp.int32)[None, :],
            engram_hash_ids=jnp.asarray(hash_ids, dtype=jnp.int32),
        )
        return np.asarray(out.logits.astype(jnp.float32))[0]


def _assert_matches_stateless_reference(model, prompt_ids, generated_ids, *, tie_tolerance: float = 5e-4) -> None:
    full = [*prompt_ids, *generated_ids]
    logits = _stateless_logits(model, full)
    assert np.isfinite(logits[: len(full)]).all()
    n_prompt = len(prompt_ids)
    mismatches = []
    for step, token in enumerate(generated_ids):
        row = logits[n_prompt + step - 1]
        gap = float(row.max() - row[int(token)])
        if int(row.argmax()) != int(token) and gap > tie_tolerance:
            mismatches.append((step, int(token), int(row.argmax()), gap))
    assert not mismatches, f"eSurge diverged from the stateless engram forward (step, engine, ref, gap): {mismatches}"


@pytest.fixture(scope="module")
def tiny_v41_engram_model():
    return _make_tiny_v41_with_engram()


def test_engram_hash_ids_change_the_reference_logits(tiny_v41_engram_model):
    """Negative control: serving without hash ids (or with wrong ones) could not match the reference."""
    model = tiny_v41_engram_model
    ids = [*TEXT_PROMPTS[0], 7, 8, 9]
    real = _stateless_logits(model, ids)
    zeros = _stateless_logits(model, ids, hash_ids=np.zeros((1, _REFERENCE_PAD_LEN, 2, 4), dtype=np.int32))
    assert float(np.abs(real[: len(ids)] - zeros[: len(ids)]).max()) > 1e-5


def test_deepseek_v41_engram_esurge_greedy_parity_and_slot_lifecycle(tiny_v41_engram_model):
    model = tiny_v41_engram_model
    harness = _ServingHarness(model)
    try:
        served = harness.generate_greedy([{"prompt_token_ids": p} for p in TEXT_PROMPTS])
        assert harness.runner._side_input_provider is not None  # the side-input path actually ran
        assert len({tuple(s) for s in served}) == 2 and all(len(set(s)) > 1 for s in served), served
        for prompt, generated in zip(TEXT_PROMPTS, served, strict=True):
            _assert_matches_stateless_reference(model, list(prompt), generated)

        skew = [
            {"prompt_token_ids": TEXT_PROMPTS[0], "max_tokens": 6},
            {"prompt_token_ids": TEXT_PROMPTS[1], "max_tokens": 18},
        ]
        skew_served = harness.generate_greedy(skew)
        for spec, generated in zip(skew, skew_served, strict=True):
            _assert_matches_stateless_reference(model, list(spec["prompt_token_ids"]), generated)
        assert skew_served[1][:MAX_NEW_TOKENS] == served[1]

        wave_prompts = ([42, 7, 99, 5], [23, 55, 8, 13, 21, 34, 2])  # reuse the slots freed by the waves above
        wave_served = harness.generate_greedy([{"prompt_token_ids": list(p)} for p in wave_prompts])
        for prompt, generated in zip(wave_prompts, wave_served, strict=True):
            _assert_matches_stateless_reference(model, list(prompt), generated)

        repeated = harness.generate_greedy([{"prompt_token_ids": p} for p in TEXT_PROMPTS])
        assert repeated == served
    finally:
        harness.close()


def test_deepseek_v41_engram_esurge_chunked_prefill_parity(tiny_v41_engram_model):
    """3-token chunks cut every n-gram window across scheduler steps; hash ids must resume from the history."""
    model = tiny_v41_engram_model
    harness = _ServingHarness(model, long_prefill_token_threshold=3)
    try:
        served = harness.generate_greedy([{"prompt_token_ids": p} for p in TEXT_PROMPTS])
        for prompt, generated in zip(TEXT_PROMPTS, served, strict=True):
            _assert_matches_stateless_reference(model, list(prompt), generated)
    finally:
        harness.close()


_CED_PROMPTS = (
    [3 + i for i in range(44)],
    [3 + (7 * i) % 250 for i in range(40)],
)


def test_ced_compact_plan_maps_live_suffixes():
    """Live tokens (a suffix of each request's chunk) become a packed stream at their real positions; each slot's
    grid columns shift by its dead prefix."""
    from easydel.modules.deepseek_v41 import modeling_deepseek_v41 as mod

    # Row 0 (slot 1): tokens 0-4, live from token 3. Row 1 (slot 0): tokens 5-7, all live. Tokens 8-9 are padding.
    live = jnp.asarray([0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    positions = jnp.asarray([[100, 101, 102, 103, 104, 7, 8, 9, 0, 0]], dtype=jnp.int32)
    plan = mod._ced_compact_plan(
        live,
        query_start_loc=jnp.asarray([0, 5, 8, 8], dtype=jnp.int32),
        num_seqs=jnp.asarray([2], dtype=jnp.int32),
        recurrent_state_indices=jnp.asarray([1, 0, 2], dtype=jnp.int32),
        position_ids=positions,
        num_slots=3,
        length=6,
    )
    assert np.asarray(plan.token_index).tolist() == [3, 4, 5, 6, 7, 10]
    assert np.asarray(plan.col_offset).tolist() == [0, 3, 0]
    valid = np.asarray(plan.meta.valid_grid)
    pos = np.asarray(plan.meta.positions_grid)
    assert valid[1, :2].all() and not valid[1, 2:].any() and pos[1, :2].tolist() == [103, 104]
    assert valid[0, :3].all() and not valid[0, 3:].any() and pos[0, :3].tolist() == [7, 8, 9]
    assert not valid[2].any()
    grid = jnp.arange(3 * 6).reshape(3, 6)
    shifted = np.asarray(mod._ced_shift_grid(grid, plan.col_offset))
    assert shifted[1, :2].tolist() == [9, 10] and shifted[0, :3].tolist() == [0, 1, 2]


def test_ced_prefill_skip_parity_and_negative_control(tiny_v41_engram_model, monkeypatch):
    """Prompts far longer than the decoder tail's reach, in 3-token chunks: steps whose tokens are all outside the
    reach skip the window-only tail, steps with at most ``num_slots`` live tokens (every decode step: 2 slots in a
    16-token bucket, and dead chunks next to a decoding request) run it on the compacted live tokens, the rest run it
    in full; greedy serving still equals the stateless forward. Negative control: an artificially short reach drops
    tokens the tail needs, and serving must diverge (the CED path really runs)."""
    from easydel.inference.esurge.multimodal.side_inputs import TOKENS_TO_END, model_side_input_specs
    from easydel.modules.deepseek_v41 import modeling_deepseek_v41 as mod

    model = tiny_v41_engram_model
    assert TOKENS_TO_END in model_side_input_specs(model)
    start, reach = mod._ced_decoder_tail(model.config)
    assert (start, reach) == (4, 15), (start, reach)
    specs = [{"prompt_token_ids": p} for p in _CED_PROMPTS]
    assert all(len(p) > 2 * reach for p in _CED_PROMPTS)

    harness = _ServingHarness(model, long_prefill_token_threshold=3)
    try:
        served = harness.generate_greedy(specs)
    finally:
        harness.close()
    for prompt, generated in zip(_CED_PROMPTS, served, strict=True):
        _assert_matches_stateless_reference(model, list(prompt), generated)

    # Different max_model_len -> different cache shapes -> a fresh trace that sees the patched reach.
    monkeypatch.setattr(mod, "_ced_decoder_tail", lambda config: (start, 1))
    harness = _ServingHarness(model, long_prefill_token_threshold=3, max_model_len=96)
    try:
        broken = harness.generate_greedy(specs)
    finally:
        harness.close()
    with pytest.raises(AssertionError):
        for prompt, generated in zip(_CED_PROMPTS, broken, strict=True):
            _assert_matches_stateless_reference(model, list(prompt), generated)


@pytest.fixture(scope="module")
def tiny_v41_host_engram_model():
    """The same tiny model with fp8 engram tables in host RAM (the real-checkpoint serving layout)."""
    import ml_dtypes

    device = jax.devices()[0]
    if "pinned_host" not in {memory.kind for memory in device.addressable_memories()}:
        pytest.skip(f"backend {device.platform!r} exposes no pinned_host memory")
    model = _make_tiny_v41_with_engram(engram_table_dtype="fp8", engram_table_memory_kind="pinned_host")
    rng = np.random.default_rng(7)
    head_dim = int(model.config.engram_head_dim)
    installed = 0
    for layer in model.model.layers:
        if layer.engram is None:
            continue
        rows = int(layer.engram.table_rows)
        weight = (rng.standard_normal((rows, head_dim)) * 0.5).astype(ml_dtypes.float8_e4m3fn)
        scale = rng.integers(125, 130, size=(rows, head_dim // 32), dtype=np.uint8)
        layer.engram.set_table(weight, scale)
        assert layer.engram.uses_host_table
        installed += 1
    assert installed == len(model.config.engram_layer_ids)
    return model


def test_host_gathered_engram_serving_parity_without_in_program_lookup(tiny_v41_host_engram_model, monkeypatch):
    """Host-resident fp8 tables: eSurge gathers the rows while preparing each step (generic side-input provider), so
    the compiled serving program never traces the in-program host lookup (``pure_callback``); greedy tokens still
    equal the stateless forward, which does use that lookup. 3-token chunks cover chunked prefill and decode."""
    from easydel.inference.esurge.multimodal.side_inputs import model_side_input_specs
    from easydel.layers.engram import GatedEngram

    model = tiny_v41_host_engram_model
    specs = model_side_input_specs(model)
    assert {"engram_rows", "engram_scales"} <= set(specs) and "engram_hash_ids" not in specs, specs

    def _forbidden(self, ids):
        raise AssertionError("the in-program host engram lookup was traced in the eSurge serving program")

    with monkeypatch.context() as patch:
        patch.setattr(GatedEngram, "_host_gather_rows", _forbidden)
        harness = _ServingHarness(model, long_prefill_token_threshold=3)
        try:
            served = harness.generate_greedy([{"prompt_token_ids": p} for p in TEXT_PROMPTS])
            assert harness.runner._side_input_provider is not None
        finally:
            harness.close()
    assert all(len(set(s)) > 1 for s in served), served
    for prompt, generated in zip(TEXT_PROMPTS, served, strict=True):
        _assert_matches_stateless_reference(model, list(prompt), generated)
