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

"""eSurge image serving of ``DeepseekV41ForConditionalGeneration`` (computation test).

A request carries a ``uint8 [H, W, 3]`` image (``EngineRequest(..., images=[img])``) and a prompt holding one
``image_token_id`` placeholder. The scheduler expands it into the image span, the runner encodes the image eagerly and
streams the merged embeddings through ``prefill_embeds``, and feeds ``token_types`` (``bias_vl`` routing) plus engram
hash ids (image positions dead). The 70x56 px image gives a 5x4 ViT patch grid, so the aligner pads both sides to a
multiple of 3 (2x2 LLM grid, 8-token span). Every engine token must equal the argmax of the stateless
``DeepseekV41ForConditionalGeneration`` forward over the same expanded sequence, a text-only request in the same batch
must match its own text forward, and a different image must change the image request's tokens only.
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
from easydel.modules.deepseek_v41 import DeepseekV41Config, DeepseekV41ForConditionalGeneration
from easydel.modules.deepseek_v41.processing_deepseek_v41 import DeepseekV41ImageExpander

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

VOCAB = 256
IMAGE_TOKEN = VOCAB - 2
MAX_NEW_TOKENS = 12
_REFERENCE_PAD_LEN = 64
IMAGE_PROMPT = [5, 6, IMAGE_TOKEN, 7, 8, 9]
TEXT_PROMPT = [10, 11, 12, 13, 14]
_ENGRAM_KEYS = (
    "engram_layer_ids",
    "engram_vocab_size",
    "engram_n_heads",
    "engram_head_dim",
    "engram_max_ngram_size",
    "engram_compressed_vocab_size",
    "engram_pad_token_id",
)


def _make_image(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(70, 56, 3), dtype=np.uint8)


def _make_tiny_vl_model() -> DeepseekV41ForConditionalGeneration:
    engram = {k: v for k, v in ref.case_config_kwargs("engram").items() if k in _ENGRAM_KEYS}
    vision = dict(ref.vl_config_kwargs()["vision_config"])
    vision.update(min_pixels=1, max_image_tokens=64)  # keep the 70x56 image at its own size
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
        image_token_id=IMAGE_TOKEN,
        max_position_embeddings=512,
        initializer_range=0.2,
        vision_config=vision,
        **engram,
    )
    assert cfg.vision_enabled
    cfg.moe_force_xla_gmm = True
    cfg.fsdp_is_ep_bound = False
    cfg.attn_dtype = jnp.float32
    cfg.kvdtype = jnp.float32
    model = DeepseekV41ForConditionalGeneration(
        config=cfg,
        rngs=spx.Rngs(0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    model.eval()
    return model


class _ServingHarness:
    def __init__(self, model, *, long_prefill_token_threshold: int | None = None):
        self.runner = eSurgeRunner(
            model=model,
            hbm_utilization=0.02,
            page_size=16,
            max_cache_tokens=2048,
            max_model_len=128,
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
        self.scheduled: dict[str, list[int]] = {}

    def generate_greedy(self, request_specs: list[dict]) -> tuple[list[list[int]], list[EngineRequest]]:
        requests = []
        for spec in request_specs:
            req = EngineRequest(
                request_id=f"req-{self._req_counter}",
                prompt_token_ids=list(spec["prompt_token_ids"]),
                sampling_params=SamplingParams(
                    max_tokens=int(spec.get("max_tokens", MAX_NEW_TOKENS)), temperature=0.0, ignore_eos=True
                ),
                eos_token_id=None,
                images=spec.get("images"),
            )
            self._req_counter += 1
            requests.append(req)
            self.scheduler.add_request(req)
        for _ in range(64 * len(requests)):
            scheduler_output = self.scheduler.schedule()
            for rid, n in scheduler_output.num_scheduled_tokens.items():
                self.scheduled.setdefault(rid, []).append(int(n))
            output = self.runner.execute_model(scheduler_output)
            self.scheduler.update_from_output(scheduler_output, output)
            if all(r.is_finished() for r in requests):
                break
        else:
            raise AssertionError("generation did not finish within the step budget")
        outputs = [list(r.output_token_ids) for r in requests]
        for spec, toks in zip(request_specs, outputs, strict=True):
            assert len(toks) == int(spec.get("max_tokens", MAX_NEW_TOKENS))
        return outputs, requests

    def close(self) -> None:
        self.runner.shutdown()
        self.runner.executor_manager.kv_pages = None
        self.runner = None
        self.scheduler = None
        gc.collect()


def _reference_logits(model, prompt_ids, generated, image: np.ndarray | None) -> tuple[np.ndarray, int]:
    """Stateless VL-forward logits ``[PAD_LEN, vocab]`` of ``prompt + generated`` and the expanded prompt length."""
    from easydel.utils.inference_mode import set_inference_mode

    patches = grid = None
    if image is None:
        ids = list(prompt_ids)
        types = np.full((len(ids),), -1, dtype=np.int64)
    else:
        expanded = DeepseekV41ImageExpander(model.config).expand(prompt_ids, [image])
        ids, types = list(expanded.token_ids), expanded.token_types
        patches = expanded.features[0].pixel_values
        grid = tuple(int(v) for v in expanded.features[0].grid_thw.reshape(-1)[1:])
    n_prompt = len(ids)
    ids = ids + [int(t) for t in generated]
    types = np.concatenate([types, np.full((len(generated),), -1, dtype=np.int64)])
    pad = _REFERENCE_PAD_LEN - len(ids)
    ids_arr = np.asarray([ids + [0] * pad], dtype=np.int64)
    types_arr = np.asarray([[*types.tolist(), *([-1] * pad)]], dtype=np.int64)
    hasher = EngramHasher.from_config(model.config, np.arange(VOCAB, dtype=np.int64))
    hashes = hasher.hash(ids_arr, types_arr >= 0)
    kwargs = {}
    if patches is not None:
        kwargs = dict(pixel_values=jnp.asarray(patches)[None], image_grid=grid)
    with model.mesh, set_inference_mode():
        out = model(
            input_ids=jnp.asarray(ids_arr, dtype=jnp.int32),
            position_ids=jnp.arange(_REFERENCE_PAD_LEN, dtype=jnp.int32)[None, :],
            engram_hash_ids=jnp.asarray(hashes, dtype=jnp.int32),
            token_types=jnp.asarray(types_arr, dtype=jnp.int32),
            **kwargs,
        )
        return np.asarray(out.logits.astype(jnp.float32))[0], n_prompt


def _assert_matches_reference(model, prompt_ids, generated, image, *, tie_tolerance: float = 5e-4) -> None:
    logits, n_prompt = _reference_logits(model, prompt_ids, generated, image)
    assert np.isfinite(logits[: n_prompt + len(generated)]).all()
    mismatches = []
    for step, token in enumerate(generated):
        row = logits[n_prompt + step - 1]
        gap = float(row.max() - row[int(token)])
        if int(row.argmax()) != int(token) and gap > tie_tolerance:
            mismatches.append((step, int(token), int(row.argmax()), gap))
    assert not mismatches, f"eSurge diverged from the stateless VL forward (step, engine, ref, gap): {mismatches}"


def _assert_no_chunk_ends_inside_span(per_step: list[int], span: tuple[int, int]) -> None:
    done = 0
    for n in per_step:
        done += n
        assert not (span[0] < done < span[1]), f"a prefill chunk ended at {done}, inside the image span {span}"


@pytest.fixture(scope="module")
def tiny_vl_model():
    return _make_tiny_vl_model()


def test_image_changes_reference_logits(tiny_vl_model):
    """Negative control on the oracle: a different image moves the logits, so parity cannot pass by ignoring it."""
    gen = [1, 2, 3]
    a, n_prompt = _reference_logits(tiny_vl_model, IMAGE_PROMPT, gen, _make_image(1))
    b, _ = _reference_logits(tiny_vl_model, IMAGE_PROMPT, gen, _make_image(2))
    assert float(np.abs(a[n_prompt - 1 : n_prompt + 2] - b[n_prompt - 1 : n_prompt + 2]).max()) > 1e-4


@pytest.mark.parametrize("threshold", [None, 5])
def test_deepseek_v41_vl_esurge_greedy_parity_and_image_sensitivity(tiny_vl_model, threshold):
    """Image + text request in one batch; ``threshold=5`` would cut the 8-token span, forcing boundary alignment."""
    model = tiny_vl_model
    image_a, image_b = _make_image(1), _make_image(2)
    harness = _ServingHarness(model, long_prefill_token_threshold=threshold)
    try:
        assert harness.scheduler.enable_prefix_caching is False
        (img_tokens, text_tokens), (img_req, _) = harness.generate_greedy(
            [
                {"prompt_token_ids": IMAGE_PROMPT, "images": [image_a]},
                {"prompt_token_ids": TEXT_PROMPT},
            ]
        )
        assert img_req.image_spans == [(2, 10)] and len(img_req.prompt_token_ids) == 13
        _assert_no_chunk_ends_inside_span(harness.scheduled[img_req.request_id], img_req.image_spans[0])
        assert len(set(img_tokens)) > 1 and len(set(text_tokens)) > 1
        _assert_matches_reference(model, IMAGE_PROMPT, img_tokens, image_a)
        _assert_matches_reference(model, TEXT_PROMPT, text_tokens, None)

        # Negative control: another image changes the image request's tokens, the text request is unchanged.
        (img_tokens_b, text_tokens_b), _ = harness.generate_greedy(
            [
                {"prompt_token_ids": IMAGE_PROMPT, "images": [image_b]},
                {"prompt_token_ids": TEXT_PROMPT},
            ]
        )
        assert img_tokens_b != img_tokens, "a different image must change the generated tokens"
        assert text_tokens_b == text_tokens
        _assert_matches_reference(model, IMAGE_PROMPT, img_tokens_b, image_b)
    finally:
        harness.close()


def test_image_span_longer_than_token_budget_is_rejected(tiny_vl_model):
    harness = _ServingHarness(tiny_vl_model)
    try:
        harness.scheduler.max_num_scheduled_tokens = 6  # the span has 8 tokens
        req = EngineRequest(
            request_id="too-big",
            prompt_token_ids=list(IMAGE_PROMPT),
            sampling_params=SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True),
            eos_token_id=None,
            images=[_make_image(1)],
        )
        with pytest.raises(ValueError, match="image span"):
            harness.scheduler.add_request(req)
    finally:
        harness.close()
