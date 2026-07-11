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
# See the License for the specific language governing permissions and
# limitations under the License.

"""eSurge serving parity tests for the new VLM families (CPU, tiny models).

Covers ``paligemma``, ``internvl``, ``hunyuan_vl``, and ``minimax_m3_vl``
end to end through the real ``eSurgeRunner`` + ``Scheduler`` (the same
harness pattern as ``test_spec_decode.py``): continuous-batching prefill and
decode over the paged KV cache with greedy sampling.

Correctness bar per family:
    * eSurge greedy decode over N tokens equals a teacher-forced *uncached*
      dense forward continuation (argmax at each step), batch >= 2.
    * Text-only requests through the vision-language task head.
    * Multimodal requests (image features + placeholder scatter at prefill)
      where the engine path is exercisable on CPU with a tiny tower.
    * Deterministic repeat on the same engine instance.

Notes:
    * KV pages default to ``bfloat16`` storage (``kvdtype = attn_dtype``);
      the tiny-model parity bar requires exact numerics, so the test configs
      pin ``attn_dtype``/``kvdtype`` to float32. Production serving keeps the
      bf16 default.
    * PaliGemma's prefix-bidirectional attention (image + text prefix attend
      bidirectionally) is driven by ``token_type_ids``, which the eSurge
      compiled step never passes — the paged prefill is causal-only, matching
      how gemma3's bidirectional image blocks serve today. The multimodal
      reference below therefore runs the dense forward without
      ``token_type_ids``; bidirectional-prefix serving is deferred (engine
      mask plumbing, not a family-level gap).
    * ``minimax_m3_vl``'s ``minimax_m3_sparse`` layer-type variant has no
      decode cache by design (``init_cache`` raises; covered in
      ``tests/modules/spmd/test_minimax_m3_vl.py``); only the default
      full-attention configuration serves.
"""

from __future__ import annotations

import gc
import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import easydel as ed
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams

VOCAB = 256
MAX_NEW_TOKENS = 8

TEXT_PROMPTS = (
    [3, 1, 4, 1, 5, 9, 2, 6],
    [10, 11, 12, 13, 14],
)


def _pin_exact_serving_dtypes(cfg) -> None:
    """Pin fp32 attention/KV dtypes so paged decode is bit-comparable to dense.

    Args:
        cfg: Composite model config; the pin is applied to it, its text
            config, and its vision config (when present).
    """
    for sub in (cfg, cfg.get_text_config(), getattr(cfg, "vision_config", None)):
        if sub is not None:
            sub.attn_dtype = jnp.float32
            sub.kvdtype = jnp.float32


def _build_model(module_name: str, cfg, task=ed.TaskType.IMAGE_TEXT_TO_TEXT):
    """Construct a tiny random-init EasyDeL module for serving tests.

    Args:
        module_name: Registry model-type name.
        cfg: Model configuration (dtypes are pinned for exact parity).
        task: Registry task the module is registered under.

    Returns:
        The eval-mode model instance.
    """
    _pin_exact_serving_dtypes(cfg)
    _, module_class = ed.get_modules_by_type(module_name, task)
    model = module_class(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    model.eval()
    return model


class _ServingHarness:
    """One compiled eSurgeRunner + Scheduler serving several greedy batches."""

    def __init__(self, model):
        self.model = model
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
            enable_prefix_caching=False,
            async_scheduling=False,
        )
        self._req_counter = 0

    def generate_greedy(self, request_specs: list[dict], max_new_tokens: int = MAX_NEW_TOKENS) -> list[list[int]]:
        """Serve one batch of requests to completion with greedy sampling.

        Args:
            request_specs: Per-request dicts with ``prompt_token_ids`` and
                optional ``pixel_values`` / ``image_grid_thw``.
            max_new_tokens: Number of tokens to decode per request.

        Returns:
            Per-request generated token-id lists (prompt excluded).
        """
        requests = []
        for spec in request_specs:
            req = EngineRequest(
                request_id=f"req-{self._req_counter}",
                prompt_token_ids=list(spec["prompt_token_ids"]),
                sampling_params=SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=True),
                eos_token_id=None,
                pixel_values=spec.get("pixel_values"),
                image_grid_thw=spec.get("image_grid_thw"),
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
        for toks in outputs:
            assert len(toks) == max_new_tokens, f"expected {max_new_tokens} tokens, got {len(toks)}"
            assert all(0 <= t < VOCAB for t in toks), f"token ids out of range: {toks}"
        return outputs

    def close(self) -> None:
        """Release the compiled runner and its KV pages."""
        self.runner.shutdown()
        self.runner.executor_manager.kv_pages = None
        self.runner = None
        self.scheduler = None
        gc.collect()


_REFERENCE_PAD_LEN = 48


def _greedy_teacher_forced(model, prompt_ids: list[int], n: int, **mm_kwargs) -> list[int]:
    """Greedy continuation from repeated uncached dense forwards.

    Each step right-pads the running sequence to a fixed length so the dense
    forward compiles once per family instead of once per length. All serving
    reference paths here are causal (PaliGemma's bidirectional prefix needs
    ``token_type_ids``, which serving never passes), so trailing pad tokens
    cannot influence logits at earlier positions.

    Args:
        model: Model whose ``__call__`` returns logits.
        prompt_ids: Prompt token ids.
        n: Number of tokens to generate.
        **mm_kwargs: Optional dense-forward multimodal inputs
            (``pixel_values``, ``image_grid_thw``, ...).

    Returns:
        The ``n`` greedily selected continuation token ids.
    """
    cur = list(prompt_ids)
    assert len(cur) + n <= _REFERENCE_PAD_LEN, "reference pad length too small for this prompt"
    toks: list[int] = []
    for _ in range(n):
        padded = cur + [0] * (_REFERENCE_PAD_LEN - len(cur))
        out = model(input_ids=jnp.asarray([padded], dtype=jnp.int32), **mm_kwargs)
        row = out.logits[0, len(cur) - 1]
        assert bool(jnp.isfinite(row).all()), "non-finite logits in teacher-forced reference"
        nxt = int(jnp.argmax(row))
        toks.append(nxt)
        cur.append(nxt)
    return toks


def _assert_family_serves(model, family: str, mm_spec: dict | None, mm_ref_kwargs: dict | None) -> None:
    """Run the full serving-parity protocol for one family on one engine.

    Phases (single compiled runner):
        1. Text-only batch of 2: eSurge greedy == teacher-forced dense.
        2. Deterministic repeat of phase 1 on the same engine.
        3. Mixed batch (multimodal + text-only): parity for both requests.

    Args:
        model: The vision-language task head under test.
        family: Family name used in assertion messages.
        mm_spec: Multimodal request spec, or None to skip phase 3.
        mm_ref_kwargs: Dense-forward kwargs reproducing the multimodal prefill.
    """
    harness = _ServingHarness(model)
    try:
        text_specs = [{"prompt_token_ids": p} for p in TEXT_PROMPTS]
        served = harness.generate_greedy(text_specs)
        expected = [_greedy_teacher_forced(model, p, MAX_NEW_TOKENS) for p in TEXT_PROMPTS]
        assert served == expected, (
            f"{family}: text-only eSurge greedy decode diverged from teacher-forced dense forward\n"
            f"  esurge={served}\n  dense ={expected}"
        )

        repeated = harness.generate_greedy(text_specs)
        assert repeated == served, (
            f"{family}: repeated generation on the same engine is not deterministic\n"
            f"  first ={served}\n  second={repeated}"
        )

        if mm_spec is not None:
            mixed_specs = [mm_spec, {"prompt_token_ids": TEXT_PROMPTS[1]}]
            served_mm = harness.generate_greedy(mixed_specs)
            expected_mm = [
                _greedy_teacher_forced(model, mm_spec["prompt_token_ids"], MAX_NEW_TOKENS, **(mm_ref_kwargs or {})),
                _greedy_teacher_forced(model, TEXT_PROMPTS[1], MAX_NEW_TOKENS),
            ]
            assert served_mm == expected_mm, (
                f"{family}: multimodal eSurge greedy decode diverged from teacher-forced dense forward\n"
                f"  esurge={served_mm}\n  dense ={expected_mm}"
            )
    finally:
        harness.close()


# paligemma — SigLIP tower + Gemma text; causal serving (see module docstring)


def _make_paligemma():
    vision_config = {
        "model_type": "siglip_vision_model",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "image_size": 28,
        "patch_size": 14,
        "vision_use_head": False,
    }
    text_config = ed.GemmaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
        head_dim=16,
        tie_word_embeddings=False,
        # A wider weight draw keeps the greedy stream varied (a tiny
        # default-init Gemma degenerates to echoing one token, which would
        # let position/mask bugs hide behind a constant sequence).
        initializer_range=0.4,
    )
    cfg = ed.PaliGemmaConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=VOCAB - 1,
        projection_dim=64,
    )
    return _build_model("paligemma", cfg)


def test_paligemma_esurge_greedy_parity():
    model = _make_paligemma()
    rng = np.random.default_rng(0)
    pixel_values = rng.standard_normal((1, 3, 28, 28)).astype(np.float32)
    # (28 / 14)^2 = 4 image placeholder tokens, image-first prompt layout.
    mm_prompt = [VOCAB - 1] * 4 + [3, 1, 4, 5, 9, 2]
    _assert_family_serves(
        model,
        "paligemma",
        mm_spec={"prompt_token_ids": mm_prompt, "pixel_values": pixel_values},
        mm_ref_kwargs={"pixel_values": jnp.asarray(pixel_values)},
    )


# internvl — InternVL ViT tower + Qwen2 text


def _make_internvl():
    vision_config = ed.InternVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=(56, 56),
        patch_size=(14, 14),
        attention_bias=True,
        use_qk_norm=True,
    )
    text_config = ed.Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    cfg = ed.InternVLConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=VOCAB - 1,
        tie_word_embeddings=False,
    )
    return _build_model("internvl", cfg)


def test_internvl_esurge_greedy_parity():
    model = _make_internvl()
    rng = np.random.default_rng(0)
    pixel_values = rng.standard_normal((1, 3, 56, 56)).astype(np.float32)
    # (56/14)^2 patches (CLS dropped) * downsample 0.5^2 = 4 image tokens.
    mm_prompt = [3, 1, 4] + [VOCAB - 1] * 4 + [5, 9, 2]
    _assert_family_serves(
        model,
        "internvl",
        mm_spec={"prompt_token_ids": mm_prompt, "pixel_values": pixel_values},
        mm_ref_kwargs={"pixel_values": jnp.asarray(pixel_values)},
    )


# hunyuan_vl — own GQA text + ViT, 3-axis mRoPE (position-consistent decode)


def _make_hunyuan_vl():
    head_dim = 16
    mrope_section = [2, 3, 3]  # sums to head_dim // 2
    cfg = ed.HunYuanVLConfig(
        text_config=dict(
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=head_dim,
            max_position_embeddings=512,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            rope_scaling={"rope_type": "default", "mrope_section": mrope_section},
            tie_word_embeddings=False,
            initializer_range=0.4,
        ),
        vision_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_channels=3,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=1,
            max_image_size=64,
            rms_norm_eps=1e-5,
        ),
        image_token_id=VOCAB - 1,
        im_start_id=VOCAB - 3,
        im_end_id=VOCAB - 2,
        im_newline_id=VOCAB - 4,
    )
    return _build_model("hunyuan_vl", cfg)


def test_hunyuan_vl_esurge_greedy_parity():
    """mRoPE serving parity: 3-axis prefill positions + rope-delta decode.

    The mixed multimodal batch checks that the runner consumes the
    ``EmbeddingInfo.position_ids``/``rope_deltas`` produced by
    ``HunYuanVLModel.compute_embedding_with_info`` (image spans get spatial
    positions, decode continues at ``max_position + delta``), matching the
    dense forward that derives positions via ``get_rope_index``.
    """
    model = _make_hunyuan_vl()
    rng = np.random.default_rng(0)
    grid_h = grid_w = 8
    merge = 2
    llm_h, llm_w = grid_h // merge, grid_w // merge
    # Grid tokens with one newline column per merged row, plus begin/end.
    num_image_tokens = llm_h * (llm_w + 1) + 2
    patch_features = 3 * 1 * 4 * 4  # channels * temporal * patch^2
    pixel_values = rng.standard_normal((grid_h * grid_w, patch_features)).astype(np.float32)
    image_grid_thw = np.array([[1, grid_h, grid_w]], dtype=np.int64)
    mm_prompt = [3, VOCAB - 3] + [VOCAB - 1] * num_image_tokens + [VOCAB - 2, 5, 9, 2]
    _assert_family_serves(
        model,
        "hunyuan_vl",
        mm_spec={
            "prompt_token_ids": mm_prompt,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        },
        mm_ref_kwargs={
            "pixel_values": jnp.asarray(pixel_values),
            "image_grid_thw": jnp.asarray(image_grid_thw, dtype=jnp.int32),
        },
    )


# minimax_m3_vl — MoE text + vision tower (full-attention config only)


def _minimax_text_config():
    return ed.MiniMaxM3VLTextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=64,
        dense_intermediate_size=128,
        shared_intermediate_size=64,
        max_position_embeddings=512,
        num_local_experts=8,
        num_experts_per_tok=2,
        routed_scaling_factor=2.0,
        partial_rotary_factor=0.5,
        mlp_layer_types=["dense", "sparse"],
    )


def _pin_minimax_moe_cpu(cfg) -> None:
    """Force the XLA grouped-matmul MoE path (no expert-parallel axes on CPU)."""
    cfg.moe_force_xla_gmm = True
    cfg.fsdp_is_ep_bound = False


def _make_minimax_m3_vl():
    vision_config = ed.MiniMaxM3VLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=56,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
    )
    text_config = _minimax_text_config()
    cfg = ed.MiniMaxM3VLConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=VOCAB - 1,
        video_token_index=VOCAB - 2,
        projector_hidden_size=128,
    )
    _pin_minimax_moe_cpu(cfg)
    _pin_minimax_moe_cpu(cfg.text_config)
    return _build_model("minimax_m3_vl", cfg)


def test_minimax_m3_vl_esurge_greedy_parity():
    model = _make_minimax_m3_vl()
    rng = np.random.default_rng(0)
    grid_t, grid_h, grid_w = 1, 4, 4
    num_image_tokens = (grid_t * grid_h * grid_w) // 4  # spatial merge 2x2
    patch_features = 3 * 2 * 14 * 14  # channels * temporal * patch^2
    pixel_values = rng.standard_normal((grid_t * grid_h * grid_w, patch_features)).astype(np.float32)
    image_grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    mm_prompt = [3, 1] + [VOCAB - 1] * num_image_tokens + [5, 9, 2]
    _assert_family_serves(
        model,
        "minimax_m3_vl",
        mm_spec={
            "prompt_token_ids": mm_prompt,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        },
        mm_ref_kwargs={
            "pixel_values": jnp.asarray(pixel_values),
            "image_grid_thw": jnp.asarray(image_grid_thw, dtype=jnp.int32),
        },
    )


def test_minimax_m3_vl_text_head_esurge_greedy_parity():
    """The family's registered CAUSAL_LM text head serves text-only requests."""
    text_config = _minimax_text_config()
    _pin_minimax_moe_cpu(text_config)
    model = _build_model("minimax_m3_vl_text", text_config, task=ed.TaskType.CAUSAL_LM)
    _assert_family_serves(model, "minimax_m3_vl_text", mm_spec=None, mm_ref_kwargs=None)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
