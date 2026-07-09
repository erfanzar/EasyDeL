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

"""End-to-end SFT-step smoke tests for the vision-language task heads.

Drives the exact trainer step (:func:`easydel.trainers.trainer._fn.training_step`)
that :class:`easydel.SFTTrainer` compiles, feeding a text-only batch that carries
the trainer-injected ``completion_mask`` field. The vision tower stays dormant
(no ``pixel_values``); the point is the
trainer -> ``prepare_inputs_for_call`` -> ``compute_loss`` -> head forward ->
inner text-model forward + backward path.

These heads carry ``**kwargs`` in ``forward``, which makes
``compute_loss`` skip its signature filter, so before the fix the trainer-only
``completion_mask`` was threaded into the inner text decoder (``TypeError``), and
the ``hunyuan_vl`` mRoPE input prep dropped ``input_ids`` / leaked a
``rope_deltas`` field that collided in the loss (``UnexpectedTracerError`` /
``TypeError``). The step runs under ``ed.ejit`` so both the trace-time
``TypeError`` and any tracer escape would fail the test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest

import easydel as ed
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.trainer._fn import training_step

try:
    from tests.modules.test_utils.model_factory import create_ed_model_only, setup_config
except ImportError:  # pragma: no cover - path shim for direct invocation
    from tests.modules.test_utils.model_factory import (  # pyright: ignore[reportImplicitRelativeImport]
        create_ed_model_only,
        setup_config,
    )


def _small_cfg() -> dict:
    """Return a tiny runtime-config dict valid on the 8-fake-device CPU mesh."""
    device_count = jax.device_count()
    tp_dim = 2 if device_count >= 2 and device_count % 2 == 0 else 1
    return {
        "batch_size": 2,
        "vocab_size": 512,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 2048,
        "sharding_axis_dims": (1, 1, 1, -1, tp_dim, 1),
        "dtype": jnp.float32,
        "precision": jax.lax.Precision.HIGHEST,
        "attn_mechanism": "vanilla",
        "blocksize_k": 128,
        "blocksize_q": 128,
        "attn_dtype": jnp.float32,
        "sequence_length": 64,
    }


def _build_paligemma(sc: dict):
    text_config = ed.GemmaConfig(
        vocab_size=sc["vocab_size"],
        hidden_size=sc["hidden_size"],
        num_hidden_layers=sc["num_hidden_layers"],
        num_attention_heads=sc["num_attention_heads"],
        num_key_value_heads=sc["num_key_value_heads"],
        intermediate_size=sc["intermediate_size"],
        max_position_embeddings=2048,
        head_dim=sc["hidden_size"] // sc["num_attention_heads"],
        tie_word_embeddings=True,
    )
    cfg = ed.PaliGemmaConfig(
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": 56,
            "patch_size": 14,
            "vision_use_head": False,
        },
        text_config=text_config,
        image_token_index=sc["vocab_size"] - 1,
        vocab_size=sc["vocab_size"],
        projection_dim=sc["hidden_size"],
        hidden_size=sc["hidden_size"],
    )
    return cfg


def _build_internvl(sc: dict):
    vision_config = ed.InternVLVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=(56, 56),
        patch_size=(14, 14),
        attention_bias=True,
        use_qk_norm=True,
    )
    text_config = ed.Qwen2Config(
        vocab_size=sc["vocab_size"],
        hidden_size=sc["hidden_size"],
        num_hidden_layers=sc["num_hidden_layers"],
        num_attention_heads=sc["num_attention_heads"],
        num_key_value_heads=sc["num_key_value_heads"],
        intermediate_size=sc["intermediate_size"],
        max_position_embeddings=2048,
    )
    cfg = ed.InternVLConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=sc["vocab_size"] - 1,
        tie_word_embeddings=False,
    )
    return cfg


def _build_hunyuan_vl(sc: dict):
    hidden_size = sc["hidden_size"]
    num_heads = sc["num_attention_heads"]
    head_dim = hidden_size // num_heads
    half = head_dim // 2
    mrope_section = [half // 4, (half - half // 4) // 2, half - half // 4 - (half - half // 4) // 2]
    vocab = sc["vocab_size"]
    cfg = ed.HunYuanVLConfig(
        text_config=dict(
            vocab_size=vocab,
            hidden_size=hidden_size,
            intermediate_size=sc["intermediate_size"],
            num_hidden_layers=sc["num_hidden_layers"],
            num_attention_heads=num_heads,
            num_key_value_heads=sc["num_key_value_heads"],
            head_dim=head_dim,
            max_position_embeddings=sc["max_position_embeddings"],
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            rope_scaling={"rope_type": "default", "mrope_section": mrope_section},
            tie_word_embeddings=True,
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
        image_token_id=vocab - 1,
        im_start_id=vocab - 3,
        im_end_id=vocab - 2,
        im_newline_id=vocab - 4,
    )
    return cfg


def _build_qwen2_vl(sc: dict):
    hidden_size = sc["hidden_size"]
    num_heads = sc["num_attention_heads"]
    head_dim = hidden_size // num_heads
    half = head_dim // 2
    # mRoPE requires head_dim == 2 * sum(mrope_section); split the half-head_dim
    # budget across the (temporal, height, width) sections.
    mrope_section = [half // 4, (half - half // 4) // 2, half - half // 4 - (half - half // 4) // 2]
    vocab = sc["vocab_size"]
    cfg = ed.Qwen2VLConfig(
        text_config=dict(
            vocab_size=vocab,
            hidden_size=hidden_size,
            intermediate_size=sc["intermediate_size"],
            num_hidden_layers=sc["num_hidden_layers"],
            num_attention_heads=num_heads,
            num_key_value_heads=sc["num_key_value_heads"],
            max_position_embeddings=sc["max_position_embeddings"],
            rms_norm_eps=1e-5,
            rope_theta=1000000.0,
            rope_scaling={"rope_type": "default", "mrope_section": mrope_section},
            tie_word_embeddings=True,
        ),
        vision_config=dict(
            depth=sc["num_hidden_layers"],
            embed_dim=hidden_size,
            hidden_size=hidden_size,
            num_heads=4,
            in_channels=3,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=1,
        ),
        image_token_id=vocab - 1,
        video_token_id=vocab - 4,
        vision_start_token_id=vocab - 3,
        vision_end_token_id=vocab - 2,
    )
    return cfg


def _build_minimax_m3_vl(sc: dict):
    num_hidden_layers = sc["num_hidden_layers"]
    text_config = ed.MiniMaxM3VLTextConfig(
        vocab_size=sc["vocab_size"],
        hidden_size=sc["hidden_size"],
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=sc["num_attention_heads"],
        num_key_value_heads=sc["num_key_value_heads"],
        head_dim=sc["hidden_size"] // sc["num_attention_heads"],
        intermediate_size=64,
        dense_intermediate_size=sc["intermediate_size"],
        shared_intermediate_size=64,
        max_position_embeddings=sc["max_position_embeddings"],
        num_local_experts=8,
        num_experts_per_tok=2,
        routed_scaling_factor=2.0,
        partial_rotary_factor=0.5,
        mlp_layer_types=["dense"] + ["sparse"] * (num_hidden_layers - 1),
    )
    vision_config = ed.MiniMaxM3VLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=56,
    )
    cfg = ed.MiniMaxM3VLConfig(text_config=text_config, vision_config=vision_config)
    return cfg


_BUILDERS = {
    "paligemma": _build_paligemma,
    "internvl": _build_internvl,
    "hunyuan_vl": _build_hunyuan_vl,
    "qwen2_vl": _build_qwen2_vl,
    "minimax_m3_vl": _build_minimax_m3_vl,
}


@pytest.mark.parametrize("family", list(_BUILDERS))
def test_vlm_head_sft_step_finite(family: str):
    """A jitted trainer step through each VLM head yields finite loss and grads.

    Exercises the SFT loss path with a trainer-injected ``completion_mask`` on a
    text-only batch. Asserts the compiled ``training_step`` returns a finite
    scalar loss and a finite, non-zero mean gradient norm, proving the
    trainer -> head -> inner-text-model forward and backward path is healthy.
    """
    sc = _small_cfg()
    cfg = _BUILDERS[family](sc)
    if family == "minimax_m3_vl":
        # XLA:CPU cannot lower the fused-MoE ragged-all-to-all with ep > 1; fold
        # the expert axis into fsdp so the expert mesh stays at ep=1 (matching
        # the family's own spmd test), while TP sharding is still exercised.
        dims = sc["sharding_axis_dims"]
        sc["sharding_axis_dims"] = (dims[0], dims[1], dims[3], 1, dims[4], dims[5])
        if hasattr(cfg, "fsdp_is_ep_bound"):
            cfg.fsdp_is_ep_bound = False

    cfg = setup_config(cfg, sc)

    with cfg.mesh:
        model = create_ed_model_only(family, ed.TaskType.IMAGE_TEXT_TO_TEXT, cfg, sc)
        state = EasyDeLState.create(model=model, tx=optax.sgd(1e-3), init_opt_state=True)

        batch_size, sequence_length = 2, 32
        rng = np.random.default_rng(0)
        input_ids = jnp.asarray(rng.integers(1, 100, (batch_size, sequence_length)), dtype="i4")
        attention_mask = jnp.ones((batch_size, sequence_length), dtype="i4")
        # completion_mask: prompt (first 8) masked out, completion supervised.
        completion_mask = jnp.concatenate(
            [
                jnp.zeros((batch_size, 8), dtype="i4"),
                jnp.ones((batch_size, sequence_length - 8), dtype="i4"),
            ],
            axis=1,
        )
        labels = jnp.where(completion_mask == 0, -100, input_ids)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "completion_mask": completion_mask,
        }

        learning_rate_fn = optax.constant_schedule(1e-3)

        @ed.ejit(static_argnums=(2, 3, 4, 5, 6, 7))  # pyright: ignore[reportUntypedFunctionDecorator]
        def step(st, b, loss_config, lr_fn, partition_spec, grad_accum, ste, loss_type):
            return training_step(st, b, loss_config, lr_fn, partition_spec, grad_accum, ste, loss_type)

        _new_state, metrics = step(state, batch, None, learning_rate_fn, None, 1, None, "nll")

        loss = float(metrics.loss)
        mean_grad_norm = float(metrics.mean_grad_norm)

    assert np.isfinite(loss), f"{family}: non-finite SFT loss {loss}"
    assert np.isfinite(mean_grad_norm), f"{family}: non-finite gradient norm {mean_grad_norm}"
    assert mean_grad_norm > 0.0, f"{family}: gradients did not flow (mean_grad_norm={mean_grad_norm})"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
