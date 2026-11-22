"""Qwen multimodal integration regressions (accelerator only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextModel
from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLTextConfig
from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModel, Qwen3VLMoeTextModel
from easydel.modules.qwen3_vl_moe.qwen3_vl_moe_configuration import Qwen3VLMoeTextConfig
from jax import numpy as jnp


class _Vision:
    spatial_merge_size = 1

    def get_dtype(self):
        return jnp.float32

    def __call__(self, pixels, *, grid_thw, max_grid_size):
        assert max_grid_size == 2
        assert tuple(grid_thw.shape) == (1, 3)
        return pixels, [pixels / 10, pixels / 5]


def _model(cls):
    # Use SpectraX allocation so public __call__ has its hook/context slots,
    # even though this fixture intentionally replaces the expensive towers.
    model = cls.__new__(cls)
    object.__setattr__(model, "config", SimpleNamespace(image_token_id=1, video_token_id=2))
    object.__setattr__(model, "visual", _Vision())
    text_cls = Qwen3VLMoeTextModel if cls is Qwen3VLMoeModel else Qwen3VLTextModel

    def decoder(**kwargs):
        hidden = kwargs["inputs_embeds"]
        for residual in kwargs["deepstack_visual_embeds"] or []:
            hidden = text_cls._deepstack_process(None, hidden, kwargs["visual_pos_masks"], residual)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(kwargs["inputs_embeds"],),
            attentions=None,
            past_key_values=None,
            router_logits=(hidden[..., :1],),
        )

    decoder.embed_tokens = lambda ids: jnp.arange(ids.size * 2, dtype=jnp.float32).reshape(*ids.shape, 2)
    object.__setattr__(model, "language_model", decoder)
    # The independent positional policy is outside this embedding/DeepStack test.
    object.__setattr__(model, "get_rope_index", lambda **kwargs: (jnp.zeros((3, 1, 6), jnp.int32), jnp.zeros((1, 1))))
    return model


@pytest.mark.parametrize("cls", [Qwen3VLModel, Qwen3VLMoeModel])
@pytest.mark.parametrize("modality", ["image", "video", "both"])
def test_pixel_and_precomputed_embedding_paths_match_sequence_order_and_deepstack(cls, modality):
    model = _model(cls)
    ids = jnp.array([[0, 2, 1, 2, 1, 0]], jnp.int32)
    text = jnp.arange(12, dtype=jnp.float32).reshape(1, 6, 2)
    image = jnp.array([[10.0, 11.0], [12.0, 13.0]])
    video = jnp.array([[20.0, 21.0], [22.0, 23.0]])
    pixels = {}
    features = {}
    expected = np.arange(12, dtype=np.float32).reshape(1, 6, 2)
    expected_residual = np.zeros_like(expected)
    for name, table, token in (("image", image, 1), ("video", video, 2)):
        if modality not in (name, "both"):
            continue
        pixel_name = "pixel_values" if name == "image" else "pixel_values_videos"
        pixels[pixel_name] = table
        pixels[f"{name}_grid_thw"] = ((1, 1, 2),)
        features[f"{name}_embeds"] = table
        mask = np.asarray(ids) == token
        expected[mask] = np.asarray(table)
        expected_residual[mask] = np.asarray(table) * 0.3

    pixel_embeddings, info = model.compute_embedding_with_info(ids, inputs_embeds=text, **pixels)
    precomputed = model.compute_embedding(ids, inputs_embeds=text, **features)
    np.testing.assert_array_equal(np.asarray(pixel_embeddings), expected)
    np.testing.assert_array_equal(np.asarray(precomputed), expected)
    assert info.position_ids.shape == (3, 1, 6)
    assert info.rope_deltas.shape == (1, 1)

    # Both the direct pixel forward and the precomputed serving path must add
    # the same two DeepStack residuals, interleaved by sequence rather than media.
    options = {"output_router_logits": True} if cls is Qwen3VLMoeModel else {}

    def forward(embeddings):
        return model.forward(
            inputs_embeds=embeddings,
            position_ids=info.position_ids,
            visual_pos_masks=info.visual_pos_masks,
            deepstack_visual_embeds=info.deepstack_visual_embeds,
            **options,
        )

    output = jax.jit(forward)(precomputed)
    direct_output = jax.jit(
        lambda token_ids: model.forward(input_ids=token_ids, position_ids=info.position_ids, **pixels, **options)
    )(ids)
    np.testing.assert_allclose(
        np.asarray(direct_output.last_hidden_state), expected + expected_residual, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(output.last_hidden_state), expected + expected_residual, rtol=1e-6, atol=1e-6)
    if options:
        np.testing.assert_allclose(np.asarray(output.router_logits[0]), (expected + expected_residual)[..., :1])


@pytest.mark.parametrize("cls", [Qwen3VLTextModel, Qwen3VLMoeTextModel])
def test_empty_deepstack_tables_are_a_jittable_noop(cls):
    config_cls = Qwen3VLMoeTextConfig if cls is Qwen3VLMoeTextModel else Qwen3VLTextConfig
    config = config_cls(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        moe_intermediate_size=8,
        num_experts=3,
        num_experts_per_tok=1,
        moe_force_xla_gmm=True,
    )
    decoder = cls(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    hidden = jnp.asarray(np.random.default_rng(0).normal(size=(1, 6, 16)), jnp.float32)
    mask = jnp.zeros((1, 6), dtype=jnp.bool_)
    empty = jnp.zeros((0, 16), dtype=jnp.float32)
    # Both families support output_hidden_states, which also keeps the layer
    # execution policy identical when the visual branch is absent. The MoE
    # decoder does not expose the dense-only `trace` keyword.
    # Compare like-for-like execution: eager-vs-JIT differences are unrelated
    # to whether an empty DeepStack table changes a decoder's output.
    for compile_forward in (False, True):

        def forward(x, *, with_deepstack):
            return decoder(
                inputs_embeds=x,
                visual_pos_masks=mask if with_deepstack else None,
                deepstack_visual_embeds=[empty, empty] if with_deepstack else None,
                output_hidden_states=True,
            ).last_hidden_state

        def reference_forward(x):
            return forward(x, with_deepstack=False)

        def visual_forward(x):
            return forward(x, with_deepstack=True)

        if compile_forward:
            reference_forward = jax.jit(reference_forward)
            visual_forward = jax.jit(visual_forward)
        with jax.default_matmul_precision("highest"):
            reference = reference_forward(hidden)
            result = visual_forward(hidden)
        np.testing.assert_allclose(np.asarray(result), np.asarray(reference), rtol=2e-5, atol=2e-6)
