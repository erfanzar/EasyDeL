"""Independent packed-linear, attention-connectivity and vision references (accelerator only)."""

from math import erf
from types import MethodType, SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.minimax import MiniMaxConfig
from easydel.modules.minimax.modeling_minimax import MiniMaxLightningAttention, MiniMaxModel
from easydel.modules.mistral import MistralConfig
from easydel.modules.mistral.modeling_mistral import MistralAttention
from easydel.modules.mistral3 import Mistral3Config
from easydel.modules.mistral3.modeling_mistral3 import Mistral3Model, Mistral3MultiModalProjector, Mistral3PatchMerger
from easydel.modules.mixtral import MixtralConfig
from easydel.modules.mixtral.modeling_mixtral import MixtralAttention
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


def _single_device(config):
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    return config


@pytest.mark.parametrize(
    "config_class,attention_class", [(MistralConfig, MistralAttention), (MixtralConfig, MixtralAttention)]
)
@jax.default_matmul_precision("highest")
def test_mixed_attention_schedule_preserves_stateless_connectivity(config_class, attention_class):
    config = _single_device(
        config_class(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            sliding_window=2,
            layer_types=["full_attention", "sliding_attention"],
            attn_mechanism="vanilla",
            attn_dtype=jnp.float32,
            attn_softmax_dtype=jnp.float32,
        )
    )
    hidden = jnp.asarray(np.random.default_rng(12).normal(size=(1, 6, 8)).astype(np.float32))
    mask = MaskInfo.from_segments(jnp.zeros((1, 6), dtype=jnp.int32))
    with config.mesh:
        for index in range(2):
            attention = attention_class(
                config,
                layer_idx=index,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                precision=jax.lax.Precision.HIGHEST,
                rngs=spx.Rngs(0),
            )
            output = attention(
                hidden,
                mask_info=mask,
                position_ids=jnp.arange(6)[None],
                mode=common_types.MODE_TRAIN,
                output_attentions=True,
                frequencies=config.get_basic_frequencies(),
            )
            weights = np.asarray(output.attention_weight)
            # Independent causal/local visibility, not a comparison with a
            # second execution of the production attention implementation.
            q, k = np.arange(6)[:, None], np.arange(6)[None, :]
            expected = k <= q
            if index == 1:
                expected &= k > q - 2
            np.testing.assert_array_equal(weights[0] > 0, np.broadcast_to(expected, (2, 6, 6)))
            np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-6)


def _vision_config(layers=-1):
    return _single_device(
        Mistral3Config(
            vision_config=dict(
                hidden_size=4, intermediate_size=8, num_hidden_layers=2, num_attention_heads=1, patch_size=2
            ),
            text_config=dict(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=16,
            ),
            vision_feature_layer=layers,
            spatial_merge_size=2,
        )
    )


@jax.default_matmul_precision("highest")
@pytest.mark.parametrize("sizes", [((4, 4), (4, 8)), ((6, 8), (8, 6)), ((2, 4), (4, 2))])
def test_multi_image_compiled_merger_matches_channel_major_unfold(sizes):
    config = _vision_config()
    grids = [np.arange((h // 2) * (w // 2) * 4, dtype=np.float32).reshape(h // 2, w // 2, 4) / 100 for h, w in sizes]
    features = np.concatenate([grid.reshape(-1, 4) for grid in grids])
    with config.mesh:
        merger = Mistral3PatchMerger(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        actual = jax.jit(lambda values: merger(values, sizes))(jnp.asarray(features))
    patches = []
    for grid in grids:
        for row in range(0, grid.shape[0] - 1, 2):
            for col in range(0, grid.shape[1] - 1, 2):
                patches.append(grid[row : row + 2, col : col + 2].transpose(2, 0, 1).reshape(-1))
    expected = np.asarray(patches, dtype=np.float32).reshape(-1, 16) @ np.asarray(merger.merging_layer.weight.value)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-6, rtol=1e-5)


@jax.default_matmul_precision("highest")
@pytest.mark.parametrize("sizes", [((4, 4), (4, 8)), ((4, 4),)])
def test_multiple_vision_layers_project_end_to_end_and_use_each_selected_layer(sizes):
    config = _vision_config([-1, -2])
    rng = np.random.default_rng(18)
    num_patches = sum((height // 2) * (width // 2) for height, width in sizes)
    first, second = (jnp.asarray(rng.normal(size=(1, num_patches, 4)).astype(np.float32)) for _ in range(2))
    with config.mesh:
        projector = Mistral3MultiModalProjector(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )

        def run(layer_a, layer_b):
            model = SimpleNamespace(
                vision_feature_layer=config.vision_feature_layer,
                vision_tower=lambda *args, **kwargs: SimpleNamespace(hidden_states=(layer_a, layer_b)),
                multi_modal_projector=projector,
            )
            return Mistral3Model.get_image_features(model, None, sizes)

        compiled = jax.jit(run)
        output = compiled(first, second)
        changed_first = compiled(first + 1, second)
        changed_second = compiled(first, second + 1)
    num_merged = sum((height // 4) * (width // 4) for height, width in sizes)
    assert output.shape == (num_merged, config.text_config.hidden_size)
    assert np.isfinite(np.asarray(output)).all()
    assert not np.allclose(output, changed_first, rtol=1e-5, atol=1e-7)
    assert not np.allclose(output, changed_second, rtol=1e-5, atol=1e-7)


@jax.default_matmul_precision("highest")
def test_precomputed_capacity_features_use_explicit_valid_length_for_text_merge():
    config = _vision_config()
    _single_device(config.text_config)
    _single_device(config.vision_config)
    with config.mesh:
        model = Mistral3Model(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        ids = jnp.asarray([[1, config.image_token_index, 2, config.image_token_index, 3]])
        features = jnp.asarray(np.arange(32, dtype=np.float32).reshape(4, 8))
        actual = model.compute_embedding(ids, image_features=features, image_features_valid_length=jnp.asarray(2))
        compiled = jax.jit(
            lambda values, count: model.compute_embedding(ids, image_features=values, image_features_valid_length=count)
        )(features, jnp.asarray(2))
        with pytest.raises(ValueError, match="Expected 2 placeholder tokens but got 4"):
            model.compute_embedding(ids, image_features=features)
    expected = np.asarray(model.get_embedding().weight.value)[np.asarray(ids)].copy()
    expected[0, 1], expected[0, 3] = np.asarray(features)[:2]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(compiled, expected)


def _unfold_reference(features, grids, k=2):
    blocks, start = [], 0
    for height, width in grids:
        count = height * width
        grid = features[start : start + count].reshape(height, width, features.shape[-1])
        for row in range(0, height - k + 1, k):
            for col in range(0, width - k + 1, k):
                blocks.append(grid[row : row + k, col : col + k].transpose(2, 0, 1).reshape(-1))
        start += count
    return np.asarray(blocks, dtype=np.float32).reshape(-1, features.shape[-1] * k * k)


@jax.default_matmul_precision("highest")
@pytest.mark.parametrize("grids", [((3, 4), (4, 3)), ((1, 4), (2, 2)), ((0, 0), (0, 0))])
def test_runtime_merger_uses_packed_patch_geometry_and_masks_capacity_padding(grids):
    config = _vision_config()
    rng = np.random.default_rng(21)
    lengths = np.prod(grids, axis=-1)
    features = rng.normal(size=(32, 4)).astype(np.float32)
    features[sum(lengths) :] = np.nan
    mask = np.arange(32)[None] < sum(lengths)
    with config.mesh:
        merger = Mistral3PatchMerger(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        compiled = jax.jit(
            lambda values, sizes, counts, patch_grids, valid: merger(
                values,
                sizes,
                patch_lengths=counts,
                patch_grid_sizes=patch_grids,
                patch_attention_mask=valid,
                return_attention_mask=True,
            )
        )
        actual, valid = compiled(
            jnp.asarray(features),
            # Metadata, not the padded image dimensions, owns the true grids.
            jnp.asarray([[8, 8], [8, 8]]),
            jnp.asarray(lengths),
            jnp.asarray(grids),
            jnp.asarray(mask),
        )
    reference = _unfold_reference(features, grids) @ np.asarray(merger.merging_layer.weight.value)
    assert actual.shape == (8, 4)
    np.testing.assert_array_equal(valid, np.arange(8) < len(reference))
    np.testing.assert_allclose(np.asarray(actual)[: len(reference)], reference, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual)[len(reference) :], 0)


@jax.default_matmul_precision("highest")
def test_runtime_selected_layers_project_metadata_prefix_and_zero_biased_padding():
    config = _vision_config([-1, -2])
    config.multimodal_projector_bias = True
    grids = np.asarray([[3, 4], [4, 3]], dtype=np.int32)
    lengths = np.prod(grids, axis=-1)
    rng = np.random.default_rng(31)
    first, second = (rng.normal(size=(1, 32, 4)).astype(np.float32) for _ in range(2))
    first[:, sum(lengths) :] = np.nan
    second[:, sum(lengths) :] = np.nan
    with config.mesh:
        projector = Mistral3MultiModalProjector(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        projector.linear_1.bias.value = jnp.full_like(projector.linear_1.bias.value, 0.3)
        projector.linear_2.bias.value = jnp.full_like(projector.linear_2.bias.value, 0.7)

        def run(layer_a, layer_b, sizes, patch_grids, counts, mask):
            model = SimpleNamespace(
                vision_feature_layer=config.vision_feature_layer,
                vision_tower=lambda *args, **kwargs: SimpleNamespace(
                    hidden_states=(layer_a, layer_b),
                    patch_grid_sizes=patch_grids,
                    patch_lengths=counts,
                    patch_attention_mask=mask,
                ),
                multi_modal_projector=projector,
            )
            return Mistral3Model.get_image_features(model, None, sizes, return_valid_length=True)

        actual, valid_length = jax.jit(run)(
            jnp.asarray(first),
            jnp.asarray(second),
            jnp.asarray([[8, 8], [8, 8]]),
            jnp.asarray(grids),
            jnp.asarray(lengths),
            jnp.asarray(np.arange(32)[None] < sum(lengths)),
        )
    selected = np.concatenate([second[0, : sum(lengths)], first[0, : sum(lengths)]], axis=-1)
    normed = selected / np.sqrt(np.mean(selected**2, axis=-1, keepdims=True) + config.text_config.rms_norm_eps)
    normed *= np.asarray(projector.norm.weight.value)
    reference = _unfold_reference(normed, grids) @ np.asarray(projector.patch_merger.merging_layer.weight.value)
    reference = reference @ np.asarray(projector.linear_1.weight.value) + np.asarray(projector.linear_1.bias.value)
    reference = reference * 0.5 * (1 + np.vectorize(erf)(reference / np.sqrt(2)))
    reference = reference @ np.asarray(projector.linear_2.weight.value) + np.asarray(projector.linear_2.bias.value)
    assert actual.shape == (8, config.text_config.hidden_size)
    assert int(valid_length) == 4
    np.testing.assert_allclose(np.asarray(actual)[:4], reference, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual)[4:], 0)


@jax.default_matmul_precision("highest")
@pytest.mark.parametrize("mode", [common_types.MODE_TRAIN, common_types.MODE_DECODE])
@pytest.mark.parametrize("block_size", [2, 3, 8])
def test_lightning_packed_boundaries_and_maskinfo_padding_match_scalar_reference(mode, block_size):
    config = _single_device(MiniMaxConfig(hidden_size=4, num_attention_heads=2, num_hidden_layers=2))
    rng = np.random.default_rng(42)
    q, k, v = (rng.normal(size=(2, 7, 2, 2)).astype(np.float32) for _ in range(3))
    qkv = jnp.asarray(np.concatenate([x.reshape(2, 7, 4) for x in (q, k, v)], axis=-1))
    segments = np.asarray([[0, 0, 1, 1, 1, -1, -1], [-1, 4, 4, 0, 0, 4, 4]], dtype=np.int32)
    slopes = np.asarray([0.1, 0.2], dtype=np.float32)
    initial_state = rng.normal(size=(2, 2, 2, 2)).astype(np.float32)
    model = SimpleNamespace(
        config=config,
        block_size=block_size,
        num_attention_heads=2,
        head_dim=2,
        qkv_proj=lambda x: qkv,
        act_fn=lambda x: x,
        norm=lambda x: x,
        output_gate=lambda x: jnp.zeros_like(x),
        out_proj=lambda x: x,
        _get_slope_rate=lambda: jnp.asarray(slopes[:, None, None]),
    )
    model._decay_factors = MethodType(MiniMaxLightningAttention._decay_factors, model)
    mask = MaskInfo.from_segments(jnp.asarray(segments))

    def run(hidden):
        cache = SimpleNamespace(
            recurrent_state=jnp.asarray(initial_state),
            update_recurrent_state=lambda state: SimpleNamespace(recurrent_state=state),
        )
        output, cache = MiniMaxLightningAttention.forward(model, hidden, None, mode, cache, mask_info=mask)
        return output, cache.recurrent_state

    with config.mesh:
        actual, actual_state = jax.jit(run)(jnp.zeros((2, 7, 4), dtype=jnp.float32))
    state = initial_state.copy()
    expected = np.zeros((2, 7, 2, 2), dtype=np.float32)
    for token in range(7):
        for batch in range(2):
            if token > 0 and segments[batch, token] != segments[batch, token - 1]:
                state[batch] = 0
            state[batch] *= np.exp(-slopes[:, None, None])
            if segments[batch, token] >= 0:
                state[batch] += k[batch, token, :, :, None] * v[batch, token, :, None, :]
                expected[batch, token] = np.einsum("hd,hdv->hv", q[batch, token], state[batch]) * 0.5
    np.testing.assert_allclose(np.asarray(actual), expected.reshape(2, 7, 4), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual_state), state, rtol=1e-5, atol=1e-6)


@jax.default_matmul_precision("highest")
def test_minimax_public_segment_ids_isolates_documents_through_linear_decoder():
    config = _single_device(
        MiniMaxConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_local_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=16,
            block_size=3,
            layer_types=["linear_attention"],
            moe_method="dense_moe",
        )
    )
    with config.mesh:
        model = MiniMaxModel(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        combined = model(
            input_ids=jnp.asarray([[1, 2, 3, 4]]), segment_ids=jnp.asarray([[0, 0, 1, 1]])
        ).last_hidden_state
        separate = model(input_ids=jnp.asarray([[3, 4]])).last_hidden_state
    np.testing.assert_allclose(np.asarray(combined[:, 2:]), np.asarray(separate), rtol=1e-5, atol=1e-5)
