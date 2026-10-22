"""Array-free generation setup and cache-budget regression contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from easydel.caching import MLARaggedPagesCacheConfig, MLARaggedPagesCacheView, TransformerCacheView
from easydel.infra.mixins import generation
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.layers.quantization import TurboQuantConfig
from easydel.modules.gemma4 import Gemma4TextConfig
from transformers import GenerationConfig

from .test_stage_cache_budget import Model

pytestmark = pytest.mark.cpu_ok


def test_homogeneous_generation_cache_preserves_real_gemma4_shared_slots(monkeypatch):
    config = Gemma4TextConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        global_head_dim=64,
        layer_types=["full_attention", "full_attention"],
        num_kv_shared_layers=1,
        attn_mechanism="vanilla",
    )
    assert config.get_kv_shared_layer_mapping() == {1: 0}

    class SharedModel(Model):
        def get_inference_cache_type(self):
            return "transformer"

        def _layer_stage_context(self, *args, **kwargs):
            return nullcontext()

    model = SharedModel(mesh=nullcontext())
    model.config.get_kv_shared_layer_mapping = config.get_kv_shared_layer_mapping
    model._quant_class = lambda **_: None
    monkeypatch.setattr(TransformerCacheView, "init", staticmethod(lambda **kw: SimpleNamespace(**kw)))
    cache = model.init_cache(batch_size=1, max_length=32)
    assert cache.views[0].layer_index == 0
    assert cache.views[1] is None


@pytest.mark.parametrize(
    "indexer_types,stages",
    [
        (["full", "shared", "full", "shared"], [0, 0, 1, 1]),
        (["full", "full", "shared", "shared"], [0, 0, 1, 1]),
        (["full", "shared", "shared", "full"], [0, 0, 0, 1]),
    ],
)
def test_mla_pipeline_pool_charges_complete_stage_footprints(monkeypatch, indexer_types, stages):
    available = 1 << 20
    monkeypatch.setattr(MLARaggedPagesCacheConfig, "_compute_free_hbm", staticmethod(lambda **_: available))

    class MLAModel(Model):
        def get_operations_cache_view(self):
            return dict.fromkeys(range(4), MLARaggedPagesCacheView)

    model = MLAModel(
        num_hidden_layers=4,
        num_attention_heads=4,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        index_kpool=2,
        index_head_dim=64,
        indexer_types=indexer_types,
    )
    plan = SimpleNamespace(is_enabled=True, layer_to_stage=dict(enumerate(stages)), max_cache_tokens=None)
    configs = model.init_operations_cache_config(batch_size=1, max_length=256, page_size=16, pipeline_plan=plan)
    # Independent byte accounting: two 128-aligned bf16 latent components;
    # full indexers additionally store token key/gate and one summary per pool.
    stage_bytes = []
    for stage in sorted(set(stages)):
        owners = [i for i, owner in enumerate(stages) if owner == stage]
        full = sum(indexer_types[i] == "full" for i in owners)
        stage_bytes.append(len(owners) * 16 * 256 * 2 + full * (16 * 128 * 2 + 8 * 64 * 2))
    assert [config.num_pages for config in configs] == [available // max(stage_bytes)] * 4


@pytest.mark.parametrize("kind", ["unified", "turboquant"])
def test_mixed_representative_retains_all_resolved_layer_configs(kind):
    model = Model(num_key_value_heads=[2, 4])
    kwargs = dict(text_config=model.config, max_length=128, page_size=16, hbm_utilization=0.9, num_pages_override=8)
    if kind == "unified":
        representative, per_layer = generation._create_mixed_standard_unified_attention_cache_configs(
            dtype=generation.jnp.bfloat16, **kwargs
        )
    else:
        representative, per_layer = generation._create_mixed_turboquant_ragged_page_cache_configs(
            turboquant_config=TurboQuantConfig(), **kwargs
        )
    assert representative._mixed_layer_configs == per_layer
    assert representative._mixed_layer_configs is not per_layer
    assert {idx: config.num_kv_heads for idx, config in representative._mixed_layer_configs.items()} == {0: 2, 1: 4}
    assert all(config.num_pages == representative.num_pages == 8 for config in per_layer.values())


class ShapeOnly:
    """Describe shape plumbing without allocating or computing JAX arrays."""

    def __init__(self, shape=()):
        self.shape = shape
        self.ndim = len(shape)

    def reshape(self, shape):
        return ShapeOnly(shape)

    def __mul__(self, other):
        return self

    def repeat(self, repeats, axis):
        shape = list(self.shape)
        shape[axis] *= repeats
        return ShapeOnly(tuple(shape))


@pytest.mark.parametrize("encoder_decoder", [False, True])
def test_beam_preparation_repeats_batch_inputs_without_flattening_sequence(monkeypatch, encoder_decoder):
    fake_numpy = SimpleNamespace(
        int32="int32",
        bool_="bool",
        array=lambda *args: ShapeOnly(),
        full=lambda shape, *args, **kwargs: ShapeOnly(shape),
        zeros=lambda shape, **kwargs: ShapeOnly(shape),
        ones=lambda shape: ShapeOnly(shape),
        tile=lambda value, reps: ShapeOnly((2, 3)),
        repeat=lambda value, repeats, axis: value.repeat(repeats, axis),
    )
    monkeypatch.setattr(generation, "jnp", fake_numpy)
    monkeypatch.setattr(generation.jax, "Array", ShapeOnly)
    monkeypatch.setattr(generation, "lax", SimpleNamespace(dynamic_update_slice=lambda value, *args: value))

    class Prepared(Exception):
        pass

    class BeamModel(EasyGenerationMixin):
        config = SimpleNamespace(is_encoder_decoder=encoder_decoder)
        generation_config = GenerationConfig(max_length=8, pad_token_id=0)
        decode = object()

        def prepare_inputs_for_generation(self, inputs, **kwargs):
            assert inputs.shape == (6, 4)
            for name in ("attention_mask", "decoder_attention_mask", "token_type_ids"):
                assert kwargs[name].shape == (6, 4)
            assert kwargs["position_ids"].shape == (3, 6, 4)
            assert kwargs["inputs_embeds"].shape == (6, 4, 16)
            mask_info = kwargs["mask_info"]
            assert mask_info.q_segment_ids.shape == (6, 4)
            assert mask_info.kv_segment_ids.shape == (6, 4)
            assert mask_info.q_positions.shape == (6, 4)
            assert mask_info._cu_seqlens_q is None and mask_info._cu_seqlens_kv is None
            if encoder_decoder:
                assert kwargs["encoder_outputs"]["last_hidden_state"].shape == (6, 7, 16)
            raise Prepared

    kwargs = {key: ShapeOnly((2, 4)) for key in ("attention_mask", "decoder_attention_mask", "token_type_ids")}
    kwargs.update(
        position_ids=ShapeOnly((3, 2, 4)),
        inputs_embeds=ShapeOnly((2, 4, 16)),
        mask_info=generation.MaskInfo(
            _q_segment_ids=ShapeOnly((2, 4)),
            _kv_segment_ids=ShapeOnly((2, 4)),
            q_positions=ShapeOnly((2, 4)),
            _cu_seqlens_q=ShapeOnly((3,)),
            _cu_seqlens_kv=ShapeOnly((3,)),
        ),
    )
    if encoder_decoder:
        kwargs["encoder_outputs"] = {"last_hidden_state": ShapeOnly((2, 7, 16))}
    with pytest.raises(Prepared):
        BeamModel()._beam_search(ShapeOnly((2, 3, 4)), model_kwargs=kwargs)


@pytest.mark.parametrize("do_sample,num_beams", [(False, 1), (True, 1), (False, 2)])
def test_public_generate_rejects_exhausted_budget(monkeypatch, do_sample, num_beams):
    class GenerationModel(EasyGenerationMixin):
        config = SimpleNamespace(is_encoder_decoder=False)

        def _validate_model_kwargs(self, kwargs):
            pass

    monkeypatch.setattr(generation, "_resolve_backend_for_esurge", lambda _: "cpu")
    with pytest.raises(ValueError, match=r"prompt length|Input length"):
        GenerationModel().generate(
            ShapeOnly((2, 3)),
            generation_config=GenerationConfig(max_length=3, pad_token_id=0, do_sample=do_sample, num_beams=num_beams),
            prng_key=object(),
        )


@pytest.mark.parametrize("method", ["_greedy_search", "_sample"])
@pytest.mark.parametrize("prompt_length,max_length", [(3, 3), (4, 3), (1, 1)])
def test_search_rejects_exhausted_budget_before_cache_or_output_allocation(method, prompt_length, max_length):
    model = EasyGenerationMixin()
    model.generation_config = GenerationConfig(pad_token_id=0)
    kwargs = {"prng_key": object()} if method == "_sample" else {}
    with pytest.raises(ValueError, match="prompt length"):
        getattr(model, method)(ShapeOnly((2, prompt_length)), max_length=max_length, **kwargs)
