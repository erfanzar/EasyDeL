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

"""Tests for Gemma4 model — exercises all Gemma4-specific features."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import transformers
from eformer import common_types
from eformer.escale import PartitionAxis, PartitionManager
from flax import nnx as nn
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh

from easydel import Gemma4RMSNorm as PublicGemma4RMSNorm
from easydel import TaskType
from easydel.caching import RaggedPagesCacheConfig, UnifiedAttentionCacheConfig
from easydel.inference.esurge.core.interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
    create_kv_cache_specs_from_config,
    estimate_runtime_page_budget,
)
from easydel.layers.quantization import TurboQuantConfig
from easydel.modules.gemma4 import (
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4Model,
    Gemma4RMSNorm,
    Gemma4TextConfig,
    Gemma4TextModel,
    Gemma4VisionConfig,
    Gemma4VisionModel,
)

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


def _make_mesh():
    return Mesh(np.array(jax.devices()[:1]), ("data",))


def _base_text_config(**overrides):
    defaults = dict(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        hidden_size_per_layer_input=0,
        enable_moe_block=False,
        num_kv_shared_layers=0,
        attention_k_eq_v=False,
        max_position_embeddings=512,
        sliding_window=64,
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def _base_vision_config(**overrides):
    defaults = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        patch_size=4,
        pooling_kernel_size=1,
        position_embedding_size=16,
    )
    defaults.update(overrides)
    return Gemma4VisionConfig(**defaults)


class TestGemma4TextModel:
    """Tests for the base Gemma4 text model."""

    def test_basic_forward(self):
        """Basic forward pass produces expected output shapes."""
        config = _base_text_config()
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((2, 16), dtype=jnp.int32))
        assert output.last_hidden_state.shape == (2, 16, 128)

    def test_output_hidden_states(self):
        """When output_hidden_states=True, returns all layer outputs."""
        config = _base_text_config()
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((1, 8), dtype=jnp.int32), output_hidden_states=True)
        # num_hidden_layers + 1 (final hidden state)
        assert len(output.hidden_states) == config.num_hidden_layers + 1

    def test_layer_types_pattern(self):
        """Verify layer types pattern is correctly generated."""
        config = _base_text_config(num_hidden_layers=6)
        # With default sliding_window_pattern=6: layers 0-4 sliding, layer 5 full
        assert config.layer_types == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]

    def test_last_layer_forced_full_attention(self):
        """Last layer is always forced to full_attention."""
        config = _base_text_config(num_hidden_layers=3)
        assert config.layer_types[-1] == "full_attention"

    def test_split_graphdef_is_hashable(self):
        """Split graphdefs must stay hashable for static JIT compilation."""
        config = _base_text_config(num_hidden_layers=6)
        model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
        graphdef, _, _ = model.split_module()
        assert isinstance(hash(graphdef), int)
        assert PublicGemma4RMSNorm is Gemma4RMSNorm

    def test_full_attention_layers_use_global_head_geometry(self):
        """Global-attention layers must build projections with global_head_dim."""
        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=128,
            intermediate_size=256,
        )
        model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))

        sliding_attn = model.layers[0].self_attn
        full_attn = model.layers[1].self_attn

        assert sliding_attn.head_dim == 32
        assert full_attn.head_dim == 64
        assert sliding_attn.q_proj.kernel.value.shape == (128, 128)
        assert full_attn.q_proj.kernel.value.shape == (128, 256)
        assert full_attn.o_proj.kernel.value.shape == (256, 128)

    def test_ragged_cache_configs_follow_per_layer_geometry(self):
        """Mixed sliding/full layers must allocate ragged pages with matching KV geometry."""
        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            attn_mechanism="ragged_page_attention_v3",
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            metadata = model.create_ragged_page_cache_config(max_length=128, page_size=16, hbm_utilization=0.1)
            views_config = model.init_operations_cache_config(
                batch_size=1,
                max_length=128,
                page_size=16,
                hbm_utilization=0.1,
                ragged_config=metadata,
            )

        sliding_cfg = views_config[0]
        full_cfg = views_config[1]

        assert sliding_cfg.num_kv_heads == 2
        assert sliding_cfg.k_headdim == 32
        assert full_cfg.num_kv_heads == 1
        assert full_cfg.k_headdim == 64
        assert sliding_cfg.num_pages == full_cfg.num_pages == metadata.num_pages

    def test_mixed_ragged_cache_replicates_kv_axis_when_gqa_is_too_small(self, monkeypatch):
        """Mixed ragged caches should stop sharding KV groups when TP cannot see enough heads."""
        from easydel.infra.mixins.generation import _create_mixed_standard_ragged_page_cache_configs

        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=256,
            global_head_dim=512,
            num_attention_heads=8,
            num_key_value_heads=1,
            attention_k_eq_v=False,
            attn_mechanism="ragged_page_attention_v3",
        )
        text_config = SimpleNamespace(
            num_hidden_layers=config.num_hidden_layers,
            layer_types=config.layer_types,
            head_dim=config.head_dim,
            num_key_value_heads=config.num_key_value_heads,
            global_head_dim=config.global_head_dim,
            attention_k_eq_v=config.attention_k_eq_v,
            num_global_key_value_heads=config.num_global_key_value_heads,
            partition_manager=PartitionManager(PartitionAxis(kv_head_axis="tp", data_parallel_axis="dp")),
            mesh=SimpleNamespace(shape={"dp": 1, "tp": 4}),
        )
        monkeypatch.setattr(
            RaggedPagesCacheConfig,
            "_compute_free_hbm",
            staticmethod(lambda **_kwargs: 131_072),
        )

        representative, per_layer_configs = _create_mixed_standard_ragged_page_cache_configs(
            text_config=text_config,
            max_length=128,
            page_size=16,
            hbm_utilization=0.1,
            dtype=jnp.bfloat16,
            version="v3",
        )

        assert representative.kv_head_shards == 1
        assert all(layer_cfg.kv_head_shards == 1 for layer_cfg in per_layer_configs.values())
        assert (
            per_layer_configs[0].get_shape_and_axes()[1][2]
            == per_layer_configs[1].get_shape_and_axes()[1][2]
            == common_types.EMPTY
        )

    def test_init_operations_cache_config_reuses_precomputed_mixed_ragged_configs(self, monkeypatch):
        """Mixed ragged config reuse should avoid rebuilding per-layer layouts during init."""
        import easydel.infra.mixins.generation as generation_mod

        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            attn_mechanism="ragged_page_attention_v3",
        )
        monkeypatch.setattr(
            RaggedPagesCacheConfig,
            "_compute_free_hbm",
            staticmethod(lambda **_kwargs: 131_072),
        )

        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            representative, _ = generation_mod._create_mixed_standard_ragged_page_cache_configs(
                text_config=model.config.get_text_config(),
                max_length=128,
                page_size=16,
                hbm_utilization=0.1,
                dtype=jnp.bfloat16,
                version="v3",
            )

            assert representative._mixed_layer_configs is not None

            monkeypatch.setattr(
                generation_mod,
                "_create_mixed_standard_ragged_page_cache_configs",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("expected init path to reuse cached mixed ragged configs")
                ),
            )

            views_config = model.init_operations_cache_config(
                batch_size=1,
                max_length=128,
                page_size=16,
                hbm_utilization=0.1,
                ragged_config=representative,
            )

        assert views_config[0].num_kv_heads == 2
        assert views_config[1].num_kv_heads == 1
        assert views_config[0].num_pages == views_config[1].num_pages == representative.num_pages

    def test_init_ragged_pages_preserves_mixed_layer_geometry(self, monkeypatch):
        """Public ragged cache init should allocate per-layer Gemma4 KV shapes."""
        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            attn_mechanism="ragged_page_attention_v3",
        )
        monkeypatch.setattr(
            RaggedPagesCacheConfig,
            "_compute_free_hbm",
            staticmethod(lambda **_kwargs: 32_768),
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            cache = model.init_ragged_pages(
                max_model_length=128,
                page_size=16,
                hbm_utilization=0.1,
            )

        sliding_view = cache.views[0]
        full_view = cache.views[1]

        assert sliding_view.metadata.num_kv_heads == 2
        assert sliding_view.metadata.k_headdim == 32
        assert full_view.metadata.num_kv_heads == 1
        assert full_view.metadata.k_headdim == 64
        assert sliding_view.metadata.num_pages == full_view.metadata.num_pages == 2

    def test_init_operations_cache_recomputes_mixed_num_pages(self, monkeypatch):
        """Auto cache init should recompute mixed-geometry page budgets from per-layer shapes."""
        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_k_eq_v=False,
            attn_mechanism="ragged_page_attention_v3",
        )
        monkeypatch.setattr(
            RaggedPagesCacheConfig,
            "_compute_free_hbm",
            staticmethod(lambda **_kwargs: 32_768),
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            metadata = model.create_ragged_page_cache_config(max_length=128, page_size=16, hbm_utilization=0.1)
            views_config = model.init_operations_cache_config(
                batch_size=1,
                max_length=128,
                page_size=16,
                hbm_utilization=0.1,
            )

        sliding_cfg = views_config[0]
        full_cfg = views_config[1]

        assert metadata.num_pages == 1
        assert sliding_cfg.num_pages == full_cfg.num_pages == metadata.num_pages

    def test_user_ragged_override_is_recomputed_after_v3_dtype_upcast(self, monkeypatch):
        """Provided mixed v3 page counts should be recomputed if sharding widens cache dtype."""
        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_k_eq_v=False,
            attn_mechanism="ragged_page_attention_v3",
        )
        monkeypatch.setattr(
            RaggedPagesCacheConfig,
            "_compute_free_hbm",
            staticmethod(lambda **_kwargs: 32_768),
        )

        import easydel.caching.ragged_page.cache as ragged_cache_mod

        monkeypatch.setattr(
            ragged_cache_mod,
            "_select_compatible_v3_kv_cache_dtype",
            lambda kvdtype, **_kwargs: jnp.float32,
        )
        provided_cfg = RaggedPagesCacheConfig(
            num_hidden_layers=2,
            max_model_length=128,
            num_kv_heads=2,
            k_headdim=32,
            v_headdim=32,
            hbm_utilization=0.1,
            page_size=16,
            num_pages=4,
            max_num_pages_per_req=8,
            version="v3",
            _kvdtype_str="bfloat16",
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            views_config = model.init_operations_cache_config(
                batch_size=1,
                max_length=128,
                page_size=16,
                hbm_utilization=0.1,
                ragged_config=provided_cfg,
            )

        sliding_cfg = views_config[0]
        full_cfg = views_config[1]

        assert sliding_cfg.kvdtype == jnp.float32
        assert full_cfg.kvdtype == jnp.float32
        assert sliding_cfg.num_pages == 1
        assert full_cfg.num_pages == 1

    def test_init_operations_cache_uses_mixed_unified_geometry(self, monkeypatch):
        """Unified cache init should preserve Gemma4's full-attention KV geometry."""
        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_k_eq_v=False,
            attn_mechanism="unified_attention",
        )
        monkeypatch.setattr(
            UnifiedAttentionCacheConfig,
            "_compute_free_hbm",
            staticmethod(lambda **_kwargs: 16_384),
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            metadata = model.create_unified_attention_cache_config(max_length=128, page_size=16, hbm_utilization=0.1)
            views_config = model.init_operations_cache_config(
                batch_size=1,
                max_length=128,
                page_size=16,
                hbm_utilization=0.1,
                unified_config=metadata,
            )

        sliding_cfg = views_config[0]
        full_cfg = views_config[1]

        assert metadata.num_pages == 1
        assert sliding_cfg.num_kv_heads == 2
        assert sliding_cfg.head_dim == 32
        assert full_cfg.num_kv_heads == 2
        assert full_cfg.head_dim == 64
        assert sliding_cfg.num_pages == full_cfg.num_pages == metadata.num_pages

    def test_init_unified_attention_cache_preserves_mixed_layer_geometry(self, monkeypatch):
        """Public unified cache init should allocate per-layer Gemma4 KV shapes."""
        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_k_eq_v=False,
            attn_mechanism="unified_attention",
        )
        monkeypatch.setattr(
            UnifiedAttentionCacheConfig,
            "_compute_free_hbm",
            staticmethod(lambda **_kwargs: 16_384),
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            cache = model.init_unified_attention_cache(
                max_model_length=128,
                page_size=16,
                hbm_utilization=0.1,
            )

        sliding_view = cache.views[0]
        full_view = cache.views[1]

        assert sliding_view.metadata.num_kv_heads == 2
        assert sliding_view.metadata.head_dim == 32
        assert sliding_view.key_cache.shape[2:] == (2, 32)
        assert full_view.metadata.num_kv_heads == 2
        assert full_view.metadata.head_dim == 64
        assert full_view.key_cache.shape[2:] == (2, 64)
        assert sliding_view.metadata.num_pages == full_view.metadata.num_pages == 1

    def test_init_operations_cache_uses_mixed_turboquant_geometry(self, monkeypatch):
        """TurboQuant ragged cache init should keep full-attention layers on global geometry."""
        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            attn_mechanism="ragged_page_attention_v3",
            kv_cache_quantization_config=TurboQuantConfig(bits=4),
        )
        monkeypatch.setattr(
            "easydel.caching.ragged_page.cache.per_device_hbm_budget_bytes",
            lambda *_args, **_kwargs: 16_384,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            cache = model.init_operations_cache(
                batch_size=1,
                max_length=128,
                page_size=16,
                hbm_utilization=0.1,
            )

        sliding_view = cache.views[0]
        full_view = cache.views[1]

        assert sliding_view.metadata.num_kv_heads == 2
        assert sliding_view.metadata.k_headdim == 32
        assert sliding_view.key_indices_pages.shape[2:] == (2, 16)
        assert full_view.metadata.num_kv_heads == 1
        assert full_view.metadata.k_headdim == 64
        assert full_view.key_indices_pages.shape[2:] == (1, 32)
        assert sliding_view.metadata.num_pages == full_view.metadata.num_pages == 6

    def test_esurge_cache_groups_use_full_attention_geometry(self):
        """eSurge cache groups should preserve Gemma4's full-attention KV geometry."""
        config = _base_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
        )
        specs = create_kv_cache_specs_from_config(
            config=config,
            page_size=16,
            num_kv_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            dtype=jnp.bfloat16,
        )

        sliding_spec = next(group.kv_cache_spec for group in specs if isinstance(group.kv_cache_spec, SlidingWindowSpec))
        full_spec = next(group.kv_cache_spec for group in specs if isinstance(group.kv_cache_spec, FullAttentionSpec))

        assert sliding_spec.num_kv_heads == 2
        assert sliding_spec.head_size == 32
        assert full_spec.num_kv_heads == 1
        assert full_spec.head_size == 64

    def test_esurge_window_aware_runtime_budget_uses_sliding_window_pages(self):
        """Hybrid Gemma4 cache budgeting should count sliding groups by live-window pages."""
        config = _base_text_config(
            num_hidden_layers=6,
            head_dim=32,
            global_head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            max_position_embeddings=512,
            sliding_window=64,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        specs = create_kv_cache_specs_from_config(
            config=config,
            page_size=16,
            num_kv_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            dtype=jnp.bfloat16,
        )

        estimate = estimate_runtime_page_budget(
            num_pages=57,
            kv_cache_groups=specs,
            max_model_len=128,
            max_num_batched_tokens=16,
        )

        # Sliding group: ceil(min(64 - 1 + 16, 128) / 16) + 1 = 6 pages
        # Full group: ceil(128 / 16) = 8 pages
        assert estimate.pages_per_request == 14
        assert estimate.max_num_seqs == 4

    def test_rms_norm_matches_hf_scaling_convention(self):
        """Gemma4 RMSNorm should use `kernel * norm(x)`, not `(1 + kernel) * norm(x)`."""
        config = _base_text_config(hidden_size=4)
        inputs = jnp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
        norm = Gemma4RMSNorm(config=config, param_dtype=jnp.float32, dim=4)

        expected = inputs / jnp.sqrt(jnp.mean(jnp.square(inputs), axis=-1, keepdims=True) + config.rms_norm_eps)
        actual = norm(inputs)

        assert jnp.allclose(actual, expected, atol=1e-6)


class TestGemma4ForCausalLM:
    """Tests for the CausalLM wrapper."""

    def test_forward_produces_logits(self):
        """Forward pass produces vocab-sized logits."""
        config = _base_text_config()
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((1, 8), dtype=jnp.int32))
        assert output.logits.shape == (1, 8, 1024)

    def test_logit_softcapping(self):
        """final_logit_softcapping bounds logit magnitudes."""
        config = _base_text_config(final_logit_softcapping=5.0)
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((1, 8), dtype=jnp.int32))
        # All logits should be bounded by [-5, 5] (approximately)
        assert jnp.all(jnp.abs(output.logits) <= 5.0 + 1e-3)

    def test_no_lm_head(self):
        """apply_lm_head=False skips logit computation."""
        config = _base_text_config()
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((1, 8), dtype=jnp.int32), apply_lm_head=False)
        assert output.logits is None
        assert output.last_hidden_state is not None

    def test_hf_forward_parity(self, small_model_config):
        """Text-only Gemma4 should numerically track the Hugging Face reference."""
        config = _base_text_config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            global_head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            sliding_window=small_model_config["sliding_window"],
            hidden_size_per_layer_input=0,
            enable_moe_block=False,
            num_kv_shared_layers=0,
            attention_k_eq_v=False,
        )

        tester = CausalLMTester()
        result = tester.run(
            module_name="gemma4_text",
            hf_class=transformers.Gemma4ForCausalLM,
            task=TaskType.CAUSAL_LM,
            config=config,
            small_model_config=small_model_config,
        )

        assert result.error_message == ""
        assert result.comparison is not None
        assert result.comparison.loss_match, result.comparison.details
        assert result.comparison.correct_percentage > 0.95, result.comparison.details


class TestGemma4RealisticParity:
    """Parity tests covering real Gemma4 architectural features.

    These tests exercise MQA (``num_key_value_heads=1``), mixed head
    dimensions (``head_dim != global_head_dim``), KV sharing, double-wide
    MLP, and per-layer input embeddings — all features present in the
    ``google/gemma-4-E2B-it`` checkpoint that the simpler parity test above
    does not cover.
    """

    @staticmethod
    def _run_parity(extra_config, *, use_cache_on_hf=True):
        """Run HF-vs-EasyDeL logits parity with realistic Gemma4 config.

        Args:
            extra_config: Dict of config overrides on top of the base.
            use_cache_on_hf: Whether to enable ``use_cache`` on HF forward
                so that KV sharing is active (default True).

        Returns:
            Tuple of ``(top1_match_ratio, max_logit_error)``.
        """
        base = dict(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=12,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=32,
            global_head_dim=64,
            hidden_activation="gelu_pytorch_tanh",
            rms_norm_eps=1e-6,
            attention_bias=False,
            sliding_window=64,
            max_position_embeddings=128,
            vocab_size=1024,
            attention_k_eq_v=False,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,
            enable_moe_block=False,
            use_double_wide_mlp=False,
            tie_word_embeddings=True,
            final_logit_softcapping=None,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=1,
            rope_parameters={
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                },
            },
        )
        base.update(extra_config)

        import torch

        hf_config = transformers.Gemma4TextConfig(**base)
        hf_model = transformers.Gemma4ForCausalLM(hf_config)
        hf_model.eval().float()

        from easydel import traversals

        hf_config = transformers.Gemma4TextConfig(**base)
        hf_model = transformers.Gemma4ForCausalLM(hf_config)
        hf_model.eval().float()

        ed_config = Gemma4TextConfig(**base)
        ed_config.attach_custom_arguments()
        device_count = jax.device_count()
        ed_config.add_basic_configurations(
            sharding_axis_dims=(1, 1, device_count, 1, 1),
            attn_mechanism="vanilla",
            attn_dtype=jnp.float32,
        )
        mesh = ed_config.mesh

        with mesh:
            ed_model = Gemma4ForCausalLM.lazy_init(
                config=ed_config,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                precision=jax.lax.Precision.HIGHEST,
                rngs=nn.Rngs(0),
            )
            ed_model = traversals.merge_model_and_tree(
                ed_model,
                tree=ed_model.transform_fn(hf_model.state_dict()),
            )
            ed_model.eval()

        np.random.seed(42)  # noqa
        ids_np = np.random.randint(0, base["vocab_size"], (1, 16))  # noqa

        with torch.no_grad():
            hf_out = hf_model(
                input_ids=torch.from_numpy(ids_np).long(),
                attention_mask=torch.ones(1, 16, dtype=torch.long),
                use_cache=use_cache_on_hf,
            )
        hf_logits = hf_out.logits.numpy()

        with mesh:
            ed_out = ed_model(
                input_ids=jnp.asarray(ids_np, dtype=jnp.int32),
                attention_mask=jnp.ones((1, 16), dtype=jnp.bool_),
            )
        ed_logits = np.array(ed_out.logits)

        max_err = float(np.abs(hf_logits - ed_logits).max())
        top1 = float(np.mean(np.argmax(hf_logits, axis=-1) == np.argmax(ed_logits, axis=-1)))
        return top1, max_err

    def test_mqa_mixed_head_dim(self):
        """MQA with different head_dim / global_head_dim must match HF."""
        top1, max_err = self._run_parity({})
        assert top1 >= 0.6, f"top-1 match {top1} < 0.6, max_err={max_err}"
        assert max_err < 0.2, f"max logit error {max_err} >= 0.2"

    def test_kv_sharing(self):
        """KV sharing must reuse donor layer K/V, matching HF use_cache=True."""
        top1, max_err = self._run_parity({"num_kv_shared_layers": 4})
        assert top1 >= 0.7, f"top-1 match {top1} < 0.7, max_err={max_err}"
        assert max_err < 0.15, f"max logit error {max_err} >= 0.15"

    def test_kv_sharing_with_double_wide_mlp(self):
        """KV sharing + double-wide MLP must match HF."""
        top1, max_err = self._run_parity({"num_kv_shared_layers": 4, "use_double_wide_mlp": True})
        assert top1 >= 0.6, f"top-1 match {top1} < 0.6, max_err={max_err}"
        assert max_err < 0.2, f"max logit error {max_err} >= 0.2"

    def test_per_layer_input(self):
        """Per-layer input embeddings must match HF."""
        top1, max_err = self._run_parity({"hidden_size_per_layer_input": 32, "vocab_size_per_layer_input": 1024})
        assert top1 >= 0.6, f"top-1 match {top1} < 0.6, max_err={max_err}"
        assert max_err < 0.2, f"max logit error {max_err} >= 0.2"

    def test_all_features_combined(self):
        """All Gemma4 features enabled simultaneously must match HF."""
        top1, max_err = self._run_parity(
            {
                "num_kv_shared_layers": 4,
                "use_double_wide_mlp": True,
                "hidden_size_per_layer_input": 32,
                "vocab_size_per_layer_input": 1024,
            }
        )
        assert top1 >= 0.6, f"top-1 match {top1} < 0.6, max_err={max_err}"
        assert max_err < 0.2, f"max logit error {max_err} >= 0.2"


class TestGemma4VisionModel:
    """Tests for the Gemma4 vision encoder."""

    def test_basic_forward(self):
        """Patchified image features should flatten to Gemma4 soft tokens."""
        config = _base_vision_config()
        mesh = _make_mesh()
        with mesh:
            model = Gemma4VisionModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(pixel_values=jnp.ones((1, 3, 8, 8), dtype=jnp.float32))
        assert output.last_hidden_state.shape == (4, config.hidden_size)

    def test_hf_checkpoint_layout(self):
        """Vision parameters should expose the Hugging Face Gemma4 tree layout."""
        config = _base_vision_config(standardize=True)
        mesh = _make_mesh()
        with mesh:
            model = Gemma4VisionModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            _, state, _ = model.split_module()

        flat = flatten_dict(state.to_pure_dict())

        assert ("patch_embedder", "input_proj", "kernel") in flat
        assert ("patch_embedder", "position_embedding_table") in flat
        assert ("encoder", "layers", 0, "self_attn", "q_proj", "linear", "kernel") in flat
        assert ("encoder", "layers", 0, "mlp", "gate_proj", "linear", "kernel") in flat
        assert ("std_bias",) in flat
        assert ("std_scale",) in flat

        assert ("embeddings", "patch_embedding", "kernel") not in flat
        assert ("layers", 0, "self_attn", "q_proj", "kernel") not in flat
        assert ("standardize_bias",) not in flat
        assert ("norm", "kernel") not in flat


class TestGemma4Multimodal:
    """Tests for the multimodal Gemma4 wrapper."""

    def test_multimodal_forward(self):
        """Gemma4 should merge image features into the text stream."""
        config = Gemma4Config(
            text_config=_base_text_config(),
            vision_config=_base_vision_config(),
            image_token_id=1020,
            video_token_id=1021,
            boi_token_id=1022,
            eoi_token_id=1023,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4Model(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(
                input_ids=jnp.array([[2, 1020, 1020, 1020, 1020, 1]], dtype=jnp.int32),
                token_type_ids=jnp.array([[0, 1, 1, 1, 1, 0]], dtype=jnp.int32),
                pixel_values=jnp.ones((1, 3, 8, 8), dtype=jnp.float32),
            )
        assert model.vision_tower is not None
        assert output.last_hidden_state.shape == (1, 6, config.text_config.hidden_size)

    def test_update_inputs_for_generation_drops_prefill_only_auxiliaries(self):
        """Decode steps should not retain prompt-length multimodal auxiliaries."""
        config = Gemma4Config(
            text_config=_base_text_config(
                hidden_size_per_layer_input=16,
                vocab_size_per_layer_input=1024,
            ),
            vision_config=_base_vision_config(),
            image_token_id=1020,
            video_token_id=1021,
            boi_token_id=1022,
            eoi_token_id=1023,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForConditionalGeneration(
                config=config,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
                rngs=nn.Rngs(0),
            )

        prompt_len = 6
        next_kwargs = model.update_inputs_for_generation(
            SimpleNamespace(past_key_values="sentinel-cache"),
            {
                "position_ids": jnp.arange(prompt_len, dtype=jnp.int32)[None, :],
                "inputs_embeds": jnp.ones((1, prompt_len, config.text_config.hidden_size), dtype=jnp.bfloat16),
                "pixel_values": jnp.ones((1, 3, 8, 8), dtype=jnp.float32),
                "token_type_ids": jnp.array([[0, 1, 1, 1, 1, 0]], dtype=jnp.int32),
                "per_layer_inputs": jnp.ones(
                    (
                        1,
                        prompt_len,
                        config.text_config.num_hidden_layers,
                        config.text_config.hidden_size_per_layer_input,
                    ),
                    dtype=jnp.bfloat16,
                ),
            },
        )

        assert next_kwargs["past_key_values"] == "sentinel-cache"
        assert next_kwargs["position_ids"].shape == (1, 1)
        assert int(next_kwargs["position_ids"][0, 0]) == prompt_len
        assert "inputs_embeds" not in next_kwargs
        assert "pixel_values" not in next_kwargs
        assert "token_type_ids" not in next_kwargs
        assert "per_layer_inputs" not in next_kwargs


class TestPerLayerInputEmbeddings:
    """Tests for Gemma4's per-layer input embedding feature."""

    def test_per_layer_input_forward(self):
        """Forward pass with per-layer input embeddings enabled."""
        config = _base_text_config(
            hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=1024,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((1, 8), dtype=jnp.int32))
        assert output.logits.shape == (1, 8, 1024)

    def test_per_layer_input_shapes(self):
        """Per-layer embeddings have correct shapes."""
        config = _base_text_config(
            hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=1024,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            input_ids = jnp.ones((2, 8), dtype=jnp.int32)
            per_layer = model.get_per_layer_inputs(input_ids)
        assert per_layer.shape == (2, 8, config.num_hidden_layers, 16)

    def test_get_per_layer_inputs_applies_gemma_embedding_scale(self):
        """Per-layer token embeddings should follow Gemma4's sqrt(hidden_size) scaling."""
        config = _base_text_config(
            num_hidden_layers=3,
            hidden_size_per_layer_input=4,
            vocab_size_per_layer_input=32,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0))
            table = jnp.arange(
                config.vocab_size_per_layer_input * config.num_hidden_layers * config.hidden_size_per_layer_input,
                dtype=jnp.float32,
            ).reshape(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            )
            model.embed_tokens_per_layer.embedding.value = table
            input_ids = jnp.array([[1, 2]], dtype=jnp.int32)
            per_layer = model.get_per_layer_inputs(input_ids)

        expected = (table[np.asarray(input_ids)] * (config.hidden_size_per_layer_input**0.5)).reshape(
            1,
            2,
            config.num_hidden_layers,
            config.hidden_size_per_layer_input,
        )
        np.testing.assert_allclose(np.asarray(per_layer), expected, atol=0.0, rtol=0.0)

    def test_inputs_embeds_path_keeps_projected_per_layer_signal(self):
        """Passing only inputs_embeds should still run the per-layer projection path."""
        config = _base_text_config(
            hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=1024,
        )
        input_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
        observed: dict[str, bool] = {}
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=nn.Rngs(0))
            inputs_embeds = model.embed_tokens(input_ids.astype("i4")) * (config.hidden_size**0.5)
            original_project = model.project_per_layer_inputs

            def _record_project_path(inputs_embeds_arg, per_layer_inputs_arg=None):
                observed["called"] = True
                observed["saw_none"] = per_layer_inputs_arg is None
                return original_project(inputs_embeds_arg, per_layer_inputs_arg)

            model.project_per_layer_inputs = _record_project_path
            _ = model(inputs_embeds=inputs_embeds)

        assert observed == {"called": True, "saw_none": True}


class TestKEqVAttention:
    """Tests for attention_k_eq_v feature."""

    def test_k_eq_v_forward(self):
        """Forward pass with k_eq_v enabled."""
        config = _base_text_config(
            attention_k_eq_v=True,
            num_global_key_value_heads=2,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((1, 8), dtype=jnp.int32))
        assert output.logits.shape == (1, 8, 1024)

    def test_k_eq_v_only_on_global_layers(self):
        """k_eq_v should only apply to global (full_attention) layers."""
        config = _base_text_config(
            attention_k_eq_v=True,
            num_global_key_value_heads=2,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
        for i, layer in enumerate(model.layers):
            attn = layer.self_attn
            if config.layer_types[i] == "full_attention":
                assert attn.use_alternative_attention is True
            else:
                assert attn.use_alternative_attention is False


class TestMoE:
    """Tests for Gemma4 MoE functionality."""

    def test_moe_forward(self):
        """Forward pass with MoE enabled."""
        config = _base_text_config(
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=64,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((1, 8), dtype=jnp.int32))
        assert output.logits.shape == (1, 8, 1024)

    def test_moe_expert_weights_shape(self):
        """Expert weight tensors have correct shapes."""
        config = _base_text_config(
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=64,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
        for layer in model.layers:
            experts = layer.experts
            assert experts.gate_proj.kernel.value.shape == (4, 128, 64)
            assert experts.up_proj.kernel.value.shape == (4, 128, 64)
            assert experts.down_proj.kernel.value.shape == (4, 64, 128)

    def test_moe_hf_forward_parity(self, small_model_config):
        """MoE Gemma4 should load HF router/expert weights and match its logits."""
        config = _base_text_config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            global_head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            sliding_window=small_model_config["sliding_window"],
            hidden_size_per_layer_input=0,
            enable_moe_block=True,
            num_experts=small_model_config["num_experts"],
            top_k_experts=small_model_config["num_experts_per_tok"],
            moe_intermediate_size=small_model_config["intermediate_size"] // 2,
            num_kv_shared_layers=0,
            attention_k_eq_v=False,
        )
        config.moe_force_xla_gmm = True

        tester = CausalLMTester()
        result = tester.run(
            module_name="gemma4_text",
            hf_class=transformers.Gemma4ForCausalLM,
            task=TaskType.CAUSAL_LM,
            config=config,
            small_model_config=small_model_config,
        )

        assert result.error_message == ""
        assert result.comparison is not None
        assert result.comparison.loss_match, result.comparison.details
        assert result.comparison.correct_percentage > 0.95, result.comparison.details


class TestKVSharing:
    """Tests for KV sharing across layers."""

    def test_kv_sharing_forward(self):
        """Forward pass with KV sharing enabled."""
        config = _base_text_config(
            num_hidden_layers=6,
            num_kv_shared_layers=2,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((1, 8), dtype=jnp.int32))
        assert output.logits.shape == (1, 8, 1024)

    def test_kv_sharing_layer_flags(self):
        """KV sharing flags are correctly set on attention layers."""
        config = _base_text_config(
            num_hidden_layers=6,
            num_kv_shared_layers=2,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
        # First 4 layers should not be KV-shared
        for i in range(4):
            assert model.layers[i].self_attn.is_kv_shared_layer is False
        # Last 2 layers should be KV-shared (if matching type found)
        for i in range(4, 6):
            attn = model.layers[i].self_attn
            layer_type = config.layer_types[i]
            # Check if a matching non-shared layer exists
            if layer_type in config.layer_types[:4]:
                assert attn.is_kv_shared_layer is True


class TestDoubleWideMLP:
    """Tests for double-wide MLP in KV-shared layers."""

    def test_double_wide_mlp_shapes(self):
        """MLP in KV-shared layers should have 2x intermediate size."""
        config = _base_text_config(
            num_hidden_layers=6,
            num_kv_shared_layers=2,
            use_double_wide_mlp=True,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
        # Non-shared layers: normal intermediate_size
        assert model.layers[0].mlp.gate_proj.kernel.value.shape[1] == 256
        # Shared layers (last 2): double-wide
        for i in range(4, 6):
            assert model.layers[i].mlp.gate_proj.kernel.value.shape[1] == 512

    def test_double_wide_forward(self):
        """Forward pass with double-wide MLP."""
        config = _base_text_config(
            num_hidden_layers=6,
            num_kv_shared_layers=2,
            use_double_wide_mlp=True,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((1, 8), dtype=jnp.int32))
        assert output.logits.shape == (1, 8, 1024)


class TestLayerScalar:
    """Tests for per-layer scalar weighting."""

    def test_layer_scalar_init(self):
        """Layer scalars are initialized to 1.0."""
        config = _base_text_config()
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
        for layer in model.layers:
            assert jnp.allclose(layer.layer_scalar.value, jnp.ones(1))


class TestVNorm:
    """Tests for value normalization."""

    def test_v_norm_exists(self):
        """Each attention layer has a v_norm."""
        config = _base_text_config()
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
        for layer in model.layers:
            assert hasattr(layer.self_attn, "v_norm")
            assert layer.self_attn.v_norm.with_scale is False


class TestRoPE:
    """Tests for Gemma4's per-layer-type RoPE."""

    def test_different_rope_for_global_vs_local(self):
        """Global and local layers use different RoPE frequencies."""
        config = _base_text_config()
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            local_freq = model.default_frequencies
            global_freq = model.global_frequencies
        assert local_freq is not global_freq

    def test_partial_rotary_factor(self):
        """Global layers keep full-width RoPE caches with a neutral no-RoPE tail."""
        config = _base_text_config(
            global_head_dim=64,
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {"rope_type": "default", "partial_rotary_factor": 0.5, "rope_theta": 1_000_000.0},
            },
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4TextModel(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            global_freq = model.global_frequencies
        freq_array = global_freq.frequencies if hasattr(global_freq, "frequencies") else global_freq
        assert freq_array.shape[-1] == 64

        cos, sin = jnp.split(freq_array, 2, axis=-1)
        assert jnp.allclose(cos[:, 16:], 1.0)
        assert jnp.allclose(sin[:, 16:], 0.0)
        assert jnp.any(jnp.abs(sin[1:, :16]) > 0)


class TestAllFeatures:
    """Integration test combining all features."""

    def test_all_features_forward(self):
        """Forward pass with all Gemma4 features enabled simultaneously."""
        config = _base_text_config(
            num_hidden_layers=6,
            hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=1024,
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=64,
            num_kv_shared_layers=2,
            use_double_wide_mlp=True,
            attention_k_eq_v=True,
            num_global_key_value_heads=2,
            final_logit_softcapping=5.0,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((2, 16), dtype=jnp.int32))
        assert output.logits.shape == (2, 16, 1024)
        assert jnp.all(jnp.abs(output.logits) <= 5.0 + 0.1)

    def test_batch_sizes(self):
        """Model handles different batch sizes."""
        config = _base_text_config()
        mesh = _make_mesh()
        with mesh:
            model = Gemma4ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            for batch_size in [1, 4]:
                output = model(input_ids=jnp.ones((batch_size, 8), dtype=jnp.int32))
                assert output.logits.shape == (batch_size, 8, 1024)


class TestGemma4Config:
    """Tests for configuration classes."""

    def test_default_rope_parameters(self):
        """Default rope_parameters are set correctly."""
        config = _base_text_config()
        assert "sliding_attention" in config.rope_parameters
        assert "full_attention" in config.rope_parameters
        assert config.rope_parameters["sliding_attention"]["rope_theta"] == 10_000.0
        assert config.rope_parameters["full_attention"]["rope_theta"] == 1_000_000.0

    def test_bidirectional_attention_adjusts_window(self):
        """use_bidirectional_attention='all' halves the sliding window."""
        config = _base_text_config(sliding_window=512, use_bidirectional_attention="all")
        assert config.sliding_window == 257  # (512 // 2) + 1

    def test_multimodal_config(self):
        """Gemma4Config wraps text and vision configs."""
        text_cfg = _base_text_config()
        vision_cfg = Gemma4VisionConfig()
        config = Gemma4Config(text_config=text_cfg, vision_config=vision_cfg)
        assert config.text_config is text_cfg
        assert config.vision_config is vision_cfg
        assert config.model_type == "gemma4"

    def test_vision_config_can_be_disabled(self):
        """Passing vision_config=None keeps the top-level config text-only."""
        text_cfg = _base_text_config()
        config = Gemma4Config(text_config=text_cfg, vision_config=None)
        assert config.text_config is text_cfg
        assert config.vision_config is None

    def test_text_only_top_level_model_initializes_without_vision_backend(self):
        """Gemma4Model should still work for text-only configs."""
        config = Gemma4Config(text_config=_base_text_config(), vision_config=None)
        mesh = _make_mesh()
        with mesh:
            model = Gemma4Model(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model(input_ids=jnp.ones((1, 8), dtype=jnp.int32))
        assert output.last_hidden_state.shape == (1, 8, config.text_config.hidden_size)

    def test_registered_vision_backend_is_used(self):
        """Explicit vision configs should instantiate a usable vision tower."""
        config = Gemma4Config(
            text_config=_base_text_config(),
            vision_config=_base_vision_config(),
            image_token_id=1020,
            video_token_id=1021,
            boi_token_id=1022,
            eoi_token_id=1023,
        )
        mesh = _make_mesh()
        with mesh:
            model = Gemma4Model(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nn.Rngs(0))
            output = model.compute_embedding(
                input_ids=jnp.array([[2, 1020, 1020, 1020, 1020, 1]], dtype=jnp.int32),
                pixel_values=jnp.ones((1, 3, 8, 8), dtype=jnp.float32),
            )

        assert model.vision_tower is not None
        assert output.shape == (1, 6, config.text_config.hidden_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
