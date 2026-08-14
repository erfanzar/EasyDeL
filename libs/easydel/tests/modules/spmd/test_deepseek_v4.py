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

"""Tests for the DeepSeek-V4 model (stateless forward + cached decode)."""

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
import transformers

try:
    from tests.modules.test_utils import CausalLMTester
    from tests.modules.test_utils.model_factory import create_ed_model, create_hf_model, setup_config
except ImportError:
    from tests.modules.test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]
    from tests.modules.test_utils.model_factory import (  # pyright: ignore[reportImplicitRelativeImport]
        create_ed_model,
        create_hf_model,
        setup_config,
    )

# transformers 5.5.0 does not ship DeepSeek-V4; when a future release does,
# this picks it up and the tester upgrades from EasyDeL-only to HF parity.
HF_DEEPSEEK_V4_CLASS = getattr(transformers, "DeepseekV4ForCausalLM", None)


def _v4_config_kwargs(small_model_config):
    """Small DeepSeek-V4 kwargs covering all three layer types + hash/moe mix."""
    return dict(
        vocab_size=small_model_config["vocab_size"],
        hidden_size=small_model_config["hidden_size"],
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        num_experts_per_tok=2,
        n_routed_experts=8,
        n_shared_experts=1,
        max_position_embeddings=small_model_config["max_position_embeddings"],
        # All three attention regimes across four layers.
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
        # Small compress rates so a 128-token sequence covers many windows.
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        hc_mult=2,
        hc_sinkhorn_iters=5,
        mlp_layer_types=["hash_moe", "moe", "moe", "moe"],
        swiglu_limit=10.0,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        # Keep the indexer in its deterministic regime (top-k >= possible compressed
        # entries). With random init roughly half the relu'd indexer scores are exactly
        # zero, and a truncating top-k orders those ties backend-dependently (HF included),
        # which is not a portable parity target — see
        # .claude/projects/port_hyv3_dsv4_mistral4.md. The full indexer path (scorer,
        # causal threshold, -1 invalidation) still executes.
        index_topk=128,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        tie_word_embeddings=False,
    )


class TestDeepseekV4:
    """Test suite for the DeepSeek-V4 model."""

    @pytest.fixture
    def v4_model_config(self, small_model_config):
        """Small-model dict with expert parallelism disabled.

        XLA:CPU cannot lower ``ragged-all-to-all`` (the MoE expert-parallel
        collective), so the default ``(1, 1, 1, -1, tp, 1)`` mesh (ep > 1, and
        fsdp folds into ep by default) fails on the CPU test host; dp x tp
        keeps a multi-device mesh without expert-parallel collectives.
        """
        import jax

        device_count = jax.device_count()
        tp_dim = 2 if device_count >= 2 and device_count % 2 == 0 else 1
        dp_dim = max(1, device_count // tp_dim)
        return {
            **small_model_config,
            "sharding_axis_dims": (1, -1, 1, 1, tp_dim, 1),
            "batch_size": max(small_model_config["batch_size"], dp_dim),
        }

    @pytest.fixture
    def deepseek_v4_config(self, v4_model_config):
        """Create a DeepSeek-V4-specific small config."""
        config = ed.DeepseekV4Config(**_v4_config_kwargs(v4_model_config))
        config.moe_force_xla_gmm = True
        return config

    def test_causal_lm(self, deepseek_v4_config, v4_model_config):
        """Test DeepseekV4ForCausalLM stateless forward (logits + loss finite)."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="deepseek_v4",
            hf_class=HF_DEEPSEEK_V4_CLASS,  # None on transformers <= 5.5.0
            task=ed.TaskType.CAUSAL_LM,
            config=deepseek_v4_config,
            small_model_config=v4_model_config,
        )
        assert result.success, f"DeepseekV4 CAUSAL_LM failed: {result.error_message or result.comparison}"

    def test_training_path_through_fused_kernel(self, small_model_config, monkeypatch):
        """TRAINING PATH: route the stateless forward through the fused op.

        With ``EASYDEL_COMPRESSED_WINDOW_ATTENTION_PLATFORM`` enabled, the
        stateless training forward runs through the differentiable
        ``compressed_window_attention`` ejkernel op (XLA reference) instead of
        the inline dense ``_attend``. Asserts (1) full-model gradient parity vs
        the dense path (the op's XLA math is the exact reference), and (2) a few
        ``value_and_grad`` + optimizer steps run with finite, changing params.
        """
        import optax
        from easydel.infra.base_state import EasyDeLState
        from easydel.modules.deepseek_v4 import modeling_deepseek_v4 as M

        config, model = self._build_fp32_model(small_model_config)
        seq_len = 24
        rng = np.random.default_rng(0)
        ids = jnp.asarray(rng.integers(0, config.vocab_size, size=(1, seq_len)), jnp.int32)
        positions = jnp.arange(seq_len)[None, :]

        with config.mesh:
            base_state = EasyDeLState.create(model=model, tx=optax.adam(1e-3), init_opt_state=True)

            def loss_fn(graphstate):
                mdl = base_state.merge(graphstate)
                mdl.eval()
                out = mdl(input_ids=ids, position_ids=positions)
                return jnp.mean(out.logits**2), out.logits

            # Gradient parity: dense inline _attend vs the fused op (XLA).
            monkeypatch.setattr(M, "_TRAIN_KERNEL_ENABLED", False)
            (_, logits_dense), grads_dense = jax.value_and_grad(loss_fn, has_aux=True)(base_state.graphstate)
            monkeypatch.setattr(M, "_TRAIN_KERNEL_ENABLED", True)
            monkeypatch.setattr(M, "_TRAIN_KERNEL_PLATFORM", "xla")
            (_, logits_fused), grads_fused = jax.value_and_grad(loss_fn, has_aux=True)(base_state.graphstate)

            logits_diff = float(jnp.max(jnp.abs(logits_dense - logits_fused)))
            gd = jax.tree_util.tree_leaves(grads_dense)
            gf = jax.tree_util.tree_leaves(grads_fused)
            grad_diff = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(gd, gf, strict=True) if a.size)
            assert logits_diff < 1e-4, f"fused-op logits diverged from dense: {logits_diff}"
            assert grad_diff < 1e-4, f"fused-op gradients diverged from dense: {grad_diff}"
            assert all(bool(jnp.isfinite(b).all()) for b in gf if b.size)

            # A few real train steps through the fused kernel path.
            state = base_state
            losses = []
            for _ in range(3):
                (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.graphstate)
                assert jnp.isfinite(loss)
                assert all(bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads) if g.size)
                state = state.apply_gradients(grads=grads)
                losses.append(float(loss))

        moved = max(
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(
                jax.tree_util.tree_leaves(base_state.graphstate),
                jax.tree_util.tree_leaves(state.graphstate),
                strict=True,
            )
            if a.size
        )
        assert moved > 0.0, "optimizer steps did not update any parameter through the fused kernel path"
        assert all(np.isfinite(losses)), losses

    def test_sliding_window_locality(self, small_model_config):
        """Masking math: with pure sliding layers, a perturbation at position 0
        cannot influence logits beyond ``num_layers * (window - 1)`` positions.

        Hash-MoE and hyper-connections are position-local, so any leak beyond
        the attention receptive field is a masking bug.
        """
        kwargs = _v4_config_kwargs(small_model_config)
        kwargs["layer_types"] = ["sliding_attention"] * 4
        kwargs["vocab_size"] = 128
        config = ed.DeepseekV4Config(**kwargs)
        config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
        config.moe_force_xla_gmm = True
        config.attach_custom_arguments()

        seq_len = 48
        window = kwargs["sliding_window"]
        num_layers = kwargs["num_hidden_layers"]
        reach = num_layers * (window - 1)  # last position influenced by token 0
        assert reach < seq_len - 1, "test config must leave positions beyond the receptive field"

        with config.mesh:
            model = ed.DeepseekV4ForCausalLM.sequential_init(
                config=config,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                precision=None,
                rngs=spx.Rngs(0),
            )
            model.eval()

            rng = np.random.default_rng(0)
            ids_a = rng.integers(0, kwargs["vocab_size"], size=(1, seq_len))
            ids_b = ids_a.copy()
            ids_b[0, 0] = (ids_b[0, 0] + 1) % kwargs["vocab_size"]
            mask = jnp.ones((1, seq_len), dtype=jnp.int32)
            positions = jnp.arange(seq_len)[None, :]

            logits_a = model(input_ids=jnp.asarray(ids_a), attention_mask=mask, position_ids=positions).logits
            logits_b = model(input_ids=jnp.asarray(ids_b), attention_mask=mask, position_ids=positions).logits

        diff = np.abs(np.asarray(logits_a) - np.asarray(logits_b)).max(axis=-1)[0]
        # Inside the receptive field the perturbation must show up somewhere...
        assert diff[: reach + 1].max() > 0.0, "perturbation did not propagate at all"
        # ...and beyond it, nothing may change (sliding-window causality).
        assert diff[reach + 1 :].max() == 0.0, (
            f"sliding-window mask leaked: positions beyond {reach} changed by {diff[reach + 1 :].max()}"
        )

    def _build_fp32_model(self, small_model_config, sharding_axis_dims=(1, 1, 1, 1, 1, 1), **config_overrides):
        """Build an fp32 model for cache-parity checks.

        ``precision=HIGHEST``, not ``None``: on TPU an f32 einsum at the default
        precision is a single bf16 MXU pass, so "fp32 model" would silently
        compute at the bf16 floor and every exactness assertion in this class
        (cached decode vs stateless forward, fused op vs dense ``_attend``)
        would compare two different precisions -- observed as a deterministic
        1.8e-2 logits gap against the ejkernel reference, which pins HIGHEST.
        On CPU this is a no-op.
        """
        kwargs = _v4_config_kwargs(small_model_config)
        kwargs.update(config_overrides)
        config = ed.DeepseekV4Config(**kwargs)
        config.sharding_axis_dims = sharding_axis_dims
        config.moe_force_xla_gmm = True
        config.attach_custom_arguments()
        with config.mesh:
            model = ed.DeepseekV4ForCausalLM.sequential_init(
                config=config,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                precision=jax.lax.Precision.HIGHEST,
                rngs=spx.Rngs(0),
            )
            model.eval()
            model = model.shard_model()
        return config, model

    @staticmethod
    def _cached_step_logits(model, ids, prefill_len, decode_steps, batch):
        """Prefill + N single-token decode steps; return per-step last logits."""
        total = prefill_len + decode_steps
        positions = jnp.broadcast_to(jnp.arange(total)[None, :], (batch, total))
        cache = model.init_cache(batch_size=batch, max_length=total + 8)
        out = model(
            input_ids=ids[:, :prefill_len],
            position_ids=positions[:, :prefill_len],
            past_key_values=cache,
        )
        cache = out.past_key_values
        step_logits = [out.logits[:, -1]]
        for k in range(decode_steps):
            pos = prefill_len + k
            out = model(
                input_ids=ids[:, pos : pos + 1],
                position_ids=positions[:, pos : pos + 1],
                past_key_values=cache,
            )
            cache = out.past_key_values
            step_logits.append(out.logits[:, -1])
        return step_logits, cache

    @pytest.mark.parametrize(
        ("prefill_len", "decode_steps", "sharding"),
        [
            # Prefill crosses the sliding window (8) and several compressor
            # windows; decode closes CSA (rate 2) and HCA (rate 4) windows
            # mid-generation at non-aligned offsets.
            (13, 9, (1, 1, 1, 1, 1, 1)),
            # Prefill shorter than one HCA window: the first HCA entry is
            # emitted entirely by decode steps.
            (3, 14, (1, 1, 1, 1, 1, 1)),
            # Multi-device tensor-parallel mesh. dp stays 1: the fused-MoE
            # shard_map cannot split a single decode token across a dp axis
            # (general fused-MoE decode constraint, not cache state).
            (13, 6, (1, 1, 1, 1, 2, 1)),
        ],
    )
    def test_cached_decode_matches_stateless(self, small_model_config, prefill_len, decode_steps, sharding):
        """PRIMARY parity: prefill + cached decode must equal the stateless forward.

        Runs the model statelessly on tokens[:T+N] and compares the logits at
        every decoded position against prefill(tokens[:T]) + N cached decode
        steps, in fp32, across all three layer types (sliding / CSA / HCA).
        """
        config, model = self._build_fp32_model(small_model_config, sharding_axis_dims=sharding)
        batch = 2
        total = prefill_len + decode_steps
        rng = np.random.default_rng(7)
        ids = jnp.asarray(rng.integers(0, config.vocab_size, size=(batch, total)), dtype=jnp.int32)

        with config.mesh:
            positions = jnp.broadcast_to(jnp.arange(total)[None, :], (batch, total))
            stateless = model(input_ids=ids, position_ids=positions).logits
            step_logits, cache = self._cached_step_logits(model, ids, prefill_len, decode_steps, batch)

        worst = 0.0
        for k in range(decode_steps + 1):
            pos = prefill_len - 1 + k
            diff = float(jnp.max(jnp.abs(stateless[:, pos] - step_logits[k])))
            worst = max(worst, diff)
        assert worst < 5e-5, f"cached decode diverged from stateless forward: max|diff|={worst}"

        # The per-row stream counters and emitted-entry accounting must line up.
        for view in cache.views:
            assert view.cache_position.shape == (batch,)
            assert bool(jnp.all(view.cache_position == total))
            if view.rate:
                assert view.compressor_entries.shape[1] >= total // view.rate

    def test_cached_greedy_matches_hf_generate(self, small_model_config):
        """SECONDARY parity: greedy tokens must match HF cached generation.

        Builds the tiny HF DeepseekV4 reference (transformers >= 5.13 ships it
        natively with the HCA/CSA cache layers), transfers weights to EasyDeL,
        and compares greedy continuations: HF ``generate`` with its
        DynamicCache vs EasyDeL cached prefill + decode.
        """
        if HF_DEEPSEEK_V4_CLASS is None:
            pytest.skip("transformers does not ship DeepseekV4ForCausalLM")
        import torch

        model_config = {
            **small_model_config,
            "sharding_axis_dims": (1, 1, 1, 1, 1, 1),
            "dtype": jnp.float32,
        }
        config = ed.DeepseekV4Config(**_v4_config_kwargs(model_config))
        config.moe_force_xla_gmm = True
        config = setup_config(config, model_config)

        hf_model = create_hf_model(HF_DEEPSEEK_V4_CLASS, config)
        prompt_len, new_tokens = 11, 16
        rng = np.random.default_rng(3)
        prompt = rng.integers(2, config.vocab_size, size=(1, prompt_len))

        hf_gen_config = transformers.GenerationConfig(
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
        hf_gen_config.eos_token_id = None  # never stop early: compare full-length continuations
        with torch.no_grad():
            hf_out = hf_model.generate(torch.asarray(prompt), generation_config=hf_gen_config)
        hf_tokens = hf_out[0, prompt_len:].numpy()
        assert hf_tokens.shape[0] == new_tokens

        with config.mesh:
            ed_model = create_ed_model(
                module_name="deepseek_v4",
                task=ed.TaskType.CAUSAL_LM,
                config=config,
                small_model_config=model_config,
                hf_model=hf_model,
            )
            ids = jnp.asarray(prompt, dtype=jnp.int32)
            cache = ed_model.init_cache(batch_size=1, max_length=prompt_len + new_tokens + 4)
            positions = jnp.arange(prompt_len, dtype=jnp.int32)[None, :]
            out = ed_model(input_ids=ids, position_ids=positions, past_key_values=cache)
            cache = out.past_key_values
            ed_tokens = []
            next_token = jnp.argmax(out.logits[:, -1], axis=-1)
            ed_tokens.append(int(next_token[0]))
            for k in range(new_tokens - 1):
                pos = prompt_len + k
                out = ed_model(
                    input_ids=next_token[:, None].astype(jnp.int32),
                    position_ids=jnp.full((1, 1), pos, dtype=jnp.int32),
                    past_key_values=cache,
                )
                cache = out.past_key_values
                next_token = jnp.argmax(out.logits[:, -1], axis=-1)
                ed_tokens.append(int(next_token[0]))

        assert list(hf_tokens) == ed_tokens, (
            f"greedy divergence over {new_tokens} steps:\nHF: {list(hf_tokens)}\nED: {ed_tokens}"
        )

    def test_generation(self, small_model_config):
        """Vanilla `generate` runs through the CompressedWindowCache path.

        Uses a tp-only mesh: the tester generates with batch 1, and the
        fused-MoE shard_map cannot split a single decode token across a dp
        axis (general fused-MoE decode constraint, not cache state).
        """
        import jax

        tp_dim = 2 if jax.device_count() >= 2 and jax.device_count() % 2 == 0 else 1
        model_config = {
            **small_model_config,
            "sharding_axis_dims": (1, 1, 1, 1, tp_dim, 1),
        }
        config = ed.DeepseekV4Config(**_v4_config_kwargs(model_config))
        config.moe_force_xla_gmm = True

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="deepseek_v4",
            hf_class=None,  # EasyDeL-only generation smoke; HF parity covered above
            task=ed.TaskType.CAUSAL_LM,
            config=config,
            small_model_config=model_config,
            max_new_tokens=8,
        )
        assert result.success, f"DeepseekV4 generation failed: {result.error_message}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
