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

"""gated_mlp_forward: routing policy, fallback equality, quantized route."""

import types

import easydel as ed
import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers import gated_mlp_forward
from easydel.layers.mlp import _quantized_fused_call
from easydel.modules.llama.modeling_llama import LlamaMLP
from jax import numpy as jnp


def _tiny_llama_mlp(hidden=64, inter=128):
    cfg = ed.LlamaConfig(
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    with cfg.mesh:
        mlp = LlamaMLP(cfg, rngs=spx.Rngs(0))
    return cfg, mlp


class TestFallbackEquality:
    def test_forward_matches_legacy_composition(self):
        """On CPU the fused route never engages; forward must equal the
        pre-helper composition exactly."""
        cfg, mlp = _tiny_llama_mlp()
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.normal(size=(2, 8, cfg.hidden_size)), jnp.bfloat16)
        with cfg.mesh:
            got = mlp(x)
            wf = mlp.gate_up_proj.weight.value
            wd = mlp.down_proj.weight.value
        gate = np.asarray(x, np.float32) @ np.asarray(wf, np.float32)[:, : cfg.intermediate_size]
        up = np.asarray(x, np.float32) @ np.asarray(wf, np.float32)[:, cfg.intermediate_size :]
        want = (jax.nn.silu(gate) * up) @ np.asarray(wd, np.float32)
        got32 = np.asarray(got, np.float32)
        rel = np.abs(got32 - want).max() / (np.abs(want).max() + 1e-9)
        assert rel < 5e-2, f"fallback diverged from composition: {rel}"

    def test_dropout_hook_applied(self):
        """The dropout callable is honored (identity at rate 0 by default)."""
        cfg, mlp = _tiny_llama_mlp()
        x = jnp.ones((1, 4, cfg.hidden_size), jnp.bfloat16)
        marker = {}

        def spy_dropout(h):
            marker["called"] = True
            return h

        with cfg.mesh:
            gated_mlp_forward(mlp, x, dropout=spy_dropout)
        assert marker.get("called") is True


class TestQuantizedRoute:
    def _fake_quant_linear(self, codes, scales, bits=8, needs_biases=False):
        lin = types.SimpleNamespace()
        lin.use_bias = False
        lin.quant_kernel = types.SimpleNamespace(value=codes)
        lin.quant_scales = types.SimpleNamespace(value=scales)
        lin._resolve_ejkernel_params = lambda: ("channelwise", codes.shape[0], bits, needs_biases)
        return lin

    def _quantize(self, rng, k, n):
        w = rng.normal(size=(k, n)).astype(np.float32)
        scale = np.abs(w).max(axis=0, keepdims=True) / 127.0
        codes = np.clip(np.round(w / scale), -127, 127)
        return jnp.asarray(codes, jnp.int8), jnp.asarray(scale, jnp.float32), w

    def test_per_channel_int8_routes_and_matches_dequant(self, monkeypatch):
        """The quantized fused route runs and matches dequantized math."""
        from easydel.layers import mlp as mlp_mod

        cfg, _real_mlp = _tiny_llama_mlp()
        monkeypatch.setattr(mlp_mod, "ParallelLinearQuantized", types.SimpleNamespace)
        rng = np.random.default_rng(1)
        k, i = cfg.hidden_size, cfg.intermediate_size
        gu_codes, gu_scales, _gu_w = self._quantize(rng, k, 2 * i)
        dn_codes, dn_scales, _dn_w = self._quantize(rng, i, k)

        fake = types.SimpleNamespace()
        fake.config = cfg
        fake.act_fn = jax.nn.silu
        fake.gate_up_proj = self._fake_quant_linear(gu_codes, gu_scales)
        fake.down_proj = self._fake_quant_linear(dn_codes, dn_scales)

        x = jnp.asarray(rng.normal(size=(16, k)), jnp.bfloat16)
        with cfg.mesh:
            out = _quantized_fused_call(fake, cfg, x)
        assert out is not None, "per-channel int8 route did not engage"

        gq = np.asarray(gu_codes, np.float32) * np.asarray(gu_scales, np.float32)
        dq = np.asarray(dn_codes, np.float32) * np.asarray(dn_scales, np.float32)
        x32 = np.asarray(x, np.float32)
        want = (jax.nn.silu(x32 @ gq[:, :i]) * (x32 @ gq[:, i:])) @ dq
        got = np.asarray(out, np.float32)
        rel = np.abs(got - want).max() / (np.abs(want).max() + 1e-9)
        assert rel < 5e-2, f"quantized route relerr {rel}"

    def test_grouped_quantization_rejected(self, monkeypatch):
        """Group-wise scales (more than one group) must NOT take the fused route."""
        from easydel.layers import mlp as mlp_mod

        cfg, _ = _tiny_llama_mlp()
        monkeypatch.setattr(mlp_mod, "ParallelLinearQuantized", types.SimpleNamespace)
        rng = np.random.default_rng(2)
        k, i = cfg.hidden_size, cfg.intermediate_size
        gu_codes, _, _ = self._quantize(rng, k, 2 * i)
        dn_codes, dn_scales, _ = self._quantize(rng, i, k)
        grouped_scales = jnp.ones((2, 2 * i), jnp.float32)

        fake = types.SimpleNamespace()
        fake.config = cfg
        fake.act_fn = jax.nn.silu
        fake.gate_up_proj = self._fake_quant_linear(gu_codes, grouped_scales)
        fake.down_proj = self._fake_quant_linear(dn_codes, dn_scales)

        x = jnp.ones((16, k), jnp.bfloat16)
        with cfg.mesh:
            assert _quantized_fused_call(fake, cfg, x) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_act_fn_identity_guard_blocks_mismatched_activation():
    """A module whose act_fn is not ACT2FN[hidden_act] must not take the
    fused route (the gemma-v1 hidden_activation hazard class)."""
    from easydel.layers.mlp import _act_name

    cfg, mlp = _tiny_llama_mlp()
    assert _act_name(cfg, mlp) == "silu"
    mlp.act_fn = jax.nn.gelu  # simulates activation derived from another field
    assert _act_name(cfg, mlp) is None


def test_declared_combine_runs_in_fallback_composition():
    """A declared clamped_swiglu combine must produce clamped math on the
    fallback path too (act_fn(gate)*up would be wrong for these families)."""
    cfg, mlp = _tiny_llama_mlp()
    object.__setattr__(mlp, "fused_act_name", "clamped_swiglu")
    object.__setattr__(mlp, "fused_act_params", (0.5,))
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(2, 8, cfg.hidden_size)) * 3.0, jnp.bfloat16)
    with cfg.mesh:
        got = gated_mlp_forward(mlp, x)
        wf = np.asarray(mlp.gate_up_proj.weight.value, np.float32)
        wd = np.asarray(mlp.down_proj.weight.value, np.float32)
    i = cfg.intermediate_size
    g = np.asarray(x, np.float32) @ wf[:, :i]
    u = np.asarray(x, np.float32) @ wf[:, i:]
    g = np.minimum(g, 0.5)
    u = np.clip(u, -0.5, 0.5)
    want = ((g / (1 + np.exp(-g))) * u) @ wd
    got32 = np.asarray(got, np.float32)
    rel = np.abs(got32 - want).max() / (np.abs(want).max() + 1e-9)
    assert rel < 5e-2, f"declared combine ignored in fallback: relerr {rel}"


def test_output_scale_applied():
    """output_scale (Falcon-H1 muP down multiplier) scales the result."""
    cfg, mlp = _tiny_llama_mlp()
    x = jnp.asarray(np.random.default_rng(4).normal(size=(1, 4, cfg.hidden_size)), jnp.bfloat16)
    with cfg.mesh:
        base = np.asarray(gated_mlp_forward(mlp, x), np.float32)
        scaled = np.asarray(gated_mlp_forward(mlp, x, output_scale=2.0), np.float32)
    np.testing.assert_allclose(scaled, base * 2.0, rtol=1e-2)


class TestRealChannelwiseQuantization:
    """End-to-end with REAL quantized linears (the production format)."""

    def _quantized_mlp(self, bits=8, activation_bits=None):
        import types as _types

        from easydel.layers.linears import ColumnParallelLinearQuantized, RowParallelLinearQuantized
        from easydel.layers.quantization import QuantizationConfig, QuantizationType

        cfg, _ref_mlp = _tiny_llama_mlp()
        qcfg = QuantizationConfig(dtype=QuantizationType.CHANNELWISE, bits=bits, activation_bits=activation_bits)
        with cfg.mesh:
            gate_up = ColumnParallelLinearQuantized(
                cfg.hidden_size, 2 * cfg.intermediate_size, use_bias=False, config=qcfg, rngs=spx.Rngs(0)
            )
            down = RowParallelLinearQuantized(
                cfg.intermediate_size, cfg.hidden_size, use_bias=False, config=qcfg, rngs=spx.Rngs(1)
            )
        fake = _types.SimpleNamespace()
        fake.config = cfg
        fake.act_fn = jax.nn.silu
        fake.gate_up_proj = gate_up
        fake.down_proj = down
        return cfg, fake

    def _dequant(self, linear):
        codes = np.asarray(linear.quant_kernel.value, np.float32)
        scales = np.asarray(linear.quant_scales.value, np.float32).reshape(1, -1)
        return codes * scales

    def test_int8_engages_and_matches_dequant(self):
        cfg, mlp = self._quantized_mlp(bits=8)
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.normal(size=(16, cfg.hidden_size)), jnp.bfloat16)
        with cfg.mesh:
            out = _quantized_fused_call(mlp, cfg, x)
        assert out is not None, "real channelwise int8 did not engage"
        gq = self._dequant(mlp.gate_up_proj)
        dq = self._dequant(mlp.down_proj)
        i = cfg.intermediate_size
        x32 = np.asarray(x, np.float32)
        want = (jax.nn.silu(x32 @ gq[:, :i]) * (x32 @ gq[:, i:])) @ dq
        rel = np.abs(np.asarray(out, np.float32) - want).max() / (np.abs(want).max() + 1e-9)
        assert rel < 5e-2, f"relerr {rel}"

    def test_int4_engages_without_packed_companion_at_small_shapes(self):
        """Small int4 layers carry NO packed twin (dead weight below the
        fused W4A4 pair gate) and still engage the W4A16/int-dot route."""
        cfg, mlp = self._quantized_mlp(bits=4)
        assert mlp.gate_up_proj.quant_kernel.value.dtype == jnp.int4
        assert mlp.gate_up_proj.quant_kernel_packed is None
        assert mlp.down_proj.quant_kernel_packed is None
        rng = np.random.default_rng(1)
        x = jnp.asarray(rng.normal(size=(16, cfg.hidden_size)), jnp.bfloat16)
        with cfg.mesh:
            out = _quantized_fused_call(mlp, cfg, x)
        assert out is not None, "real channelwise int4 did not engage"
        gq = self._dequant(mlp.gate_up_proj)
        dq = self._dequant(mlp.down_proj)
        i = cfg.intermediate_size
        x32 = np.asarray(x, np.float32)
        want = (jax.nn.silu(x32 @ gq[:, :i]) * (x32 @ gq[:, i:])) @ dq
        rel = np.abs(np.asarray(out, np.float32) - want).max() / (np.abs(want).max() + 1e-9)
        assert rel < 8e-2, f"relerr {rel}"

    def test_packed_companion_gate_mirrors_fused_route(self):
        """Only layers that can pass the fused W4A4 pair gate get a twin.

        The pair gate needs gate_up + down packed bytes >=
        ``W4A4_FUSED_PAIR_MIN_PACKED_BYTES``; the smallest possible member of
        a qualifying pair (the down projection) packs one third of that, so
        anything smaller — every 7B attention projection — must not
        materialize a companion.
        """
        from easydel.layers.linears import W4A4_FUSED_PAIR_MIN_PACKED_BYTES
        from easydel.layers.linears._linear_quantized import should_pack_int4_companion

        member_floor = W4A4_FUSED_PAIR_MIN_PACKED_BYTES // 3
        # 7B attention shapes: far below the floor -> no companion.
        assert not should_pack_int4_companion(4096, 4096)
        assert not should_pack_int4_companion(4096, 4096 + 2 * 1024)  # fused qkv
        # 7B gate_up [4096, 2*11008] packs 45MB and its down packs 22.5MB:
        # the pair passes the 48MB gate, so both members keep their twins.
        assert should_pack_int4_companion(4096, 2 * 11008)
        assert should_pack_int4_companion(11008, 4096)
        # Exactly at the per-member floor.
        assert should_pack_int4_companion(2, member_floor)
        assert not should_pack_int4_companion(2, member_floor - 2)

    def test_packed_companion_matches_codes_for_large_layer(self):
        """A large-enough int4 layer packs a uint8 twin that decodes back to
        the stored codes."""
        from easydel.layers.linears import ColumnParallelLinearQuantized
        from easydel.layers.quantization import QuantizationConfig, QuantizationType

        qcfg = QuantizationConfig(dtype=QuantizationType.CHANNELWISE, bits=4)
        # [2048, 16384] packs exactly 16 MiB = the per-member floor.
        linear = ColumnParallelLinearQuantized(2048, 16384, use_bias=False, config=qcfg, rngs=spx.Rngs(0))
        assert linear.quant_kernel_packed is not None
        packed = np.asarray(linear.quant_kernel_packed.value)
        assert packed.dtype == np.uint8
        codes = np.asarray(linear.quant_kernel.value.astype(jnp.int8))
        low = (packed & 0xF).astype(np.int8)
        high = ((packed >> 4) & 0xF).astype(np.int8)
        low[low > 7] -= 16
        high[high > 7] -= 16
        rebuilt = np.stack([low, high], axis=1).reshape(codes.shape)
        np.testing.assert_array_equal(rebuilt, codes)

    def test_linear_forward_matches_dequant(self):
        """The quantized LINEAR itself (not the helper) uses the channelwise op."""
        cfg, mlp = self._quantized_mlp(bits=8)
        rng = np.random.default_rng(2)
        x = jnp.asarray(rng.normal(size=(4, 16, cfg.hidden_size)), jnp.bfloat16)
        with cfg.mesh:
            out = np.asarray(mlp.gate_up_proj(x), np.float32)
        want = np.asarray(x, np.float32) @ self._dequant(mlp.gate_up_proj)
        rel = np.abs(out - want).max() / (np.abs(want).max() + 1e-9)
        assert rel < 5e-2, f"linear forward relerr {rel}"

    def test_config_roundtrip(self):
        from easydel.layers.quantization import QuantizationConfig, QuantizationType

        qcfg = QuantizationConfig(dtype=QuantizationType.CHANNELWISE, bits=4, activation_bits=4)
        again = QuantizationConfig.from_dict(qcfg.to_dict())
        assert str(again.dtype) == "channelwise" and again.bits == 4 and again.activation_bits == 4


class TestFusedRouteTrainingGates:
    """M5/M6 gates: name-based remat + quant-under-training + FORCE_NATIVE_RUNTIME."""

    def _spy_routes(self, monkeypatch):
        from easydel.layers import mlp as mlp_mod

        calls = {"dense": 0, "quant": 0}

        def dense_spy(*a, **k):
            calls["dense"] += 1
            return None

        def quant_spy(*a, **k):
            calls["quant"] += 1
            return None

        monkeypatch.setattr(mlp_mod, "_dense_fused_call", dense_spy)
        monkeypatch.setattr(mlp_mod, "_quantized_fused_call", quant_spy)
        return calls

    def _run(self, cfg, mlp):
        x = jnp.ones((1, 4, cfg.hidden_size), jnp.bfloat16)
        with cfg.mesh:
            return gated_mlp_forward(mlp, x)

    def test_name_based_remat_forces_unfused_for_training(self, monkeypatch):
        """Training under MLP_NOTSAVEABLE must never enter the fused routes:
        the fused custom_vjp drops the checkpoint_name tags the policy keys on."""
        from easydel.infra.etils import EasyDeLGradientCheckPointers

        cfg, mlp = _tiny_llama_mlp()
        cfg.gradient_checkpointing = EasyDeLGradientCheckPointers.MLP_NOTSAVEABLE
        calls = self._spy_routes(monkeypatch)
        self._run(cfg, mlp)
        assert calls == {"dense": 0, "quant": 0}

    def test_name_based_remat_keeps_fused_for_inference(self, monkeypatch):
        """The same policy under set_inference_mode keeps the fused routes
        (no backward pass -> remat policies are irrelevant)."""
        from easydel.infra.etils import EasyDeLGradientCheckPointers
        from easydel.utils.inference_mode import set_inference_mode

        cfg, mlp = _tiny_llama_mlp()
        cfg.gradient_checkpointing = EasyDeLGradientCheckPointers.MLP_NOTSAVEABLE
        calls = self._spy_routes(monkeypatch)
        with set_inference_mode():
            self._run(cfg, mlp)
        assert calls["dense"] == 1
        assert calls["quant"] == 1

    def test_training_keeps_dense_fused_but_declines_quant(self, monkeypatch):
        """With a non-name-based policy, training keeps the bf16 fused route
        but never the quant route (frozen-weight custom_vjp would hard-zero
        trainable quant_scales gradients)."""
        from easydel.infra.etils import EasyDeLGradientCheckPointers

        cfg, mlp = _tiny_llama_mlp()
        cfg.gradient_checkpointing = EasyDeLGradientCheckPointers.NOTHING_SAVEABLE
        calls = self._spy_routes(monkeypatch)
        self._run(cfg, mlp)
        assert calls["dense"] == 1
        assert calls["quant"] == 0

    def test_force_native_runtime_forces_unfused(self, monkeypatch):
        """FORCE_NATIVE_RUNTIME (the repo-wide XLA-reference flag) must bisect
        the MLP route to the legacy composition even in inference mode."""
        from easydel.utils.inference_mode import set_inference_mode

        cfg, mlp = _tiny_llama_mlp()
        calls = self._spy_routes(monkeypatch)
        monkeypatch.setenv("FORCE_NATIVE_RUNTIME", "1")
        with set_inference_mode():
            self._run(cfg, mlp)
        assert calls == {"dense": 0, "quant": 0}


class TestDeclaredCombineMatchesConfiguredActivation:
    """A declared combine hard-codes its elementwise activation, so a family
    may only declare one when ``config.hidden_act`` actually agrees with it.

    Both families below run the *same* gate/up math in two places — the dense
    MLP and the routed experts — so a declaration that ignores ``hidden_act``
    silently desynchronizes them mid-model.
    """

    @staticmethod
    def _deepseek_v4_mlp(act):
        from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config
        from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MLP

        cfg = DeepseekV4Config(
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=64,
            hidden_act=act,
            sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        )
        cfg.mlp_bias = False
        with cfg.mesh:
            return cfg, DeepseekV4MLP(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))

    @pytest.mark.parametrize("act", ["silu", "gelu", "relu"])
    def test_deepseek_v4_dense_mlp_matches_expert_math(self, act):
        """The dense MLP must equal ``clamp -> ACT2FN[hidden_act] -> down``,
        which is exactly what ``DeepseekV4Experts.forward`` computes. Declaring
        ejkernel's silu-only ``clamped_swiglu`` for a non-silu ``hidden_act``
        made the dense half of the model compute silu while its experts kept
        the configured activation."""
        from easydel.infra.utils import ACT2FN
        from easydel.layers.layouts import split_fused_gate_up_projection

        cfg, mlp = self._deepseek_v4_mlp(act)
        x = jnp.asarray(np.random.default_rng(0).normal(size=(1, 4, cfg.hidden_size)) * 3.0, jnp.float32)
        with cfg.mesh:
            got = mlp(x)
            gate, up = split_fused_gate_up_projection(mlp.gate_up_proj(x), config=cfg)
            gate = jnp.clip(gate, max=cfg.swiglu_limit)
            up = jnp.clip(up, min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
            want = mlp.down_proj(ACT2FN[act](gate) * up)
        np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("act", ["gelu", "relu"])
    def test_deepseek_v4_non_silu_is_not_silu(self, act):
        """Guards the assertion above against being vacuous: the configured
        activation must be observably different from silu at these shapes."""
        from easydel.infra.utils import ACT2FN
        from easydel.layers.layouts import split_fused_gate_up_projection

        cfg, mlp = self._deepseek_v4_mlp(act)
        x = jnp.asarray(np.random.default_rng(0).normal(size=(1, 4, cfg.hidden_size)) * 3.0, jnp.float32)
        with cfg.mesh:
            gate, up = split_fused_gate_up_projection(mlp.gate_up_proj(x), config=cfg)
            gate = jnp.clip(gate, max=cfg.swiglu_limit)
            up = jnp.clip(up, min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
            silu_out = mlp.down_proj(ACT2FN["silu"](gate) * up)
            got = mlp(x)
        assert not np.allclose(np.asarray(got, np.float32), np.asarray(silu_out, np.float32), rtol=1e-4, atol=1e-5)

    def test_deepseek_v4_declares_fused_combine_only_for_silu(self):
        """silu keeps the fused declaration (the checkpoint default); other
        activations fall back to the explicit clamped composition."""
        assert self._deepseek_v4_mlp("silu")[1].fused_act_name == "clamped_swiglu"
        assert getattr(self._deepseek_v4_mlp("gelu")[1], "fused_act_name", None) is None


class TestKimiSituParamsSingleSource:
    """SITU's two scales are read by the dense MLP's declared combine, its
    fallback composition, and the routed experts' ``ffn_activation``; all
    three must resolve them identically."""

    @staticmethod
    def _cfg(beta, linear_beta):
        from easydel.modules.kimi_linear.modeling_kimi_linear import SITU_ACTIVATION

        return types.SimpleNamespace(
            hidden_act=SITU_ACTIVATION,
            activation_situ_beta=beta,
            activation_situ_linear_beta=linear_beta,
        )

    @pytest.mark.parametrize(
        ("beta", "linear_beta"),
        [(None, None), (2.0, None), (2.0, 3.0), (1.0, 0.0), (0.5, 4.0)],
    )
    def test_declared_params_and_composed_activation_agree(self, beta, linear_beta):
        from easydel.modules.kimi_linear.modeling_kimi_linear import resolve_gated_activation, resolve_situ_params
        from ejkernel.modules import resolve_mlp_combine

        cfg = self._cfg(beta, linear_beta)
        rng = np.random.default_rng(7)
        gate = jnp.asarray(rng.normal(size=(4, 8)) * 2.0, jnp.float32)
        up = jnp.asarray(rng.normal(size=(4, 8)) * 2.0, jnp.float32)
        composed = resolve_gated_activation(cfg)(gate, up)
        declared = resolve_mlp_combine("situ", resolve_situ_params(cfg))(gate, up)
        np.testing.assert_array_equal(np.asarray(composed), np.asarray(declared))

    @pytest.mark.parametrize("beta", [0.0, -1.0])
    def test_non_positive_beta_rejected(self, beta):
        """The gate branch divides by beta; a non-positive value produced NaN
        activations instead of an error."""
        from easydel.modules.kimi_linear.modeling_kimi_linear import resolve_situ_params

        with pytest.raises(ValueError, match="activation_situ_beta"):
            resolve_situ_params(self._cfg(beta, None))


def test_resolved_specs_follow_the_decode_scope_not_the_shape():
    """Inside a ``decode_mode_specs`` scope the fused MLP's in_specs must be
    the decode-layout specs the rest of the trace is constrained to.

    ``_resolved_specs`` feeds ``shard_map`` in_specs. eSurge packs decode as
    ``[1, N]`` tokens, so deriving the mode from ``shape[1] == 1`` classified
    a decode window as training and described a tp-sharded residual stream
    that ``decode_hidden_state_axis=None`` had already replicated.
    """
    from easydel.infra.sharding import decode_mode_specs
    from easydel.layers.mlp import _resolved_specs

    # Both axes must be > 1 for the assertions to mean anything, and the mesh
    # has to fit the live device set: the CPU trio fakes 8 devices while a
    # v5p-8 has 4, so derive the dims instead of hardcoding a device count.
    if jax.device_count() < 4:
        pytest.skip("needs at least 4 devices for a tp>1, sp>1 mesh")
    tp = 2
    sp = jax.device_count() // tp
    cfg = ed.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        sharding_axis_dims=(1, 1, 1, 1, tp, sp),
    )
    with cfg.mesh:
        mlp = LlamaMLP(cfg, rngs=spx.Rngs(0))
        packed_decode = jnp.zeros((1, 16, cfg.hidden_size), jnp.bfloat16)
        train_x_spec = _resolved_specs(mlp, cfg, packed_decode)[0]
        with decode_mode_specs(True):
            decode_x_spec, decode_wf, decode_wd = _resolved_specs(mlp, cfg, packed_decode)
        single_token_spec = _resolved_specs(mlp, cfg, jnp.zeros((1, 1, cfg.hidden_size), jnp.bfloat16))[0]
        with decode_mode_specs(True):
            train_weight_specs = _resolved_specs(mlp, cfg, packed_decode)[1:]

    assert decode_x_spec == single_token_spec, "packed [1, N] decode must resolve like a [1, 1] decode step"
    assert decode_x_spec != train_x_spec, "the scope must actually change the activation spec on this mesh"
    assert decode_x_spec[-1] is None, f"decode hidden must be replicated over tp, got {decode_x_spec}"
    # Weight layouts are declared MODE_TRAIN and do not move between modes.
    assert (decode_wf, decode_wd) == tuple(train_weight_specs)
