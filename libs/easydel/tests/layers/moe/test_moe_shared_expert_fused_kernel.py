"""``moe_shared_expert_fused_kernel``: the folded W8A8 shared expert through eJKernel's ``shared_expert_w8a8``.

On four TPU devices (ep4) with a W8A8 GLM-5-Next MoE block whose shared expert folds into the inference-mode
``moe_defer_tp_reduce`` dispatch, these tests pin: the switch is off by default; switched off, or outside inference
mode (training), the lowered program is identical to the default one; switched on under inference mode the program
runs the Pallas kernel and the output stays within the documented approximation of the default path.
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu" or jax.device_count() < 4, reason="requires 4 TPU devices"
)

AXES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _block(fused):
    import easydel as ed
    from easydel.layers.quantization import EasyQuantizer, QuantizationConfig
    from easydel.modules.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    config = ed.Glm5NextTextConfig(
        hidden_size=1024, num_attention_heads=2, num_key_value_heads=2, q_lora_rank=16, qk_rope_head_dim=0,
        index_topk=8, index_kpool=4, moe_intermediate_size=256, n_routed_experts=16, num_experts_per_tok=4,
        n_shared_experts=1, n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5,
        swiglu_limit=10.0,
    )  # fmt: skip
    config.add_basic_configurations(
        sharding_axis_dims=(1, 1, 1, 4, 1, 1), sharding_axis_names=AXES, moe_method="fused_moe",
        moe_defer_tp_reduce=True, moe_shared_expert_fused_kernel=fused,
    )  # fmt: skip
    block = Glm5NextTextMoE(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, precision=None,
                            rngs=spx.Rngs(3))  # fmt: skip
    pattern = r"(?:experts\.(?:gate_proj|up_proj|down_proj)|shared_experts\.(?:gate_up_proj|down_proj))$"
    return EasyQuantizer(quantization_config=QuantizationConfig(dtype="w8a8", pattern=pattern)).apply_quantization(block)


def _run(fused, inference):
    from easydel.infra.sharding import decode_mode_specs
    from easydel.utils.inference_mode import set_inference_mode

    block = _block(fused)
    graphdef, state = spx.export(block)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 32, 1024), jnp.float32).astype(jnp.bfloat16)
    with contextlib.ExitStack() as stack:
        stack.enter_context(block.config.mesh)
        if inference:
            stack.enter_context(set_inference_mode())
            stack.enter_context(decode_mode_specs(True))
        fn = jax.jit(lambda s, h: spx.bind(graphdef, s)(h)[0])
        lowered = fn.lower(state, x)
        text = lowered.as_text()
        out = np.asarray(lowered.compile()(state, x), np.float32)
    return text, out


def test_switch_defaults_off():
    from easydel.infra.base_config import EasyDeLBaseConfig

    assert EasyDeLBaseConfig().moe_shared_expert_fused_kernel is False
    config = EasyDeLBaseConfig()
    config.add_basic_configurations(moe_shared_expert_fused_kernel=True)
    assert config.moe_shared_expert_fused_kernel is True


def test_switch_off_and_training_unchanged():
    text_default, _ = _run(False, inference=True)
    assert "shared_expert_w8a8" not in text_default
    text_train_on, out_train_on = _run(True, inference=False)
    text_train_off, out_train_off = _run(False, inference=False)
    assert text_train_on == text_train_off
    assert np.array_equal(out_train_on.view(np.uint32), out_train_off.view(np.uint32))


def test_switch_on_runs_kernel_and_is_close():
    _, out_off = _run(False, inference=True)
    text_on, out_on = _run(True, inference=True)
    assert "ejkernel_shared_expert_w8a8_gate_up" in text_on and "ejkernel_shared_expert_w8a8_down" in text_on
    assert np.isfinite(out_on).all()
    scale = float(np.abs(out_off).max())
    assert float(np.abs(out_on - out_off).max()) <= 0.05 * scale
