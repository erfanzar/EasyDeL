import jax.numpy as jnp
from easydel.infra import EasyDeLBaseConfig
from easydel.operations import OperationMetadata
from ejkernel.modules.operations.configs import GatedDeltaRuleConfig, RaggedGatedDeltaRuleV2Config


def test_base_config_explicit_operation_configs_override_checkpoint_default():
    gdr_cfg = GatedDeltaRuleConfig(
        platform="pallas",
        backend="tpu",
        chunk_size=128,
        use_input_dtype_phase1_outputs=True,
        use_input_dtype_state=True,
    )

    cfg = EasyDeLBaseConfig(operation_configs={"gated_delta_rule": gdr_cfg})

    assert cfg.operation_configs == {"gated_delta_rule": gdr_cfg}


def test_operation_metadata_inherits_operation_configs_from_base_config():
    gdr_cfg = GatedDeltaRuleConfig(platform="xla", backend="any", chunk_size=64, use_chunked=False)
    cfg = EasyDeLBaseConfig(operation_configs={"gated_delta_rule": gdr_cfg})

    metadata = OperationMetadata(
        runtime_dtype=jnp.bfloat16,
        runtime_softmax_dtype=jnp.float32,
        base_config=cfg,
    )

    assert metadata.get_operation_config("gated_delta_rule") is gdr_cfg


def test_operation_metadata_preserves_ragged_gdr_v2_config_knobs():
    ragged_cfg = RaggedGatedDeltaRuleV2Config(
        platform="pallas",
        backend="tpu",
        chunk_size=32,
        kernel_tile_policy="b4",
        use_fused_gdn_decode=True,
    )
    cfg = EasyDeLBaseConfig(operation_configs={"ragged_gated_delta_rule_v2": ragged_cfg})

    metadata = OperationMetadata(
        runtime_dtype=jnp.bfloat16,
        runtime_softmax_dtype=jnp.float32,
        base_config=cfg,
    )

    assert metadata.get_operation_config("ragged_gated_delta_rule_v2") is ragged_cfg
