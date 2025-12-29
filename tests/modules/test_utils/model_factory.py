"""Model creation and weight transfer utilities for EasyDeL testing.

This module provides functions to create matched EasyDeL and HuggingFace model pairs,
transfer weights between them, and load model classes from HuggingFace Hub.
"""

import copy
import gc
from typing import Any

import transformers
from flax import nnx as nn

import easydel as ed
from easydel.infra.etils import EasyDeLGradientCheckPointers


def get_hf_model_from_hub(
    repo_id: str,
    small_model_config: dict,
    factory: type = transformers.AutoModelForCausalLM,
) -> tuple[type, Any]:
    """Load HuggingFace model class from hub.

    Args:
        repo_id: HuggingFace repository ID
        small_model_config: Config dict with model parameters to apply
        factory: AutoModel factory class to use

    Returns:
        Tuple of (model_class, config)
    """
    conf = transformers.AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    for k, v in small_model_config.items():
        if isinstance(v, bool | str | float | type(None) | int):
            setattr(conf, k, v)
    model = type(factory.from_config(conf, trust_remote_code=True))
    return model, conf


def create_base_config(
    module_config_class: type,
    small_model_config: dict,
) -> Any:
    """Create a base configuration from small_model_config.

    Args:
        module_config_class: EasyDeL config class
        small_model_config: Base config dictionary

    Returns:
        Configured config instance
    """
    config = module_config_class(
        num_experts_per_tok=small_model_config.get("num_experts_per_tok", 4),
        num_experts=small_model_config.get("num_experts", 16),
        num_local_experts=small_model_config.get("num_local_experts", 16),
        vocab_size=small_model_config["vocab_size"],
        hidden_size=small_model_config["hidden_size"],
        num_attention_heads=small_model_config["num_attention_heads"],
        num_hidden_layers=small_model_config["num_hidden_layers"],
        num_layers=small_model_config["num_hidden_layers"],
        gradient_checkpointing=small_model_config.get("gradient_checkpointing", EasyDeLGradientCheckPointers.NONE),
        max_position_embeddings=small_model_config["max_position_embeddings"],
        max_context_length=small_model_config["max_position_embeddings"],
        num_key_value_heads=small_model_config.get("num_key_value_heads"),
        scan_mlp_chunk_size=small_model_config.get("scan_mlp_chunk_size", 64),
        intermediate_size=small_model_config["intermediate_size"],
        rotary_dim=small_model_config.get("rotary_dim", 32),
        rms_norm_eps=small_model_config.get("rms_norm_eps", 1e-6),
        layer_norm_eps=small_model_config.get("layer_norm_eps", 1e-6),
        head_dim=small_model_config.get("head_dim", 8),
        new_decoder_architecture=True,
        num_kv_heads=small_model_config.get("num_key_value_heads"),
        multi_query=True,
        num_ln_in_parallel_attn=1,
        parallel_attn=True,
        use_expert_tensor_mode=small_model_config.get("use_expert_tensor_mode", False),
        use_parallel_residual=small_model_config.get("use_parallel_residual", True),
        qk_layernorm=small_model_config.get("qk_layernorm", False),
        rope_scaling=small_model_config.get("rope_scaling"),
        platform=small_model_config.get("platform"),
        use_scan_mlp=small_model_config.get("use_scan_mlp", False),
        scan_mlp=small_model_config.get("use_scan_mlp", False),
        use_pallas_group_matmul=small_model_config.get("use_pallas_group_matmul", True),
        moe_method=small_model_config.get("moe_method", "standard_moe"),
    )
    return config


def setup_config(
    config: Any,
    small_model_config: dict,
) -> Any:
    """Setup config with sharding and attention parameters.

    Args:
        config: Config instance to setup
        small_model_config: Config dictionary with parameters

    Returns:
        Configured config instance
    """
    config.sharding_axis_dims = small_model_config["sharding_axis_dims"]
    config.pad_token_id = 0

    # Set head_dim if not already set
    if not hasattr(config, "head_dim") or config.head_dim is None:
        if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            config.head_dim = config.hidden_size // config.num_attention_heads

    config.attach_custom_arguments()
    config.add_basic_configurations(
        use_sharding_constraint=small_model_config.get("use_sharding_constraint", False),
        scan_mlp_chunk_size=small_model_config.get("scan_mlp_chunk_size", 64),
    )
    config.add_basic_configurations(
        attn_mechanism=small_model_config["attn_mechanism"],
        blocksize_k=small_model_config["blocksize_k"],
        blocksize_q=small_model_config["blocksize_q"],
        attn_dtype=small_model_config["attn_dtype"],
    )

    return config


def create_hf_model(
    hf_class: type,
    config: Any,
) -> Any:
    """Create HuggingFace model instance.

    Args:
        hf_class: HuggingFace model class
        config: Model configuration

    Returns:
        HuggingFace model instance in eval mode
    """
    hf_config = copy.deepcopy(config)

    # Ensure deterministic attention backend for strict parity checks.
    # HF defaults to SDPA which can introduce numerical differences vs EasyDeL/JAX.
    if getattr(hf_config, "model_type", None) in {"glm4v", "glm4v_moe", "glm46v", "gemma3"}:

        def _force_eager(cfg: Any) -> None:
            if cfg is None:
                return
            if hasattr(cfg, "_attn_implementation"):
                try:
                    cfg._attn_implementation = "eager"
                except Exception:
                    pass
            if hasattr(cfg, "attn_implementation"):
                try:
                    cfg.attn_implementation = "eager"
                except Exception:
                    pass

        _force_eager(hf_config)
        for _sub_cfg_name in ("text_config", "vision_config", "audio_config", "encoder_config", "decoder_config"):
            _force_eager(getattr(hf_config, _sub_cfg_name, None))

    hf_model = hf_class(config=hf_config)
    hf_model.eval()
    hf_model = hf_model.float()
    return hf_model


def create_ed_model(
    module_name: str,
    task: ed.TaskType,
    config: Any,
    small_model_config: dict,
    hf_model: Any | None = None,
) -> Any:
    """Create and initialize EasyDeL model.

    Args:
        module_name: Name of the module (e.g., 'llama', 'arctic')
        task: Task type (CAUSAL_LM, BASE_MODULE, etc.)
        config: Model configuration
        small_model_config: Base config dictionary
        hf_model: Optional HuggingFace model to transfer weights from

    Returns:
        EasyDeL model instance with optional weight transfer
    """
    _, module_class = ed.get_modules_by_type(module_name, task)

    ed_model = module_class.lazy_init(
        config=config,
        dtype=small_model_config["dtype"],
        param_dtype=small_model_config["dtype"],
        precision=small_model_config["precision"],
        rngs=nn.Rngs(0),
    )

    if hf_model is not None:
        ed_model = transfer_weights(ed_model, hf_model)

    ed_model.eval()
    ed_model = ed_model.shard_model()

    return ed_model


def create_ed_model_only(
    module_name: str,
    task: ed.TaskType,
    config: Any,
    small_model_config: dict,
) -> Any:
    """Create EasyDeL model without HuggingFace comparison.

    Uses sequential_init instead of lazy_init.

    Args:
        module_name: Name of the module
        task: Task type
        config: Model configuration
        small_model_config: Base config dictionary

    Returns:
        EasyDeL model instance
    """
    _, module_class = ed.get_modules_by_type(module_name, task)

    ed_model = module_class.sequential_init(
        config=config,
        dtype=small_model_config["dtype"],
        param_dtype=small_model_config["dtype"],
        precision=small_model_config["precision"],
        rngs=nn.Rngs(0),
    )

    ed_model.eval()
    ed_model = ed_model.shard_model()

    return ed_model


def transfer_weights(ed_model: Any, hf_model: Any) -> Any:
    """Transfer weights from HuggingFace to EasyDeL model.

    Args:
        ed_model: EasyDeL model (lazily initialized)
        hf_model: HuggingFace model with weights

    Returns:
        EasyDeL model with transferred weights
    """
    return ed.traversals.merge_model_and_tree(
        ed_model,
        tree=ed_model.transform_fn(hf_model.state_dict()),
    )


def create_model_pair(
    module_name: str,
    hf_class: type,
    task: ed.TaskType,
    config: Any,
    small_model_config: dict,
) -> tuple[Any, Any, Any]:
    """Create matched EasyDeL + HuggingFace model pair with transferred weights.

    Args:
        module_name: Name of the module
        hf_class: HuggingFace model class
        task: Task type
        config: Model configuration
        small_model_config: Base config dictionary

    Returns:
        Tuple of (ed_model, hf_model, config)
    """
    # Setup config
    config = setup_config(config, small_model_config)

    # Create HF model
    hf_model = create_hf_model(hf_class, config)

    # Create ED model with weight transfer
    with config.mesh:
        ed_model = create_ed_model(
            module_name=module_name,
            task=task,
            config=config,
            small_model_config=small_model_config,
            hf_model=hf_model,
        )

    return ed_model, hf_model, config


def cleanup_models(*models: Any) -> None:
    """Cleanup models and free memory.

    Args:
        *models: Model instances to delete
    """
    for model in models:
        del model
    gc.collect()


def get_module_classes(
    module_name: str,
    task: ed.TaskType,
) -> tuple[type, type]:
    """Get EasyDeL config and module classes for a given module and task.

    Args:
        module_name: Name of the module
        task: Task type

    Returns:
        Tuple of (config_class, module_class)
    """
    return ed.get_modules_by_type(module_name, task)
