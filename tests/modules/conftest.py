"""Shared pytest fixtures for model testing.

This module provides reusable fixtures for testing EasyDeL models across different
configurations, attention mechanisms, and hardware setups.
"""

import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx as nn

import easydel as ed
from easydel.infra.etils import EasyDeLGradientCheckPointers


@pytest.fixture(scope="session")
def test_seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session")
def small_model_config():
    """Standard small configuration for fast model testing."""
    return {
        "batch_size": 2,
        "vocab_size": 32000,
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_experts_per_tok": 4,
        "num_experts": 16,
        "num_local_experts": 16,
        "rms_norm_eps": 1e-6,
        "layer_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "use_cache": True,
        "use_pallas_group_matmul": True,
        # "moe_method": "standard_moe",
        "moe_method": "fused_moe",
        "sharding_axis_dims": (1, 1, 1, -1, 1),
        "use_expert_tensor_mode": True,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attention_dropout": 0.0,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "gradient_checkpointing": EasyDeLGradientCheckPointers.NONE,
        "fcm_min_ratio": -1,
        "fcm_max_ratio": -1,
        "rope_scaling": None,
        "use_scan_mlp": False,
        "bits": None,
        "hidden_act": "silu",
        "scan_layers": False,
        "shard_attention_computation": True,
        "rotary_dim": 32,
        "dtype": jnp.float32,
        "precision": jax.lax.Precision.HIGHEST,
        "attn_mechanism": "vanilla",
        "blocksize_k": 128,
        "blocksize_q": 128,
        "sequence_length": 128,
        "sliding_window": 64,
        "use_sliding_window": True,
        "head_dim": 32,
        "use_parallel_residual": True,
        "qk_layernorm": True,
        "max_position_embeddings": 128,
        "use_sharding_constraint": False,
        "pad_token_id": 0,
        "attn_dtype": jnp.bfloat16,
        "attn_softmax_dtype": jnp.bfloat16,
        "platform": None,
    }


@pytest.fixture(scope="session")
def jax_mesh():
    """Create a JAX mesh for distributed computation tests."""
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((1, 1, -1, 1, 1)),
        ("dp", "fsdp", "tp", "sp", "pp"),
    )
    return mesh


@pytest.fixture
def random_input_ids(test_seed, small_model_config):
    """Generate random input IDs for testing."""
    np.random.seed(test_seed)  # noqa: NPY002
    torch.manual_seed(test_seed)

    batch_size = small_model_config["batch_size"]
    sequence_length = small_model_config["sequence_length"]
    vocab_size = small_model_config["vocab_size"]

    np_input_ids = np.random.randint(0, vocab_size, (batch_size, sequence_length))  # noqa: NPY002
    torch_input_ids = torch.from_numpy(np_input_ids).to(torch.long)
    jax_input_ids = jnp.asarray(np_input_ids, dtype="i4")

    return torch_input_ids, jax_input_ids


@pytest.fixture
def model_factory(small_model_config, jax_mesh):
    """Factory for creating EasyDeL and HuggingFace model pairs.

    Returns a function that takes model_name, hf_class, and task_type
    and returns configured models ready for testing.
    """

    def _create_models(
        module_name: str,
        hf_module_class: type,
        task: ed.TaskType,
        custom_config: dict | None = None,
        extra_kwargs: dict | None = None,
    ):
        """Create matched EasyDeL and HuggingFace models.

        Args:
            module_name: Name of the model module (e.g., "llama", "mistral")
            hf_module_class: HuggingFace model class
            task: Task type (CAUSAL_LM, IMAGE_TEXT_TO_TEXT, etc.)
            custom_config: Optional custom configuration dict
            extra_kwargs: Optional extra kwargs for forward pass

        Returns:
            Tuple of (ed_model, hf_model, config, extra_kwargs_dict)
        """
        module_config, module_class = ed.get_modules_by_type(module_name, task)

        # Use custom config or create from small_model_config
        if custom_config is not None:
            config = custom_config
        else:
            config = module_config(**small_model_config)

        config.sharding_axis_dims = small_model_config["sharding_axis_dims"]
        config.pad_token_id = 0

        # Create HuggingFace model
        hf_model = hf_module_class(config=config)
        hf_model.eval()
        hf_model = hf_model.float()

        # Prepare EasyDeL config
        config.attach_custom_arguments()
        config.add_basic_configurations(
            shard_attention_computation=small_model_config["shard_attention_computation"],
            use_sharding_constraint=small_model_config["use_sharding_constraint"],
            scan_mlp_chunk_size=small_model_config.get("scan_mlp_chunk_size", 64),
        )

        config.add_basic_configurations(
            attn_mechanism=small_model_config["attn_mechanism"],
            blocksize_k=small_model_config["blocksize_k"],
            blocksize_q=small_model_config["blocksize_q"],
            attn_dtype=small_model_config["attn_dtype"],
        )

        # Create EasyDeL model
        with config.mesh:
            ed_model = module_class.lazy_init(
                config=config,
                dtype=small_model_config["dtype"],
                param_dtype=small_model_config["dtype"],
                precision=small_model_config["precision"],
                rngs=nn.Rngs(42),
            )
            ed_model = ed.traversals.merge_model_and_tree(ed_model, tree=ed_model.transform_fn(hf_model.state_dict()))
            ed_model.eval()
            ed_model = ed_model.shard_model()

        # Prepare extra kwargs
        kwargs_dict = {}
        if extra_kwargs:
            for k, v in extra_kwargs.items():
                kwargs_dict[k] = {
                    "jax": jnp.ones(v["shape"], dtype=getattr(jnp, v["dtype"])),
                    "torch": torch.ones(v["shape"], dtype=getattr(torch, v["dtype"])),
                }

        return ed_model, hf_model, config, kwargs_dict

    return _create_models


@pytest.fixture
def model_tester(small_model_config, random_input_ids):
    """Helper for testing model forward pass correctness.

    Compares EasyDeL and HuggingFace outputs and returns comparison results.
    """

    def _test_forward_pass(
        ed_model,
        hf_model,
        config,
        extra_kwargs: dict | None = None,
        atol: float = 0.125,
        rtol: float = 0.0,
        test_generation: bool = False,
    ):
        """Test forward pass of both models and compare outputs.

        Returns:
            dict with comparison results including:
            - outputs_match: bool
            - loss_match: bool
            - max_error: float
            - correct_percentage: float
            - ed_output: model output
            - hf_output: model output
        """
        torch_input_ids, jax_input_ids = random_input_ids

        kwargs_torch = {}
        kwargs_jax = {}
        if extra_kwargs:
            for k, v in extra_kwargs.items():
                kwargs_torch[k] = v["torch"]
                kwargs_jax[k] = v["jax"]

        # HuggingFace forward pass
        with torch.no_grad():
            try:
                hf_output = hf_model(
                    input_ids=torch_input_ids,
                    attention_mask=torch.ones_like(torch_input_ids),
                    labels=torch_input_ids,
                    output_router_logits=True,
                    past_key_values=None,
                    use_cache=test_generation,
                    **kwargs_torch,
                )
            except Exception:
                hf_output = hf_model(
                    input_ids=torch_input_ids,
                    attention_mask=torch.ones_like(torch_input_ids),
                    labels=torch_input_ids,
                    past_key_values=None,
                    use_cache=test_generation,
                    **kwargs_torch,
                )

        # EasyDeL forward pass
        with config.mesh:
            try:

                @ed.ejit(static_argnums=(1,))
                def jited(ids, gd, gs, go):
                    return nn.merge(gd, gs, go).compute_loss(
                        input_ids=ids,
                        attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                        output_router_logits=True,
                        **kwargs_jax,
                    )

                ed_output, _metrics = jited(jax_input_ids, *ed_model.split_module())
            except Exception:

                @ed.ejit(static_argnums=(1,))
                def jited(ids, gd, gs, go):
                    return nn.merge(gd, gs, go).compute_loss(
                        input_ids=ids,
                        attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                        **kwargs_jax,
                    )

                ed_output, _metrics = jited(jax_input_ids, *ed_model.split_module())

        # Compare outputs
        to = hf_output.logits.cpu().detach().numpy()
        jo = ed_output.logits

        outputs_match = jnp.allclose(to, jo, atol=atol, rtol=rtol)

        # Compare loss
        jux = getattr(ed_output, "aux_loss", 0) or 0
        tux = getattr(hf_output, "aux_loss", 0) or 0  # noqa
        ed_loss = ed_output.loss - jux
        hf_loss = hf_output.loss.cpu().detach().numpy()
        loss_match = jnp.allclose(hf_loss, ed_loss, atol=atol, rtol=rtol)

        # Calculate metrics
        max_error = float(jnp.abs(to - jo).max())
        correct_percentage = float(jnp.mean(jnp.where(jnp.isclose(to, jo, atol=atol, rtol=rtol), 1, 0)))

        # Cleanup
        del hf_model
        gc.collect()

        return {
            "outputs_match": outputs_match,
            "loss_match": loss_match,
            "max_error": max_error,
            "correct_percentage": correct_percentage,
            "ed_output": ed_output,
            "hf_output": hf_output,
            "torch_logits": to,
            "jax_logits": jo,
        }

    return _test_forward_pass


@pytest.fixture(params=["flash_attn2", "vanilla"])
def attention_mechanism(request):
    """Parametrized fixture for different attention mechanisms."""
    return request.param


@pytest.fixture(params=[jnp.float32, jnp.bfloat16])
def model_dtype(request):
    """Parametrized fixture for different data types."""
    return request.param


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Automatically cleanup memory after each test."""
    yield
    gc.collect()
    jax.clear_caches()
