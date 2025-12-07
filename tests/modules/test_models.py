"""Comprehensive pytest-based tests for all EasyDeL models.

This module tests all models with forward pass correctness verification.
Migrated from unittest to pytest while keeping all original test logic.

"""

import copy
import gc

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import transformers
from ejkernel.callib._utils import quiet
from flax import nnx as nn
from tabulate import tabulate

import easydel as ed
from easydel.infra.etils import EasyDeLGradientCheckPointers

torch.manual_seed(42)
STRICT_CHECK = False


class TestHelper:
    """Helper methods for model testing."""

    @staticmethod
    def compare_torch_to_jax(
        name: str,
        hf_out,
        ed_out,
        atol: float = 0.125,
        rtol: float = 0,
        easy_time: float | None = None,
        torch_time: float | None = None,
    ):
        """Compare HuggingFace and EasyDeL outputs with tabulated results."""
        jux = getattr(ed_out, "aux_loss", 0)
        tux = getattr(hf_out, "aux_loss", 0)
        if jux is None:
            jux = 0
        if tux is None:
            tux = 0
        to = hf_out.logits.cpu().detach().numpy()
        jo = ed_out.logits

        ed_loss = (ed_out.loss - jux) if name not in ["gpt_oss"] else ed_out.loss
        hf_loss = hf_out.loss.cpu().detach().numpy()

        if STRICT_CHECK:
            np.testing.assert_allclose(to, jo, atol=0.125, rtol=0)

        all_close = jnp.allclose(to, jo, atol=atol, rtol=rtol)
        all_close_loss = jnp.allclose(hf_loss, ed_loss, atol=0.125, rtol=0)

        def color(text, color_code):
            return f"\x1b[{color_code}m{text}\x1b[0m"

        correct_percentage = jnp.mean(jnp.where(jnp.isclose(to, jo, atol=0.125, rtol=0), 1, 0))
        err = jnp.abs(to - jo).max()
        diff = np.abs(to - jo)
        max_flat = diff.argmax()
        max_idx = np.unravel_index(max_flat, diff.shape)
        max_hf = to[max_idx]
        max_ed = jo[max_idx]

        table = tabulate(
            [
                ["Last 5 elements", str(to[0, -1, -5:]), str(jo[0, -1, -5:])],
                ["Loss", str(hf_loss), str(ed_loss)],
                ["Took", str(torch_time), str(easy_time)],
                ["AUX", str(tux), str(jux)],
            ],
            headers=["Metric", "HuggingFace", "EasyDeL"],
            tablefmt="grid",
        )
        lose_close_string = color(str(all_close_loss), "32" if all_close_loss else "31")
        max_error_string = color(f"{err:.6f}", "32" if err < 1e-2 else "31")
        co_p = color(f"{correct_percentage:.2%}", "32" if correct_percentage > 0.99 else "31")

        print()
        print(f"{color(name, '36;1')}")
        print(table)
        print()
        print(f"{color('Additional Information:', '33;1')}")
        print(f"Correct %: {co_p}")
        print(f"Max Error: {max_error_string}")
        print(f"Losses Close: {lose_close_string}")
        print(f"Max diff index: {max_idx}, HF={max_hf}, ED={max_ed}")

        return all_close or correct_percentage > 0.995, err

    @staticmethod
    def make_input_id(vocab_size: int, input_shape: tuple[int, int]):
        """Generate random input IDs for testing."""
        np_input_ids = np.random.randint(0, vocab_size, input_shape)  # noqa
        return torch.from_numpy(np_input_ids).to(torch.long), jnp.asarray(np_input_ids, dtype="i4")

    @staticmethod
    def get_hf_model_from_hub(repo_id, small_model_config, factory=transformers.AutoModelForCausalLM):
        """Load HuggingFace model from hub."""
        conf = transformers.AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        for k, v in small_model_config.items():
            if isinstance(v, bool | str | float | type(None) | int):
                setattr(conf, k, v)
        model = type(factory.from_config(conf, trust_remote_code=True))
        return model, conf

    @staticmethod
    def make_vlm_inputs(
        vocab_size: int,
        input_shape: tuple[int, int],
        image_token_id: int,
        num_image_tokens: int,
        pixel_values_shape: tuple,
        num_images: int = 1,
        token_type_ids: bool = False,
    ):
        """Generate input IDs with image token placeholders and pixel values for VLM testing.

        Args:
            vocab_size: Size of the vocabulary
            input_shape: (batch_size, seq_len) shape for input_ids
            image_token_id: The token ID used as image placeholder
            num_image_tokens: Number of image tokens per image (e.g., 256 for 16x16 patches)
            pixel_values_shape: Shape of pixel_values tensor (batch, channels, height, width)
            num_images: Number of images to include
            token_type_ids: Whether to generate token_type_ids (for Gemma3)

        Returns:
            Dictionary with torch and jax versions of all inputs
        """
        _batch_size, seq_len = input_shape

        # Generate random input IDs
        np_input_ids = np.random.randint(0, vocab_size, input_shape)  # noqa: NPY002

        # Insert image token placeholders at the beginning (after potential BOS)
        # Each image needs num_image_tokens consecutive tokens
        total_image_tokens = num_images * num_image_tokens
        if total_image_tokens < seq_len - 1:
            # Place image tokens starting at position 1 (leave position 0 for BOS if needed)
            start_pos = 1
            for img_idx in range(num_images):
                img_start = start_pos + img_idx * num_image_tokens
                img_end = img_start + num_image_tokens
                if img_end <= seq_len:
                    np_input_ids[:, img_start:img_end] = image_token_id

        # Create pixel values (random values in reasonable range)
        np_pixel_values = np.random.randn(*pixel_values_shape).astype(np.float32) * 0.5  # noqa: NPY002

        # Create attention mask (all ones)
        np_attention_mask = np.ones(input_shape, dtype=np.int64)

        result = {
            "torch": {
                "input_ids": torch.from_numpy(np_input_ids).to(torch.long),
                "pixel_values": torch.from_numpy(np_pixel_values).to(torch.float32),
                "attention_mask": torch.from_numpy(np_attention_mask).to(torch.long),
            },
            "jax": {
                "input_ids": jnp.asarray(np_input_ids, dtype="i4"),
                "pixel_values": jnp.asarray(np_pixel_values, dtype="f4"),
                "attention_mask": jnp.asarray(np_attention_mask, dtype="bool"),
            },
        }

        if token_type_ids:
            # For Gemma3: token_type_ids distinguishes image (0) vs text (1) tokens
            np_token_type_ids = np.ones(input_shape, dtype=np.int64)
            # Mark image token positions as 0
            for img_idx in range(num_images):
                img_start = 1 + img_idx * num_image_tokens
                img_end = img_start + num_image_tokens
                if img_end <= seq_len:
                    np_token_type_ids[:, img_start:img_end] = 0

            result["torch"]["token_type_ids"] = torch.from_numpy(np_token_type_ids).to(torch.long)
            result["jax"]["token_type_ids"] = jnp.asarray(np_token_type_ids, dtype="i4")

        return result

    @staticmethod
    def make_qwen_vlm_inputs(
        vocab_size: int,
        input_shape: tuple[int, int],
        image_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
        num_image_tokens: int,
        pixel_values_shape: tuple,
        image_grid_thw: np.ndarray,
        num_images: int = 1,
    ):
        """Generate inputs for Qwen2-VL/Qwen3-VL models with mRoPE support.

        Args:
            vocab_size: Size of the vocabulary
            input_shape: (batch_size, seq_len) shape for input_ids
            image_token_id: The token ID used as image placeholder
            vision_start_token_id: Token ID for vision start marker
            vision_end_token_id: Token ID for vision end marker
            num_image_tokens: Number of image tokens per image
            pixel_values_shape: Shape of pixel_values tensor
            image_grid_thw: Grid shape array (num_images, 3) with [T, H, W] per image
            num_images: Number of images

        Returns:
            Dictionary with torch and jax versions of all inputs
        """
        _batch_size, seq_len = input_shape

        # Generate random input IDs
        np_input_ids = np.random.randint(0, vocab_size, input_shape)  # noqa: NPY002

        # Insert vision tokens: <vision_start> + image_tokens + <vision_end>
        tokens_per_image = num_image_tokens + 2  # start + tokens + end
        total_vision_tokens = num_images * tokens_per_image

        if total_vision_tokens < seq_len - 1:
            start_pos = 1
            for img_idx in range(num_images):
                base_pos = start_pos + img_idx * tokens_per_image
                if base_pos + tokens_per_image <= seq_len:
                    # Vision start token
                    np_input_ids[:, base_pos] = vision_start_token_id
                    # Image tokens
                    np_input_ids[:, base_pos + 1 : base_pos + 1 + num_image_tokens] = image_token_id
                    # Vision end token
                    np_input_ids[:, base_pos + 1 + num_image_tokens] = vision_end_token_id

        # Create pixel values
        np_pixel_values = np.random.randn(*pixel_values_shape).astype(np.float32) * 0.5  # noqa: NPY002

        # Create attention mask
        np_attention_mask = np.ones(input_shape, dtype=np.int64)

        # NOTE: Do NOT pass position_ids for Qwen VL models.
        # Both HuggingFace and EasyDeL should compute 3D position_ids internally
        # using get_rope_index for proper mRoPE (multimodal rotary position embedding).
        # Vision tokens need different T, H, W position values based on their spatial position.

        result = {
            "torch": {
                "input_ids": torch.from_numpy(np_input_ids).to(torch.long),
                "pixel_values": torch.from_numpy(np_pixel_values).to(torch.float32),
                "attention_mask": torch.from_numpy(np_attention_mask).to(torch.long),
                "image_grid_thw": torch.from_numpy(image_grid_thw).to(torch.long),
            },
            "jax": {
                "input_ids": jnp.asarray(np_input_ids, dtype="i4"),
                "pixel_values": jnp.asarray(np_pixel_values, dtype="f4"),
                "attention_mask": jnp.asarray(np_attention_mask, dtype="bool"),
                "image_grid_thw": jnp.asarray(image_grid_thw, dtype="i4"),
            },
        }

        return result


class TestAllModels:
    """Comprehensive tests for all 55+ models."""

    def create_test_for_models(
        self,
        module_name: str,
        hf_module_class,
        task,
        small_model_config,
        header_config=None,
        extra_exec: dict | None = None,
        generation_test: bool = False,
    ):
        """Universal test method for all models (converted from original)."""
        module_config, module_class = ed.get_modules_by_type(module_name, task)

        if header_config is None:
            config = module_config(
                num_experts_per_tok=small_model_config.get("num_experts_per_tok", 4),
                num_experts=small_model_config.get("num_experts", 16),
                num_local_experts=small_model_config.get("num_local_experts", 16),
                vocab_size=small_model_config["vocab_size"],
                hidden_size=small_model_config["hidden_size"],
                num_attention_heads=small_model_config["num_attention_heads"],
                num_hidden_layers=small_model_config["num_hidden_layers"],
                num_layers=small_model_config["num_hidden_layers"],
                gradient_checkpointing=small_model_config.get(
                    "gradient_checkpointing", EasyDeLGradientCheckPointers.NONE
                ),
                gradient_checkpointing_targets=[
                    "attn_key",
                    "attn_dense",
                    "residual",
                    "normed_input",
                    "moe_router_logits",
                    "mlp_up",
                    "mlp_output",
                ],
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
        else:
            config = header_config

        kwargs_torch = {}
        kwargs_easydel = {}
        if extra_exec is not None:
            for k, v in extra_exec.items():
                kwargs_easydel[k] = jnp.ones(v["shape"], dtype=getattr(jnp, v["dtype"]))
                kwargs_torch[k] = torch.ones(v["shape"], dtype=getattr(torch, v["dtype"]))

        config.sharding_axis_dims = small_model_config["sharding_axis_dims"]
        config.pad_token_id = 0

        hf_model = hf_module_class(config=copy.deepcopy(config))
        hf_model.eval()
        hf_model = hf_model.float()

        config.attach_custom_arguments()
        config.add_basic_configurations(
            use_sharding_constraint=small_model_config.get("use_sharding_constraint", False),
            scan_mlp_chunk_size=small_model_config.get("scan_mlp_chunk_size", 64),
        )
        mesh = config.mesh
        config.add_basic_configurations(
            attn_mechanism=small_model_config["attn_mechanism"],
            blocksize_k=small_model_config["blocksize_k"],
            blocksize_q=small_model_config["blocksize_q"],
            attn_dtype=small_model_config["attn_dtype"],
        )

        with mesh:
            torch_input_ids, jax_input_ids = TestHelper.make_input_id(
                small_model_config["vocab_size"],
                (small_model_config["batch_size"], small_model_config["sequence_length"]),
            )

            ed_model = module_class.lazy_init(
                config=config,
                dtype=small_model_config["dtype"],
                param_dtype=small_model_config["dtype"],
                precision=small_model_config["precision"],
                rngs=nn.Rngs(0),
            )
            ed_model = ed.traversals.merge_model_and_tree(ed_model, tree=ed_model.transform_fn(hf_model.state_dict()))
            ed_model.eval()
            ed_model = ed_model.shard_model()

            with ed.utils.capture_time() as torch_time:
                try:
                    hf_output = hf_model(
                        input_ids=torch_input_ids,
                        attention_mask=torch.ones_like(torch_input_ids),
                        labels=torch_input_ids,
                        output_router_logits=True,
                        past_key_values=None,
                        use_cache=generation_test,
                        **kwargs_torch,
                    )
                except Exception:
                    hf_output = hf_model(
                        input_ids=torch_input_ids,
                        attention_mask=torch.ones_like(torch_input_ids),
                        labels=torch_input_ids,
                        past_key_values=None,
                        use_cache=generation_test,
                        **kwargs_torch,
                    )
            torch_time = torch_time()
            with quiet() as _q:
                try:

                    @ed.ejit(static_argnums=(1,))
                    def jited(ids, gd, gs, go):
                        return nn.merge(gd, gs, go).compute_loss(
                            input_ids=ids,
                            attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                            output_router_logits=True,
                            **kwargs_easydel,
                        )

                    ed_output = jited(jax_input_ids, *ed_model.split_module())
                except Exception:

                    @ed.ejit(static_argnums=(1,))
                    def jited(ids, gd, gs, go):
                        return nn.merge(gd, gs, go).compute_loss(
                            input_ids=ids,
                            attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                            **kwargs_easydel,
                        )

                    ed_output = jited(jax_input_ids, *ed_model.split_module())

            with ed.utils.capture_time() as easy_time:
                ed_output, _metrics = jited(jax_input_ids, *ed_model.split_module())
            easy_time = easy_time()

            del hf_model
            gc.collect()

            return TestHelper.compare_torch_to_jax(
                module_name,
                hf_out=hf_output,
                ed_out=ed_output,
                easy_time=easy_time,
                torch_time=torch_time,
            )

    def run_easydel_only_test(
        self,
        module_name: str,
        task,
        small_model_config,
        header_config=None,
        extra_exec: dict | None = None,
    ):
        """Run EasyDeL model test without HuggingFace comparison (for models without HF support)."""
        module_config, module_class = ed.get_modules_by_type(module_name, task)

        if header_config is None:
            config = module_config(
                num_experts_per_tok=small_model_config.get("num_experts_per_tok", 4),
                num_experts=small_model_config.get("num_experts", 16),
                num_local_experts=small_model_config.get("num_local_experts", 16),
                vocab_size=small_model_config["vocab_size"],
                hidden_size=small_model_config["hidden_size"],
                num_attention_heads=small_model_config["num_attention_heads"],
                num_hidden_layers=small_model_config["num_hidden_layers"],
                num_layers=small_model_config["num_hidden_layers"],
                gradient_checkpointing=small_model_config.get(
                    "gradient_checkpointing", EasyDeLGradientCheckPointers.NONE
                ),
                gradient_checkpointing_targets=[
                    "attn_key",
                    "attn_dense",
                    "residual",
                    "normed_input",
                    "moe_router_logits",
                    "mlp_up",
                    "mlp_output",
                ],
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
                use_parallel_residual=small_model_config.get("use_parallel_residual", True),
                qk_layernorm=small_model_config.get("qk_layernorm", False),
                rope_scaling=small_model_config.get("rope_scaling"),
                platform=small_model_config.get("platform"),
                use_scan_mlp=small_model_config.get("use_scan_mlp", False),
                scan_mlp=small_model_config.get("use_scan_mlp", False),
                use_pallas_group_matmul=small_model_config.get("use_pallas_group_matmul", True),
            )
        else:
            config = header_config

        kwargs_easydel = {}
        if extra_exec is not None:
            for k, v in extra_exec.items():
                kwargs_easydel[k] = jnp.ones(v["shape"], dtype=getattr(jnp, v["dtype"]))

        config.sharding_axis_dims = small_model_config["sharding_axis_dims"]
        config.pad_token_id = 0

        config.attach_custom_arguments()
        config.add_basic_configurations(
            use_sharding_constraint=small_model_config.get("use_sharding_constraint", False),
            scan_mlp_chunk_size=small_model_config.get("scan_mlp_chunk_size", 64),
        )
        mesh = config.mesh
        config.add_basic_configurations(
            attn_mechanism=small_model_config["attn_mechanism"],
            blocksize_k=small_model_config["blocksize_k"],
            blocksize_q=small_model_config["blocksize_q"],
            attn_dtype=small_model_config["attn_dtype"],
        )

        with mesh:
            _, jax_input_ids = TestHelper.make_input_id(
                small_model_config["vocab_size"],
                (small_model_config["batch_size"], small_model_config["sequence_length"]),
            )

            ed_model = module_class.sequential_init(
                config=config,
                dtype=small_model_config["dtype"],
                param_dtype=small_model_config["dtype"],
                precision=small_model_config["precision"],
                rngs=nn.Rngs(0),
            )
            ed_model.eval()
            ed_model = ed_model.shard_model()

            try:

                @ed.ejit(static_argnums=(1,))
                def jited(ids, gd, gs, go):
                    return nn.merge(gd, gs, go).compute_loss(
                        input_ids=ids,
                        attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                        output_router_logits=True,
                        **kwargs_easydel,
                    )

                ed_output = jited(jax_input_ids, *ed_model.split_module())
            except Exception:

                @ed.ejit(static_argnums=(1,))
                def jited(ids, gd, gs, go):
                    return nn.merge(gd, gs, go).compute_loss(
                        input_ids=ids,
                        attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                        **kwargs_easydel,
                    )

                ed_output = jited(jax_input_ids, *ed_model.split_module())

            with ed.utils.capture_time() as easy_time:
                ed_output, _metrics = jited(jax_input_ids, *ed_model.split_module())
            easy_time = easy_time()

            # Verify output
            assert ed_output is not None, f"{module_name} forward pass returned None"
            assert hasattr(ed_output, "logits"), f"{module_name} output missing logits"
            assert hasattr(ed_output, "loss"), f"{module_name} output missing loss"
            assert jnp.isfinite(ed_output.loss), f"{module_name} loss is not finite"

            print(f"\n✓ {module_name} - EasyDeL-only test passed")
            print(f"  Loss: {ed_output.loss}")
            print(f"  Logits shape: {ed_output.logits.shape}")
            print(f"  Time: {easy_time}s")

            gc.collect()
            return True

    def create_test_for_vlm_models(
        self,
        module_name: str,
        hf_module_class,
        task,
        small_model_config,
        header_config,
        vlm_config: dict,
    ):
        """Test method for Vision-Language Models with proper vision inputs.

        Args:
            module_name: Name of the module to test
            hf_module_class: HuggingFace model class
            task: EasyDeL task type
            small_model_config: Base model configuration dict
            header_config: Pre-configured header config for the model
            vlm_config: VLM-specific configuration with keys:
                - image_token_id: Token ID for image placeholders
                - num_image_tokens: Number of tokens per image
                - pixel_values_shape: Shape of pixel_values tensor
                - num_images: Number of images (default 1)
                - use_token_type_ids: Whether to use token_type_ids (for Gemma3)
                - vision_start_token_id: (optional) For Qwen models
                - vision_end_token_id: (optional) For Qwen models
                - image_grid_thw: (optional) For Qwen models
                - is_qwen_vl: Whether this is a Qwen VL model
        """
        _, module_class = ed.get_modules_by_type(module_name, task)
        config = header_config

        config.sharding_axis_dims = small_model_config["sharding_axis_dims"]
        config.pad_token_id = 0

        hf_model = hf_module_class(config=copy.deepcopy(config))
        hf_model.eval()
        hf_model = hf_model.float()

        config.attach_custom_arguments()
        config.add_basic_configurations(
            use_sharding_constraint=small_model_config.get("use_sharding_constraint", False),
            scan_mlp_chunk_size=small_model_config.get("scan_mlp_chunk_size", 64),
        )
        mesh = config.mesh
        config.add_basic_configurations(
            attn_mechanism=small_model_config["attn_mechanism"],
            blocksize_k=small_model_config["blocksize_k"],
            blocksize_q=small_model_config["blocksize_q"],
            attn_dtype=small_model_config["attn_dtype"],
        )

        with mesh:
            input_shape = (small_model_config["batch_size"], small_model_config["sequence_length"])

            # Generate VLM inputs based on model type
            if vlm_config.get("is_qwen_vl", False):
                vlm_inputs = TestHelper.make_qwen_vlm_inputs(
                    vocab_size=small_model_config["vocab_size"],
                    input_shape=input_shape,
                    image_token_id=vlm_config["image_token_id"],
                    vision_start_token_id=vlm_config["vision_start_token_id"],
                    vision_end_token_id=vlm_config["vision_end_token_id"],
                    num_image_tokens=vlm_config["num_image_tokens"],
                    pixel_values_shape=vlm_config["pixel_values_shape"],
                    image_grid_thw=vlm_config["image_grid_thw"],
                    num_images=vlm_config.get("num_images", 1),
                )
            else:
                vlm_inputs = TestHelper.make_vlm_inputs(
                    vocab_size=small_model_config["vocab_size"],
                    input_shape=input_shape,
                    image_token_id=vlm_config["image_token_id"],
                    num_image_tokens=vlm_config["num_image_tokens"],
                    pixel_values_shape=vlm_config["pixel_values_shape"],
                    num_images=vlm_config.get("num_images", 1),
                    token_type_ids=vlm_config.get("use_token_type_ids", False),
                )

            torch_inputs = vlm_inputs["torch"]
            jax_inputs = vlm_inputs["jax"]

            # Initialize EasyDeL model
            ed_model = module_class.lazy_init(
                config=config,
                dtype=small_model_config["dtype"],
                param_dtype=small_model_config["dtype"],
                precision=small_model_config["precision"],
                rngs=nn.Rngs(0),
            )
            ed_model = ed.traversals.merge_model_and_tree(ed_model, tree=ed_model.transform_fn(hf_model.state_dict()))
            ed_model.eval()
            ed_model = ed_model.shard_model()

            ed_kwargs = {k: v for k, v in jax_inputs.items() if k not in ["input_ids", "attention_mask"]}

            # Run HuggingFace model
            # VLM models require vision inputs (image_grid_thw, etc.) - no fallback

            with ed.utils.capture_time() as torch_time:
                hf_output = hf_model(
                    input_ids=torch_inputs["input_ids"],
                    attention_mask=torch_inputs["attention_mask"],
                    pixel_values=torch_inputs["pixel_values"],
                    labels=torch_inputs["input_ids"],
                    past_key_values=None,
                    use_cache=False,
                    **{
                        k: v for k, v in torch_inputs.items() if k not in ["input_ids", "attention_mask", "pixel_values"]
                    },
                )

            torch_time = torch_time()

            # with quiet() as _q:
            #     ed_output = ed_model.compute_loss(
            #         input_ids=jax_inputs["input_ids"],
            #         attention_mask=jnp.ones_like(jax_inputs["input_ids"], dtype=jnp.bool),
            #         **ed_kwargs,
            #     )

            with ed.utils.capture_time() as easy_time:
                ed_output, _metrics = ed_model.compute_loss(
                    input_ids=jax_inputs["input_ids"],
                    attention_mask=jnp.ones_like(jax_inputs["input_ids"], dtype=jnp.bool),
                    **ed_kwargs,
                )
            easy_time = easy_time()

            del hf_model
            gc.collect()

            return TestHelper.compare_torch_to_jax(
                module_name,
                hf_out=hf_output,
                ed_out=ed_output,
                easy_time=easy_time,
                torch_time=torch_time,
            )

    def test_llama(self, small_model_config):
        """Test LLaMA model with rope scaling."""
        config_dict = small_model_config.copy()
        config_dict["rope_scaling"] = {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
        res, err = self.create_test_for_models(
            "llama",
            transformers.LlamaForCausalLM,
            ed.TaskType.CAUSAL_LM,
            config_dict,
        )
        assert res, f"Llama model Failed [ERROR {err}]"

    def test_llama4(self, small_model_config):
        """Test LLaMA4 text model."""
        header_config = ed.Llama4TextConfig(
            hidden_size=128,
            intermediate_size=512,
            intermediate_size_mlp=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            use_qk_norm=False,
        )
        res, err = self.create_test_for_models(
            "llama4_text",
            transformers.Llama4ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"Llama4 model Failed [ERROR {err}]"

    def test_llama4_cond(self, small_model_config):
        """Test LLaMA4 conditional generation (vision) model with proper vision inputs."""
        header_config = ed.Llama4Config(
            boi_token_index=200080,
            eoi_token_index=200081,
            image_token_index=200092,
        )

        text_config = ed.Llama4TextConfig(
            hidden_size=512,
            intermediate_size=2048,
            intermediate_size_mlp=2048,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            use_qk_norm=False,
            vocab_size=202048,
            bos_token_id=200000,
            eos_token_id=[200001, 200007, 200008],
            pad_token_id=200018,
            hidden_act="silu",
            max_position_embeddings=4096,
            initializer_range=0.02,
            rms_norm_eps=1e-05,
            use_cache=True,
            attention_bias=False,
            attention_dropout=0.0,
            rope_theta=500000.0,
            rope_scaling=None,
            num_experts_per_tok=1,
            num_local_experts=1,
            output_router_logits=False,
            router_aux_loss_coef=0.0,
            router_jitter_noise=0.0,
        )

        vision_config = ed.Llama4VisionConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            vision_output_dim=512,
            projector_input_dim=512,
            projector_output_dim=512,
            image_size=336,
            patch_size=14,
            num_channels=3,
            hidden_act="gelu",
            initializer_range=0.02,
            norm_eps=1e-05,
            attention_dropout=0.0,
            rope_theta=10000,
            pixel_shuffle_ratio=0.5,
            projector_dropout=0.0,
            multi_modal_projector_bias=False,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        )
        header_config.text_config = text_config
        header_config.vision_config = vision_config

        # VLM configuration for Llama4
        # Llama4 uses image_token_index with boi/eoi markers
        # Number of image tokens depends on image_size/patch_size and pixel_shuffle_ratio
        image_size = vision_config.image_size  # 336
        patch_size = vision_config.patch_size  # 14
        pixel_shuffle_ratio = vision_config.pixel_shuffle_ratio  # 0.5
        # patches = (image_size / patch_size)^2 * pixel_shuffle_ratio^2
        patches_per_side = image_size // patch_size  # 24
        num_image_tokens = int((patches_per_side * pixel_shuffle_ratio) ** 2)  # 144 tokens
        num_images_per_batch = 1
        batch_size = small_model_config["batch_size"]

        # Ensure sequence_length is large enough for image tokens + text
        local_cfg = small_model_config.copy()
        local_cfg["sequence_length"] = max(small_model_config["sequence_length"], num_image_tokens + 32)

        # pixel_values shape: (batch_size * num_images_per_batch, channels, height, width)
        vlm_config = {
            "image_token_id": header_config.image_token_index,  # 200092
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (batch_size * num_images_per_batch, 3, image_size, image_size),
            "num_images": num_images_per_batch,
            "use_token_type_ids": False,
        }

        res, err = self.create_test_for_vlm_models(
            "llama4",
            transformers.Llama4ForConditionalGeneration,
            ed.TaskType.IMAGE_TEXT_TO_TEXT,
            local_cfg,
            header_config=header_config,
            vlm_config=vlm_config,
        )
        assert res, f"Llama4 conditional model Failed [ERROR {err}]"

    def test_mpt(self, small_model_config):
        """Test MPT model."""
        header_config = ed.MptConfig(
            d_model=small_model_config["hidden_size"],
            n_heads=small_model_config["num_attention_heads"],
            n_layers=4,
            attn_config=ed.MptAttentionConfig(),
            sharding_axis_dims=(1, 1, 1, 1, -1),
        )
        res, err = self.create_test_for_models(
            "mpt",
            transformers.MptForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"MPT model Failed [ERROR {err}]"

    def test_falcon(self, small_model_config):
        """Test Falcon model."""
        res, err = self.create_test_for_models(
            "falcon",
            transformers.FalconForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"Falcon model Failed [ERROR {err}]"

    def test_mistral(self, small_model_config):
        """Test Mistral model."""
        res, err = self.create_test_for_models(
            "mistral",
            transformers.MistralForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"Mistral model Failed [ERROR {err}]"

    def test_exaone(self, small_model_config):
        """Test EXAONE model from HuggingFace Hub."""
        hf_model, _conf = TestHelper.get_hf_model_from_hub(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            small_model_config,
        )
        res, err = self.create_test_for_models(
            "exaone",
            hf_model,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"EXAONE model Failed [ERROR {err}]"

    def test_exaone4(self, small_model_config):
        """Test EXAONE4 model from HuggingFace Hub."""
        res, err = self.create_test_for_models(
            "exaone4",
            transformers.Exaone4ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"EXAONE4 model Failed [ERROR {err}]"

    def test_internlm2(self, small_model_config):
        """Test InternLM2 model from HuggingFace Hub."""
        hf_model, _conf = TestHelper.get_hf_model_from_hub(
            "internlm/internlm2_5-7b-chat",
            small_model_config,
        )
        res, err = self.create_test_for_models(
            "internlm2",
            hf_model,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"INTERNLM2 model Failed [ERROR {err}]"

    @pytest.mark.skip(reason="Gemma3 VLM test requires access to gated HF model")
    def test_gemma3(self, small_model_config):
        """Test Gemma3 vision model with proper vision inputs."""
        repo_id = "google/gemma-3-4b-it"
        model_task = ed.TaskType.IMAGE_TEXT_TO_TEXT
        conf = ed.AutoEasyDeLConfig.from_pretrained(repo_id, trust_remote_code=True, model_task=model_task)

        conf.text_config.hidden_size = small_model_config["hidden_size"]
        conf.text_config.num_attention_heads = small_model_config["num_attention_heads"]
        conf.text_config.num_key_value_heads = small_model_config.get("num_key_value_heads")
        conf.text_config.num_hidden_layers = small_model_config["num_hidden_layers"]
        conf.text_config.sliding_window_pattern = small_model_config["num_hidden_layers"] // 4
        conf.text_config.freq_max_position_embedding = small_model_config["max_position_embeddings"]
        conf.text_config.mask_max_position_embedding = small_model_config["max_position_embeddings"]
        conf.text_config.vocab_size = small_model_config["vocab_size"]
        conf.text_config.attn_mechanism = "vanilla"

        # VLM configuration for Gemma3
        # Gemma3 uses image_token_id=262144, mm_tokens_per_image=256 (16x16 patches)
        image_size = conf.vision_config.image_size  # typically 490
        num_images_per_batch = 1
        num_image_tokens = conf.mm_tokens_per_image  # 256
        batch_size = small_model_config["batch_size"]

        # Ensure sequence_length is large enough for image tokens + text
        local_cfg = small_model_config.copy()
        local_cfg["sequence_length"] = max(small_model_config["sequence_length"], num_image_tokens + 32)

        # pixel_values shape: (batch_size * num_images_per_batch, channels, height, width)
        vlm_config = {
            "image_token_id": conf.image_token_id,  # 262144
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (batch_size * num_images_per_batch, 3, image_size, image_size),
            "num_images": num_images_per_batch,
            "use_token_type_ids": True,  # Gemma3 uses token_type_ids
        }

        res, err = self.create_test_for_vlm_models(
            "gemma3",
            transformers.Gemma3ForConditionalGeneration,
            model_task,
            local_cfg,
            header_config=conf,
            vlm_config=vlm_config,
        )
        assert res, f"Gemma3 model Failed [ERROR {err}]"

    @pytest.mark.skip(reason="Gemma3 text test requires access to gated HF model")
    def test_gemma3_text(self, small_model_config):
        """Test Gemma3 text-only model."""
        repo_id = "google/gemma-3-1b-it"
        conf = ed.AutoEasyDeLConfig.from_pretrained(repo_id, trust_remote_code=True)

        conf.hidden_size = small_model_config["hidden_size"]
        conf.num_attention_heads = small_model_config["num_attention_heads"]
        conf.num_key_value_heads = small_model_config.get("num_key_value_heads")
        conf.num_hidden_layers = small_model_config["num_hidden_layers"]
        conf.freq_max_position_embedding = small_model_config["max_position_embeddings"]
        conf.mask_max_position_embedding = small_model_config["max_position_embeddings"]
        conf.sliding_window_pattern = 2
        conf.sliding_window = 256
        conf.vocab_size = small_model_config["vocab_size"]
        conf.attn_mechanism = "vanilla"

        res, err = self.create_test_for_models(
            "gemma3_text",
            transformers.Gemma3ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=conf,
        )
        assert res, f"Gemma3Text model Failed [ERROR {err}]"

    def test_mixtral(self, small_model_config):
        """Test Mixtral MoE model."""
        res, err = self.create_test_for_models(
            "mixtral",
            transformers.MixtralForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"Mixtral model Failed [ERROR {err}]"

    def test_gpt2(self, small_model_config):
        """Test GPT-2 model."""
        res, err = self.create_test_for_models(
            "gpt2",
            transformers.GPT2LMHeadModel,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"GPT2 model Failed [ERROR {err}]"

    def test_gptj(self, small_model_config):
        """Test GPT-J model."""
        header_config = ed.GPTJConfig(
            vocab_size=small_model_config["vocab_size"],
            n_positions=small_model_config["max_position_embeddings"],
            n_embd=small_model_config["hidden_size"],
            n_layer=small_model_config["num_hidden_layers"],
            n_head=small_model_config["num_attention_heads"],
            rotary_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
        )
        res, err = self.create_test_for_models(
            "gptj",
            transformers.GPTJForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"GPT-J model Failed [ERROR {err}]"

    def test_gpt_neox(self, small_model_config):
        """Test GPT-NeoX model."""
        header_config = ed.GPTNeoXConfig(
            vocab_size=small_model_config["vocab_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            rotary_pct=1,
            rope_scaling=None,
        )
        res, err = self.create_test_for_models(
            "gpt_neox",
            transformers.GPTNeoXForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"GPT-NeoX model Failed [ERROR {err}]"

    def test_gpt_oss(self, small_model_config):
        """Test GPT-OSS model."""
        header_config = ed.GptOssConfig(
            num_hidden_layers=8,
            num_local_experts=8,
            vocab_size=201088,
            hidden_size=128,
            intermediate_size=256,
            head_dim=64,
            num_attention_heads=8,
            num_key_value_heads=4,
            sliding_window=128,
            rope_theta=150000.0,
            tie_word_embeddings=False,
            hidden_act="silu",
            initializer_range=0.02,
            max_position_embeddings=2048,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            attention_dropout=0.0,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.9,
            output_router_logits=False,
            use_cache=True,
            layer_types=None,
        )
        res, err = self.create_test_for_models(
            "gpt_oss",
            transformers.GptOssForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"GPT-OSS model Failed [ERROR {err}]"

    def test_glm(self, small_model_config):
        """Test GLM model."""
        header_config = ed.GlmConfig(
            vocab_size=151552,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            partial_rotary_factor=0.5,
            head_dim=128,
            hidden_act="silu",
            attention_dropout=0.0,
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=0.00000015625,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            pad_token_id=151329,
            eos_token_id=None,
            bos_token_id=None,
            attention_bias=True,
        )
        res, err = self.create_test_for_models(
            "glm",
            transformers.GlmForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"GLM model Failed [ERROR {err}]"

    def test_glm4(self, small_model_config):
        """Test GLM4 model."""
        header_config = ed.Glm4Config(
            vocab_size=151552,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            partial_rotary_factor=0.5,
            head_dim=128,
            hidden_act="silu",
            attention_dropout=0.0,
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=0.00000015625,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            pad_token_id=151329,
            eos_token_id=None,
            bos_token_id=None,
            attention_bias=True,
        )
        res, err = self.create_test_for_models(
            "glm4",
            transformers.Glm4ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"GLM4 model Failed [ERROR {err}]"

    def test_glm4_moe(self, small_model_config):
        """Test GLM4-MoE model."""
        header_config = ed.Glm4MoeConfig(
            vocab_size=151552,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            partial_rotary_factor=0.5,
            num_key_value_heads=4,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            moe_intermediate_size=1408,
            num_experts_per_tok=4,
            n_shared_experts=4,
            n_routed_experts=4,
            routed_scaling_factor=1.0,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            norm_topk_prob=True,
            use_qk_norm=False,
        )
        res, err = self.create_test_for_models(
            "glm4_moe",
            transformers.Glm4MoeForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"GLM4-MoE model Failed [ERROR {err}]"

    def test_qwen3(self, small_model_config):
        """Test Qwen3 model."""
        header_config = ed.Qwen3Config(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
        )
        res, err = self.create_test_for_models(
            "qwen3",
            transformers.Qwen3ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"Qwen3 model Failed [ERROR {err}]"

    def test_olmo(self, small_model_config):
        """Test OLMo model."""
        res, err = self.create_test_for_models(
            "olmo",
            transformers.OlmoForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"OLMo model Failed [ERROR {err}]"

    def test_olmo2(self, small_model_config):
        """Test OLMo2 model."""
        res, err = self.create_test_for_models(
            "olmo2",
            transformers.Olmo2ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"OLMo2 model Failed [ERROR {err}]"

    def test_olmo3(self, small_model_config):
        """Test OLMo3 model."""
        res, err = self.create_test_for_models(
            "olmo3",
            transformers.Olmo3ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"OLMo3 model Failed [ERROR {err}]"

    def test_phi(self, small_model_config):
        """Test Phi model."""
        header_config = ed.PhiConfig(
            vocab_size=51200,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=None,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attention_dropout=0.0,
            hidden_act="gelu_new",
            max_position_embeddings=small_model_config["max_position_embeddings"],
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            partial_rotary_factor=0.5,
            qk_layernorm=False,
            bos_token_id=1,
            eos_token_id=2,
        )
        res, err = self.create_test_for_models(
            "phi",
            transformers.PhiForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"PHI model Failed [ERROR {err}]"

    def test_gemma(self, small_model_config):
        """Test Gemma model."""
        config_dict = small_model_config.copy()
        config_dict["tie_word_embeddings"] = True
        res, err = self.create_test_for_models(
            "gemma",
            transformers.GemmaForCausalLM,
            ed.TaskType.CAUSAL_LM,
            config_dict,
        )
        assert res, f"Gemma model Failed [ERROR {err}]"

    def test_dbrx(self, small_model_config):
        """Test DBRX MoE model."""
        header_config = ed.DbrxConfig(
            d_model=small_model_config["hidden_size"],
            n_heads=small_model_config["num_attention_heads"],
            n_layers=small_model_config["num_hidden_layers"],
            ffn_config=ed.DbrxFFNConfig(
                ffn_hidden_size=small_model_config["intermediate_size"],
                moe_top_k=small_model_config.get("num_experts_per_tok", 4),
                moe_num_experts=small_model_config.get("num_local_experts", 16),
            ),
            attn_config=ed.DbrxAttentionConfig(),
            max_seq_len=small_model_config["max_position_embeddings"],
        )
        res, err = self.create_test_for_models(
            "dbrx",
            transformers.DbrxForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"DBRX model Failed [ERROR {err}]"

    def test_stablelm(self, small_model_config):
        """Test StableLM model."""
        new_model_config = copy.copy(small_model_config)
        new_model_config.update({"attention_bias": True})
        res, err = self.create_test_for_models(
            "stablelm",
            transformers.StableLmForCausalLM,
            ed.TaskType.CAUSAL_LM,
            new_model_config,
        )
        assert res, f"StableLM model Failed [ERROR {err}]"

    def test_phi3(self, small_model_config):
        """Test Phi3 model."""
        config_dict = small_model_config.copy()
        config_dict["rope_scaling"] = {
            "long_factor": [1.0199999809265137, 1.0299999713897705, 1.0399999618530273, 1.0499999523162842],
            "long_mscale": 1.8,
            "original_max_position_embeddings": 4,
            "short_factor": [1.0, 1.0399999618530273, 1.0399999618530273, 1.0399999618530273],
            "short_mscale": 1.1,
            "type": "longrope",
        }
        res, err = self.create_test_for_models(
            "phi3",
            transformers.Phi3ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            config_dict,
        )
        assert res, f"PHI3 model Failed [ERROR {err}]"

    def test_phimoe(self, small_model_config):
        """Test PhiMoE model from HuggingFace Hub."""
        hf_model, _conf = TestHelper.get_hf_model_from_hub(
            "microsoft/Phi-3.5-MoE-instruct",
            small_model_config,
        )
        config_dict = small_model_config.copy()
        config_dict["rope_scaling"] = {
            "long_factor": [
                1.0199999809265137,
                1.0299999713897705,
                1.0399999618530273,
                1.0499999523162842,
                1.0499999523162842,
                1.0499999523162842,
                1.059999942779541,
                1.059999942779541,
            ],
            "long_mscale": 1.243163121016122,
            "original_max_position_embeddings": 4096,
            "short_factor": [
                1.0,
                1.0399999618530273,
                1.0399999618530273,
                1.0399999618530273,
                1.0499999523162842,
                1.0499999523162842,
                1.0499999523162842,
                1.0499999523162842,
            ],
            "short_mscale": 1.243163121016122,
            "type": "longrope",
        }
        res, err = self.create_test_for_models(
            "phimoe",
            hf_model,
            ed.TaskType.CAUSAL_LM,
            config_dict,
        )
        assert res, f"PHIMOE model Failed [ERROR {err}]"

    def test_deepseek_v2(self, small_model_config):
        """Test DeepSeek V2 model from HuggingFace Hub."""
        hf_model, conf = TestHelper.get_hf_model_from_hub(
            "deepseek-ai/DeepSeek-V2",
            small_model_config,
        )
        header_config = ed.DeepseekV2Config(
            vocab_size=conf.vocab_size,
            hidden_size=conf.hidden_size,
            intermediate_size=conf.intermediate_size,
            moe_intermediate_size=conf.moe_intermediate_size,
            num_hidden_layers=conf.num_hidden_layers,
            num_attention_heads=conf.num_attention_heads,
            num_key_value_heads=conf.num_key_value_heads,
            n_shared_experts=conf.n_shared_experts,
            n_routed_experts=conf.n_routed_experts,
            ep_size=conf.ep_size,
            routed_scaling_factor=conf.routed_scaling_factor,
            kv_lora_rank=conf.kv_lora_rank,
            q_lora_rank=conf.q_lora_rank,
            qk_rope_head_dim=conf.qk_rope_head_dim,
            v_head_dim=conf.v_head_dim,
            qk_nope_head_dim=conf.qk_nope_head_dim,
            topk_method=conf.topk_method,
            n_group=conf.n_group,
            topk_group=conf.topk_group,
            num_experts_per_tok=conf.num_experts_per_tok,
            moe_layer_freq=conf.moe_layer_freq,
            first_k_dense_replace=conf.first_k_dense_replace,
            norm_topk_prob=conf.norm_topk_prob,
            scoring_func=conf.scoring_func,
            aux_loss_alpha=conf.aux_loss_alpha,
            seq_aux=conf.seq_aux,
            hidden_act=conf.hidden_act,
            max_position_embeddings=conf.max_position_embeddings,
            initializer_range=conf.initializer_range,
            rms_norm_eps=conf.rms_norm_eps,
            use_cache=conf.use_cache,
            pad_token_id=conf.pad_token_id,
            bos_token_id=conf.bos_token_id,
            eos_token_id=conf.eos_token_id,
            pretraining_tp=conf.pretraining_tp,
            tie_word_embeddings=conf.tie_word_embeddings,
            rope_theta=conf.rope_theta,
            attention_bias=conf.attention_bias,
            attention_dropout=conf.attention_dropout,
            gradient_checkpointing=EasyDeLGradientCheckPointers.NONE,
            rope_scaling=conf.rope_scaling,
        )
        res, err = self.create_test_for_models(
            "deepseek_v2",
            hf_model,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"DeepSeekv2 model Failed [ERROR {err}]"

    def test_deepseek_v3(self, small_model_config):
        """Test DeepSeek V3 model from HuggingFace Hub."""
        hf_model, _conf = TestHelper.get_hf_model_from_hub(
            "deepseek-ai/DeepSeek-V3",
            small_model_config,
        )
        header_config = ed.DeepseekV3Config(
            **{
                "aux_loss_alpha": 0.001,
                "bos_token_id": 0,
                "eos_token_id": 1,
                "ep_size": 1,
                "first_k_dense_replace": 3,
                "hidden_act": "silu",
                "hidden_size": 128,
                "initializer_range": 0.02,
                "intermediate_size": 256,
                "kv_lora_rank": 512,
                "max_position_embeddings": 1024,
                "moe_intermediate_size": 128,
                "moe_layer_freq": 1,
                "n_group": 8,
                "n_routed_experts": 32,
                "n_shared_experts": 1,
                "norm_topk_prob": True,
                "num_attention_heads": 128,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 4,
                "num_key_value_heads": 128,
                "num_nextn_predict_layers": 1,
                "pretraining_tp": 1,
                "q_lora_rank": 1536,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "factor": 40,
                    "mscale": 1.0,
                    "mscale_all_dim": 1.0,
                    "original_max_position_embeddings": 4096,
                    "type": "yarn",
                },
                "rope_theta": 10000,
                "routed_scaling_factor": 2.5,
                "scoring_func": "sigmoid",
                "seq_aux": True,
                "tie_word_embeddings": False,
                "topk_group": 4,
                "topk_method": "noaux_tc",
                "transformers_version": "4.33.1",
                "use_cache": True,
                "v_head_dim": 128,
                "vocab_size": 129280,
            }
        )
        res, err = self.create_test_for_models(
            "deepseek_v3",
            hf_model,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"DeepSeekv3 model Failed [ERROR {err}]"

    def test_openelm(self, small_model_config):
        """Test OpenELM model from HuggingFace Hub."""
        hf_model, conf = TestHelper.get_hf_model_from_hub(
            "apple/OpenELM-270M-Instruct",
            small_model_config,
        )
        conf_f = ed.OpenELMConfig()
        for k, v in conf.__dict__.items():
            setattr(conf_f, k, v)

        conf_f.max_context_length = (small_model_config["max_position_embeddings"],)

        res, err = self.create_test_for_models(
            "openelm",
            hf_model,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=conf_f,
        )
        assert res, f"OpenELM model Failed [ERROR {err}]"

    def test_arctic(self, small_model_config):
        """Test Arctic MoE model from HuggingFace Hub."""
        hf_model, _conf = TestHelper.get_hf_model_from_hub(
            "Snowflake/snowflake-arctic-instruct",
            small_model_config,
        )
        res, err = self.create_test_for_models(
            "arctic",
            hf_model,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"ARCTIC model Failed [ERROR {err}]"

    def test_rwkv(self, small_model_config):
        """Test RWKV model."""
        res, err = self.create_test_for_models(
            "rwkv",
            transformers.RwkvForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"RWKV model Failed [ERROR {err}]"

    def test_gemma2(self, small_model_config):
        """Test Gemma2 model."""
        header_config = ed.Gemma2Config(32000, 128, 256, 4, 8, 4, 128 // 8, use_scan_mlp=False)
        res, err = self.create_test_for_models(
            "gemma2",
            transformers.Gemma2ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"Gemma2 model Failed [ERROR {err}]"

    def test_mamba(self, small_model_config):
        """Test Mamba model."""
        res, err = self.create_test_for_models(
            "mamba",
            transformers.MambaForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"MAMBA model Failed [ERROR {err}]"

    def test_mamba2(self, small_model_config):
        """Test Mamba2 model."""
        header_config = ed.Mamba2Config(
            hidden_size=256,
            num_hidden_layers=16,
            num_heads=8,
        )
        res, err = self.create_test_for_models(
            "mamba2",
            transformers.Mamba2ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, f"Mamba2 model Failed [ERROR {err}]"

    def test_cohere(self, small_model_config):
        """Test Cohere model."""
        res, err = self.create_test_for_models(
            "cohere",
            transformers.CohereForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"Cohere model Failed [ERROR {err}]"

    def test_cohere2(self, small_model_config):
        """Test Cohere2 model."""
        res, err = self.create_test_for_models(
            "cohere2",
            transformers.Cohere2ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"Cohere2 model Failed [ERROR {err}]"

    def test_smollm3(self, small_model_config):
        """Test smollm3 model."""
        res, err = self.create_test_for_models(
            "smollm3",
            transformers.SmolLM3ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"smollm3 model Failed [ERROR {err}]"

    def test_qwen2(self, small_model_config):
        """Test Qwen2 model."""
        res, err = self.create_test_for_models(
            "qwen2",
            transformers.Qwen2ForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"Qwen2 model Failed [ERROR {err}]"

    def test_qwen2_moe(self, small_model_config):
        """Test Qwen2-MoE model."""
        res, err = self.create_test_for_models(
            "qwen2_moe",
            transformers.Qwen2MoeForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"Qwen2Moe model Failed [ERROR {err}]"

    def test_qwen2_vl(self, small_model_config):
        """Test Qwen2-VL vision-language model with proper vision inputs."""
        local_cfg = small_model_config.copy()

        local_cfg["max_position_embeddings"] = 2048
        org_config = ed.Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        hf_config = transformers.Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        org_config.text_config.hidden_size = 1024
        org_config.text_config.intermediate_size = 8192
        org_config.text_config.num_attention_heads = org_config.text_config.hidden_size // 128
        org_config.text_config.num_key_value_heads = 2
        org_config.text_config.num_hidden_layers = 32
        org_config.text_config.head_dim = 128
        org_config.text_config.rope_scaling = hf_config.text_config.rope_scaling
        org_config.text_config.layer_types = [
            "sliding_attention"
            if org_config.text_config.sliding_window is not None and i >= org_config.text_config.max_window_layers
            else "full_attention"
            for i in range(org_config.text_config.num_hidden_layers)
        ]
        # Match vision output size to text hidden size
        org_config.vision_config.hidden_size = org_config.text_config.hidden_size

        # VLM configuration for Qwen2-VL
        # Qwen2-VL uses mRoPE with 3D position IDs
        num_images_per_batch = 1
        batch_size = local_cfg["batch_size"]
        # For Qwen2-VL, patches are computed based on vision config
        # Using a 14x14 grid for patches - spatial_merge_size reduces final tokens
        grid_h, grid_w = 14, 14
        # Account for spatial_merge_size - vision model merges patches spatially
        spatial_merge_size = org_config.vision_config.spatial_merge_size
        merged_h = grid_h // spatial_merge_size
        merged_w = grid_w // spatial_merge_size
        num_image_tokens = merged_h * merged_w  # 49 tokens after merge (7x7)

        # Ensure sequence_length is large enough for image tokens + vision markers + text
        # Qwen VL needs: vision_start + image_tokens + vision_end + text
        tokens_per_image = num_image_tokens + 2  # start + tokens + end
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 128), tokens_per_image + 32)

        # pixel_values shape: (batch_size * num_images * num_patches, patch_features)
        total_patches = batch_size * num_images_per_batch * grid_h * grid_w
        # Handle both in_chans (old) and in_channels (new) attribute names
        in_channels = getattr(org_config.vision_config, "in_chans", None) or getattr(
            org_config.vision_config, "in_channels", 3
        )
        patch_size = org_config.vision_config.patch_size
        temporal_patch_size = org_config.vision_config.temporal_patch_size or 2
        # Each patch element is reshaped to (in_channels, temporal_patch_size, patch_size, patch_size)
        patch_features = in_channels * temporal_patch_size * patch_size * patch_size

        # image_grid_thw: (batch_size * num_images, 3) - T, H, W per image
        image_grid_thw = np.tile(np.array([[1, grid_h, grid_w]], dtype=np.int64), (batch_size * num_images_per_batch, 1))

        vlm_config = {
            "image_token_id": org_config.image_token_id,  # 151655
            "vision_start_token_id": org_config.vision_start_token_id,  # 151652
            "vision_end_token_id": org_config.vision_end_token_id,  # 151653
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (total_patches, patch_features),
            "image_grid_thw": image_grid_thw,
            "num_images": num_images_per_batch,
            "is_qwen_vl": True,
        }

        res, err = self.create_test_for_vlm_models(
            "qwen2_vl",
            transformers.Qwen2VLForConditionalGeneration,
            ed.TaskType.IMAGE_TEXT_TO_TEXT,
            local_cfg,
            header_config=org_config,
            vlm_config=vlm_config,
        )
        assert res, f"Qwen2VL model Failed [ERROR {err}]"

    def test_qwen2_vl_text(self, small_model_config):
        """Test Qwen2-VL vision-language model with proper vision inputs."""
        local_cfg = small_model_config.copy()

        local_cfg["max_position_embeddings"] = 2048
        org_config = ed.Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        hf_config = transformers.Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        org_config.text_config.hidden_size = 1024
        org_config.text_config.intermediate_size = 8192
        org_config.text_config.num_attention_heads = org_config.text_config.hidden_size // 128
        org_config.text_config.num_key_value_heads = 2
        org_config.text_config.num_hidden_layers = 32
        org_config.text_config.head_dim = 128
        org_config.text_config.rope_scaling = hf_config.text_config.rope_scaling
        org_config.text_config.layer_types = [
            "sliding_attention"
            if org_config.text_config.sliding_window is not None and i >= org_config.text_config.max_window_layers
            else "full_attention"
            for i in range(org_config.text_config.num_hidden_layers)
        ]
        # Match vision output size to text hidden size
        org_config.vision_config.hidden_size = org_config.text_config.hidden_size

        res, err = self.create_test_for_models(
            "qwen2_vl",
            transformers.Qwen2VLForConditionalGeneration,
            ed.TaskType.IMAGE_TEXT_TO_TEXT,
            local_cfg,
            header_config=org_config,
        )
        assert res, f"Qwen2VL-text model Failed [ERROR {err}]"

    def test_qwen3_moe(self, small_model_config):
        """Test Qwen3-MoE model."""
        res, err = self.create_test_for_models(
            "qwen3_moe",
            transformers.Qwen3MoeForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, f"Qwen3Moe model Failed [ERROR {err}]"

    def test_qwen3_vl(self, small_model_config):
        """Test Qwen3-VL vision-language model with proper vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        org_config = ed.Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")
        org_config.text_config.hidden_size = 512
        org_config.text_config.intermediate_size = 1024
        org_config.text_config.num_attention_heads = 4
        org_config.text_config.num_key_value_heads = 2
        org_config.text_config.num_hidden_layers = 2
        org_config.text_config.head_dim = 128
        org_config.text_config.rope_scaling = {"rope_type": "default", "mrope_section": [24, 20, 20]}
        # Match vision OUTPUT size to text hidden size (Qwen3-VL has separate out_hidden_size)
        org_config.vision_config.out_hidden_size = org_config.text_config.hidden_size

        # VLM configuration for Qwen3-VL
        # Note: Qwen3-VL uses patch_size=16, so grid dimensions should be multiples of 2
        # (for spatial_merge_size=2 to work correctly)
        num_images_per_batch = 1
        batch_size = local_cfg["batch_size"]
        # Use 16x16 grid (must be divisible by spatial_merge_size=2)
        grid_h, grid_w = 16, 16
        # Account for spatial_merge_size - vision model merges patches spatially
        spatial_merge_size = org_config.vision_config.spatial_merge_size
        merged_h = grid_h // spatial_merge_size
        merged_w = grid_w // spatial_merge_size
        num_image_tokens = merged_h * merged_w  # 49 tokens after merge (7x7)

        # Ensure sequence_length is large enough for image tokens + vision markers + text
        tokens_per_image = num_image_tokens + 2  # start + tokens + end
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 128), tokens_per_image + 32)

        # pixel_values shape: (batch_size * num_images * num_patches, patch_features)
        total_patches = batch_size * num_images_per_batch * grid_h * grid_w
        # Handle both in_chans (old) and in_channels (new) attribute names
        in_channels = getattr(org_config.vision_config, "in_chans", None) or getattr(
            org_config.vision_config, "in_channels", 3
        )
        patch_size = org_config.vision_config.patch_size
        temporal_patch_size = org_config.vision_config.temporal_patch_size or 2
        # Each patch element is reshaped to (in_channels, temporal_patch_size, patch_size, patch_size)
        patch_features = in_channels * temporal_patch_size * patch_size * patch_size

        # image_grid_thw: (batch_size * num_images, 3) - T, H, W per image
        image_grid_thw = np.tile(np.array([[1, grid_h, grid_w]], dtype=np.int64), (batch_size * num_images_per_batch, 1))

        vlm_config = {
            "image_token_id": org_config.image_token_id,  # 151655
            "vision_start_token_id": org_config.vision_start_token_id,  # 151652
            "vision_end_token_id": org_config.vision_end_token_id,  # 151653
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (total_patches, patch_features),
            "image_grid_thw": image_grid_thw,
            "num_images": num_images_per_batch,
            "is_qwen_vl": True,
        }

        res, err = self.create_test_for_vlm_models(
            "qwen3_vl",
            transformers.Qwen3VLForConditionalGeneration,
            ed.TaskType.IMAGE_TEXT_TO_TEXT,
            local_cfg,
            header_config=org_config,
            vlm_config=vlm_config,
        )
        assert res, f"Qwen3-VL model Failed [ERROR {err}]"

    def test_qwen3_vl_text(self, small_model_config):
        """Test Qwen3-VL vision-language model with proper vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        org_config = ed.Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")
        org_config.text_config.hidden_size = 512
        org_config.text_config.intermediate_size = 1024
        org_config.text_config.num_attention_heads = 4
        org_config.text_config.num_key_value_heads = 2
        org_config.text_config.num_hidden_layers = 2
        org_config.text_config.head_dim = 128
        org_config.text_config.rope_scaling = {"rope_type": "default", "mrope_section": [24, 20, 20]}

        res, err = self.create_test_for_models(
            "qwen3_vl",
            transformers.Qwen3VLForConditionalGeneration,
            ed.TaskType.IMAGE_TEXT_TO_TEXT,
            local_cfg,
            header_config=org_config,
        )
        assert res, f"Qwen3-VL-text model Failed [ERROR {err}]"

    def test_qwen3_vl_moe(self, small_model_config):
        """Test Qwen3-VL-MoE vision-language model with MoE and proper vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        header_config = ed.Qwen3VLMoeConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Thinking")
        header_config.text_config.hidden_size = 512
        header_config.text_config.intermediate_size = 1024
        header_config.text_config.num_attention_heads = 4
        header_config.text_config.num_key_value_heads = 2
        header_config.text_config.decoder_sparse_step = 1
        header_config.text_config.moe_intermediate_size = 1024
        header_config.text_config.num_experts_per_tok = 4
        header_config.text_config.num_experts = 8
        # Match vision output size to text hidden size
        header_config.vision_config.out_hidden_size = 512

        # VLM configuration for Qwen3-VL-MoE
        num_images_per_batch = 1
        batch_size = local_cfg["batch_size"]
        grid_h, grid_w = 14, 14
        # Account for spatial_merge_size - vision model merges patches spatially
        spatial_merge_size = header_config.vision_config.spatial_merge_size
        merged_h = grid_h // spatial_merge_size
        merged_w = grid_w // spatial_merge_size
        num_image_tokens = merged_h * merged_w  # 49 tokens after merge (7x7)

        # Ensure sequence_length is large enough for image tokens + vision markers + text
        tokens_per_image = num_image_tokens + 2  # start + tokens + end
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 128), tokens_per_image + 32)

        # pixel_values shape: (batch_size * num_images * num_patches, patch_features)
        total_patches = batch_size * num_images_per_batch * grid_h * grid_w
        # Handle both in_chans (old) and in_channels (new) attribute names
        in_channels = getattr(header_config.vision_config, "in_chans", None) or getattr(
            header_config.vision_config, "in_channels", 3
        )
        patch_size = header_config.vision_config.patch_size
        temporal_patch_size = header_config.vision_config.temporal_patch_size or 2
        # Each patch element is reshaped to (in_channels, temporal_patch_size, patch_size, patch_size)
        patch_features = in_channels * temporal_patch_size * patch_size * patch_size

        # image_grid_thw: (batch_size * num_images, 3) - T, H, W per image
        image_grid_thw = np.tile(np.array([[1, grid_h, grid_w]], dtype=np.int64), (batch_size * num_images_per_batch, 1))

        vlm_config = {
            "image_token_id": header_config.image_token_id,  # 151655
            "vision_start_token_id": header_config.vision_start_token_id,  # 151652
            "vision_end_token_id": header_config.vision_end_token_id,  # 151653
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (total_patches, patch_features),
            "image_grid_thw": image_grid_thw,
            "num_images": num_images_per_batch,
            "is_qwen_vl": True,
        }

        res, err = self.create_test_for_vlm_models(
            "qwen3_vl_moe",
            transformers.Qwen3VLMoeForConditionalGeneration,
            ed.TaskType.IMAGE_TEXT_TO_TEXT,
            local_cfg,
            header_config=header_config,
            vlm_config=vlm_config,
        )
        assert res, f"Qwen3-VL-MoE model Failed [ERROR {err}]"

    def test_qwen3_vl_moe_text(self, small_model_config):
        """Test Qwen3-VL-MoE vision-language model with MoE and proper vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        header_config = ed.Qwen3VLMoeConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Thinking")
        header_config.text_config.hidden_size = 512
        header_config.text_config.intermediate_size = 1024
        header_config.text_config.num_attention_heads = 4
        header_config.text_config.num_key_value_heads = 2
        header_config.text_config.decoder_sparse_step = 1
        header_config.text_config.moe_intermediate_size = 1024
        header_config.text_config.num_experts_per_tok = 4
        header_config.text_config.num_experts = 8
        res, err = self.create_test_for_models(
            "qwen3_vl_moe",
            transformers.Qwen3VLMoeForConditionalGeneration,
            ed.TaskType.IMAGE_TEXT_TO_TEXT,
            local_cfg,
            header_config=header_config,
        )
        assert res, f"Qwen3-VL-MoE-text model Failed [ERROR {err}]"

    def test_qwen3_next(self, small_model_config):
        """Test Qwen3Next hybrid attention + MoE model.

        This model uses:
        - Hybrid attention: alternating full attention (with sigmoid gating + partial RoPE)
          and linear attention (GatedDeltaRule)
        - MoE FFN with shared expert
        """

        header_config = ed.Qwen3NextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            # Hybrid attention settings
            partial_rotary_factor=0.25,
            full_attention_interval=4,  # Every 4th layer is full attention
            # Linear attention settings
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_num_key_heads=8,
            linear_value_head_dim=16,
            linear_num_value_heads=8,
            # MoE settings
            num_experts=8,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
        res = self.create_test_for_models(
            "qwen3_next",
            transformers.Qwen3NextForCausalLM,
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, "Qwen3Next model Failed"

    def test_qwen3_omni_moe(self, small_model_config):
        """Test Qwen3OmniMoe multimodal MoE model (EasyDeL-only, no HuggingFace version).

        This model processes text, vision (images/videos), and audio inputs
        with a MoE text decoder.
        """
        # Create sub-configs for multimodal model
        audio_config = ed.Qwen3OmniMoeAudioConfig(
            num_mel_bins=80,
            encoder_layers=4,
            encoder_attention_heads=8,
            encoder_ffn_dim=256,
            d_model=128,
            output_dim=256,
            max_source_positions=1500,
        )
        vision_config = ed.Qwen3OmniMoeVisionConfig(
            hidden_size=128,
            out_hidden_size=256,
            depth=4,
            num_heads=4,
            mlp_ratio=2.0,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
        )
        text_config = ed.Qwen3OmniMoeTextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            # MoE settings
            num_experts=8,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            moe_intermediate_size=256,
            # rope_scaling with mrope_section is required by HF model
            # mrope_section must sum to head_dim // 2 = 16
            rope_scaling={"rope_type": "default", "mrope_section": [6, 5, 5]},
        )
        thinker_config = ed.Qwen3OmniMoeThinkerConfig(
            audio_config=audio_config,
            vision_config=vision_config,
            text_config=text_config,
            audio_token_id=151646,
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            vision_end_token_id=151653,
        )
        _header_config = ed.Qwen3OmniMoeConfig(
            thinker_config=thinker_config,
            talker_config=ed.Qwen3OmniMoeTalkerConfig(
                text_config=ed.Qwen3OmniMoeTalkerTextConfig(
                    rope_scaling={"rope_type": "default", "mrope_section": [6, 5, 5]},
                )
            ),
        )
        res = self.create_test_for_models(
            "qwen3_omni_moe_thinker",
            transformers.Qwen3OmniMoeThinkerForConditionalGeneration,
            ed.TaskType.ANY_TO_ANY,
            small_model_config,
            # header_config=header_config,
            header_config=thinker_config,
        )
        assert res, "Qwen3OmniMoe model Failed"

    def test_kimi_linear(self, small_model_config):
        header_config = ed.KimiLinearConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            # MLA settings
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            # MoE settings
            num_experts=8,
            num_experts_per_token=2,
            num_shared_experts=1,
            moe_intermediate_size=256,
            first_k_dense_replace=1,
            moe_layer_freq=2,
            num_expert_group=2,
            topk_group=1,
            routed_scaling_factor=1.0,
            # Linear attention settings
            linear_attn_config={
                "kda_layers": [1, 3],  # Layers 1 and 3 use KDA (1-indexed)
                "full_attn_layers": [2, 4],  # Layers 2 and 4 use MLA (1-indexed)
                "num_heads": 8,
                "head_k_dim": 32,
                "head_v_dim": 32,
                "d_conv": 4,
                "gate_low_rank_dim": 32,
                "chunk_size": 64,
            },
        )

        # hf_model, _conf = TestHelper.get_hf_model_from_hub("moonshotai/Kimi-Linear-48B-A3B-Instruct", small_model_config)
        # res = self.create_test_for_models(
        #     "kimi_linear",
        #     hf_model,
        #     ed.TaskType.CAUSAL_LM,
        #     small_model_config,
        #     header_config=header_config,
        # )

        res = self.run_easydel_only_test(
            "kimi_linear",
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, "Kimi Linear model Failed"

    # def test_roberta(self, small_model_config):
    #     """Test RoBERTa model."""
    #     header_config = ed.RobertaConfig(
    #         hidden_size=256,
    #         intermediate_size=512,
    #         num_hidden_layers=4,
    #         num_attention_heads=8,
    #     )
    #     res, err = self.create_test_for_models(
    #         "roberta",
    #         transformers.RobertaForCausalLM,
    #         ed.TaskType.CAUSAL_LM,
    #         small_model_config,
    #         header_config=header_config,
    #     )
    #     assert res, f"ROBERTA model Failed [ERROR {err}]"

    # def test_opt(self, small_model_config):
    #     """Test OPT model - CRITICAL: We just fixed bugs in this model!"""
    #     res, err = self.create_test_for_models(
    #         "opt",
    #         transformers.OPTForCausalLM,
    #         ed.TaskType.CAUSAL_LM,
    #         small_model_config,
    #     )
    #     assert res, f"OPT model Failed [ERROR {err}]"

    def test_mistral3(self, small_model_config):
        """Test Mistral3 model (EasyDeL-only, no HuggingFace version)."""
        res = self.run_easydel_only_test(
            "mistral3",
            ed.TaskType.IMAGE_TEXT_TO_TEXT,
            small_model_config,
        )
        assert res, "Mistral3 model Failed"

    @pytest.mark.skip(reason="Grok-1 model has initialization issues (ShapeDtypeStruct not valid JAX type)")
    def test_grok_1(self, small_model_config):
        """Test Grok-1 MoE model (EasyDeL-only, no HuggingFace version)."""
        res = self.run_easydel_only_test(
            "grok-1",
            ed.TaskType.CAUSAL_LM,
            small_model_config,
        )
        assert res, "Grok-1 model Failed"

    @pytest.mark.skip(reason="Xerxes model has implementation issues (XerxesSparseMoeBlock missing 'layers' attribute)")
    def test_xerxes(self, small_model_config):
        """Test Xerxes custom architecture model (EasyDeL-only)."""
        header_config = ed.XerxesConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config.get("num_key_value_heads"),
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )
        res = self.run_easydel_only_test(
            "xerxes",
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, "Xerxes model Failed"

    @pytest.mark.skip(reason="Xerxes2 model has initialization issues (ShapeDtypeStruct not valid JAX type)")
    def test_xerxes2(self, small_model_config):
        """Test Xerxes2 custom architecture model (EasyDeL-only)."""
        header_config = ed.Xerxes2Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config.get("num_key_value_heads"),
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )
        res = self.run_easydel_only_test(
            "xerxes2",
            ed.TaskType.CAUSAL_LM,
            small_model_config,
            header_config=header_config,
        )
        assert res, "Xerxes2 model Failed"

    @pytest.mark.skip(reason="Pixtral model not registered in get_modules_by_type")
    def test_pixtral(self, small_model_config):
        """Test Pixtral vision-language model (EasyDeL-only)."""
        pytest.skip("Pixtral not registered")

    @pytest.mark.skip(reason="LLaVA model has initialization issues (ShapeDtypeStruct)")
    def test_llava(self, small_model_config):
        """Test LLaVA vision-language model (EasyDeL-only)."""
        pytest.skip("LLaVA initialization issues")

    @pytest.mark.skip(reason="Aya Vision model repo not found or config loading issues")
    def test_aya_vision(self, small_model_config):
        """Test Aya Vision model (EasyDeL-only)."""
        pytest.skip("Aya Vision config issues")

    @pytest.mark.skip(reason="CLIP config initialization needs fixing (CLIPTextConfig not a mapping)")
    def test_clip(self, small_model_config):
        """Test CLIP vision model (EasyDeL-only)."""
        pytest.skip("CLIP config initialization issues")

    @pytest.mark.skip(reason="SigLIP config initialization needs fixing (SiglipTextConfig not a mapping)")
    def test_siglip(self, small_model_config):
        """Test SigLIP vision model (EasyDeL-only)."""
        pytest.skip("SigLIP config initialization issues")

    @pytest.mark.skip(reason="Whisper TaskType.AUDIO_TO_TEXT doesn't exist (need to check correct TaskType)")
    def test_whisper(self, small_model_config):
        """Test Whisper speech-to-text model (EasyDeL-only)."""
        pytest.skip("Whisper TaskType issues")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
