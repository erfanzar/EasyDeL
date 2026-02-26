"""Task-specific test classes for EasyDeL model testing.

This module provides tester classes for different model task types,
including CAUSAL_LM, BASE_MODULE, SEQUENCE_CLASSIFICATION, VLM, etc.
"""

# pyright: reportPrivateLocalImportUsage=false

import traceback
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from flax import nnx as nn

import easydel as ed

from .comparators import ComparisonResult, compare_hidden_states, compare_logits
from .input_generators import (
    make_audio_inputs,
    make_classification_inputs,
    make_qwen_vlm_inputs,
    make_seq2seq_inputs,
    make_text_inputs,
    make_vision_inputs,
    make_vlm_inputs,
)
from .model_factory import (
    cleanup_models,
    create_ed_model,
    create_ed_model_only,
    create_hf_model,
    setup_config,
)


@dataclass
class TestResult:
    """Result of a model test."""

    success: bool
    comparison: ComparisonResult | None = None
    ed_time: float = 0.0
    hf_time: float = 0.0
    error_message: str = ""
    extra_info: dict = field(default_factory=dict)


class BaseTester:
    """Base class for task-specific testers."""

    def _run_hf_forward(
        self,
        hf_model: Any,
        inputs: dict,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
    ) -> tuple[Any, float]:
        """Run HuggingFace forward pass with timing.

        Returns:
            Tuple of (output, time_taken)
        """
        with ed.utils.capture_time() as timer:
            call_variants: list[dict[str, Any]] = []
            if output_router_logits:
                call_variants.append(
                    {
                        "output_hidden_states": output_hidden_states,
                        "output_router_logits": output_router_logits,
                        "past_key_values": None,
                        "use_cache": False,
                    }
                )
                call_variants.append(
                    {
                        "output_hidden_states": output_hidden_states,
                        "output_router_logits": output_router_logits,
                    }
                )

            call_variants.append(
                {
                    "output_hidden_states": output_hidden_states,
                    "past_key_values": None,
                    "use_cache": False,
                }
            )
            call_variants.append({"output_hidden_states": output_hidden_states})
            call_variants.append({})

            last_exc: Exception | None = None
            output = None
            for extra_kwargs in call_variants:
                try:
                    output = hf_model(**inputs, **extra_kwargs)
                    break
                except (TypeError, ValueError) as exc:
                    last_exc = exc
            if output is None:
                raise last_exc or RuntimeError("HuggingFace forward failed with no exception captured.")
        return output, timer()

    def _run_ed_forward(
        self,
        ed_model: Any,
        inputs: dict,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
    ) -> tuple[Any, float]:
        """Run EasyDeL forward pass with timing.

        Returns:
            Tuple of (output, time_taken)
        """
        try:

            @ed.ejit(static_argnums=(1,))  # pyright: ignore[reportUntypedFunctionDecorator]
            def jited(ids, gd, gs, go, attention_mask, **kwargs):
                model = nn.merge(gd, gs, go)
                return model.compute_loss(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    output_router_logits=output_router_logits,
                    output_hidden_states=output_hidden_states,
                    **kwargs,
                )

            # Warmup
            extra_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}
            _ = jited(
                inputs["input_ids"],
                *ed_model.split_module(),
                attention_mask=inputs["attention_mask"],
                **extra_kwargs,
            )

            with ed.utils.capture_time() as timer:
                output, _metrics = jited(
                    inputs["input_ids"],
                    *ed_model.split_module(),
                    attention_mask=inputs["attention_mask"],
                    **extra_kwargs,
                )
            return output, timer()
        except Exception:
            # Fallback without router logits (handles TypeError, ValueError, etc.)

            @ed.ejit(static_argnums=(1,))  # pyright: ignore[reportUntypedFunctionDecorator]
            def jited_fallback(ids, gd, gs, go, attention_mask, **kwargs):
                model = nn.merge(gd, gs, go)
                return model.compute_loss(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                    **kwargs,
                )

            extra_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}
            _ = jited_fallback(
                inputs["input_ids"],
                *ed_model.split_module(),
                attention_mask=inputs["attention_mask"],
                **extra_kwargs,
            )

            with ed.utils.capture_time() as timer:
                output, _metrics = jited_fallback(
                    inputs["input_ids"],
                    *ed_model.split_module(),
                    attention_mask=inputs["attention_mask"],
                    **extra_kwargs,
                )
            return output, timer()


class CausalLMTester(BaseTester):
    """Test CAUSAL_LM models by comparing logits and loss."""

    def run(
        self,
        module_name: str,
        hf_class: type | None,
        task: ed.TaskType,
        config: Any,
        small_model_config: dict,
    ) -> TestResult:
        """Run forward pass test and compare logits + loss.

        Args:
            module_name: Name of the module
            hf_class: HuggingFace model class
            task: Task type (should be CAUSAL_LM)
            config: Model configuration
            small_model_config: Base config dictionary

        Returns:
            TestResult with comparison details
        """
        try:
            # Setup config
            config = setup_config(config, small_model_config)

            # Handle EasyDeL-only models (no HF comparison)
            if hf_class is None:
                with config.mesh:
                    ed_model = create_ed_model_only(
                        module_name=module_name,
                        task=task,
                        config=config,
                        small_model_config=small_model_config,
                    )

                    # Generate inputs
                    inputs = make_text_inputs(
                        vocab_size=small_model_config["vocab_size"],
                        batch_size=small_model_config["batch_size"],
                        seq_len=small_model_config["sequence_length"],
                    )

                    # Run ED forward only
                    ed_inputs = inputs["jax"]
                    ed_output, ed_time = self._run_ed_forward(ed_model, ed_inputs, output_router_logits=True)

                    # Verify output has expected attributes
                    assert hasattr(ed_output, "logits"), "Output missing logits"
                    assert hasattr(ed_output, "loss"), "Output missing loss"

                return TestResult(
                    success=True,
                    ed_time=ed_time,
                    extra_info={"easydel_only": True},
                )

            # Create models with HF comparison
            hf_model = create_hf_model(hf_class, config)

            with config.mesh:
                ed_model = create_ed_model(
                    module_name=module_name,
                    task=task,
                    config=config,
                    small_model_config=small_model_config,
                    hf_model=hf_model,
                )

                # Generate inputs
                inputs = make_text_inputs(
                    vocab_size=small_model_config["vocab_size"],
                    batch_size=small_model_config["batch_size"],
                    seq_len=small_model_config["sequence_length"],
                )

                # Run HF forward
                hf_inputs = {
                    **inputs["torch"],
                    "labels": inputs["torch"]["input_ids"],
                }
                hf_output, hf_time = self._run_hf_forward(hf_model, hf_inputs, output_router_logits=True)

                # Run ED forward
                ed_inputs = inputs["jax"]
                ed_output, ed_time = self._run_ed_forward(ed_model, ed_inputs, output_router_logits=True)

                # Get aux losses
                hf_aux = getattr(hf_output, "aux_loss", 0)
                ed_aux = getattr(ed_output, "aux_loss", 0)

                # Compare outputs
                comparison = compare_logits(
                    name=module_name,
                    hf_logits=hf_output.logits.cpu().detach().numpy(),
                    ed_logits=ed_output.logits,
                    hf_loss=float(hf_output.loss.cpu().detach().numpy()),
                    ed_loss=float(ed_output.loss),
                    hf_aux_loss=float(hf_aux) if hf_aux else 0,
                    ed_aux_loss=float(ed_aux) if ed_aux else 0,
                )

            cleanup_models(hf_model)

            return TestResult(
                success=comparison.success,
                comparison=comparison,
                ed_time=ed_time,
                hf_time=hf_time,
            )

        except Exception as e:
            traceback.print_exc()
            return TestResult(
                success=False,
                error_message=str(e),
            )

    def test_generation(
        self,
        module_name: str,
        hf_class: type | None,
        config: Any,
        small_model_config: dict,
        max_new_tokens: int = 16,
        task: ed.TaskType = ed.TaskType.CAUSAL_LM,
    ) -> TestResult:
        """Test that generation runs without errors.

        Args:
            module_name: Name of the module
            hf_class: HuggingFace model class
            config: Model configuration
            small_model_config: Base config dictionary
            max_new_tokens: Number of tokens to generate
            task: Task type used to look up the EasyDeL module.

        Returns:
            TestResult indicating if generation succeeded
        """
        from transformers import GenerationConfig

        try:
            # Setup config
            config = setup_config(config, small_model_config)
            # Generation tests should be portable across backends. Some MoE paths
            # use collectives that are not available on XLA:CPU when sharding on EP,
            # so prefer placing the extra device on TP.
            config.sharding_axis_dims = (1, 1, 1, -1, 1)
            # Handle EasyDeL-only models (no HF model needed for generation test)
            if hf_class is None:
                with config.mesh:
                    ed_model = create_ed_model_only(
                        module_name=module_name,
                        task=task,
                        config=config,
                        small_model_config=small_model_config,
                    )

                    # Set generation_config
                    if not hasattr(ed_model, "generation_config") or ed_model.generation_config is None:
                        ed_model.generation_config = GenerationConfig(
                            max_length=small_model_config.get("max_position_embeddings", 256),
                            max_new_tokens=max_new_tokens,
                            pad_token_id=getattr(config, "pad_token_id", 0) or 0,
                            eos_token_id=getattr(config, "eos_token_id", 2) or 2,
                            bos_token_id=getattr(config, "bos_token_id", 1) or 1,
                        )

                    # Generate inputs (shorter for generation)
                    inputs = make_text_inputs(
                        vocab_size=small_model_config["vocab_size"],
                        batch_size=1,
                        seq_len=16,
                    )

                    # Run generation
                    with ed.utils.capture_time() as timer:
                        output = ed_model.generate(
                            input_ids=inputs["jax"]["input_ids"],
                            attention_mask=inputs["jax"]["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    ed_time = timer()

                # Verify output
                assert output is not None, "Generation returned None"
                assert hasattr(output, "sequences"), "Output missing sequences"
                assert output.sequences.shape[1] > inputs["jax"]["input_ids"].shape[1], "No tokens generated"

                return TestResult(
                    success=True,
                    ed_time=ed_time,
                    extra_info={
                        "generated_length": int(output.sequences.shape[1]),
                        "input_length": int(inputs["jax"]["input_ids"].shape[1]),
                        "easydel_only": True,
                    },
                )

            # Create models with HF for weight transfer
            hf_model = create_hf_model(hf_class, config)

            with config.mesh:
                ed_model = create_ed_model(
                    module_name=module_name,
                    task=task,
                    config=config,
                    small_model_config=small_model_config,
                    hf_model=hf_model,
                )

                # Set generation_config if not present
                if not hasattr(ed_model, "generation_config") or ed_model.generation_config is None:
                    ed_model.generation_config = GenerationConfig(
                        max_length=small_model_config.get("max_position_embeddings", 256),
                        max_new_tokens=max_new_tokens,
                        pad_token_id=getattr(config, "pad_token_id", 0) or 0,
                        eos_token_id=getattr(config, "eos_token_id", 2) or 2,
                        bos_token_id=getattr(config, "bos_token_id", 1) or 1,
                    )

                # Generate inputs (shorter for generation)
                inputs = make_text_inputs(
                    vocab_size=small_model_config["vocab_size"],
                    batch_size=1,  # Single batch for generation
                    seq_len=16,  # Short prompt
                )

                # Run generation
                with ed.utils.capture_time() as timer:
                    output = ed_model.generate(
                        input_ids=inputs["jax"]["input_ids"],
                        attention_mask=inputs["jax"]["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Greedy for reproducibility
                    )
                ed_time = timer()

            cleanup_models(hf_model)

            # Verify output
            assert output is not None, "Generation returned None"
            assert hasattr(output, "sequences"), "Output missing sequences"
            assert output.sequences.shape[1] > inputs["jax"]["input_ids"].shape[1], "No tokens generated"

            return TestResult(
                success=True,
                ed_time=ed_time,
                extra_info={
                    "generated_length": int(output.sequences.shape[1]),
                    "input_length": int(inputs["jax"]["input_ids"].shape[1]),
                },
            )

        except Exception as e:
            traceback.print_exc()
            return TestResult(
                success=False,
                error_message=str(e),
            )


class BaseModuleTester(BaseTester):
    """Test BASE_MODULE by comparing hidden states."""

    def run(
        self,
        module_name: str,
        hf_class: type,
        task: ed.TaskType,
        config: Any,
        small_model_config: dict,
    ) -> TestResult:
        """Run forward pass and compare hidden states.

        Args:
            module_name: Name of the module
            hf_class: HuggingFace model class (base model, not ForCausalLM)
            task: Task type (should be BASE_MODULE)
            config: Model configuration
            small_model_config: Base config dictionary

        Returns:
            TestResult with comparison details
        """
        try:
            # Setup config
            config = setup_config(config, small_model_config)

            # Create models
            hf_model = create_hf_model(hf_class, config)

            with config.mesh:
                ed_model = create_ed_model(
                    module_name=module_name,
                    task=task,
                    config=config,
                    small_model_config=small_model_config,
                    hf_model=hf_model,
                )

                # Generate inputs
                if task == ed.TaskType.BASE_VISION:
                    image_size = getattr(config, "image_size", 224)
                    num_channels = getattr(config, "num_channels", 3)
                    inputs = make_vision_inputs(
                        batch_size=small_model_config["batch_size"],
                        image_size=image_size,
                        num_channels=num_channels,
                    )
                else:
                    inputs = make_text_inputs(
                        vocab_size=small_model_config["vocab_size"],
                        batch_size=small_model_config["batch_size"],
                        seq_len=small_model_config["sequence_length"],
                    )

                # Run HF forward with hidden states
                hf_inputs = inputs["torch"]
                hf_output, hf_time = self._run_hf_forward(hf_model, hf_inputs, output_hidden_states=True)

                # Run ED forward with hidden states
                ed_inputs = inputs["jax"]

                @ed.ejit(static_argnums=(0,))  # pyright: ignore[reportUntypedFunctionDecorator]
                def jited(gd, gs, go, **kwargs):
                    model = nn.merge(gd, gs, go)
                    return model(**kwargs, output_hidden_states=True)

                _ = jited(*ed_model.split_module(), **ed_inputs)
                with ed.utils.capture_time() as timer:
                    ed_output = jited(*ed_model.split_module(), **ed_inputs)
                ed_time = timer()

                # Compare hidden states
                hf_hidden = hf_output.last_hidden_state.cpu().detach().numpy()
                ed_hidden = ed_output.last_hidden_state

                comparison = compare_hidden_states(
                    name=module_name,
                    hf_hidden=hf_hidden,
                    ed_hidden=ed_hidden,
                )

            cleanup_models(hf_model)

            return TestResult(
                success=comparison.success,
                comparison=comparison,
                ed_time=ed_time,
                hf_time=hf_time,
            )

        except Exception as e:
            return TestResult(
                success=False,
                error_message=str(e),
            )


class SequenceClassificationTester(BaseTester):
    """Test SEQUENCE_CLASSIFICATION models."""

    def run(
        self,
        module_name: str,
        hf_class: type,
        task: ed.TaskType,
        config: Any,
        small_model_config: dict,
        num_labels: int = 2,
    ) -> TestResult:
        """Run forward pass and compare classification logits.

        Args:
            module_name: Name of the module
            hf_class: HuggingFace model class
            task: Task type (should be SEQUENCE_CLASSIFICATION)
            config: Model configuration
            small_model_config: Base config dictionary
            num_labels: Number of classification labels

        Returns:
            TestResult with comparison details
        """
        try:
            # Set num_labels in config
            config.num_labels = num_labels

            # Setup config
            config = setup_config(config, small_model_config)

            # Create models
            hf_model = create_hf_model(hf_class, config)

            with config.mesh:
                ed_model = create_ed_model(
                    module_name=module_name,
                    task=task,
                    config=config,
                    small_model_config=small_model_config,
                    hf_model=hf_model,
                )

                # Generate inputs
                inputs = make_classification_inputs(
                    vocab_size=small_model_config["vocab_size"],
                    batch_size=small_model_config["batch_size"],
                    seq_len=small_model_config["sequence_length"],
                    num_labels=num_labels,
                )

                # Run HF forward
                hf_inputs = inputs["torch"]
                hf_output, hf_time = self._run_hf_forward(hf_model, hf_inputs)

                # Run ED forward
                ed_inputs = inputs["jax"]
                ed_output, ed_time = self._run_ed_forward(ed_model, ed_inputs)

                # Compare outputs
                comparison = compare_logits(
                    name=f"{module_name}_classification",
                    hf_logits=hf_output.logits.cpu().detach().numpy(),
                    ed_logits=ed_output.logits,
                    hf_loss=float(hf_output.loss.cpu().detach().numpy()) if hf_output.loss is not None else None,
                    ed_loss=float(ed_output.loss) if ed_output.loss is not None else None,
                )

            cleanup_models(hf_model)

            return TestResult(
                success=comparison.success,
                comparison=comparison,
                ed_time=ed_time,
                hf_time=hf_time,
            )

        except Exception as e:
            return TestResult(
                success=False,
                error_message=str(e),
            )


class VisionLanguageTester(BaseTester):
    """Test IMAGE_TEXT_TO_TEXT (Vision-Language) models."""

    @staticmethod
    def _build_vlm_jit_inputs(ed_model: Any, ed_inputs: dict, vlm_config: dict) -> tuple[dict, dict]:
        """Prepare inputs for a jittable VLM forward pass.

        Computes `inputs_embeds` outside of `ejit` (including multimodal merge) and
        passes only JIT-friendly tensors (e.g. `inputs_embeds`, `position_ids`,
        `deepstack_visual_embeds`) to the compiled forward.
        """
        input_ids = ed_inputs["input_ids"]
        attention_mask = ed_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool)
        else:
            attention_mask = attention_mask.astype(jnp.bool)

        embedding_kwargs = {k: v for k, v in ed_inputs.items() if k not in ["input_ids", "attention_mask"]}
        if vlm_config.get("is_qwen_vl", False):
            inputs_embeds, embed_info = ed_model.compute_embedding_with_info(
                input_ids,
                attention_mask=attention_mask,
                **embedding_kwargs,
            )
        else:
            inputs_embeds, embed_info = ed_model.compute_embedding_with_info(input_ids, **embedding_kwargs)

        forward_kwargs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        loss_kwargs: dict[str, Any] = {"labels": input_ids}

        token_type_ids = ed_inputs.get("token_type_ids")
        if token_type_ids is not None:
            forward_kwargs["token_type_ids"] = token_type_ids

        if embed_info is not None:
            position_ids = getattr(embed_info, "position_ids", None)
            if position_ids is not None:
                forward_kwargs["position_ids"] = jnp.asarray(position_ids, dtype="i4")

            visual_pos_masks = getattr(embed_info, "visual_pos_masks", None)
            deepstack_visual_embeds = getattr(embed_info, "deepstack_visual_embeds", None)
            if deepstack_visual_embeds is not None:
                forward_kwargs["visual_pos_masks"] = visual_pos_masks
                forward_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds

        if vlm_config.get("is_qwen_vl", False) and hasattr(ed_model, "base_model"):
            base_model = ed_model.base_model
            if hasattr(base_model, "get_rope_index"):
                if "position_ids" not in forward_kwargs:
                    position_ids, _rope_deltas = base_model.get_rope_index(
                        input_ids=input_ids,
                        image_grid_thw=ed_inputs.get("image_grid_thw"),
                        video_grid_thw=ed_inputs.get("video_grid_thw"),
                        attention_mask=attention_mask,
                    )
                    forward_kwargs["position_ids"] = jnp.asarray(position_ids, dtype="i4")

        return forward_kwargs, loss_kwargs

    def run(
        self,
        module_name: str,
        hf_class: type | None,
        task: ed.TaskType,
        config: Any,
        small_model_config: dict,
        vlm_config: dict,
    ) -> TestResult:
        """Run forward pass with vision inputs.

        Args:
            module_name: Name of the module
            hf_class: HuggingFace model class
            task: Task type (should be IMAGE_TEXT_TO_TEXT)
            config: Model configuration
            small_model_config: Base config dictionary
            vlm_config: VLM-specific configuration with:
                - image_token_id: Token ID for image placeholders
                - num_image_tokens: Number of tokens per image
                - pixel_values_shape: Shape of pixel_values tensor
                - num_images: Number of images (default 1)
                - use_token_type_ids: Whether to use token_type_ids
                - is_qwen_vl: Whether this is a Qwen VL model
                - vision_start_token_id: (for Qwen) Start token
                - vision_end_token_id: (for Qwen) End token
                - image_grid_thw: (for Qwen) Grid shape array

        Returns:
            TestResult with comparison details
        """
        try:
            # Setup config
            config = setup_config(config, small_model_config)

            # Generate VLM inputs based on model type
            if vlm_config.get("is_qwen_vl", False):
                inputs = make_qwen_vlm_inputs(
                    vocab_size=small_model_config["vocab_size"],
                    batch_size=small_model_config["batch_size"],
                    seq_len=small_model_config["sequence_length"],
                    image_token_id=vlm_config["image_token_id"],
                    vision_start_token_id=vlm_config["vision_start_token_id"],
                    vision_end_token_id=vlm_config["vision_end_token_id"],
                    num_image_tokens=vlm_config["num_image_tokens"],
                    pixel_values_shape=vlm_config["pixel_values_shape"],
                    image_grid_thw=vlm_config["image_grid_thw"],
                    num_images=vlm_config.get("num_images", 1),
                )
            else:
                inputs = make_vlm_inputs(
                    vocab_size=small_model_config["vocab_size"],
                    batch_size=small_model_config["batch_size"],
                    seq_len=small_model_config["sequence_length"],
                    image_token_id=vlm_config["image_token_id"],
                    num_image_tokens=vlm_config["num_image_tokens"],
                    pixel_values_shape=vlm_config["pixel_values_shape"],
                    num_images=vlm_config.get("num_images", 1),
                    token_type_ids=vlm_config.get("use_token_type_ids", False),
                    image_grid_hws=vlm_config.get("image_grid_hws"),
                    image_grid_thw=vlm_config.get("image_grid_thw"),
                    video_grid_thw=vlm_config.get("video_grid_thw"),
                )

            # Handle EasyDeL-only VLM models (no HF comparison)
            if hf_class is None:
                with config.mesh:
                    ed_model = create_ed_model_only(
                        module_name=module_name,
                        task=task,
                        config=config,
                        small_model_config=small_model_config,
                    )

                    # Run ED forward only
                    ed_inputs = inputs["jax"]
                    forward_kwargs, loss_kwargs = self._build_vlm_jit_inputs(ed_model, ed_inputs, vlm_config)

                    @ed.ejit(static_argnums=(1,))  # pyright: ignore[reportUntypedFunctionDecorator]
                    def jited(embeds, gd, gs, go, attention_mask, labels, **kwargs):
                        model = nn.merge(gd, gs, go)
                        return model.compute_loss(
                            labels=labels,
                            inputs_embeds=embeds,
                            attention_mask=attention_mask,
                            **kwargs,
                        )

                    # Warmup
                    _ = jited(
                        forward_kwargs["inputs_embeds"],
                        *ed_model.split_module(),
                        attention_mask=forward_kwargs["attention_mask"],
                        labels=loss_kwargs["labels"],
                        **{k: v for k, v in forward_kwargs.items() if k not in ["inputs_embeds", "attention_mask"]},
                    )

                    with ed.utils.capture_time() as timer:
                        ed_output, _metrics = jited(
                            forward_kwargs["inputs_embeds"],
                            *ed_model.split_module(),
                            attention_mask=forward_kwargs["attention_mask"],
                            labels=loss_kwargs["labels"],
                            **{k: v for k, v in forward_kwargs.items() if k not in ["inputs_embeds", "attention_mask"]},
                        )
                    ed_time = timer()

                    # Verify output has expected attributes
                    assert hasattr(ed_output, "logits"), "Output missing logits"

                return TestResult(
                    success=True,
                    ed_time=ed_time,
                    extra_info={"easydel_only": True},
                )

            # Create models with HF comparison
            hf_model = create_hf_model(hf_class, config)

            with config.mesh:
                ed_model = create_ed_model(
                    module_name=module_name,
                    task=task,
                    config=config,
                    small_model_config=small_model_config,
                    hf_model=hf_model,
                )

                # Run HF forward
                hf_inputs = {
                    **inputs["torch"],
                    "labels": inputs["torch"]["input_ids"],
                }
                with ed.utils.capture_time() as timer:
                    hf_output = hf_model(
                        **hf_inputs,
                        past_key_values=None,
                        use_cache=False,
                    )
                hf_time = timer()

                # Run ED forward
                ed_inputs = inputs["jax"]
                forward_kwargs, loss_kwargs = self._build_vlm_jit_inputs(ed_model, ed_inputs, vlm_config)

                @ed.ejit(static_argnums=(1,))  # pyright: ignore[reportUntypedFunctionDecorator]
                def jited(embeds, gd, gs, go, attention_mask, labels, **kwargs):
                    model = nn.merge(gd, gs, go)
                    return model.compute_loss(
                        labels=labels,
                        inputs_embeds=embeds,
                        attention_mask=attention_mask,
                        **kwargs,
                    )

                # Warmup
                _ = jited(
                    forward_kwargs["inputs_embeds"],
                    *ed_model.split_module(),
                    attention_mask=forward_kwargs["attention_mask"],
                    labels=loss_kwargs["labels"],
                    **{k: v for k, v in forward_kwargs.items() if k not in ["inputs_embeds", "attention_mask"]},
                )

                with ed.utils.capture_time() as timer:
                    ed_output, _metrics = jited(
                        forward_kwargs["inputs_embeds"],
                        *ed_model.split_module(),
                        attention_mask=forward_kwargs["attention_mask"],
                        labels=loss_kwargs["labels"],
                        **{k: v for k, v in forward_kwargs.items() if k not in ["inputs_embeds", "attention_mask"]},
                    )
                ed_time = timer()

                # Compare outputs
                comparison = compare_logits(
                    name=module_name,
                    hf_logits=hf_output.logits.cpu().detach().numpy(),
                    ed_logits=ed_output.logits,
                    hf_loss=float(hf_output.loss.cpu().detach().numpy()) if hf_output.loss is not None else None,
                    ed_loss=float(ed_output.loss) if ed_output.loss is not None else None,
                )

            cleanup_models(hf_model)

            return TestResult(
                success=comparison.success,
                comparison=comparison,
                ed_time=ed_time,
                hf_time=hf_time,
            )

        except Exception as e:
            traceback.print_exc()
            return TestResult(success=False, error_message=str(e))


class Seq2SeqTester(BaseTester):
    """Test SEQUENCE_TO_SEQUENCE (encoder-decoder) models."""

    def run(
        self,
        module_name: str,
        hf_class: type,
        task: ed.TaskType,
        config: Any,
        small_model_config: dict,
    ) -> TestResult:
        """Run forward pass and compare encoder-decoder outputs.

        Args:
            module_name: Name of the module
            hf_class: HuggingFace model class
            task: Task type (should be SEQUENCE_TO_SEQUENCE)
            config: Model configuration
            small_model_config: Base config dictionary

        Returns:
            TestResult with comparison details
        """
        try:
            # Setup config
            config = setup_config(config, small_model_config)

            # Create models
            hf_model = create_hf_model(hf_class, config)

            with config.mesh:
                ed_model = create_ed_model(
                    module_name=module_name,
                    task=task,
                    config=config,
                    small_model_config=small_model_config,
                    hf_model=hf_model,
                )

                if task == ed.TaskType.SPEECH_SEQUENCE_TO_SEQUENCE:
                    audio_length = int(getattr(config, "max_source_positions", small_model_config["sequence_length"]))
                    audio_length *= 2  # Whisper expects 2x downsampling (e.g., 1500 -> 3000 mel frames)
                    audio_inputs = make_audio_inputs(
                        batch_size=small_model_config["batch_size"],
                        audio_length=audio_length,
                        num_mel_bins=getattr(config, "num_mel_bins", 80),
                    )
                    decoder_inputs = make_text_inputs(
                        vocab_size=small_model_config["vocab_size"],
                        batch_size=small_model_config["batch_size"],
                        seq_len=small_model_config["sequence_length"] // 2,
                    )

                    hf_inputs = {
                        **audio_inputs["torch"],
                        "decoder_input_ids": decoder_inputs["torch"]["input_ids"],
                        "labels": decoder_inputs["torch"]["input_ids"],
                    }
                    hf_output, hf_time = self._run_hf_forward(hf_model, hf_inputs)

                    ed_inputs = {
                        **audio_inputs["jax"],
                        "decoder_input_ids": decoder_inputs["jax"]["input_ids"],
                    }

                    @ed.ejit(static_argnums=(1,))  # pyright: ignore[reportUntypedFunctionDecorator]
                    def jited(input_features, gd, gs, go, decoder_input_ids):
                        model = nn.merge(gd, gs, go)
                        return model.compute_loss(
                            labels=decoder_input_ids,
                            input_features=input_features,
                            decoder_input_ids=decoder_input_ids,
                        )

                    _ = jited(
                        ed_inputs["input_features"],
                        *ed_model.split_module(),
                        decoder_input_ids=ed_inputs["decoder_input_ids"],
                    )

                    with ed.utils.capture_time() as timer:
                        ed_output, _metrics = jited(
                            ed_inputs["input_features"],
                            *ed_model.split_module(),
                            decoder_input_ids=ed_inputs["decoder_input_ids"],
                        )
                    ed_time = timer()
                else:
                    # Generate inputs
                    inputs = make_seq2seq_inputs(
                        vocab_size=small_model_config["vocab_size"],
                        batch_size=small_model_config["batch_size"],
                        src_len=small_model_config["sequence_length"],
                        tgt_len=small_model_config["sequence_length"] // 2,
                    )

                    # Run HF forward
                    hf_inputs = {
                        **inputs["torch"],
                        "labels": inputs["torch"]["decoder_input_ids"],
                    }
                    hf_output, hf_time = self._run_hf_forward(hf_model, hf_inputs)

                    # Run ED forward
                    ed_inputs = {**inputs["jax"], "labels": inputs["jax"]["decoder_input_ids"]}

                    @ed.ejit(static_argnums=(1,))  # pyright: ignore[reportUntypedFunctionDecorator]
                    def jited(input_ids, gd, gs, go, attention_mask, labels, **kwargs):
                        model = nn.merge(gd, gs, go)
                        return model.compute_loss(
                            labels=labels,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **kwargs,
                        )

                    extra_kwargs = {
                        k: v for k, v in ed_inputs.items() if k not in ["input_ids", "attention_mask", "labels"]
                    }
                    _ = jited(
                        ed_inputs["input_ids"],
                        *ed_model.split_module(),
                        attention_mask=ed_inputs["attention_mask"],
                        labels=ed_inputs["labels"],
                        **extra_kwargs,
                    )
                    with ed.utils.capture_time() as timer:
                        ed_output, _metrics = jited(
                            ed_inputs["input_ids"],
                            *ed_model.split_module(),
                            attention_mask=ed_inputs["attention_mask"],
                            labels=ed_inputs["labels"],
                            **extra_kwargs,
                        )
                    ed_time = timer()

                # Compare outputs
                comparison = compare_logits(
                    name=module_name,
                    hf_logits=hf_output.logits.cpu().detach().numpy(),
                    ed_logits=ed_output.logits,
                    hf_loss=float(hf_output.loss.cpu().detach().numpy()) if hf_output.loss is not None else None,
                    ed_loss=float(ed_output.loss) if ed_output.loss is not None else None,
                )

            cleanup_models(hf_model)

            return TestResult(
                success=comparison.success,
                comparison=comparison,
                ed_time=ed_time,
                hf_time=hf_time,
            )

        except Exception as e:
            return TestResult(
                success=False,
                error_message=str(e),
            )

    def test_generation(
        self,
        module_name: str,
        hf_class: type,
        config: Any,
        small_model_config: dict,
        max_new_tokens: int = 16,
        task: ed.TaskType = ed.TaskType.SEQUENCE_TO_SEQUENCE,
    ) -> TestResult:
        """Test encoder-decoder generation.

        Args:
            module_name: Name of the module
            hf_class: HuggingFace model class
            config: Model configuration
            small_model_config: Base config dictionary
            max_new_tokens: Number of tokens to generate
            task: Task type used to look up the EasyDeL module.

        Returns:
            TestResult indicating if generation succeeded
        """
        try:
            # Setup config
            config = setup_config(config, small_model_config)

            # Create models
            hf_model = create_hf_model(hf_class, config)

            with config.mesh:
                ed_model = create_ed_model(
                    module_name=module_name,
                    task=task,
                    config=config,
                    small_model_config=small_model_config,
                    hf_model=hf_model,
                )

                if task == ed.TaskType.SPEECH_SEQUENCE_TO_SEQUENCE:
                    from transformers import GenerationConfig

                    if not hasattr(ed_model, "generation_config") or ed_model.generation_config is None:
                        ed_model.generation_config = GenerationConfig(
                            max_length=getattr(config, "max_target_positions", max_new_tokens + 1),
                            max_new_tokens=max_new_tokens,
                            pad_token_id=getattr(config, "pad_token_id", 0) or 0,
                            eos_token_id=getattr(config, "eos_token_id", 2) or 2,
                            bos_token_id=getattr(config, "bos_token_id", 1) or 1,
                            decoder_start_token_id=getattr(
                                config, "decoder_start_token_id", getattr(config, "bos_token_id", 1) or 1
                            ),
                        )

                    audio_length = int(getattr(config, "max_source_positions", small_model_config["sequence_length"]))
                    audio_length *= 2  # Whisper expects 2x downsampling (e.g., 1500 -> 3000 mel frames)
                    inputs = make_audio_inputs(
                        batch_size=1,
                        audio_length=audio_length,
                        num_mel_bins=getattr(config, "num_mel_bins", 80),
                    )

                    # Run generation
                    with ed.utils.capture_time() as timer:
                        output = ed_model.generate(
                            input_features=inputs["jax"]["input_features"],
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    ed_time = timer()
                else:
                    # Generate inputs (encoder only for generation)
                    inputs = make_text_inputs(
                        vocab_size=small_model_config["vocab_size"],
                        batch_size=1,
                        seq_len=32,
                    )

                    # Run generation
                    with ed.utils.capture_time() as timer:
                        output = ed_model.generate(
                            input_ids=inputs["jax"]["input_ids"],
                            attention_mask=inputs["jax"]["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    ed_time = timer()

            cleanup_models(hf_model)

            # Verify output
            assert output is not None, "Generation returned None"
            assert hasattr(output, "sequences"), "Output missing sequences"

            return TestResult(
                success=True,
                ed_time=ed_time,
                extra_info={
                    "generated_length": int(output.sequences.shape[1]),
                },
            )

        except Exception as e:
            return TestResult(
                success=False,
                error_message=str(e),
            )


class EasyDeLOnlyTester:
    """Test EasyDeL models without HuggingFace comparison.

    Used for models that don't have HuggingFace equivalents.
    """

    def run(
        self,
        module_name: str,
        task: ed.TaskType,
        config: Any,
        small_model_config: dict,
    ) -> TestResult:
        """Run forward pass and verify basic output properties.

        Args:
            module_name: Name of the module
            task: Task type
            config: Model configuration
            small_model_config: Base config dictionary

        Returns:
            TestResult indicating if forward pass succeeded
        """
        try:
            from .model_factory import create_ed_model_only

            # Setup config
            config = setup_config(config, small_model_config)

            with config.mesh:
                ed_model = create_ed_model_only(
                    module_name=module_name,
                    task=task,
                    config=config,
                    small_model_config=small_model_config,
                )

                # Generate inputs
                inputs = make_text_inputs(
                    vocab_size=small_model_config["vocab_size"],
                    batch_size=small_model_config["batch_size"],
                    seq_len=small_model_config["sequence_length"],
                )

                # Run forward pass
                with ed.utils.capture_time() as timer:

                    @ed.ejit(static_argnums=(0,))  # pyright: ignore[reportUntypedFunctionDecorator]
                    def jited(gd, gs, go, input_ids, attention_mask):
                        model = nn.merge(gd, gs, go)
                        return model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )

                    ed_output = jited(
                        *ed_model.split_module(),
                        inputs["jax"]["input_ids"],
                        inputs["jax"]["attention_mask"],
                    )
                ed_time = timer()

            # Verify output
            assert ed_output is not None, f"{module_name} forward pass returned None"
            assert hasattr(ed_output, "last_hidden_state"), f"{module_name} output missing last_hidden_state"
            assert bool(jnp.isfinite(ed_output.last_hidden_state).all()), f"{module_name} last_hidden_state not finite"

            return TestResult(
                success=True,
                ed_time=ed_time,
                extra_info={
                    "last_hidden_state_shape": list(ed_output.last_hidden_state.shape),
                },
            )

        except Exception as e:
            return TestResult(
                success=False,
                error_message=str(e),
            )
