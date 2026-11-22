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

"""Sparse (gray-box) knowledge distillation trainer.

The student generates completions on-policy, the teacher scores them with
top-k logprobs, and the student minimises partial KL divergence between
its own distribution and the teacher's sparse top-k distribution.
"""

from __future__ import annotations

import contextlib
import typing as tp
from functools import partial

import jax
import numpy as np
import spectrax as spx
from eformer.loggings import get_logger
from jax import numpy as jnp
from spectrax import with_sharding_constraint

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.helpers import capture_time

from .._shared import drop_optimizer_state
from ..metrics import _host_mean_float
from ..model_loading import disable_state_dropout, reject_string_model_id
from ..prompt_transforms import GRPOPreprocessTransform
from ..trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import (
    compile_trainer_step,
    filter_kwargs_for_callable,
    resolve_straight_through_emulator,
    sanitize_model_call_kwargs,
)
from ._fn import sparse_distillation_step
from .sparse_distillation_config import SparseDistillationConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

SparseTeacherFn = tp.Callable[
    [jax.Array, jax.Array, list[str]],
    tuple[jax.Array, jax.Array],
]

logger = get_logger(__name__)


@Registry.register("trainer", "sparse_distillation")
class SparseDistillationTrainer(Trainer):
    """Sparse (gray-box) knowledge distillation trainer.

    The student generates completions on-policy, the teacher scores them with
    top-k logprobs, and the student minimises partial KL divergence.  This
    works with API-based teachers that expose limited logprob information.

    Two teacher modes are supported:
        1. **Local teacher model** (``teacher_model``): An EasyDeL model whose
           full logits are computed, then only the top-k are retained.
        2. **External teacher function** (``teacher_fn``): A callable
           ``(input_ids, attention_mask, prompt_texts) -> (top_k_indices, top_k_logprobs)``
           for API-based teachers.  ``prompt_texts`` is a ``list[str]``
           of decoded prompts so the function can forward raw text to an
           external API when the teacher uses a different tokenizer.

    Training loop:
        1. Sample prompts from the dataset
        2. Student generates completions (on-policy)
        3. Teacher scores the full sequences → top-k logprobs
        4. Student minimises partial KL on the generated tokens

    Attributes:
        arguments (SparseDistillationConfig): Resolved sparse-distillation
            configuration carried from initialisation.
        teacher_state (EasyDeLState | None): In-process teacher state
            whose top-k logprobs are computed locally; ``None`` when an
            external ``teacher_fn`` is used instead.
        teacher_fn (SparseTeacherFn | None): External callable mapping
            ``(input_ids, attention_mask, prompt_texts)`` to
            ``(top_k_indices, top_k_logprobs)`` for API-based teachers;
            ``None`` when ``teacher_state`` is used instead.
        processing_class (ProcessingClassType): Tokenizer / processor
            used for prompt encoding and (for the callable teacher
            mode) prompt-text decoding.
        padding_value (int): Pad token id used by the collators; falls
            back to ``0`` when the tokenizer exposes no pad token.

    Example:
        >>> config = SparseDistillationConfig(
        ...     top_k_teacher=20,
        ...     temperature=3.0,
        ...     alpha=0.9,
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ...     learning_rate=2e-5,
        ... )
        >>> trainer = SparseDistillationTrainer(
        ...     arguments=config,
        ...     student_model=student,
        ...     teacher_model=teacher,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer,
        ... )
        >>> trainer.train()
    """

    supports_sequence_packing: tp.ClassVar[bool] = False

    teacher_state: EasyDeLState | None
    teacher_fn: SparseTeacherFn | None
    arguments: SparseDistillationConfig

    def __init__(
        self,
        arguments: SparseDistillationConfig,
        processing_class: ProcessingClassType,
        student_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_fn: SparseTeacherFn | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
    ):
        """Initialize the sparse-distillation trainer.

        Exactly one of ``teacher_model`` or ``teacher_fn`` must be
        provided. If both are ``None`` a :class:`ValueError` is raised.

        Args:
            arguments (SparseDistillationConfig): Trainer configuration.
            processing_class (ProcessingClassType): Tokenizer / processor
                used for prompt encoding and (where applicable) decoding
                for ``teacher_fn``.
            student_model (EasyDeLBaseModule | EasyDeLState | None):
                Student model under training.
            teacher_model (EasyDeLBaseModule | EasyDeLState | None):
                Optional in-process teacher whose top-k logprobs are
                computed locally.
            teacher_fn (SparseTeacherFn | None): Optional callable
                returning ``(top_k_indices, top_k_logprobs)`` arrays
                given ``(input_ids, attention_mask, prompt_texts)``.
            train_dataset (Dataset | IterableDataset | ShardedDataSource | None):
                Training dataset of prompts.
            eval_dataset (Dataset | IterableDataset | ShardedDataSource |
                dict[str, Dataset] | None): Optional evaluation
                dataset(s).

        Raises:
            TypeError: If ``arguments`` is not a
                :class:`SparseDistillationConfig`.
            ValueError: If neither ``teacher_model`` nor ``teacher_fn``
                is provided.
        """
        tokenizer = processing_class
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token

        if not isinstance(arguments, SparseDistillationConfig):
            raise TypeError("passed argument must be a `SparseDistillationConfig`.")
        if teacher_model is None and teacher_fn is None:
            raise ValueError("Either `teacher_model` or `teacher_fn` must be provided.")

        self.arguments = arguments

        if isinstance(student_model, str):
            reject_string_model_id(student_model, role="student model")
        if student_model is None:
            raise ValueError("`student_model` must be provided for SparseDistillationTrainer.")
        if isinstance(teacher_model, str):
            reject_string_model_id(teacher_model, role="teacher model")

        if not isinstance(student_model, EasyDeLState):
            student_model = student_model.to_state(trainable_selector=arguments.trainable_selector)
        if getattr(arguments, "disable_dropout", False):
            student_model = disable_state_dropout(student_model)

        if teacher_model is not None:
            if not isinstance(teacher_model, EasyDeLState):
                teacher_model.eval()
                teacher_model = teacher_model.to_state()
            teacher_model = disable_state_dropout(teacher_model)

        self.teacher_state = (
            None
            if teacher_model is None
            else self._share_esurge_cache_scope_if_compatible(
                drop_optimizer_state(teacher_model),
                primary_state=student_model,
            )
        )
        self._eval_teacher_graphdef = None
        self.teacher_fn = teacher_fn
        self.processing_class = processing_class

        pad_token_id = getattr(processing_class, "pad_token_id", None)
        if pad_token_id is None and hasattr(processing_class, "tokenizer"):
            pad_token_id = getattr(processing_class.tokenizer, "pad_token_id", None)
        self.padding_value = 0 if pad_token_id is None else int(pad_token_id)

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=student_model,
            data_collator=None,
            processing_class=processing_class,
        )
        if self.teacher_state is not None:
            self.teacher_state = self._share_esurge_cache_scope_if_compatible(
                self.teacher_state,
                primary_state=self.model_state,
            )

    def _get_preprocess_transform(self) -> GRPOPreprocessTransform | None:
        """Build the lazy prompt-only preprocessing transform for sparse distillation.

        Sparse distillation runs the student on-policy over prompts and
        scores completions through the teacher, so the dataset only
        needs to provide prompts. The :class:`GRPOPreprocessTransform`
        applies the chat template (unless
        ``skip_apply_chat_template`` is set) and tokenises to
        ``arguments.max_prompt_length`` lazily inside the data loader.

        Returns:
            A :class:`GRPOPreprocessTransform`, or ``None`` when the
            source already exposes ``input_ids`` (per
            :meth:`_is_pretokenized`).
        """
        if self._is_pretokenized():
            return None
        return GRPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            skip_apply_chat_template=self.arguments.skip_apply_chat_template,
        )

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Construct the Grain collator that left-pads prompt-only batches.

        Sparse distillation operates on prompt-only batches; on-policy
        completions are sampled inside :meth:`_preprocess_batch_input`.
        The collator left-pads tokenised prompts to
        ``arguments.max_prompt_length`` using ``self.padding_value`` so
        student generation receives a contiguous, right-aligned prefix.

        Args:
            max_sequence_length: Accepted for compatibility with
                :class:`Trainer`; ignored.
            truncation_mode: Accepted for compatibility with
                :class:`Trainer`; ignored.

        Returns:
            A freshly built :class:`GRPODataCollatorGrain`.
        """
        from ..utils import GRPODataCollatorGrain

        return GRPODataCollatorGrain(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Construct the TFDS collator that left-pads prompt-only batches.

        TFDS analogue of :meth:`create_grain_collect_function`.

        Args:
            max_sequence_length: Accepted for compatibility with
                :class:`Trainer`; ignored.
            truncation_mode: Accepted for compatibility with
                :class:`Trainer`; ignored.

        Returns:
            A freshly built :class:`GRPODataCollatorTFDS`.
        """
        from ..utils import GRPODataCollatorTFDS

        return GRPODataCollatorTFDS(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def _configure_eval_config_override(self):
        """Apply evaluation config overrides to the local frozen teacher as well."""
        super()._configure_eval_config_override()
        self._eval_teacher_graphdef = None
        overrides = getattr(self.arguments, "eval_config_overrides", None)
        if self._eval_graphdef is not None and overrides and self.teacher_state is not None:
            teacher_graphstate = self.teacher_state.graphstate
            if hasattr(teacher_graphstate, "items"):
                selected_keys = {(collection, path) for collection, path, _ in teacher_graphstate.items()}

                def teacher_selector(v, path):
                    return (v.kind, path) in selected_keys
            else:
                teacher_selector = None
            self._eval_teacher_graphdef = self._build_config_override_graphdef(
                model=self.teacher_state.model,
                overrides=overrides,
                reference_graphstate=teacher_graphstate,
                trainable_selector=teacher_selector,
                label="eval_config_overrides (teacher)",
                eval_mode=True,
            )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Compile the sparse-distillation training and evaluation step functions.

        Sparse distillation differs from the dense distillation path in
        that the teacher contributes only its top-k log-probabilities;
        :func:`sparse_distillation_step` consumes those tensors out of
        the prepared batch instead of doing a teacher forward inside the
        compiled graph. This compile-step therefore wires:

        * the resolved straight-through emulator for quantisation-aware
          training (when configured);
        * static-argument tuples for both train and eval phases that
          freeze ``loss_config``, ``scheduler``, ``step_partition_spec``,
          ``gradient_accumulation_steps``, the ``is_training`` flag,
          ``temperature`` (KL softening), and ``alpha`` (mixing
          coefficient between hard-label CE and KL terms);
        * two compiled step functions over :func:`sparse_distillation_step`
          that share the same input/output shardings -- state donated,
          batch and metrics replicated.

        When a local teacher state is available the method also adds the
        teacher-forward FLOPs (with loss accounting, no backward) to
        ``self._extra_forward_flops_per_token`` so the trainer's
        throughput math accounts for the per-step teacher pass.

        Returns:
            A :class:`TrainerConfigureFunctionOutput` with the compiled
            step functions, the active mesh, and the streaming
            checkpoint manager.
        """
        mesh = self.model.mesh
        empty_sharding = replicated_named_sharding(mesh)

        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_group_size=self.arguments.quantization_group_size,
            quantization_bits=self.arguments.quantization_bits,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )

        self._train_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_training
            self.arguments.temperature,
            self.arguments.alpha,
            straight_through_emulator,
        )

        static_argnames = tuple(range(2, 10))

        self._runtime_trace("train.compile_wrapper.begin")
        sharded_training_step_function = compile_trainer_step(
            sparse_distillation_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
        )
        self._runtime_trace("train.compile_wrapper.end")

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_training
            self.arguments.temperature,
            self.arguments.alpha,
            None,  # straight_through_emulator
        )

        self._runtime_trace("eval.compile_wrapper.begin")
        sharded_evaluation_step_function = compile_trainer_step(
            sparse_distillation_step,
            in_shardings=(self.eval_state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
        )
        self._runtime_trace("eval.compile_wrapper.end")

        self._compute_teacher_topk_train = None
        self._compute_teacher_topk_eval = None
        if self.teacher_state is not None:
            top_k_teacher = int(self.arguments.top_k_teacher)
            teacher_mesh = self.teacher_state.model.mesh
            teacher_empty_sharding = replicated_named_sharding(teacher_mesh)

            def _score_teacher_topk(
                graphtree: spx.State,
                graphother: spx.State,
                input_ids: jax.Array,
                attention_mask: jax.Array,
                graphdef: spx.GraphDef,
            ) -> tuple[jax.Array, jax.Array]:
                graphother = jax.tree_util.tree_map(
                    lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
                    graphother,
                )
                teacher_module = spx.bind(graphdef, graphtree.merge(graphother, copy=False))
                with teacher_module.mesh:
                    input_ids = with_sharding_constraint(
                        input_ids,
                        self.arguments.step_partition_spec,
                        mesh=teacher_module.mesh,
                        ignore_mpmd=True,
                    )
                    attention_mask = with_sharding_constraint(
                        attention_mask,
                        self.arguments.step_partition_spec,
                        mesh=teacher_module.mesh,
                        ignore_mpmd=True,
                    )
                    teacher_kwargs: dict[str, tp.Any] = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                    teacher_kwargs = filter_kwargs_for_callable(
                        getattr(teacher_module, "forward", teacher_module),
                        teacher_kwargs,
                    )
                    teacher_kwargs = sanitize_model_call_kwargs(teacher_kwargs)
                    teacher_outputs = teacher_module(**teacher_kwargs)
                    teacher_log_probs = jax.nn.log_softmax(
                        teacher_outputs.logits.astype(jnp.float32),
                        axis=-1,
                    )
                    top_k_logprobs, top_k_indices = jax.lax.top_k(teacher_log_probs, top_k_teacher)
                    return top_k_indices, top_k_logprobs

            self._compute_teacher_topk_train = compile_trainer_step(
                partial(_score_teacher_topk, graphdef=self.teacher_state.graphdef),
                mesh=teacher_mesh,
                static_argnames=("graphdef",),
                in_shardings=(
                    self.teacher_state.shardings.graphstate,
                    self.teacher_state.shardings.graphother,
                    teacher_empty_sharding,
                    teacher_empty_sharding,
                ),
                out_shardings=(teacher_empty_sharding, teacher_empty_sharding),
            )
            eval_teacher_graphdef = getattr(self, "_eval_teacher_graphdef", None)
            self._compute_teacher_topk_eval = (
                compile_trainer_step(
                    partial(_score_teacher_topk, graphdef=eval_teacher_graphdef),
                    mesh=teacher_mesh,
                    static_argnames=("graphdef",),
                    in_shardings=(
                        self.teacher_state.shardings.graphstate,
                        self.teacher_state.shardings.graphother,
                        teacher_empty_sharding,
                        teacher_empty_sharding,
                    ),
                    out_shardings=(teacher_empty_sharding, teacher_empty_sharding),
                )
                if eval_teacher_graphdef is not None
                else self._compute_teacher_topk_train
            )

            flops_per_tkn = self.teacher_state.model.flops_per_token(include_loss=True, include_backward=False)
            self._extra_forward_flops_per_token = flops_per_tkn
            self._extra_backward_flops_per_token = 0.0

        self.arguments.ensure_checkpoint_path()
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=self.arguments.get_streaming_checkpointer(),
        )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, tp.Any],
        is_train: bool,
    ) -> tuple[dict[str, tp.Any], dict[str, float | int | str]]:
        """Generate completions and score with teacher to get top-k logprobs.

        Steps:
            1. Student generates completions (on-policy).
            2. Full sequences (prompt + completion) are assembled.
            3. Teacher scores the sequences and the top-k logprobs are
               extracted (locally via ``teacher_state`` or via the
               ``teacher_fn`` callable).
            4. The batch returned to the step function contains the
               pre-computed teacher top-k data alongside the student
               inputs.

        Args:
            state (EasyDeLState): Current student state used as the
                generator.
            batch (dict[str, tp.Any]): Raw input batch carrying prompt
                token ids and attention mask.
            is_train (bool): Whether this preprocessing runs for
                training (currently behavior-equivalent for eval).

        Returns:
            tuple[dict[str, tp.Any], dict[str, float | int | str]]: A
            pair of (training_batch, auxiliary_metrics). The training
            batch contains ``input_ids``, ``attention_mask``,
            ``completion_mask``, ``teacher_top_k_indices``, and
            ``teacher_top_k_logprobs``.
        """
        batch = self._apply_user_data_collator(batch)
        batch = self._purify_batch(batch)

        scoring_time: float = 0.0
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

            with capture_time() as generation_time_fn:
                results = self.rollout(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                )
                prompt_ids = results.prompt_ids
                prompt_mask = results.prompt_mask
                completion_ids = results.completion_ids
            generation_time = generation_time_fn()

            completion_mask = self._make_attn_mask(completion_ids)
            if results.completion_mask is not None:
                completion_mask = completion_mask * results.completion_mask.astype(completion_mask.dtype)

            # Handle generation factor (num_generations_per_prompt > 1)
            generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
            generation_factor = max(generation_factor, 1)
            expanded_prompt_ids = prompt_ids.repeat(generation_factor, 0) if generation_factor > 1 else prompt_ids
            expanded_prompt_mask = prompt_mask.repeat(generation_factor, 0) if generation_factor > 1 else prompt_mask

            input_ids_full = jnp.concatenate([expanded_prompt_ids, completion_ids], axis=1)
            attention_mask_full = jnp.concatenate([expanded_prompt_mask, completion_mask], axis=1)

            # Loss mask: zero for prompt, completion_mask for generated tokens
            prompt_zero_mask = jnp.zeros_like(expanded_prompt_mask)
            loss_mask = jnp.concatenate([prompt_zero_mask, completion_mask], axis=1)

            with capture_time() as scoring_time_fn:
                if self.teacher_fn is not None:
                    prompt_texts = self.processing_class.batch_decode(
                        np.asarray(jax.device_get(expanded_prompt_ids)),
                        skip_special_tokens=True,
                    )
                    top_k_indices, top_k_logprobs = self.teacher_fn(
                        input_ids_full,
                        attention_mask_full,
                        prompt_texts,
                    )
                    top_k_indices = jnp.asarray(top_k_indices)
                    top_k_logprobs = jnp.asarray(top_k_logprobs)
                else:
                    scorer = getattr(
                        self,
                        "_compute_teacher_topk_train" if is_train else "_compute_teacher_topk_eval",
                        None,
                    )
                    if scorer is not None:
                        top_k_indices, top_k_logprobs = scorer(
                            self.teacher_state.graphstate,
                            self.teacher_state.graphother,
                            input_ids_full,
                            attention_mask_full,
                        )
                    else:
                        teacher_model = self.teacher_state.model
                        teacher_mesh = getattr(teacher_model, "mesh", None)
                        mesh_ctx = teacher_mesh if teacher_mesh is not None else contextlib.nullcontext()
                        with mesh_ctx:
                            teacher_kwargs: dict[str, tp.Any] = {
                                "input_ids": input_ids_full,
                                "attention_mask": attention_mask_full,
                            }
                            teacher_kwargs = filter_kwargs_for_callable(
                                getattr(teacher_model, "forward", teacher_model),
                                teacher_kwargs,
                            )
                            teacher_kwargs = sanitize_model_call_kwargs(teacher_kwargs)
                            teacher_outputs = teacher_model(**teacher_kwargs)

                            k = self.arguments.top_k_teacher
                            teacher_log_probs = jax.nn.log_softmax(
                                teacher_outputs.logits.astype(jnp.float32),
                                axis=-1,
                            )
                            top_k_logprobs, top_k_indices = jax.lax.top_k(teacher_log_probs, k)  # both [B, L, K]
            scoring_time = scoring_time_fn()

            # Gather to replicated sharding
            input_ids_full = self._all_gather(input_ids_full)
            attention_mask_full = self._all_gather(attention_mask_full)
            loss_mask = self._all_gather(loss_mask)
            top_k_indices = self._all_gather(top_k_indices)
            top_k_logprobs = self._all_gather(top_k_logprobs)

        preprocessing_time = preprocessing_time_fn()

        completion_length = jnp.sum(completion_mask, -1)
        metrics_dict: dict[str, float | int | str] = {
            "completion_length": _host_mean_float(completion_length),
            "generation_time": generation_time,
            "scoring_time": scoring_time,
            "preprocessing_time": preprocessing_time,
        }
        if is_train:
            self._log_training_generations_to_wandb(
                state=state,
                prompts=expanded_prompt_ids,
                prompt_mask=expanded_prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                generation_time=generation_time,
                source="student",
            )

        return (
            {
                "input_ids": input_ids_full,
                "attention_mask": attention_mask_full,
                "completion_mask": loss_mask,
                "teacher_top_k_indices": top_k_indices,
                "teacher_top_k_logprobs": top_k_logprobs,
            },
            metrics_dict,
        )
