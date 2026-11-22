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

"""Generalized Knowledge Distillation (GKD) trainer.

Implements :class:`GKDTrainer`, an :class:`SFTTrainer` subclass that
mixes ground-truth supervised fine-tuning with on-policy distillation
from a frozen teacher model. Each training step is one of two modes
selected stochastically per batch:

* **Supervised step** (probability ``1 - lmbda``) -- standard SFT loss
  on the dataset reference answers.
* **On-policy distillation step** (probability ``lmbda``) -- the
  student samples completions from its own policy, then the teacher
  scores those samples and the student is trained to match the
  teacher's distribution under a generalized JSD divergence with
  parameter ``beta`` (``0`` recovers forward KL, ``1`` recovers reverse
  KL, ``0.5`` recovers symmetric JSD).

The trainer keeps a frozen teacher state alongside the trainable
student state, builds a generation harness for on-policy sampling, and
delegates the actual loss/gradient computation to :func:`gkd_step` in
``_fn.py``.
"""

from __future__ import annotations

import random
import typing as tp

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from .._shared import copy_frozen_policy, drop_optimizer_state
from ..async_rollout_mixin import AsyncRolloutMixin
from ..model_loading import disable_state_dropout, reject_string_model_id
from ..supervised_fine_tuning_trainer import SFTTrainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import compile_trainer_step, resolve_straight_through_emulator
from ..utils import DataCollatorForCompletionOnlyLM
from ._fn import gkd_step
from .gkd_config import GKDConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]

logger = get_logger(__name__)

_STALE_SEQUENCE_KEYS = (
    "token_type_ids",
    "segment_ids",
    "assistant_masks",
    "loss_mask",
    "loss_token_weights",
    "loss_group_ids",
    "dataset_ids",
    "mask_info",
)


def _selector_matching_graphstate(reference_graphstate: tp.Any) -> tp.Any:
    """Return a SpectraX selector predicate matching ``reference_graphstate``'s exact leaves."""
    paths_fn = getattr(reference_graphstate, "paths", None)
    if not callable(paths_fn):
        return None
    selected = frozenset(paths_fn())
    return lambda variable, path: (getattr(variable, "kind", None), path) in selected


@Registry.register("trainer", "gkd")
class GKDTrainer(AsyncRolloutMixin, SFTTrainer):
    """Trainer for Generalized Knowledge Distillation (GKD).

    Implements the GKD objective from Agarwal et al. 2024: a
    generalized Jensen-Shannon divergence loss between the student
    and teacher next-token distributions, interpolated by ``beta``
    (``0`` -> forward KL, ``1`` -> reverse KL, ``0.5`` -> standard
    JSD). Optionally injects *on-policy* training data by sampling
    student completions with probability ``lmbda`` and replacing the
    supervised batch with those rollouts, mitigating the
    distribution-shift problem of offline distillation. With
    ``seq_kd=True`` the trainer also pre-generates the batch with
    the teacher (sequence-level KD) before any student-side
    processing.

    Inherits the SFT data plumbing (chat-template tokenisation,
    packing, ``response_template`` masking, ...) from
    :class:`SFTTrainer`.

    See :func:`generalized_jsd_loss` for the loss form and
    :func:`gkd_step` for the per-step pipeline.

    Attributes:
        arguments: :class:`GKDConfig` controlling JSD/teacher knobs and
            the inherited ``SFTConfig``/``TrainingArguments`` surface.
        teacher_state: Frozen teacher :class:`EasyDeLState`. Required
            for the loss; constructed from ``teacher_model`` in
            ``__init__``.
        lmbda: Cached on-policy sampling probability per batch.
        seq_kd: Cached sequence-level KD flag.
    """

    supports_sequence_packing: tp.ClassVar[bool] = False  # RL/online or paired-preference: warn-and-ignore packing

    arguments: GKDConfig

    def __init__(
        self,
        arguments: GKDConfig,
        processing_class: ProcessingClassType,
        model: EasyDeLBaseModule | EasyDeLState | None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        formatting_func: tp.Callable | None = None,
        data_collator: DataCollatorForCompletionOnlyLM | None = None,
    ):
        """Initialize the GKD trainer.

        Resolves the student and teacher into :class:`EasyDeLState`
        objects, configures the on-policy mixing ratio ``lmbda``, the
        sequence-KD flag, and forwards construction to the SFT trainer
        which provides completion-only-loss preprocessing.

        Args:
            arguments: GKD-specific training configuration.
            processing_class: Tokenizer/processor for SFT preprocessing.
            model: Student module or state.
            teacher_model: Frozen teacher module/state.
            train_dataset: SFT-style training dataset.
            eval_dataset: Optional evaluation dataset.
            formatting_func: Optional formatting callable forwarded to
                the SFT preprocessing pipeline.
            data_collator: Optional custom completion-only collator.

        Raises:
            TypeError: If ``arguments`` is not a :class:`GKDConfig`.
        """
        if not isinstance(arguments, GKDConfig):
            raise TypeError(f"`arguments` must be a `GKDConfig`, received {type(arguments)}.")

        if model is None:
            raise ValueError("`model` must be provided for GKDTrainer.")
        if isinstance(model, str):
            reject_string_model_id(model, role="student model")

        if getattr(arguments, "packing", False):
            logger.warning(
                "GKDTrainer does not support `packing=True` (its on-policy / prompt-extraction data path is "
                "incompatible); ignoring the flag."
            )
            arguments.packing = False
            if getattr(arguments, "eval_packing", None):
                arguments.eval_packing = False

        if isinstance(model, EasyDeLState):
            student_state = model
        else:
            student_state = model.to_state(trainable_selector=arguments.trainable_selector)

        self.lmbda = None if arguments.lmbda is None else float(arguments.lmbda)
        self.seq_kd = bool(arguments.seq_kd)
        self._on_policy_rng = random.Random(getattr(arguments, "seed", None))
        self._warned_missing_prompt = False

        teacher_model = self._resolve_teacher_model(
            teacher_model=teacher_model,
        )

        self_distillation = teacher_model is None
        if self_distillation:
            teacher_state = student_state
        elif isinstance(teacher_model, EasyDeLState):
            teacher_state = teacher_model
        else:
            teacher_model.eval()
            teacher_state = teacher_model.to_state()

        if arguments.disable_dropout:
            student_state = disable_state_dropout(student_state)
            teacher_state = student_state if self_distillation else disable_state_dropout(teacher_state)

        # Self-distillation borrows the student state until the base initializer
        # has finalized `self.model_state` (optimizer init, resharding, resume).
        # `configure_functions` only reads shardings/flops off the teacher, so no
        # copy is needed yet -- and copying here would leave a second teacher
        # resident for the whole run once the real copy is taken below.
        self.teacher_state = drop_optimizer_state(teacher_state)
        if not self_distillation and self.teacher_state is not None:
            self.teacher_state = self._share_esurge_cache_scope_if_compatible(
                self.teacher_state,
                primary_state=student_state,
            )

        super().__init__(
            arguments=arguments,
            processing_class=processing_class,
            model=student_state,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_func,
            data_collator=data_collator,
        )

        if self_distillation:
            # `self.model_state` now carries optimizer slots, so a whole-pytree
            # deep copy would duplicate those too; the teacher only needs params.
            self.teacher_state = copy_frozen_policy(self.model_state)
        elif self.teacher_state is not None:
            self.teacher_state = self._share_esurge_cache_scope_if_compatible(
                self.teacher_state,
                primary_state=self.model_state,
            )

        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            raise ValueError("GKDTrainer requires a tokenizer with a defined `pad_token_id`.")

    def _generation_overrides(self) -> dict[str, tp.Any]:
        """Sampling of GKD rollouts: ``temperature``, at most ``max_new_tokens``, no top-k/top-p, no penalties.

        Every key is set explicitly so the trainer-wide ``generation_*`` fallbacks (top-p 0.95, top-k 64) do
        not apply.
        """
        return {
            "max_new_tokens": int(self.arguments.max_new_tokens),
            "temperature": float(self.arguments.temperature),
            "do_sample": True,
            "top_k": 0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "num_return_sequences": 1,
        }

    @staticmethod
    def _resolve_teacher_model(
        *,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None,
    ) -> EasyDeLBaseModule | EasyDeLState | None:
        """Resolve an initialized GKD teacher."""
        if isinstance(teacher_model, str):
            reject_string_model_id(teacher_model, role="teacher model")
        return teacher_model

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Build the JIT-compiled GKD training/evaluation step functions.

        Resolves the optional QAT straight-through emulator, captures
        the JSD ``beta`` / ``temperature`` plus loss / partition spec
        arguments, and compiles :func:`gkd_step` once for training
        (with student state donation and the active MPMD pipeline
        schedule) and once for evaluation. The teacher state's
        sharding spec is threaded into the input shardings so the
        compiled step can run the teacher forward in place.

        Returns:
            ``TrainerConfigureFunctionOutput`` with the sharded
            training / evaluation step callables, the model mesh,
            and the streaming checkpoint manager.
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
            True,
            float(self.arguments.beta),
            float(self.arguments.temperature),
            straight_through_emulator,
        )

        static_argnums = (3, 4, 5, 6, 7, 8, 9, 10)
        self._runtime_trace("train.compile_wrapper.begin")
        sharded_training_step_function = compile_trainer_step(
            gkd_step,
            in_shardings=(self.state_shardings, empty_sharding, self.teacher_state.shardings),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
        )
        self._runtime_trace("train.compile_wrapper.end")
        sharded_training_step_function.static_argnums_ = static_argnums

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,
            float(self.arguments.beta),
            float(self.arguments.temperature),
            None,
        )

        self._runtime_trace("eval.compile_wrapper.begin")
        sharded_evaluation_step_function = compile_trainer_step(
            gkd_step,
            in_shardings=(self.eval_state_shardings, empty_sharding, self._eval_shared_fn_extra_args[0].shardings),
            out_shardings=empty_sharding,
            static_argnums=static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
        )
        self._runtime_trace("eval.compile_wrapper.end")
        sharded_evaluation_step_function.static_argnums_ = static_argnums

        self.sharded_training_step_function = sharded_training_step_function
        self.sharded_evaluation_step_function = sharded_evaluation_step_function

        teacher_flops = self.teacher_state.model.flops_per_token(include_loss=True, include_backward=False)
        self._extra_forward_flops_per_token = teacher_flops

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    @property
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any, ...]:
        """Forward the *live* teacher state alongside the shared training step.

        Resolved per call rather than captured once at compile time: in the
        self-distillation path the teacher is copied from the finalized student
        after ``configure_functions`` has already run, and a cached tuple would
        keep feeding the loss the pre-initialization state (the pre-resume
        weights, for a resumed run).
        """
        return (self.teacher_state,)

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any, ...]:
        """Forward live teacher buffers with matching evaluation static metadata."""
        graphdef = getattr(self, "_eval_teacher_graphdef", None)
        teacher = self.teacher_state
        if graphdef is not None:
            teacher = teacher.replace(graphdef=graphdef)
        return (teacher,)

    def _apply_preprocess_transforms(self) -> None:
        """Disable legacy ``packing`` before delegating to :class:`SFTTrainer`."""
        if getattr(self.arguments, "packing", False):
            logger.warning(
                "GKDTrainer does not support `packing=True` (its on-policy / prompt-extraction data path is "
                "incompatible); ignoring the flag."
            )
            self.arguments.packing = False
            if getattr(self.arguments, "eval_packing", None):
                self.arguments.eval_packing = False
        super()._apply_preprocess_transforms()

    def _configure_eval_config_override(self):
        """Apply evaluation config overrides to the teacher forward as well."""
        super()._configure_eval_config_override()
        self._eval_teacher_graphdef = None
        overrides = getattr(self.arguments, "eval_config_overrides", None)
        if self._eval_graphdef is not None and overrides and self.teacher_state is not None:
            self._eval_teacher_graphdef = self._build_config_override_graphdef(
                model=self.teacher_state.model,
                overrides=overrides,
                reference_graphstate=self.teacher_state.graphstate,
                trainable_selector=_selector_matching_graphstate(self.teacher_state.graphstate),
                label="eval_config_overrides (teacher)",
                eval_mode=True,
            )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Optionally generate on-policy samples before training.

        Args:
            state: Current model state.
            batch: Input batch.
            is_train: Whether this is a training step.

        Returns:
            Processed batch and metrics dictionary.
        """
        batch, info = super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)
        info = dict(info)
        if not is_train:
            return batch, info

        batch_copy = dict(batch)

        use_student = self._should_use_student_sampling()
        if use_student:
            generated, meta = self._apply_generation(state, batch_copy, source="student")
            if generated is not None:
                batch_copy = generated
                info.update(meta)
            elif self.seq_kd:
                generated, meta = self._apply_generation(self.teacher_state, batch_copy, source="teacher")
                if generated is not None:
                    batch_copy = generated
                    info.update(meta)
        elif self.seq_kd:
            generated, meta = self._apply_generation(self.teacher_state, batch_copy, source="teacher")
            if generated is not None:
                batch_copy = generated
                info.update(meta)

        return batch_copy, info

    def _should_use_student_sampling(self) -> bool:
        """Determine whether to use student on-policy sampling for this batch.

        Returns:
            True if student sampling should be performed.
        """
        if self.lmbda is None or self.lmbda <= 0:
            return False
        return self._on_policy_rng.random() <= self.lmbda

    def _apply_generation(
        self,
        generator_state: EasyDeLState,
        batch: dict[str, jax.Array],
        *,
        source: str,
    ) -> tuple[dict[str, jax.Array] | None, dict[str, float]]:
        """Generate completions using the specified model.

        The prompt rows go to eSurge as token ids through :meth:`rollout`, on ``generator_state``'s engine.

        Args:
            generator_state: Model state to use for generation.
            batch: Input batch containing prompts.
            source: Source identifier for logging ('teacher' or 'student').

        Returns:
            Generated batch and generation metrics.
        """
        prompts = self._extract_prompts(batch)
        if prompts is None:
            if source == "teacher":
                raise ValueError("Sequence-KD requires prompt fields, completion_mask, or prompt-masked labels.")
            if not self._warned_missing_prompt:
                logger.warning(
                    "Unable to derive prompts from batch; on-policy sampling is disabled until prompts are available."
                )
                self._warned_missing_prompt = True
            return None, {}
        prompt_ids, prompt_mask, _prompt_lengths = prompts
        prompt_seq_len = int(prompt_ids.shape[1])

        try:
            results = self.rollout(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                state=generator_state,
                apply_chat_template=False,
                config_overrides=self._generation_overrides(),
            )
            sequences = jax.block_until_ready(results.sequences)
        except Exception as exc:  # pragma: no cover - generation failures are rare
            if source == "teacher":
                raise RuntimeError("Sequence-KD teacher generation failed.") from exc
            logger.warning("Failed to generate %s continuations: %s", source, exc)
            return None, {}

        new_batch = self._build_batch_from_sequences(
            batch,
            sequences,
            prompt_seq_len=prompt_seq_len,
            prompt_mask=getattr(results, "prompt_mask", None),
            completion_mask=getattr(results, "completion_mask", None),
        )
        completion_mask_np = np.asarray(new_batch["completion_mask"], dtype=np.int32)
        attention_mask_np = np.asarray(new_batch["attention_mask"], dtype=np.int32)
        completion_lengths = completion_mask_np.sum(axis=-1)
        kept_prompt_lengths = np.maximum(attention_mask_np - completion_mask_np, 0).sum(axis=-1)
        info = {
            f"gkd_completion_length_{source}": float(np.mean(completion_lengths)),
            f"gkd_prompt_length_{source}": float(np.mean(kept_prompt_lengths)),
        }
        return new_batch, info

    def _extract_prompts(self, batch: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array, np.ndarray] | None:
        """Extract prompt token IDs and masks from batch.

        Args:
            batch: Input batch.

        Returns:
            Tuple of (prompt_ids, prompt_mask, prompt_lengths) or None if prompts cannot be extracted.
        """
        if "prompt_input_ids" in batch and "prompt_attention_mask" in batch:
            prompt_ids = jnp.asarray(batch["prompt_input_ids"])
            prompt_mask = jnp.asarray(batch["prompt_attention_mask"]).astype(jnp.int32)
            prompt_lengths = np.asarray(np.sum(np.asarray(prompt_mask), axis=-1), dtype=np.int32)
            return prompt_ids, prompt_mask, prompt_lengths

        input_ids = np.asarray(batch["input_ids"])
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask_np = (input_ids != self.pad_token_id).astype(np.int32)
        else:
            attention_mask_np = np.asarray(attention_mask)
        valid_tokens = attention_mask_np.astype(bool)

        completion_mask = batch.get("completion_mask")
        assistant_masks = batch.get("assistant_masks")
        labels = batch.get("labels")
        if completion_mask is not None:
            comp_bool = valid_tokens & np.asarray(completion_mask).astype(bool)
        elif assistant_masks is not None:
            comp_bool = valid_tokens & np.asarray(assistant_masks).astype(bool)
        elif labels is not None:
            comp_bool = valid_tokens & (np.asarray(labels) != -100)
        else:
            return None

        batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        row_tokens: list[np.ndarray] = []
        raw_lengths: list[int] = []
        for idx in range(batch_size):
            comp_indices = np.flatnonzero(comp_bool[idx])
            if comp_indices.size > 0:
                first_comp = int(comp_indices[0])
                tokens = input_ids[idx, :first_comp][valid_tokens[idx, :first_comp]]
            else:
                tokens = input_ids[idx][valid_tokens[idx]]
            row_tokens.append(tokens)
            raw_lengths.append(int(tokens.size))

        if not raw_lengths or max(raw_lengths) <= 0:
            return None

        max_length = getattr(self.arguments, "max_length", None)
        max_new_tokens = int(getattr(self.arguments, "max_new_tokens", 0) or 0)
        if max_length is not None and int(max_length) > 0:
            if int(max_length) > max_new_tokens:
                target_prompt_width = int(max_length) - max_new_tokens
            else:
                target_prompt_width = int(max_length)
        else:
            target_prompt_width = seq_len

        prompt_ids_np = np.full((batch_size, target_prompt_width), self.pad_token_id, dtype=input_ids.dtype)
        prompt_mask_np = np.zeros((batch_size, target_prompt_width), dtype=np.int32)
        prompt_lengths = np.zeros((batch_size,), dtype=np.int32)
        for idx, tokens in enumerate(row_tokens):
            if tokens.size == 0:
                continue
            length = int(min(tokens.size, target_prompt_width))
            prompt_ids_np[idx, -length:] = tokens[-length:]
            prompt_mask_np[idx, -length:] = 1
            prompt_lengths[idx] = length

        return jnp.asarray(prompt_ids_np), jnp.asarray(prompt_mask_np), prompt_lengths

    def _build_batch_from_sequences(
        self,
        original_batch: dict[str, jax.Array],
        sequences: jax.Array,
        *,
        prompt_seq_len: int,
        prompt_mask: jax.Array | np.ndarray | None = None,
        completion_mask: jax.Array | np.ndarray | None = None,
    ) -> dict[str, jax.Array]:
        """Construct training batch from generated sequences.

        Args:
            original_batch: Original input batch.
            sequences: Generated token sequences.
            prompt_seq_len: Prompt token array length used during generation (includes padding).
            prompt_mask: Optional prompt attention mask from ``rollout()`` preserving EOS when
                ``pad_token_id == eos_token_id``.
            completion_mask: Optional completion attention mask from ``rollout()`` preserving EOS
                and masking right-padded tails when ``pad_token_id == eos_token_id``.

        Returns:
            New batch with generated completions and appropriate masks/labels.
        """
        seq_np = np.asarray(sequences)
        seq_len = int(seq_np.shape[1])
        max_len = self.arguments.max_length
        prompt_cutoff = int(max(prompt_seq_len, 0))

        pm_np = None if prompt_mask is None else (np.asarray(prompt_mask) != 0).astype(np.int32)
        cm_np = None if completion_mask is None else (np.asarray(completion_mask) != 0).astype(np.int32)

        if pm_np is not None and cm_np is not None and pm_np.shape[1] + cm_np.shape[1] == seq_len:
            attention_mask = np.concatenate([pm_np, cm_np], axis=1)
            comp_mask = np.concatenate([np.zeros_like(pm_np), cm_np], axis=1)
        else:
            attention_mask = (seq_np != self.pad_token_id).astype(np.int32)
            if pm_np is not None and pm_np.shape[1] <= seq_len:
                attention_mask[:, : pm_np.shape[1]] = pm_np
            if cm_np is not None and prompt_cutoff + cm_np.shape[1] <= seq_len:
                attention_mask[:, prompt_cutoff : prompt_cutoff + cm_np.shape[1]] = cm_np
            comp_mask = np.zeros_like(attention_mask)
            if cm_np is not None and prompt_cutoff + cm_np.shape[1] <= seq_len:
                comp_mask[:, prompt_cutoff : prompt_cutoff + cm_np.shape[1]] = cm_np
            elif prompt_cutoff < seq_len:
                comp_mask[:, prompt_cutoff:] = 1
            comp_mask *= attention_mask

        if max_len is not None and seq_len > max_len:
            start = seq_len - max_len
            seq_np = seq_np[:, start:]
            attention_mask = attention_mask[:, start:]
            comp_mask = comp_mask[:, start:]
            prompt_cutoff = max(prompt_cutoff - start, 0)
            seq_len = int(max_len)

        labels_dtype = np.asarray(original_batch.get("labels", seq_np)).dtype
        # Labels are masked in place; generated tokens may share a read-only
        # device buffer and must never be overwritten by the ignore index.
        labels = seq_np.astype(labels_dtype, copy=True)
        labels[(attention_mask == 0) | (comp_mask == 0)] = -100

        new_batch = dict(original_batch)
        for stale_key in _STALE_SEQUENCE_KEYS:
            new_batch.pop(stale_key, None)
        new_batch["input_ids"] = jnp.asarray(seq_np.astype(np.asarray(original_batch["input_ids"]).dtype, copy=False))
        attn_dtype = np.asarray(original_batch.get("attention_mask", attention_mask)).dtype
        new_batch["attention_mask"] = jnp.asarray(attention_mask.astype(attn_dtype, copy=False))
        new_batch["completion_mask"] = jnp.asarray(comp_mask.astype(attn_dtype, copy=False))
        new_batch["labels"] = jnp.asarray(labels)
        if "position_ids" in original_batch:
            pos = np.cumsum(attention_mask, axis=-1, dtype=np.int32) - 1
            position_ids = np.where(attention_mask > 0, pos, 0).astype(np.int32)
            new_batch["position_ids"] = jnp.asarray(position_ids)
        return new_batch
