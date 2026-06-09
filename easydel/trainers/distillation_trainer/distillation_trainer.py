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
"""Offline knowledge-distillation trainer.

Trains a "student" model to match a frozen "teacher" via a combination of
temperature-scaled KL divergence on token logits and an optional supervised
cross-entropy term mixed by ``alpha``. The configured loss may additionally
include:

- Masked MSE on per-layer hidden states (``hidden_state_loss_weight`` /
  ``hidden_state_layers`` in :class:`DistillationConfig`).
- Masked MSE on per-layer attention probabilities, optionally L1-normalised
  (``attention_loss_weight`` / ``attention_layers`` / ``attention_normalize``).
- A memory-saving chunked KL path that streams logits in slices of
  ``logits_chunk_size`` tokens, with optional ``jax.checkpoint`` rematerialisation
  controlled by ``checkpoint_kl_loss``.
- Quantization-aware straight-through emulation for the student
  (``quantization_mode`` / ``quantization_bits`` / ``tensor_straight_through`` /
  ``straight_through_emulator``).

Both ejit-style ``spx.jit`` and MPMD-scheduled compilation paths are supported
based on whether ``arguments.mpmd_scheduler`` is set.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import spectrax as spx
from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..model_loading import disable_state_dropout, reject_string_model_id
from ..prompt_transforms import SFTPreprocessTransform
from ..trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import compile_trainer_step, resolve_straight_through_emulator
from ..utils import DataCollatorForCompletionOnlyLM
from ._fn import distillation_step
from .distillation_config import DistillationConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]

logger = get_logger(__name__)


@Registry.register("trainer", "distillation")
class DistillationTrainer(Trainer):
    """Knowledge distillation trainer for model compression.

    Transfers knowledge from a frozen teacher to a trainable student by minimising
    a temperature-scaled KL term on logits, optionally mixed with a hard-label
    cross-entropy term and optional hidden-state / attention auxiliary losses.

    Key features:
        - Temperature-scaled KL on full or chunked logits
        - α / (1-α) mix between KL and supervised CE
        - Optional masked-MSE losses on hidden states and attention probabilities
        - Optional QAT straight-through emulation on the student forward
        - Two compile backends: ``spx.jit`` (default) and MPMD-scheduled
          (``arguments.mpmd_scheduler is not None``)

    Loss (with hard labels, no auxiliaries):
        ``Loss = α * (E_t[-log p_s] - E_t[-log p_t]) * T² + (1 - α) * CE(student, labels)``

    With auxiliary terms:
        ``Loss += hidden_state_loss_weight * Σ_l MSE(h_s[l], h_t[l]; mask)``
        ``Loss += attention_loss_weight   * Σ_l MSE(a_s[l], a_t[l]; mask)``

    Attributes:
        teacher_state: Frozen :class:`EasyDeLState` of the teacher model.
        arguments: :class:`DistillationConfig` instance carrying all knobs.

    Example:
        >>> config = DistillationConfig(
        ...     temperature=3.0,
        ...     alpha=0.7,
        ...     learning_rate=2e-5,
        ... )
        >>> trainer = DistillationTrainer(
        ...     arguments=config,
        ...     student_model=student,
        ...     teacher_model=teacher,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer,
        ... )
        >>> trainer.train()
    """

    teacher_state: EasyDeLState
    arguments: DistillationConfig  # type hinting

    def __init__(
        self,
        arguments: DistillationConfig,
        processing_class: ProcessingClassType,
        student_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        data_collator: DataCollatorForCompletionOnlyLM | None = None,
    ):
        """Initialize the offline distillation trainer.

        Workflow:
            1. Resolve the processor's ``pad_token`` to ``eos_token`` when unset
               (mutates the caller's tokenizer in place).
            2. Convert ``student_model`` to :class:`EasyDeLState` via
               ``to_state(trainable_selector=arguments.trainable_selector)`` so
               only the selected parameter collection participates in optimizer
               updates.
            3. Put ``teacher_model`` into eval mode and convert it to a state
               (no ``trainable_selector`` — the teacher is frozen wholesale).
            4. Delegate to :class:`Trainer.__init__` for dataloaders, optimizer,
               scheduler, and the shared training loop.

        Args:
            arguments: Distillation-specific training configuration.
            processing_class: Tokenizer/processor used for SFT-style preprocessing.
                Its ``pad_token`` may be mutated to match ``eos_token``.
            student_model: Trainable student module or pre-built state. Required.
            teacher_model: Frozen teacher module or pre-built state. Required.
                A module input is forced into eval mode before state export; a
                pre-built state is taken as-is.
            train_dataset: Training dataset of completion examples.
            eval_dataset: Optional evaluation dataset (single or named-split dict).
            data_collator: Optional custom collator; otherwise the default
                completion-only collator is used.

        Raises:
            TypeError: If ``arguments`` is not a :class:`DistillationConfig`.
            AttributeError: If ``student_model`` or ``teacher_model`` is ``None``
                (they default to ``None`` for legacy signature compatibility but
                must be supplied — the conversion to state then fails).
        """
        tokenizer = processing_class
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token
        if not isinstance(arguments, DistillationConfig):
            raise TypeError("passed argument must be a `DistillationConfig`.")

        self.arguments = arguments
        student_model = self._resolve_student_model(student_model)
        teacher_model = self._resolve_teacher_model(
            teacher_model=teacher_model,
            teacher_model_revision=arguments.teacher_model_revision,
        )

        if not isinstance(student_model, EasyDeLState):
            student_model = student_model.to_state(trainable_selector=arguments.trainable_selector)
        if not isinstance(teacher_model, EasyDeLState):
            teacher_model.eval()
            teacher_model = teacher_model.to_state()
        if arguments.disable_dropout:
            student_model = disable_state_dropout(student_model)
            teacher_model = disable_state_dropout(teacher_model)

        self.teacher_state = teacher_model

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=student_model,
            data_collator=data_collator,
            processing_class=processing_class,
        )

    @staticmethod
    def _resolve_student_model(
        student_model: EasyDeLBaseModule | EasyDeLState | None,
    ) -> EasyDeLBaseModule | EasyDeLState:
        """Reject accidental string student ids."""
        if student_model is None:
            raise ValueError("`student_model` must be provided for EasyDeL distillation.")
        if isinstance(student_model, str):
            reject_string_model_id(student_model, role="student model")
        return student_model

    @staticmethod
    def _resolve_teacher_model(
        *,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None,
        teacher_model_revision: str | None,
    ) -> EasyDeLBaseModule | EasyDeLState:
        """Resolve an initialized teacher model/state."""
        if teacher_model_revision is not None:
            raise ValueError("`teacher_model_revision` is only metadata for externally loaded teachers.")
        if isinstance(teacher_model, str):
            reject_string_model_id(teacher_model, role="teacher model")
        if teacher_model is None:
            raise ValueError("`teacher_model` must be provided for distillation.")
        return teacher_model

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Build the JIT-compiled distillation training/evaluation step functions.

        Steps:
            1. Resolve the optional QAT straight-through emulator and capture all
               loss knobs (temperature, alpha, hidden-state / attention weights and
               layer indices, logits chunk size, KL-checkpoint flag).
            2. Pack the captured knobs into ``_train_shared_fn_static_args`` and
               ``_eval_shared_fn_static_args`` (identical except ``is_train`` and
               the straight-through emulator slot).
            3. Compile :func:`distillation_step` twice — once for training (with
               student state donation via ``donate_argnums=(0,)``) and once for
               evaluation (no donation, ``out_shardings=empty_sharding``).
            4. Pick the compile backend based on ``arguments.mpmd_scheduler``:
               ``spx.jit`` directly when ``None``; otherwise
               :func:`compile_trainer_step` with the MPMD schedule and mesh.
            5. Record the teacher's per-token forward FLOPs as the extra forward
               compute (no backward — the teacher is frozen).

        The teacher state's sharding spec is threaded through ``in_shardings`` so
        the compiled step receives ``teacher_state`` as a regular input and runs
        its forward (with ``jax.lax.stop_gradient`` applied inside ``loss_fn``)
        without a separate auxiliary program.

        Emits debug-level runtime traces for the compile-wrapper begin/end of both
        train and eval paths.

        Returns:
            :class:`TrainerConfigureFunctionOutput` with the sharded training and
            evaluation step callables, the model mesh, and the streaming
            checkpoint manager.
        """
        self._runtime_trace(
            "configure_functions.distillation",
            mpmd_scheduler=self.arguments.mpmd_scheduler,
            logits_chunk_size=self.arguments.logits_chunk_size,
            gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
        )
        mesh = self.model.mesh

        # Master MTP switch. A checkpoint that ships an MTP head keeps training it via the student's
        # self-supervised MTP cross-entropy, which the model folds into ``outputs.aux_loss`` (gated by
        # ``mtp_loss_coef``) and the step adds unconditionally -- independent of the trainer's MTP-KD term.
        # So when MTP distillation is OFF, freeze the head (``mtp_loss_coef=0``) so ``mtp_distillation=False``
        # means *no* MTP loss at all. (``mtp_distillation=True`` leaves ``mtp_loss_coef`` untouched; pair it
        # with ``mtp_kd_weight=0`` if you want the self-supervised MTP CE WITHOUT the soft-KD term.) The MoE
        # router aux loss is added separately and is unaffected. ``text_config`` covers the VLM student,
        # whose MTP head lives on the inner text model.
        if not bool(self.arguments.mtp_distillation):
            _froze_mtp = False
            for _cfg in (self.model.config, getattr(self.model.config, "text_config", None)):
                if _cfg is not None and float(getattr(_cfg, "mtp_loss_coef", 0.0) or 0.0) > 0.0:
                    _cfg.mtp_loss_coef = 0.0
                    _froze_mtp = True
            if _froze_mtp:
                logger.debug(
                    "`mtp_distillation=False` -> froze the student MTP head (mtp_loss_coef=0); "
                    "no MTP loss is trained or distilled."
                )

        # Zero-config large-vocab fit on a tensor-parallel mesh: enable the memory-safe LM-head paths
        # automatically (explicit user values are always preserved). On TP>1 the row-parallel LM head
        # otherwise materializes full ``[B, S, V]`` logits -- the distillation-KL and MTP-aux OOM:
        #   * ``logits_chunk_size`` -> single full-sequence pass through the COLUMN-PARALLEL projection
        #     so the distillation KL never all-reduces the full vocabulary;
        #   * ``lmhead_chunksize``  -> chunk + ``jax.checkpoint`` the LM-head / MTP-aux projection so the
        #     vocab-sized logits are recomputed in the backward instead of held as residuals.
        try:
            _tp_size = int(mesh.shape["tp"]) if (mesh is not None and "tp" in getattr(mesh, "axis_names", ())) else 1
        except Exception:
            _tp_size = 1
        if _tp_size > 1:
            _adv_vocab = (
                self.arguments.beta is not None
                or int(self.arguments.loss_top_k or 0) > 0
                or bool(self.arguments.loss_add_tail)
            )
            _seq_len = int(getattr(self.arguments, "max_length", None) or 0)
            if (
                self.arguments.logits_chunk_size is None
                and _seq_len > 0
                and not bool(self.arguments.mtp_distillation)
                and not _adv_vocab
            ):
                self.arguments.logits_chunk_size = _seq_len
                logger.info(
                    "Auto-enabled column-parallel distillation KL (logits_chunk_size=%d) on a TP=%d mesh.",
                    _seq_len,
                    _tp_size,
                )
            if getattr(self.model.config, "lmhead_chunksize", None) is None:
                _lh = min(_seq_len, 2048) if _seq_len > 0 else 2048
                try:
                    self.model.config.lmhead_chunksize = _lh
                    _t_model = getattr(self.teacher_state, "model", None)
                    if _t_model is not None and getattr(_t_model.config, "lmhead_chunksize", None) is None:
                        _t_model.config.lmhead_chunksize = _lh
                    logger.debug(
                        "Auto-enabled chunked+checkpointed LM-head/MTP projection (lmhead_chunksize=%d) on a TP=%d mesh.",
                        _lh,
                        _tp_size,
                    )
                except Exception as exc:
                    logger.warning("Could not auto-set lmhead_chunksize: %s", exc)

        empty_sharding = replicated_named_sharding(mesh)

        hidden_layers = self.arguments.hidden_state_layers
        attention_layers = self.arguments.attention_layers
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
            True,  # is_train
            self.arguments.temperature,
            self.arguments.alpha,
            0.0 if self.arguments.hidden_state_loss_weight is None else float(self.arguments.hidden_state_loss_weight),
            hidden_layers,
            self.arguments.hidden_state_loss,
            0.0 if self.arguments.attention_loss_weight is None else float(self.arguments.attention_loss_weight),
            attention_layers,
            bool(self.arguments.attention_normalize),
            straight_through_emulator,
            self.arguments.logits_chunk_size,
            bool(self.arguments.checkpoint_kl_loss),
            self.arguments.beta,
            int(self.arguments.loss_top_k),
            bool(self.arguments.loss_add_tail),
            bool(self.arguments.mtp_distillation),
            float(self.arguments.mtp_kd_weight),
            int(self.arguments.mtp_draft_tokens),
        )

        static_argnums = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
        self._runtime_trace("train.compile_wrapper.begin")
        if self.arguments.mpmd_scheduler is None:
            sharded_training_step_function = spx.jit(
                distillation_step,
                in_shardings=(self.state_shardings, empty_sharding, self.teacher_state.shardings),
                out_shardings=(self.state_shardings, empty_sharding),
                donate_argnums=(0,),
                static_argnums=static_argnums,
            )
        else:
            sharded_training_step_function = compile_trainer_step(
                distillation_step,
                in_shardings=(self.state_shardings, empty_sharding, self.teacher_state.shardings),
                out_shardings=(self.state_shardings, empty_sharding),
                donate_argnums=(0,),
                static_argnums=static_argnums,
                mesh=self.model.mesh,
                schedule=self.arguments.mpmd_scheduler,
            )
        self._runtime_trace("train.compile_wrapper.end")

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
            self.arguments.temperature,
            self.arguments.alpha,
            0.0 if self.arguments.hidden_state_loss_weight is None else float(self.arguments.hidden_state_loss_weight),
            hidden_layers,
            self.arguments.hidden_state_loss,
            0.0 if self.arguments.attention_loss_weight is None else float(self.arguments.attention_loss_weight),
            attention_layers,
            bool(self.arguments.attention_normalize),
            None,
            self.arguments.logits_chunk_size,
            bool(self.arguments.checkpoint_kl_loss),
            self.arguments.beta,
            int(self.arguments.loss_top_k),
            bool(self.arguments.loss_add_tail),
            bool(self.arguments.mtp_distillation),
            float(self.arguments.mtp_kd_weight),
            int(self.arguments.mtp_draft_tokens),
        )

        self._runtime_trace("eval.compile_wrapper.begin")
        if self.arguments.mpmd_scheduler is None:
            sharded_evaluation_step_function = spx.jit(
                distillation_step,
                in_shardings=(self.state_shardings, empty_sharding, self.teacher_state.shardings),
                out_shardings=empty_sharding,
                static_argnums=static_argnums,
            )
        else:
            sharded_evaluation_step_function = compile_trainer_step(
                distillation_step,
                in_shardings=(self.state_shardings, empty_sharding, self.teacher_state.shardings),
                out_shardings=empty_sharding,
                static_argnums=static_argnums,
                mesh=self.model.mesh,
                schedule=self.arguments.mpmd_scheduler,
            )
        self._runtime_trace("eval.compile_wrapper.end")

        sharded_training_step_function.static_argnums_ = static_argnums
        sharded_evaluation_step_function.static_argnums_ = static_argnums

        # Teacher is frozen: it contributes a forward pass only, no backward.
        teacher_forward_flops = self.teacher_state.model.flops_per_token(
            include_loss=False,
            include_backward=False,
        )
        self._extra_forward_flops_per_token = teacher_forward_flops
        self._extra_backward_flops_per_token = 0.0

        self.arguments.ensure_checkpoint_path()
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=self.arguments.get_streaming_checkpointer(),
        )

    def _get_preprocess_transform(self) -> SFTPreprocessTransform | None:
        """Build the SFT-style tokenisation transform when the dataset is raw text.

        Uses ``dataset_text_field`` to pick the input column and the
        legacy ``completion_only_loss`` flag (with
        ``assistant_only_loss`` as the canonical fallback) to decide
        whether the prompt portion should be masked out.

        Returns:
            An :class:`SFTPreprocessTransform` wired with the
            tokenizer and configured max length, or ``None`` when the
            source is already pretokenised.
        """
        if self._is_pretokenized():
            return None
        text_field = getattr(self.arguments, "dataset_text_field", None) or "text"
        mask_prompt = bool(getattr(self.arguments, "assistant_only_loss", False))
        completion_only_loss = getattr(self.arguments, "completion_only_loss", None)
        if completion_only_loss is not None:
            mask_prompt = bool(completion_only_loss)
        return SFTPreprocessTransform(
            tokenizer=self.processing_class,
            max_length=self.arguments.max_length,
            text_field=text_field,
            mask_prompt=mask_prompt,
            padding=False
            if getattr(self.arguments, "sequence_packing", False) or self._user_data_collator
            else "max_length",
            truncation_mode=self.arguments.truncation_mode,
        )

    def _is_pretokenized(self) -> bool:
        """Detect whether the training source already yields tokenised samples.

        Returns:
            ``True`` when the first sample of the first shard exposes
            an ``input_ids`` column; ``False`` when the source is
            absent or the shard is empty.
        """
        if self._train_source is None:
            return False
        if self._source_requires_row_preprocessing(self._train_source):
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "input_ids" in sample
        except (StopIteration, IndexError):
            return False

    @staticmethod
    def _source_requires_row_preprocessing(source: tp.Any) -> bool:
        """Return whether a source wrapper can expose heterogeneous row schemas."""
        current = source
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if type(current).__name__ in {"MixedShardedSource", "ShuffledShardedSource"}:
                return True
            current = getattr(current, "_source", None)
        return False

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, tp.Any],
        is_train: bool,
    ) -> tuple[dict[str, tp.Any], dict[str, float | int | str]]:
        """Normalize completion masks and labels for distillation batches.

        Operates host-side after the parent's preprocessing. Handles three cases
        so downstream JIT receives a consistent shape regardless of whether the
        upstream collator produced ``assistant_masks``, ``completion_mask``,
        and/or ``labels``:

        1. ``assistant_masks`` is rebranded as ``completion_mask`` and dropped
           from the batch (the student forward must not receive it as a kwarg).
        2. If ``completion_mask`` is present, it is anded with ``attention_mask``
           and re-cast to the attention mask's dtype. When ``labels`` are absent,
           they are synthesised from ``input_ids`` with ``-100`` written at
           positions where ``completion_mask == 0`` or ``attention_mask == 0``.
        3. If ``labels`` are present but ``completion_mask`` isn't, a derived
           completion mask is built from ``labels != -100`` (anded with
           ``attention_mask`` when available).

        All work is on NumPy host arrays — no device synchronisation is forced
        on inputs that are already CPU-resident.

        Args:
            state: Current student state (unused here; threaded by the parent
                signature).
            batch: Host-side batch dict mutated in place by this method.
            is_train: Whether this is a training step; forwarded to the parent.

        Returns:
            ``(batch, infos)`` where ``infos`` is the parent's auxiliary
            information dict.
        """
        batch, infos = super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)

        if "assistant_masks" in batch:
            if "completion_mask" not in batch:
                batch["completion_mask"] = batch["assistant_masks"]
            # Keep assistant mask only as training-time supervision metadata;
            # model forwards must not receive this key.
            batch.pop("assistant_masks", None)

        attention_mask = batch.get("attention_mask")
        completion_mask = batch.get("completion_mask")

        if completion_mask is not None:
            completion_mask_np = np.asarray(completion_mask)
            if attention_mask is not None:
                completion_mask_np = completion_mask_np * np.asarray(attention_mask)
            completion_dtype = (
                np.asarray(attention_mask).dtype if attention_mask is not None else completion_mask_np.dtype
            )
            batch["completion_mask"] = completion_mask_np.astype(completion_dtype, copy=False)

            if "labels" not in batch and "input_ids" in batch:
                labels = np.asarray(batch["input_ids"]).astype(np.int32, copy=True)
                labels[completion_mask_np == 0] = -100
                if attention_mask is not None:
                    labels[np.asarray(attention_mask) == 0] = -100
                batch["labels"] = labels

        if "labels" in batch and "completion_mask" not in batch:
            labels_np = np.asarray(batch["labels"])
            if (labels_np == -100).any():
                completion_mask_np = (labels_np != -100).astype(np.int32)
                if attention_mask is not None:
                    completion_mask_np = completion_mask_np * np.asarray(attention_mask)
                batch["completion_mask"] = completion_mask_np

        return batch, infos

    @property
    def _train_shared_fn_extra_args(self) -> tuple[EasyDeLState]:
        """Extra positional args appended to every compiled training step call.

        Returns the single teacher :class:`EasyDeLState` that
        :func:`distillation_step` consumes as its third positional argument.
        Matches main-branch behaviour: the teacher is passed in directly rather
        than precomputed outside the compiled step.
        """
        return (self.teacher_state,)

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[EasyDeLState]:
        """Extra positional args appended to every compiled evaluation step call.

        Same as :attr:`_train_shared_fn_extra_args` — the evaluation step uses
        the same compiled function with ``is_train=False`` in its static args.
        """
        return (self.teacher_state,)
