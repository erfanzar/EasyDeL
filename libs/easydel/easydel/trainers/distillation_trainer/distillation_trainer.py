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

from easydel.data.transforms.mixture import DATASET_ID_FIELD, DATASET_ID_TOKEN_FIELD
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from .._shared import drop_optimizer_state
from ..assistant_only_mask import AssistantOnlyMaskMixin
from ..execution_accumulation import build_execution_accumulation
from ..model_loading import disable_state_dropout, reject_string_model_id
from ..prompt_transforms import SFTPreprocessTransform
from ..trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import compile_trainer_step, resolve_straight_through_emulator
from ..utils import DataCollatorForCompletionOnlyLM
from ._fn import distillation_step
from .distillation_config import DistillationConfig
from .loss_groups import (
    LOSS_GROUP_IDS_KEY,
    LOSS_TOKEN_WEIGHTS_KEY,
    loss_group_ids_from_dataset_ids,
    loss_group_token_weights,
    loss_supervision_mask,
)

if tp.TYPE_CHECKING:
    from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]

    from ..buckets import BucketRule, TrainingBucket

logger = get_logger(__name__)


@Registry.register("trainer", "distillation")
class DistillationTrainer(AssistantOnlyMaskMixin, Trainer):
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
    # Distillation only folds `assistant_masks` into a MISSING completion_mask (no all-zero fallback).
    _assistant_mask_fallback_on_empty: tp.ClassVar[bool] = False

    def __init__(
        self,
        arguments: DistillationConfig,
        processing_class: ProcessingClassType,
        student_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        data_collator: DataCollatorForCompletionOnlyLM | None = None,
        buckets: list[TrainingBucket] | None = None,
        bucket_rule: BucketRule | None = None,
        bucket_datasets: list | None = None,
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
            buckets: Optional list of :class:`~easydel.trainers.buckets.TrainingBucket`
                configs; forwarded to :class:`BaseTrainer` (one compiled distillation
                step + dataloader per bucket). Falls back to ``arguments.buckets``.
            bucket_rule: :class:`~easydel.trainers.buckets.BucketRule` selecting the
                active bucket per step; required when ``buckets`` is set.
            bucket_datasets: Optional per-bucket datasets/sources, one per bucket.

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
        teacher_cached = bool(getattr(arguments, "teacher_from_cache", False))
        if teacher_cached and teacher_model is not None:
            raise ValueError(
                "`teacher_from_cache=True` and a `teacher_model` were both given: the cached mode never runs the "
                "teacher. Pass `teacher_model=None`, or disable `teacher_from_cache`."
            )
        if not teacher_cached:
            teacher_model = self._resolve_teacher_model(
                teacher_model=teacher_model,
                teacher_model_revision=arguments.teacher_model_revision,
            )

        if not isinstance(student_model, EasyDeLState):
            student_model = student_model.to_state(trainable_selector=arguments.trainable_selector)
        if teacher_model is not None and not isinstance(teacher_model, EasyDeLState):
            teacher_model.eval()
            teacher_model = teacher_model.to_state()
        if arguments.disable_dropout:
            student_model = disable_state_dropout(student_model)
            if teacher_model is not None:
                teacher_model = disable_state_dropout(teacher_model)

        self.teacher_state = None if teacher_model is None else drop_optimizer_state(teacher_model)
        self._teacher_cache_reader = None
        if teacher_cached:
            from easydel.data.teacher_cache import TeacherCacheReader

            expect: dict[str, tp.Any] = {"segment_align": int(arguments.teacher_cache_segment_align)}
            if arguments.teacher_cache_teacher_uri:
                expect["teacher_uri"] = arguments.teacher_cache_teacher_uri
            if arguments.teacher_cache_tokenizer:
                expect["tokenizer"] = arguments.teacher_cache_tokenizer
            self._teacher_cache_reader = TeacherCacheReader(
                arguments.teacher_cache_path, expect=expect, topk=int(arguments.teacher_cache_topk or 0)
            )
            logger.info(
                "offline teacher cache: %d records, cache k=%d, training k=%d.",
                len(self._teacher_cache_reader),
                self._teacher_cache_reader.k_cache,
                self._teacher_cache_reader.topk,
            )

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=student_model,
            data_collator=data_collator,
            processing_class=processing_class,
            buckets=buckets,
            bucket_rule=bucket_rule,
            bucket_datasets=bucket_datasets,
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
                or self.arguments.cakld_gamma is not None
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
            self.arguments.cakld_gamma,
            int(self.arguments.loss_top_k),
            bool(self.arguments.loss_add_tail),
            bool(self.arguments.mtp_distillation),
            float(self.arguments.mtp_kd_weight),
            int(self.arguments.mtp_draft_tokens),
            bool(self.arguments.teacher_matched_compilation),
            *self._teacher_cache_static_args(),
            self._num_loss_groups(),
        )

        static_argnums = tuple(range(3, 3 + len(self._train_shared_fn_static_args)))
        # Bucketed training compiles one distillation step per bucket via the
        # BaseTrainer hooks below; they need this flavor's static-arg layout.
        self._bucket_static_argnums = static_argnums
        # xla_compiler_options must reach the direct spx.jit branches too; the
        # compile_trainer_step branches get it from arguments=self.arguments.
        _step_jit_options: dict = {}
        if getattr(self.arguments, "xla_compiler_options", None):
            _step_jit_options["compiler_options"] = dict(self.arguments.xla_compiler_options)
        self._runtime_trace("train.compile_wrapper.begin")
        if self.arguments.mpmd_scheduler is None:
            sharded_training_step_function = spx.jit(
                distillation_step,
                in_shardings=(self.state_shardings, empty_sharding, self._teacher_in_shardings()),
                out_shardings=(self.state_shardings, empty_sharding),
                donate_argnums=(0,),
                static_argnums=static_argnums,
                **_step_jit_options,
            )
        else:
            sharded_training_step_function = compile_trainer_step(
                distillation_step,
                in_shardings=(self.state_shardings, empty_sharding, self._teacher_in_shardings()),
                out_shardings=(self.state_shardings, empty_sharding),
                donate_argnums=(0,),
                static_argnums=static_argnums,
                mesh=self.model.mesh,
                schedule=self.arguments.mpmd_scheduler,
                arguments=self.arguments,
            )
        self._runtime_trace("train.compile_wrapper.end")
        self._execution_accumulation = build_execution_accumulation(
            self,
            step_fn=distillation_step,
            in_shardings=(self.state_shardings, empty_sharding, self._teacher_in_shardings()),
            static_argnums=static_argnums,
            empty_sharding=empty_sharding,
            jit_options=_step_jit_options,
        )

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
            self.arguments.cakld_gamma,
            int(self.arguments.loss_top_k),
            bool(self.arguments.loss_add_tail),
            bool(self.arguments.mtp_distillation),
            float(self.arguments.mtp_kd_weight),
            int(self.arguments.mtp_draft_tokens),
            bool(self.arguments.teacher_matched_compilation),
            *self._teacher_cache_static_args(),
            self._num_loss_groups(),
        )

        # The eval step may run under `eval_config_overrides` (e.g. vanilla
        # attention): compile it against the eval-graphdef shardings for BOTH
        # the student (eval_state_shardings) and the teacher (the swapped
        # teacher state built by _configure_eval_config_override) so the
        # sharding trees match the states passed at call time.
        eval_teacher_state = getattr(self, "_eval_teacher_state", None) or self.teacher_state
        self._runtime_trace("eval.compile_wrapper.begin")
        if self.arguments.mpmd_scheduler is None:
            sharded_evaluation_step_function = spx.jit(
                distillation_step,
                in_shardings=(self.eval_state_shardings, empty_sharding, self._teacher_in_shardings(eval_teacher_state)),
                out_shardings=empty_sharding,
                static_argnums=static_argnums,
                **_step_jit_options,
            )
        else:
            sharded_evaluation_step_function = compile_trainer_step(
                distillation_step,
                in_shardings=(self.eval_state_shardings, empty_sharding, self._teacher_in_shardings(eval_teacher_state)),
                out_shardings=empty_sharding,
                static_argnums=static_argnums,
                mesh=self.model.mesh,
                schedule=self.arguments.mpmd_scheduler,
                arguments=self.arguments,
            )
        self._runtime_trace("eval.compile_wrapper.end")

        sharded_training_step_function.static_argnums_ = static_argnums
        sharded_evaluation_step_function.static_argnums_ = static_argnums

        # Teacher is frozen: it contributes a forward pass only, no backward (none at all when cached).
        teacher_forward_flops = (
            0.0
            if self.teacher_state is None
            else self.teacher_state.model.flops_per_token(include_loss=False, include_backward=False)
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
        return self._build_sft_preprocess_transform(
            max_length=self.arguments.max_length,
            padding_free=self._user_data_collator,
        )

    def _build_sft_preprocess_transform(self, *, max_length: int, padding_free: bool) -> SFTPreprocessTransform:
        """Construct the SFT tokenisation transform at a given length / padding mode.

        Shared by the primary-source path (:meth:`_get_preprocess_transform`) and
        the per-bucket path (:meth:`_get_bucket_preprocess_transform`) so both
        honour the same ``dataset_text_field`` / prompt-masking policy while
        differing only in ``max_length`` and whether rows are padded (a bucket
        with its own packing collator wants unpadded rows).
        """
        text_field = getattr(self.arguments, "dataset_text_field", None) or "text"
        mask_prompt = bool(getattr(self.arguments, "assistant_only_loss", False))
        completion_only_loss = getattr(self.arguments, "completion_only_loss", None)
        if completion_only_loss is not None:
            mask_prompt = bool(completion_only_loss)
        packing = bool(getattr(self.arguments, "sequence_packing", False))
        return SFTPreprocessTransform(
            tokenizer=self.processing_class,
            max_length=max_length,
            text_field=text_field,
            mask_prompt=mask_prompt,
            padding=False if (packing or padding_free) else "max_length",
            truncation_mode=self.arguments.truncation_mode,
        )

    def _get_bucket_preprocess_transform(self, source, bucket=None):
        """Per-bucket SFT tokenisation, gated on the bucket source's own schema.

        Unlike the base implementation (which gates on the primary train source),
        this inspects ``source`` directly: a distillation run can mix a
        pretokenized long-context bucket with a raw-text SFT/VLM bucket. Returns
        ``None`` when the bucket source is already tokenised; otherwise builds a
        transform sized to the bucket's ``max_length`` and emitting unpadded rows
        when the bucket carries its own collator (which packs/pads itself).
        """
        if self._source_is_pretokenized(source):
            return None
        max_length = self.arguments.max_length
        if bucket is not None and getattr(bucket, "max_length", None):
            max_length = bucket.max_length
        padding_free = self._user_data_collator or (bucket is not None and bucket.data_collator is not None)
        return self._build_sft_preprocess_transform(max_length=max_length, padding_free=padding_free)

    def _source_is_pretokenized(self, source) -> bool:
        """Whether ``source`` already yields tokenised rows (exposes ``input_ids``).

        Heterogeneous wrappers (mixed / shuffled sources) always report ``False``
        so their raw rows still flow through the transform (which passes
        already-tokenised rows through untouched).
        """
        if source is None:
            return False
        if self._source_requires_row_preprocessing(source):
            return False
        try:
            sample = next(iter(source.open_shard(source.shard_names[0])))
            return "input_ids" in sample
        except (StopIteration, IndexError):
            return False

    def _is_pretokenized(self) -> bool:
        """Detect whether the primary training source already yields tokenised samples.

        Returns:
            ``True`` when the first sample of the first shard exposes
            an ``input_ids`` column; ``False`` when the source is
            absent or the shard is empty.
        """
        return self._source_is_pretokenized(self._train_source)

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
        infos = dict(infos)
        group_source = None
        if self._num_loss_groups() and isinstance(batch, dict):
            group_source = (batch.get(DATASET_ID_TOKEN_FIELD), batch.get(DATASET_ID_FIELD))
        # Without per-dataset carriers this is the legacy rule (assistant_masks fill a missing
        # completion_mask, masks AND attention, labels synthesised from the mask, or the mask recovered
        # from -100 labels); with carriers the per-dataset assembly of AssistantOnlyMaskMixin runs.
        batch = self._apply_assistant_supervision(batch, infos)
        if group_source is not None:
            batch = self._attach_loss_groups(batch, infos, *group_source, is_train=is_train)
        if getattr(self, "_teacher_cache_reader", None) is not None and isinstance(batch, dict):
            batch = self._join_teacher_cache(batch, infos)
        return batch, infos

    def _num_loss_groups(self) -> int:
        """Number of loss groups (``0`` when ``loss_group_weights`` is unset)."""
        weights = getattr(getattr(self, "arguments", None), "loss_group_weights", None)
        return len(weights) if weights else 0

    def _attach_loss_groups(
        self,
        batch: dict[str, tp.Any],
        infos: dict[str, tp.Any],
        token_ids: tp.Any,
        row_ids: tp.Any,
        *,
        is_train: bool,
    ) -> dict[str, tp.Any]:
        """Add ``loss_group_ids`` (and, in training, fp32 ``loss_token_weights``) for the full step batch.

        Raises:
            ValueError: When the batch carries no dataset attribution or a supervised token has no group.
        """
        input_ids = np.asarray(batch.get("input_ids", batch.get("inputs_embeds")))
        if token_ids is not None:
            dataset_ids = np.asarray(token_ids)
        elif row_ids is not None:
            rows = np.asarray(row_ids).reshape(-1).astype(np.int64)
            dataset_ids = np.repeat(rows[:, None], int(input_ids.shape[1]), axis=1)
        else:
            raise ValueError(
                "`loss_group_weights` is set but the batch carries no dataset attribution (per-token "
                f"`{DATASET_ID_TOKEN_FIELD}` or per-row `{DATASET_ID_FIELD}`)."
            )
        group_ids = loss_group_ids_from_dataset_ids(
            dataset_ids,
            num_groups=self._num_loss_groups(),
            dataset_map=getattr(self.arguments, "loss_group_dataset_map", None),
        )
        loss_config = getattr(self.arguments, "loss_config", None)
        shift_tokens = bool(getattr(loss_config, "shift_tokens", True)) if loss_config is not None else True
        supervised = loss_supervision_mask(batch, shift_tokens=shift_tokens)
        token_weights, stats = loss_group_token_weights(
            group_ids,
            supervised,
            self.arguments.loss_group_weights,
            max_multiplier=getattr(self.arguments, "loss_group_max_multiplier", None),
        )
        batch[LOSS_GROUP_IDS_KEY] = group_ids
        if is_train:
            batch[LOSS_TOKEN_WEIGHTS_KEY] = token_weights
        for g in range(self._num_loss_groups()):
            for name, vec in stats.items():
                infos[f"loss_group/{g}/{name}"] = float(vec[g])
        return batch

    def _teacher_cache_static_args(self) -> tuple[bool, int, bool]:
        """Trailing static args of :func:`distillation_step` for the offline-teacher-cache mode."""
        return (
            bool(getattr(self.arguments, "teacher_from_cache", False)),
            int(getattr(self.arguments, "teacher_cache_topk", 0) or 0),
            bool(getattr(self.arguments, "teacher_cache_add_tail", True)),
        )

    def _teacher_in_shardings(self, teacher_state: EasyDeLState | None = None):
        """Input sharding of the teacher-state argument (``None`` when the teacher is served from a cache)."""
        state = self.teacher_state if teacher_state is None else teacher_state
        return None if state is None else state.shardings

    def _join_teacher_cache(self, batch: dict[str, tp.Any], infos: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Add the cached teacher top-k tensors for every document of a host-side batch.

        Documents are located by the collator's layout sidecar (``teacher_layout_keys`` /
        ``teacher_layout_span``) when present, otherwise derived from ``segment_ids`` (packed: one document per
        contiguous segment) or ``attention_mask`` (unpacked: the real tokens of each row). Records are joined by
        row content (:func:`easydel.data.teacher_cache.row_key`); a missing record raises
        :class:`~easydel.data.teacher_cache.TeacherCacheMiss` -- there is no live-teacher fallback.
        """
        from easydel.data.teacher_cache import (
            KEY_BYTES,
            LAYOUT_KEYS_FIELD,
            LAYOUT_SPAN_FIELD,
            key_to_array,
            row_key,
        )

        reader = self._teacher_cache_reader
        input_ids = np.asarray(batch["input_ids"])
        n_windows, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        keys = batch.pop(LAYOUT_KEYS_FIELD, None)
        span = batch.pop(LAYOUT_SPAN_FIELD, None)
        if keys is None or span is None:
            docs: list[list[tuple[int, int]]] = []
            segment_ids = batch.get("segment_ids")
            attention = batch.get("attention_mask")
            for w in range(n_windows):
                row_docs: list[tuple[int, int]] = []
                if segment_ids is not None:
                    seg = np.asarray(segment_ids)[w]
                    valid = np.flatnonzero(seg >= 0)
                    if valid.size:
                        bounds = np.flatnonzero(np.diff(seg[valid]) != 0) + 1
                        for part in np.split(valid, bounds):
                            row_docs.append((int(part[0]), int(part[-1]) - int(part[0]) + 1))
                else:
                    real = np.flatnonzero(np.ones(seq_len) if attention is None else np.asarray(attention)[w] > 0)
                    if real.size:
                        row_docs.append((int(real[0]), int(real[-1]) - int(real[0]) + 1))
                docs.append(row_docs)
            slots = max(max((len(d) for d in docs), default=0), 1)
            keys = np.zeros((n_windows, slots, KEY_BYTES), dtype=np.uint8)
            span = np.zeros((n_windows, slots, 2), dtype=np.int32)
            for w, row_docs in enumerate(docs):
                for d, (off, n) in enumerate(row_docs):
                    keys[w, d] = key_to_array(row_key(input_ids[w, off : off + n]))
                    span[w, d] = (off, n)
        filled = reader.fill_windows(np.asarray(keys), np.asarray(span), seq_len=seq_len, input_ids=input_ids)
        batch.update(filled)
        infos["teacher_cache_documents"] = float(int((np.asarray(span)[..., 1] > 0).sum()))
        return batch

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
        the same compiled function with ``is_train=False`` in its static args —
        except under ``eval_config_overrides``, where the teacher state carrying
        the eval graphdef (e.g. vanilla attention) is passed instead so the
        teacher forward also runs on the overridden config.
        """
        eval_teacher_state = getattr(self, "_eval_teacher_state", None)
        if eval_teacher_state is not None:
            return (eval_teacher_state,)
        return (self.teacher_state,)

    def _configure_eval_config_override(self):
        """Extend the base override with a matching teacher-side eval graphdef.

        Distillation's compiled eval step consumes ``teacher_state`` as a
        regular input; evaluating the student on an overridden config while the
        teacher keeps the training config would score the eval KL/entropy
        against a mismatched reference, so the same overrides are applied to a
        teacher graphdef variant (buffers shared, teacher stays frozen).
        """
        super()._configure_eval_config_override()
        self._eval_teacher_state = None
        if self._eval_graphdef is None or self.teacher_state is None:
            return
        teacher_graphdef = self._build_config_override_graphdef(
            model=self.teacher_state.model,
            overrides=self.arguments.eval_config_overrides,
            reference_graphstate=self.teacher_state.graphstate,
            trainable_selector=None,
            label="eval_config_overrides (teacher)",
            eval_mode=True,
        )
        self._eval_teacher_state = self.teacher_state.replace(graphdef=teacher_graphdef)

    # -- Training-bucket hooks (see BaseTrainer._compile_bucket_step_functions) --
    # Without these overrides, buckets compiled the BASE Trainer's
    # `training_step` (plain LM loss) — a bucketed distillation run would
    # silently train without the teacher.

    def _resolve_bucket_step_args(self, i: int) -> tuple[tuple, tuple]:
        """Per-bucket ``(extra, static)`` args for the compiled distillation step.

        ``extra`` carries ``teacher_state`` (positional arg 2 of
        :func:`distillation_step`); ``static`` is this trainer's shared
        static-arg tuple with the bucket's ``loss_config`` /
        ``step_partition_spec`` / ``gradient_accumulation_steps`` overrides
        applied at their positional slots.
        """
        bucket = self._buckets[i]
        static = list(self._train_shared_fn_static_args)
        if bucket.loss_config is not None:
            static[0] = bucket.loss_config
        if bucket.step_partition_spec is not None:
            static[2] = bucket.step_partition_spec
        if bucket.gradient_accumulation_steps is not None:
            static[3] = bucket.gradient_accumulation_steps
        return (self.teacher_state,), tuple(static)

    def _bucket_step_compile_kwargs(self, i: int) -> dict:
        """Extend the base compile kwargs with the teacher-state input sharding."""
        kwargs = super()._bucket_step_compile_kwargs(i)
        state_shardings, empty_sharding = kwargs["in_shardings"]
        kwargs["in_shardings"] = (state_shardings, empty_sharding, self._teacher_in_shardings())
        return kwargs

    def _compile_one_bucket_step(self, i: int):
        """Compile bucket ``i``'s :func:`distillation_step` with teacher wiring."""
        return compile_trainer_step(distillation_step, **self._bucket_step_compile_kwargs(i))
