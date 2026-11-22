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

"""Llama decoder with separate, optionally tied discrete-vision embeddings.

Text checkpoint paths and shapes are identical to Llama. Joint token IDs put
vision codes after the text vocabulary; no image encoder or continuous-pixel
representation is required. The ordinary causal decoder and cache/generation
implementations are reused for interleaved text and image-code sequences.
"""

import typing as tp

import jax
import spectrax as spx
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array
from transformers import GenerationConfig

from easydel.inference.logits_process import LogitsProcessor, LogitsProcessorList
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.layers import ColumnParallelLinear, Embed
from easydel.modules._base import BaseCausalLMModule

from .llama_configuration import VisionLlamaConfig
from .modeling_llama import LlamaModel


class _JointVocabularyProjection(spx.Module):
    """Exportable hidden-to-joint-vocabulary map without a decoder capture.

    The projection borrows modules, rather than copying parameters or attaching
    a new child to the model, preserving all checkpoint paths and weight ties.
    """

    def __init__(self, text_head, text_embedding, vision_head, vision_tied, dtype, precision, postprocess):
        self.text_head = text_head
        self.text_embedding = text_embedding
        self.vision_head = vision_head
        self.vision_tied = vision_tied
        self.dtype = dtype
        self.precision = precision
        self.postprocess = postprocess

    def forward(self, hidden_states):
        """Project both namespaces in text-then-vision order and transform once."""
        weight = self.text_embedding.weight.value.T if self.text_embedding is not None else None
        text_logits = self.text_head.native_forward(hidden_states, w=weight)
        if self.vision_tied:
            vision_logits = jnp.dot(
                hidden_states.astype(self.dtype),
                self.vision_head.weight.value.T.astype(self.dtype),
                precision=self.precision,
            )
        else:
            vision_logits = self.vision_head.native_forward(hidden_states)
        logits = jnp.concatenate((text_logits, vision_logits), axis=-1)
        return logits if self.postprocess is None else self.postprocess(logits)


class _VisionVocabularyLogitsProcessor(LogitsProcessor):
    """Constrain generated joint IDs without renumbering either vocabulary."""

    def __init__(self, text_vocab_size: int, sample_mode: str):
        self.text_vocab_size = text_vocab_size
        self.sample_mode = sample_mode

    def __call__(self, input_ids, scores, cur_len):
        del input_ids, cur_len
        if self.sample_mode in ("all", "random"):
            return scores
        is_vision = jnp.arange(scores.shape[-1]) >= self.text_vocab_size
        allowed = is_vision if self.sample_mode == "vision" else ~is_vision
        return jnp.where(allowed, scores, -jnp.inf)


@register_module(
    TaskType.BASE_MODULE,
    config=VisionLlamaConfig,
    model_type="vision_llama",
    embedding_layer_names=["embed_tokens", "vision_embed_tokens"],
)
class VisionLlamaModel(LlamaModel):
    """Llama backbone accepting text IDs and offset discrete-vision codebook IDs."""

    _config_class = VisionLlamaConfig

    def __init__(
        self,
        config: VisionLlamaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(config, dtype, param_dtype, precision, rngs=rngs)
        with self.assign_layer_stage(0, total_layers=config.num_hidden_layers):
            self.vision_embed_tokens = Embed(
                config.vision_vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=jax.nn.initializers.normal(config.initializer_range),
                rngs=rngs,
            )

    def embed_input_ids(self, input_ids: Array) -> Array:
        """Embed joint IDs while keeping text and vision parameter tables separate.

        Args:
            input_ids: Integer IDs in ``[0, config.total_vocab_size)``.

        Returns:
            Embeddings of shape ``(*input_ids.shape, config.hidden_size)``.
        """
        is_vision = input_ids >= self.config.vision_token_offset
        text_ids = jnp.where(is_vision, 0, input_ids)
        vision_ids = jnp.where(is_vision, input_ids - self.config.vision_token_offset, 0)
        return jnp.where(
            is_vision[..., None],
            self.vision_embed_tokens(vision_ids.astype(jnp.int32)),
            self.embed_tokens(text_ids.astype(jnp.int32)),
        )

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        mask_info=None,
        position_ids=None,
        mode=None,
        past_key_values=None,
        cache_metadata=None,
        output_attentions=None,
        output_hidden_states=None,
        trace=False,
    ):
        """Decode an interleaved discrete text/vision sequence with ordinary Llama caches."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_input_ids(input_ids), "embeddings")
        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            trace=trace,
        )


@register_module(
    TaskType.CAUSAL_LM,
    config=VisionLlamaConfig,
    model_type="vision_llama",
    embedding_layer_names=["model.embed_tokens", "model.vision_embed_tokens"],
    layernorm_names=["input_layernorm", "post_attention_layernorm", "norm"],
)
class VisionLlamaForCausalLM(BaseCausalLMModule[VisionLlamaModel, VisionLlamaConfig]):
    """Registered joint-vocabulary causal LM for ``VisionLlamaConfig``.

    The text backbone/head retain the ``model.*`` and ``lm_head.*`` paths of
    Llama. Vision parameters live at ``model.vision_embed_tokens.weight`` and,
    when not tied, ``vision_lm_head.*``. Tying projects directly with the vision
    input table, so optimizer updates and checkpoint reloads cannot break ties.
    """

    _task_type = TaskType.CAUSAL_LM
    _config_class = VisionLlamaConfig

    def __init__(
        self,
        config: VisionLlamaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=VisionLlamaModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            tie_word_embeddings=config.tie_word_embeddings,
            lm_head_bias=False,
        )
        # Direct construction must support generation as well as checkpoint
        # loading (the latter may subsequently replace these defaults).
        self.generation_config = GenerationConfig.from_model_config(config)
        if not config.tie_vision_embeddings:
            self.vision_lm_head = self._create_task_head_on_last_stage(
                lambda: ColumnParallelLinear(
                    config.hidden_size,
                    config.vision_vocab_size,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    use_bias=False,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    rngs=rngs,
                )
            )

    def get_lm_head_projection(self) -> spx.Module:
        """Export the complete joint head for eSurge's single and pipeline splits.

        Only projection modules and a parameter-free postprocessing snapshot
        are retained. Text and vision ties refer to their original embeddings;
        the decoder and its unrelated parameters are not part of this graph.
        """
        return _JointVocabularyProjection(
            text_head=self.get_lm_head(),
            text_embedding=self.get_embedding() if self._tie_embeddings_enabled() else None,
            vision_head=self.model.vision_embed_tokens if self.config.tie_vision_embeddings else self.vision_lm_head,
            vision_tied=self.config.tie_vision_embeddings,
            dtype=self.dtype,
            precision=self.precision,
            postprocess=self.get_lm_logit_postprocess(),
        )

    def _make_raw_lm_head_fn(self, vocab_shard_stage: int | None = None) -> tp.Callable[[Array], Array]:
        """Capture both raw projections without prematurely transforming text logits."""
        text_projection = EasyDeLBaseModule.make_lm_head_fn(self, vocab_shard_stage=vocab_shard_stage)
        if self.config.tie_vision_embeddings:
            weight = self.model.vision_embed_tokens.weight.value.T.astype(self.dtype)
            dtype, precision = self.dtype, self.precision

            def vision_projection(hidden_states):
                return jnp.dot(hidden_states.astype(dtype), weight, precision=precision)
        else:
            vision_projection = self.vision_lm_head.native_forward

        def project(hidden_states):
            return jnp.concatenate((text_projection(hidden_states), vision_projection(hidden_states)), axis=-1)

        return project

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Project raw joint logits; ``compute_lm_logits`` applies the final transform."""
        return self._make_raw_lm_head_fn()(hidden_states)

    def make_lm_head_fn(self, vocab_shard_stage: int | None = None) -> tp.Callable[[Array], Array]:
        """Return trace-safe joint logits with the authoritative transform applied once."""
        project = self._make_raw_lm_head_fn(vocab_shard_stage=vocab_shard_stage)
        postprocess = self.get_lm_logit_postprocess()
        if postprocess is None:
            return project

        def transformed(hidden_states):
            return postprocess(project(hidden_states))

        return transformed

    def generate(
        self,
        input_ids,
        generation_config=None,
        prng_key=None,
        trace=True,
        logits_processor=None,
        sample_mode: str | None = None,
        **kwargs,
    ):
        """Generate joint IDs using the config's discrete-vision sampling strategy.

        Args:
            input_ids: Prompt in the joint text/vision namespace.
            generation_config: Optional standard generation configuration.
            prng_key: Random key used for categorical sampling.
            trace: Whether to compile the generation loop.
            logits_processor: Additional caller-provided generation constraints.
            sample_mode: Override ``config.sample_mode`` for this call only.
                ``all`` uses the requested generation strategy on both vocabularies;
                ``random`` defaults to categorical sampling on both. ``text`` and
                ``vision`` constrain generated IDs to the respective vocabulary.
            **kwargs: Standard generation options, including an explicit
                ``do_sample`` override for the random-mode default.

        Returns:
            The standard greedy, sampled or beam generation output with joint IDs.
        """
        sample_mode = self.config.sample_mode if sample_mode is None else sample_mode
        self.config.validate_sample_mode(sample_mode)
        processors = LogitsProcessorList(logits_processor or ())
        # The generation mixin may call generate again on a remat-compatible
        # graph. Carry the policy in the processor list so a per-call override
        # is preserved, and avoid duplicate-processor errors in that recursion.
        if not any(isinstance(processor, _VisionVocabularyLogitsProcessor) for processor in processors):
            processors.append(_VisionVocabularyLogitsProcessor(self.config.vocab_size, sample_mode))
            if sample_mode == "random":
                kwargs.setdefault("do_sample", True)
        return super().generate(
            input_ids=input_ids,
            generation_config=generation_config,
            prng_key=prng_key,
            trace=trace,
            logits_processor=processors,
            **kwargs,
        )
