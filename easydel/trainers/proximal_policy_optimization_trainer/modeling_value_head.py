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

"""Value head model wrapper for PPO training.

This module provides the CausalLMWithValueHead class that wraps a causal language
model with a scalar value head, which is essential for PPO-style RLHF training.
The value head predicts the expected return from each state, enabling advantage
estimation and variance reduction during policy optimization.
"""

from __future__ import annotations

import typing as tp

import spectrax as spx

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.layers import ParallelLinear


def _infer_hidden_size(config) -> int:
    """Infer the hidden size from a model configuration.

    Searches for common hidden size attribute names across different model
    architectures (GPT, T5, BERT, etc.) to determine the embedding dimension.

    Args:
        config: Model configuration object with hidden size attributes.

    Returns:
        The hidden size (embedding dimension) of the model.

    Raises:
        ValueError: If no recognized hidden size attribute is found.
    """
    for name in ("hidden_size", "n_embd", "d_model", "model_dim", "dim", "embed_dim"):
        val = getattr(config, name, None)
        if isinstance(val, int) and val > 0:
            return val
    raise ValueError(
        "Unable to infer hidden size from config. Expected one of: "
        "`hidden_size`, `n_embd`, `d_model`, `model_dim`, `dim`, `embed_dim`."
    )


class CausalLMWithValueHead(EasyDeLBaseModule):
    """Causal LM augmented with a scalar value-head adapter for PPO.

    PPO needs both a stochastic policy ``pi_theta`` (the language
    model) and a value function ``V_phi(s)`` that predicts the
    expected discounted return from the current state. This wrapper
    keeps the language model untouched -- so generation, FLOP
    accounting, and the eSurge inference path all delegate to the
    base module -- and adds a single trainable head:

    * **Architecture**: a bias-free :class:`ParallelLinear` projection
      ``hidden_size -> 1``, kernel initialised to zeros so that
      ``V_phi`` is exactly ``0`` at the start of training. This is the
      standard "warmup" initialisation that prevents the early-training
      value loss from dominating the policy gradient. The projection
      is sharded by the same strategy as the rest of the model
      (``ParallelLinear`` resolves its kernel partition spec from the
      mesh).
    * **Input/output**: given hidden states of shape
      ``(batch, seq_len, hidden_size)`` produced by the underlying
      model's last decoder layer, the head returns scalar values of
      shape ``(batch, seq_len, 1)`` that the PPO step squeezes to
      ``(batch, seq_len)`` and uses as the per-token baseline in GAE.
    * **Training treatment**: PPO trains ``V_phi`` with a clipped
      regression loss (see :class:`PPOConfig.cliprange_value` and
      :class:`PPOConfig.vf_coef`) jointly with the clipped policy
      surrogate. The base model parameters are differentiated through
      the same call so the policy and value-head share representations.

    The wrapper is *not* constructible from ``config`` alone; the
    eSurge inference path falls back to the underlying model's
    constructible variant for attention-implementation swaps.

    Attributes:
        config_class: Mirror of ``base_model.config_class`` -- exposed
            so downstream code can introspect the model family without
            unwrapping.
        base_model_prefix: Parameter-tree prefix (``"model"``) under
            which the base LM's parameters live; the value-head
            parameters live under ``"value_head"``.
        model: The wrapped causal LM whose forward/generate APIs are
            delegated.
        value_head: Zero-initialised :class:`ParallelLinear` projecting
            ``hidden_size -> 1`` that produces ``V_phi``.
    """

    config_class: tp.Any = None
    base_model_prefix: str = "model"

    def __init__(
        self,
        base_model: EasyDeLBaseModule,
        *,
        rngs: spx.Rngs | None = None,
    ):
        """Initialize the CausalLMWithValueHead wrapper.

        Args:
            base_model: The underlying causal language model to wrap.
            rngs: Optional random number generators for initialization.
        """
        self.config_class = getattr(base_model, "config_class", None)
        super().__init__(
            config=base_model.config,
            dtype=base_model.dtype,
            param_dtype=base_model.param_dtype,
            precision=base_model.precision,
            rngs=rngs or base_model.rngs,
        )

        self.model = base_model
        hidden_size = _infer_hidden_size(base_model.config)
        self.value_head = ParallelLinear(
            hidden_size,
            1,
            use_bias=False,
            dtype=base_model.dtype,
            param_dtype=base_model.param_dtype,
            precision=base_model.precision,
            kernel_init=jax.nn.initializers.zeros,
            rngs=self.rngs,
        )

    def forward(self, *args, **kwargs):
        """Run the wrapped causal LM forward pass.

        The value head is *not* applied here -- callers obtain the
        base model's :class:`ModelOutput` (logits, hidden states, etc.)
        and feed ``hidden_states[-1]`` into ``self.value_head`` separately
        when they need ``V_phi``. This split keeps the wrapper compatible
        with EasyDeL helpers that expect an unmodified causal-LM forward
        signature.

        Args:
            *args: Positional arguments forwarded to the base module.
            **kwargs: Keyword arguments forwarded to the base module.

        Returns:
            The base model's forward output -- typically a model-output
            dataclass carrying ``logits`` and (when requested)
            ``hidden_states`` / ``last_hidden_state``.
        """
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Sample completions via the underlying language model.

        The value head is irrelevant for sampling and is bypassed
        entirely. Used by the PPO trainer to produce on-policy rollouts
        before computing log-probs and rewards.

        Args:
            *args: Positional arguments forwarded to ``base_model.generate``.
            **kwargs: Keyword arguments forwarded to ``base_model.generate``.

        Returns:
            Whatever the base module's ``generate`` returns -- typically
            a dataclass with ``sequences``/``scores``/``logits``.
        """
        return self.model.generate(*args, **kwargs)

    def flops_per_token(self, *args, **kwargs):
        """Forward to the base model's FLOPs-per-token estimator.

        The value head's contribution (a single 1-row dense projection)
        is negligible relative to the LM's hidden-to-vocab projection
        and is intentionally *not* added to the estimate so the trainer's
        throughput accounting matches the unwrapped model's numbers.

        Args:
            *args: Positional arguments forwarded to
                ``base_model.flops_per_token``.
            **kwargs: Keyword arguments forwarded to
                ``base_model.flops_per_token``.

        Returns:
            FLOPs per token reported by the base model.
        """
        return self.model.flops_per_token(*args, **kwargs)

    # eSurge is an inference-only path that relies on instantiating
    # `self.__class__(config=...)` when swapping attention mechanisms.
    # This wrapper is not constructible from `config`, so we delegate all eSurge
    # APIs to the underlying base model.

    @property
    def esurge_graphdef(self):  # pragma: no cover
        """eSurge graph definition produced by the underlying base model.

        The wrapper is not directly constructible from a config (it
        needs a pre-built ``base_model``), so the eSurge attention-swap
        machinery operates on the base module's graph definition. This
        property surfaces that graph definition transparently.
        """
        return self.model.esurge_graphdef

    @property
    def esurge_compatible_model(self):  # pragma: no cover
        """eSurge-compatible model handle from the wrapped base module.

        eSurge-compatible models expose APIs (paged KV cache, ragged
        attention, etc.) needed by the inference engine. This property
        exposes the base model's variant without re-wrapping it -- the
        value head is dropped during inference.
        """
        return self.model.esurge_compatible_model

    def get_esurge(self, *args, **kwargs):  # pragma: no cover
        """Construct or fetch an eSurge engine via the base model.

        Forwarded so PPO rollout code can opt into eSurge-accelerated
        sampling without unwrapping the value head.
        """
        return self.model.get_esurge(*args, **kwargs)

    def esurge_generate(self, *args, **kwargs):  # pragma: no cover
        """Generate completions using the base model's eSurge engine.

        Mirrors :meth:`generate` but routes through the eSurge inference
        runtime; the value head is not consulted.
        """
        return self.model.esurge_generate(*args, **kwargs)

    def _call_esurge_engine(self, *args, **kwargs):  # pragma: no cover
        """Invoke a pre-resolved eSurge engine on the underlying model.

        Internal helper used by the rollout path when an engine instance
        has already been built and only needs to be called.
        """
        return self.model._call_esurge_engine(*args, **kwargs)

    def pause_esurge(self, *args, **kwargs):  # pragma: no cover
        """Pause an active eSurge inference engine on the base model.

        Used by trainers that share the eSurge engine across rollout
        phases and wish to release worker capacity during the PPO update.
        """
        return self.model.pause_esurge(*args, **kwargs)

    def resume_esurge(self, *args, **kwargs):  # pragma: no cover
        """Resume an eSurge engine previously paused via :meth:`pause_esurge`."""
        return self.model.resume_esurge(*args, **kwargs)

    def list_esurge_engines(self, *args, **kwargs):  # pragma: no cover
        """Enumerate active eSurge engines registered against the base model."""
        return self.model.list_esurge_engines(*args, **kwargs)
