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

"""Reusable Loop Transformer execution mixin, checkpoint upgrade hook, and telemetry."""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from eformer.loggings import get_logger
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn, with_sharding_constraint

from easydel.infra.sharding import resolve_stage_mesh
from easydel.infra.utils import ArrayParam

logger = get_logger(__name__)


def materialize_missing_loop_state(
    config: tp.Any,
    state: dict[tuple[tp.Any, ...], tp.Any],
    expected_leaves: dict[tuple[tp.Any, ...], tp.Any],
) -> dict[tuple[tp.Any, ...], tp.Any]:
    """Initialize missing Loop Transformer leaves when loading a non-looped checkpoint.

    Used by :meth:`LoopTransformerMixin._upgrade_legacy_native_state` so any
    standard EasyDeL checkpoint can be loaded directly into a looped config with
    zero offline weight conversion: every base weight loads verbatim, while
    absent ``loop_gates.*`` and ``loop_iter_embeddings`` leaves receive their
    exact identity-preserving initialization. Already-trained looped checkpoints
    that include those keys are returned untouched.

    Args:
        config: Model configuration (or composite VLM config exposing ``get_text_config``).
        state: Flattened checkpoint leaf dictionary loaded from disk.
        expected_leaves: Flattened leaf template from the target model's state.

    Returns:
        Updated ``state`` dictionary with any missing loop leaves materialized.
    """
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    if not getattr(text_config, "is_looped", False):
        return state

    init_biases = text_config.resolved_loop_gate_init_biases
    added = 0
    for key, leaf in expected_leaves.items():
        if key in state or not hasattr(leaf, "shape"):
            continue
        str_parts = tuple(str(p) for p in key)
        if "loop_gates" in str_parts:
            gate_pos = str_parts.index("loop_gates")
            if gate_pos + 1 >= len(str_parts):
                continue
            gate_idx = int(str_parts[gate_pos + 1])
            if len(str_parts) >= 2 and str_parts[-2:] == ("proj", "bias"):
                bias_val = float(init_biases[gate_idx])
                state[key] = np.full(leaf.shape, bias_val, dtype=np.float32)
            else:
                state[key] = np.zeros(leaf.shape, dtype=np.float32)
            added += 1
        elif "loop_iter_embeddings" in str_parts:
            state[key] = np.zeros(leaf.shape, dtype=np.float32)
            added += 1

    if added:
        logger.info(
            "Initialized %d missing Loop Transformer leaf/leaves for layers [%d, %d) x %d (init_biases=%s).",
            added,
            int(text_config.loop_start_layer),
            int(text_config.loop_end_layer),
            int(text_config.num_loop_iterations),
            init_biases,
        )
    return state


class LoopTransformerMixin:
    """Mixin adding dynamic Loop Transformer execution to :class:`EasyDeLBaseModule`.

    When ``config.is_looped`` is ``False`` (the default), this mixin allocates
    no parameters and adds zero runtime overhead. When ``config.is_looped`` is
    ``True``, a decoder model opts in with two lines:

    1. Call ``self._init_loop_modules(rngs=rngs)`` in ``__init__`` (creating
       ``self.loop_gates`` and optional ``self.loop_iter_embeddings``).
    2. In ``forward``, branch ``if self.config.is_looped:`` into
       ``self._run_looped_layers(...)`` with a per-layer ``step_fn``.

    Supports:
        - Arbitrary Prelude / ``K``-pass gated Core / Coda ranges
          (``[loop_start_layer, loop_end_layer) x num_loop_iterations``).
        - Both ``"scalar"`` and ``"channel"`` per-token progressive update gates.
        - Optional zero-initialized per-iteration embeddings.
        - Both ``"separate"`` and GLM-5.4 ``"hybrid_sliding_window"`` KV caching
          across ``scan_layers=True`` + ``auto_remat``, ``TransformerCache`` /
          ``HybridCache``, and eSurge ``RaggedPagesCache``.
        - Zero-conversion checkpoint loading from non-looped base checkpoints via
          :meth:`_upgrade_legacy_native_state`.
    """

    def _init_loop_modules(
        self,
        *,
        rngs: spx.Rngs,
        norm_cls: type[spx.Module] | None = None,
    ) -> None:
        """Initialize ``self.loop_gates`` and ``self.loop_iter_embeddings`` from ``self.config``.

        Args:
            rngs: SpectraX PRNG container.
            norm_cls: Optional custom RMSNorm class forwarded to :class:`LoopGate`.
                Defaults to zero-centered float32-scaled :class:`RMSNorm`.
        """
        self.config.validate_loop_config()
        if self.config.is_looped:
            from easydel.layers.loop import LoopGate

            init_biases = self.config.resolved_loop_gate_init_biases
            self.loop_gates = nn.ModuleList(
                [
                    LoopGate(
                        config=self.config,
                        init_bias=init_biases[it],
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                        precision=self.precision,
                        pass_index=it,
                        norm_cls=norm_cls,
                        rngs=rngs,
                    )
                    for it in range(self.config.num_loop_iterations)
                ]
            )
            if self.config.loop_use_iteration_embedding:
                self.loop_iter_embeddings = ArrayParam.bound(
                    shape=(self.config.num_loop_iterations, self.config.hidden_size),
                    dtype=self.param_dtype,
                    init_method="zeros",
                    key=rngs.param,
                )
            else:
                self.loop_iter_embeddings = None
        else:
            self.loop_gates = None
            self.loop_iter_embeddings = None

    def _upgrade_legacy_native_state(
        self,
        state: dict[tuple[tp.Any, ...], tp.Any],
        expected_leaves: dict[tuple[tp.Any, ...], tp.Any],
    ) -> dict[tuple[tp.Any, ...], tp.Any]:
        """Auto-initialize missing loop gate/embedding leaves when loading a non-looped checkpoint."""
        return materialize_missing_loop_state(self.config, state, expected_leaves)

    def _block_is_full_attention(self, block: tp.Any, *, all_core_full: bool) -> bool:
        """Return whether a decoder block executes full self-attention inside a scanned loop."""
        if all_core_full:
            return True
        explicit = getattr(block, "is_full_attention", None)
        if explicit is not None:
            return bool(explicit)
        self_attn = getattr(block, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "sliding_window"):
            return self_attn.sliding_window is None
        return True

    def _run_looped_layers(
        self,
        *,
        layers: nn.ModuleList,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        step_fn: tp.Callable[..., tp.Any],
        mask_info: tp.Any = None,
        cache_views: tp.Any = None,
        past_key_values: tp.Any = None,
        trace_layers: bool = False,
        can_scan: bool = False,
        scan_scope: str = "text",
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> tuple[
        Float[Array, "batch seq_len hidden_dim"],
        tuple[Float[Array, "batch seq_len hidden_dim"], ...] | None,
        tuple[tp.Any, ...] | None,
        tuple[tp.Any, ...] | None,
    ]:
        """Execute Prelude + ``K``-pass gated Loop Core + Coda over ``layers``.

        Args:
            layers: Physical decoder layer stack (unstacked ``nn.ModuleList``).
            hidden_states: Input residual stream ``[batch, seq_len, hidden_dim]``.
            step_fn: Callable ``step_fn(block, hidden_states, cache_view, shared_global_kv, return_shared_kv)``
                returning a ``DecoderLayerOutput`` (or ``(DecoderLayerOutput, captured_kv)`` when
                ``return_shared_kv=True``).
            mask_info: Attention ``MaskInfo`` forwarded to Pass 2+ dual-stream attention in scan mode.
            cache_views: Active cache view list when ``trace_layers=True``, else ``None``.
            past_key_values: Owning cache container passed to ``_layer_cache_view_at`` /
                ``_layer_cache_view_update``.
            trace_layers: Whether cache views are actively threaded through layers.
            can_scan: Whether the caller permits scanned execution (typically ``not trace and not needs_trace_cache``).
            scan_scope: Scope key passed to ``_layer_scan_kwargs``. Defaults to ``"text"``.
            output_hidden_states: Whether to collect per-step hidden states.
            output_attentions: Whether to collect per-step attention weights.
            output_router_logits: Whether to collect per-step MoE router logits.

        Returns:
            Tuple ``(hidden_states, all_hidden_states, all_attentions, all_router_logits)``.

        Raises:
            ValueError: If pipeline parallelism (``pp > 1``) is enabled alongside looping.
            RuntimeError: If ``self.loop_gates`` was not initialized via :meth:`_init_loop_modules`.
        """
        if self._pipeline_stage_count() > 1:
            raise ValueError(
                "Looped transformer execution requires pp=1 (pipeline_stage_count == 1) because "
                "re-entering the shared loop core creates a cycle across pipeline stages."
            )
        if getattr(self, "loop_gates", None) is None:
            raise RuntimeError(
                f"{type(self).__name__} has config.is_looped=True but loop_gates were not initialized; "
                "call self._init_loop_modules(rngs=rngs) in __init__."
            )

        cfg = self.config
        loop_start = int(cfg.loop_start_layer)  # type: ignore[arg-type]
        loop_end = int(cfg.loop_end_layer)  # type: ignore[arg-type]
        num_loops = int(cfg.num_loop_iterations)
        use_hybrid_kv = getattr(cfg, "loop_kv_cache_mode", "separate") == "hybrid_sliding_window"
        iter_embeddings = getattr(self, "loop_iter_embeddings", None)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        use_scan = (
            bool(can_scan)
            and bool(cfg.scan_layers)
            and not output_hidden_states
            and not output_attentions
            and not output_router_logits
        )

        if use_scan:
            scan_kwargs = self._layer_scan_kwargs(scan_scope)

            def _plain_scan_step(block, h):
                return step_fn(block, h, None, None, False).hidden_states

            # 1. Scanned Prelude: layers [0, loop_start)
            if loop_start > 0:
                hidden_states = nn.ModuleList(list(layers[:loop_start])).scan(
                    _plain_scan_step,
                    hidden_states,
                    trace=False,
                    **scan_kwargs,
                )

            # 2. Scanned Shared Loop Core: layers [loop_start, loop_end) x num_loops
            core_list = nn.ModuleList(list(layers[loop_start:loop_end]))
            n_core_full = sum(1 for i in range(loop_start, loop_end) if cfg.is_full_attention_layer(i))
            all_core_full = n_core_full == (loop_end - loop_start)
            pass1_k_buf = None
            pass1_v_buf = None
            if use_hybrid_kv and n_core_full > 0:
                num_kv_heads = int(getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)
                head_dim = int(getattr(cfg, "head_dim", None) or (int(cfg.hidden_size) // int(cfg.num_attention_heads)))
                kv_shape = (
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    num_kv_heads,
                    head_dim,
                )
                kv_spec = cfg.runtime_sharding_resolver.resolve(
                    dynamic_axes=common_types.AttnKVSharding,
                    shape=kv_shape,
                )
                buf_spec = PartitionSpec(None, *tuple(kv_spec))
                stage_mesh = resolve_stage_mesh(cfg.mesh, arr=hidden_states) or cfg.mesh
                pass1_k_buf = jnp.zeros((n_core_full, *kv_shape), dtype=hidden_states.dtype)
                pass1_v_buf = jnp.zeros((n_core_full, *kv_shape), dtype=hidden_states.dtype)
                if stage_mesh is not None:
                    pass1_k_buf = with_sharding_constraint(pass1_k_buf, buf_spec, mesh=stage_mesh)
                    pass1_v_buf = with_sharding_constraint(pass1_v_buf, buf_spec, mesh=stage_mesh)

            for loop_it in range(num_loops):
                x_prev = hidden_states
                if iter_embeddings is not None:
                    hidden_states = hidden_states + iter_embeddings.value[loop_it].astype(hidden_states.dtype)
                if use_hybrid_kv and n_core_full > 0 and loop_it == 0:
                    assert pass1_k_buf is not None and pass1_v_buf is not None

                    def _pass1_scan_step(block, carry):
                        h_in, k_arr, v_arr, f_idx = carry
                        if self._block_is_full_attention(block, all_core_full=all_core_full):
                            layer_out, captured = step_fn(block, h_in, None, None, True)
                            k_g, v_g, _, _ = captured
                            k_arr = jax.lax.dynamic_update_index_in_dim(k_arr, k_g.astype(k_arr.dtype), f_idx, axis=0)
                            v_arr = jax.lax.dynamic_update_index_in_dim(v_arr, v_g.astype(v_arr.dtype), f_idx, axis=0)
                            return layer_out.hidden_states, k_arr, v_arr, f_idx + jnp.asarray(1, dtype=jnp.int32)
                        layer_out = step_fn(block, h_in, None, None, False)
                        return layer_out.hidden_states, k_arr, v_arr, f_idx

                    hidden_states, pass1_k_buf, pass1_v_buf, _ = core_list.scan(
                        _pass1_scan_step,
                        (hidden_states, pass1_k_buf, pass1_v_buf, jnp.asarray(0, dtype=jnp.int32)),
                        trace=False,
                        **scan_kwargs,
                    )
                elif use_hybrid_kv and n_core_full > 0 and loop_it > 0:
                    assert pass1_k_buf is not None and pass1_v_buf is not None
                    k_ref = pass1_k_buf
                    v_ref = pass1_v_buf

                    def _pass2_scan_step(block, carry, *, _k_ref=k_ref, _v_ref=v_ref):
                        h_in, f_idx = carry
                        if self._block_is_full_attention(block, all_core_full=all_core_full):
                            k_g = jax.lax.dynamic_index_in_dim(_k_ref, f_idx, axis=0, keepdims=False)
                            v_g = jax.lax.dynamic_index_in_dim(_v_ref, f_idx, axis=0, keepdims=False)
                            layer_out = step_fn(block, h_in, None, (k_g, v_g, mask_info, None), False)
                            return layer_out.hidden_states, f_idx + jnp.asarray(1, dtype=jnp.int32)
                        layer_out = step_fn(block, h_in, None, None, False)
                        return layer_out.hidden_states, f_idx

                    hidden_states, _ = core_list.scan(
                        _pass2_scan_step,
                        (hidden_states, jnp.asarray(0, dtype=jnp.int32)),
                        trace=False,
                        **scan_kwargs,
                    )
                else:
                    hidden_states = core_list.scan(
                        _plain_scan_step,
                        hidden_states,
                        trace=False,
                        **scan_kwargs,
                    )

                hidden_states = self.loop_gates[loop_it](x_prev, hidden_states)
                hidden_states = apply_logical_sharding(
                    hidden_states,
                    dynamic_axes=common_types.HiddenStateSharding,
                    partition_manager=cfg.runtime_sharding_resolver,
                )

            # 3. Scanned Coda: layers [loop_end, num_hidden_layers)
            if loop_end < len(layers):
                hidden_states = nn.ModuleList(list(layers[loop_end:])).scan(
                    _plain_scan_step,
                    hidden_states,
                    trace=False,
                    **scan_kwargs,
                )
            return hidden_states, all_hidden_states, all_attentions, all_router_logits

        pass1_kv_streams: dict[int, tuple] = {}
        cv = cache_views
        cache_slot = 0

        def _run_layer(
            phys_idx: int,
            slot: int,
            h: Array,
            cv_in: tp.Any,
            *,
            shared_global_kv: tp.Any = None,
            return_shared_kv: bool = False,
        ):
            nonlocal all_hidden_states, all_attentions, all_router_logits
            if output_hidden_states:
                all_hidden_states += (h,)
            layer_res = step_fn(
                layers[phys_idx],
                h,
                self._layer_cache_view_at(cv_in, slot, enabled=trace_layers, cache=past_key_values),
                shared_global_kv,
                return_shared_kv,
            )
            if return_shared_kv:
                layer_outputs, captured_kv = layer_res
            else:
                layer_outputs, captured_kv = layer_res, None
            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)
            cv_out = self._layer_cache_view_update(
                cv_in,
                slot,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )
            if output_router_logits and getattr(layer_outputs, "router_logits", None) is not None:
                all_router_logits += (layer_outputs.router_logits,)
            return layer_outputs.hidden_states, cv_out, captured_kv

        # 1. Prelude: layers [0, loop_start)
        for phys_idx in range(0, loop_start):
            hidden_states, cv, _ = _run_layer(phys_idx, cache_slot, hidden_states, cv)
            cache_slot += 1

        # 2. Shared Loop Core: layers [loop_start, loop_end) x num_loops with Gated Progressive Updates
        for loop_it in range(num_loops):
            x_prev = hidden_states
            if iter_embeddings is not None:
                hidden_states = hidden_states + iter_embeddings.value[loop_it].astype(hidden_states.dtype)
            for phys_idx in range(loop_start, loop_end):
                is_full_attn = cfg.is_full_attention_layer(phys_idx)
                capture_kv = use_hybrid_kv and is_full_attn and loop_it == 0
                shared_kv = pass1_kv_streams.get(phys_idx) if (use_hybrid_kv and is_full_attn and loop_it > 0) else None
                hidden_states, cv, captured_kv = _run_layer(
                    phys_idx,
                    cache_slot,
                    hidden_states,
                    cv,
                    shared_global_kv=shared_kv,
                    return_shared_kv=capture_kv,
                )
                if capture_kv and captured_kv is not None:
                    pass1_kv_streams[phys_idx] = captured_kv
                cache_slot += 1
            hidden_states = self.loop_gates[loop_it](x_prev, hidden_states)
            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=cfg.runtime_sharding_resolver,
            )

        # 3. Coda: layers [loop_end, num_hidden_layers)
        for phys_idx in range(loop_end, len(layers)):
            hidden_states, cv, _ = _run_layer(phys_idx, cache_slot, hidden_states, cv)
            cache_slot += 1

        return hidden_states, all_hidden_states, all_attentions, all_router_logits


def _resolve_loop_layers(text_model: tp.Any):
    """Return the physical decoder layer sequence on ``text_model`` (``layers``, ``h``, or ``blocks``)."""
    for attr in ("layers", "h", "blocks"):
        seq = getattr(text_model, attr, None)
        if seq is not None:
            return seq
    raise AttributeError(f"{type(text_model).__name__} has no 'layers', 'h', or 'blocks' container.")


def _resolve_loop_embed(text_model: tp.Any, input_ids: Array) -> Array:
    """Embed ``input_ids`` across standard EasyDeL text backbones."""
    for attr in ("embed_tokens", "wte", "word_embeddings", "embed_in", "tok_embeddings"):
        emb = getattr(text_model, attr, None)
        if emb is not None:
            return emb(input_ids.astype("i4"))
    if hasattr(text_model, "get_embedding"):
        return text_model.get_embedding()(input_ids.astype("i4"))
    raise AttributeError(f"Unable to locate token embedding layer on {type(text_model).__name__}.")


def _resolve_loop_final_norm(text_model: tp.Any, hidden_states: Array) -> Array:
    """Apply final normalization on ``text_model`` if present."""
    for attr in ("norm", "ln_f", "norm_f", "final_layer_norm", "final_layernorm"):
        norm = getattr(text_model, attr, None)
        if norm is not None:
            return norm(hidden_states)
    return hidden_states


def _unwrap_loop_text_model(model: tp.Any) -> tp.Any:
    """Unwrap causal-LM / VLM wrappers down to the underlying decoder trunk owning ``loop_gates``."""
    tm = model
    for _ in range(4):
        if getattr(tm, "loop_gates", None) is not None:
            return tm
        advanced = False
        for attr in ("model", "transformer", "gpt_neox", "language_model", "decoder"):
            sub = getattr(tm, attr, None)
            if sub is not None and sub is not tm and hasattr(sub, "config"):
                tm = sub
                advanced = True
                break
        if not advanced:
            break
    return tm


def compute_loop_gate_stats(
    model: tp.Any,
    input_ids: Int[Array, "batch seq_len"],
    attention_mask: Bool[Array, "batch seq_len"] | None = None,
    *,
    assistant_mask: Bool[Array, "batch seq_len"] | None = None,
    include_knockout: bool = False,
) -> dict[str, tp.Any]:
    """Run a diagnostic forward pass and collect Loop Transformer gate & knockout telemetry.

    Works on any looped :class:`EasyDeLBaseModule`, causal-LM wrapper
    (``*ForCausalLM``), or multimodal wrapper (``*ForConditionalGeneration``)
    whose text trunk defines ``embed_tokens``, ``layers``, ``loop_gates``, and ``norm``.

    Args:
        model: Looped EasyDeL model or task wrapper.
        input_ids: Token IDs ``[batch, seq_len]``.
        attention_mask: Optional boolean valid-token mask ``[batch, seq_len]``.
        assistant_mask: Optional boolean assistant-token mask ``[batch, seq_len]``
            used for assistant-only NLL in knockout comparisons.
        include_knockout: If ``True``, also runs two loop-knockout ablations
            reusing the Prelude representation ``x_0``:
            - ``gate_zero``: clamps early refinement gates ``beta_1 .. beta_{K-1} = 0``,
              running only the final pass from ``x_0`` (keeping Pass-1 shared KV in
              hybrid mode).
            - ``single_pass``: runs ``layers[loop_start:loop_end]`` once without
              looping (exact non-looped trunk with the same weights).
            Positive ``delta_nll = knockout_nll - full_nll`` measures how much the
            recurrent loop directly improves predictions beyond trunk fine-tuning.

    Returns:
        Dictionary with ``per_pass``, ``pass1_loss_corr``, ``mean_nll``, and
        (when ``include_knockout=True``) ``knockout`` metrics.
    """
    text_model = _unwrap_loop_text_model(model)

    cfg = text_model.config
    if not getattr(cfg, "is_looped", False) or getattr(text_model, "loop_gates", None) is None:
        raise ValueError("compute_loop_gate_stats requires a looped EasyDeL model with initialized loop_gates.")

    @spx.jit
    def _run_telemetry(m, ids, attn_mask, asst_mask):
        lm_head = getattr(m, "lm_head", None)
        tm = _unwrap_loop_text_model(m)

        tcfg = tm.config
        mask_info = MaskInfo.dynamic_init(
            mask_info=None,
            input_ids=ids,
            attention_mask=attn_mask,
        )
        position_ids = mask_info.q_position_ids
        hidden_states = _resolve_loop_embed(tm, ids)
        tm_layers = _resolve_loop_layers(tm)
        frequencies = (
            tm.frequencies
            if hasattr(tm, "frequencies")
            else (tcfg.get_basic_frequencies() if hasattr(tcfg, "get_basic_frequencies") else None)
        )
        mode = common_types.MODE_TRAIN

        loop_start = int(tcfg.loop_start_layer)
        loop_end = int(tcfg.loop_end_layer)
        num_loops = int(tcfg.num_loop_iterations)
        use_hybrid_kv = getattr(tcfg, "loop_kv_cache_mode", "separate") == "hybrid_sliding_window"
        pass1_kv_streams: dict[int, tuple] = {}

        def _call_layer(phys_idx: int, h_in: Array, *, shared_kv=None, capture_kv: bool = False):
            layer_kwargs: dict[str, tp.Any] = {
                "hidden_states": h_in,
                "mask_info": mask_info,
                "position_ids": position_ids,
                "mode": mode,
                "frequencies": frequencies,
            }
            if shared_kv is not None:
                layer_kwargs["shared_global_kv"] = shared_kv
            if capture_kv:
                layer_kwargs["return_shared_kv"] = True
            return tm_layers[phys_idx](**layer_kwargs)

        def _run_coda(h_in):
            h = h_in
            for phys_idx in range(loop_end, len(tm_layers)):
                h = _call_layer(phys_idx, h).hidden_states
            return _resolve_loop_final_norm(tm, h)

        def _score_hidden_nll(h_normed):
            if hasattr(m, "compute_lm_logits"):
                logits = m.compute_lm_logits(h_normed[:, :-1, :]).astype(jnp.float32)
            else:
                logits = lm_head(h_normed[:, :-1, :]).astype(jnp.float32)
            targets = ids[:, 1:].astype(jnp.int32)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]

        def _run_loop_core(h_in, *, loop_it: int, capture_pass1_kv: bool, use_shared_kv: bool):
            h_cur = h_in
            iter_emb = getattr(tm, "loop_iter_embeddings", None)
            if iter_emb is not None:
                h_cur = h_cur + iter_emb.value[loop_it].astype(h_cur.dtype)
            for phys_idx in range(loop_start, loop_end):
                is_full_attn = tcfg.is_full_attention_layer(phys_idx)
                capture_kv = capture_pass1_kv and is_full_attn
                shared_kv = pass1_kv_streams.get(phys_idx) if (use_shared_kv and is_full_attn) else None
                layer_res = _call_layer(phys_idx, h_cur, shared_kv=shared_kv, capture_kv=capture_kv)
                if capture_kv:
                    layer_out, captured_kv = layer_res
                    if captured_kv is not None:
                        pass1_kv_streams[phys_idx] = captured_kv
                else:
                    layer_out = layer_res
                h_cur = layer_out.hidden_states
            return h_cur

        # 1. Prelude (shared across full forward and knockout ablations)
        for phys_idx in range(0, loop_start):
            hidden_states = _call_layer(phys_idx, hidden_states).hidden_states
        x0_prelude = hidden_states

        # 2. Loop core with gate capture
        per_pass_arrays: list[tuple[Array, ...]] = []
        per_pass_token_betas: list[Array] = []
        valid_mask = (
            jnp.asarray(attn_mask, dtype=jnp.bool_) if attn_mask is not None else jnp.ones(ids.shape, dtype=jnp.bool_)
        )
        vm = valid_mask.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(vm), 1.0)
        eps = 1e-8

        for loop_it in range(num_loops):
            x_prev = hidden_states
            h_cand = _run_loop_core(
                x_prev,
                loop_it=loop_it,
                capture_pass1_kv=use_hybrid_kv and loop_it == 0,
                use_shared_kv=use_hybrid_kv and loop_it > 0,
            )

            gate = tm.loop_gates[loop_it]
            beta = gate.compute_beta(x_prev, h_cand)  # [B, S, 1 or H] in fp32
            delta_f32 = h_cand.astype(jnp.float32) - x_prev.astype(jnp.float32)
            x_prev_f32 = x_prev.astype(jnp.float32)
            x_out_f32 = x_prev_f32 + beta * delta_f32
            hidden_states = x_out_f32.astype(x_prev.dtype)

            token_beta = jnp.mean(beta, axis=-1)  # [B, S]
            per_pass_token_betas.append(token_beta)

            b_mean = jnp.sum(token_beta * vm) / denom
            b_var = jnp.sum(((token_beta - b_mean) ** 2) * vm) / denom
            b_std = jnp.sqrt(jnp.maximum(b_var, 0.0))
            b_min = jnp.min(jnp.where(valid_mask, token_beta, jnp.inf))
            b_max = jnp.max(jnp.where(valid_mask, token_beta, -jnp.inf))

            rms_prev = jnp.sqrt(jnp.mean(x_prev_f32**2, axis=-1) + eps)
            rms_out = jnp.sqrt(jnp.mean(x_out_f32**2, axis=-1) + eps)
            rms_delta = jnp.sqrt(jnp.mean(delta_f32**2, axis=-1) + eps)
            norm_ratio = jnp.sum((rms_out / rms_prev) * vm) / denom
            delta_ratio = jnp.sum((rms_delta / rms_prev) * vm) / denom
            per_pass_arrays.append((b_mean, b_std, b_min, b_max, norm_ratio, delta_ratio))

        # 3. Coda + final norm
        hidden_states = _run_coda(hidden_states)

        can_score_logits = hasattr(m, "compute_lm_logits") or lm_head is not None
        if can_score_logits and ids.shape[1] >= 2 and per_pass_token_betas:
            nll = _score_hidden_nll(hidden_states)
            pair_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
            pv = pair_valid.astype(jnp.float32)
            p_denom = jnp.maximum(jnp.sum(pv), 1.0)
            mean_nll_val = jnp.sum(nll * pv) / p_denom

            has_asst = asst_mask is not None
            av = (jnp.asarray(asst_mask, dtype=jnp.bool_)[:, 1:] & pair_valid).astype(jnp.float32) if has_asst else pv
            a_denom = jnp.maximum(jnp.sum(av), 1.0)
            full_asst_nll = jnp.sum(nll * av) / a_denom

            nll_c = (nll - mean_nll_val) * pv
            nll_ss = jnp.sum(nll_c**2)
            per_pass_corrs: list[Array] = []
            for tb in per_pass_token_betas:
                b_i = tb[:, :-1]
                b_mean_i = jnp.sum(b_i * pv) / p_denom
                b_c = (b_i - b_mean_i) * pv
                cov = jnp.sum(b_c * nll_c)
                std_prod = jnp.sqrt(jnp.sum(b_c**2) * nll_ss)
                per_pass_corrs.append(jnp.where(std_prod > 1e-12, cov / jnp.maximum(std_prod, 1e-12), 0.0))
            corr_val = per_pass_corrs[0]

            knockout_arrays = None
            if include_knockout:
                final_it = num_loops - 1
                h_cand_gz = _run_loop_core(
                    x0_prelude,
                    loop_it=final_it,
                    capture_pass1_kv=False,
                    use_shared_kv=use_hybrid_kv,
                )
                h_gz = tm.loop_gates[final_it](x0_prelude, h_cand_gz)
                nll_gz = _score_hidden_nll(_run_coda(h_gz))
                gz_all = jnp.sum(nll_gz * pv) / p_denom
                gz_asst = jnp.sum(nll_gz * av) / a_denom

                h_sp = x0_prelude
                for phys_idx in range(loop_start, loop_end):
                    h_sp = _call_layer(phys_idx, h_sp).hidden_states
                nll_sp = _score_hidden_nll(_run_coda(h_sp))
                sp_all = jnp.sum(nll_sp * pv) / p_denom
                sp_asst = jnp.sum(nll_sp * av) / a_denom

                knockout_arrays = (full_asst_nll, gz_all, gz_asst, sp_all, sp_asst)

            return per_pass_arrays, mean_nll_val, corr_val, per_pass_corrs, knockout_arrays
        return per_pass_arrays, None, None, None, None

    per_pass_arrays, mean_nll_arr, corr_arr, per_pass_corr_arrs, knockout_arrs = _run_telemetry(
        model, input_ids, attention_mask, assistant_mask
    )
    per_pass = [
        {
            "pass_index": idx + 1,
            "gate_mean": float(vals[0]),
            "gate_std": float(vals[1]),
            "gate_min": float(vals[2]),
            "gate_max": float(vals[3]),
            "norm_ratio": float(vals[4]),
            "delta_ratio": float(vals[5]),
            "loss_corr": float(per_pass_corr_arrs[idx]) if per_pass_corr_arrs is not None else None,
        }
        for idx, vals in enumerate(per_pass_arrays)
    ]
    out: dict[str, tp.Any] = {
        "per_pass": per_pass,
        "pass1_loss_corr": float(corr_arr) if corr_arr is not None else None,
        "mean_nll": float(mean_nll_arr) if mean_nll_arr is not None else None,
    }
    if include_knockout and knockout_arrs is not None and mean_nll_arr is not None:
        full_all = float(mean_nll_arr)
        full_asst, gz_all, gz_asst, sp_all, sp_asst = [float(x) for x in knockout_arrs]
        out["knockout"] = {
            "full_all_nll": full_all,
            "full_asst_nll": full_asst if assistant_mask is not None else None,
            "gate_zero": {
                "all_nll": gz_all,
                "all_delta_nll": gz_all - full_all,
                "asst_nll": gz_asst if assistant_mask is not None else None,
                "asst_delta_nll": (gz_asst - full_asst) if assistant_mask is not None else None,
            },
            "single_pass": {
                "all_nll": sp_all,
                "all_delta_nll": sp_all - full_all,
                "asst_nll": sp_asst if assistant_mask is not None else None,
                "asst_delta_nll": (sp_asst - full_asst) if assistant_mask is not None else None,
            },
        }
    return out


__all__ = (
    "LoopTransformerMixin",
    "compute_loop_gate_stats",
    "materialize_missing_loop_state",
)
