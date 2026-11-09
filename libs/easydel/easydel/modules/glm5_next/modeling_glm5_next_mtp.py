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

"""GLM-5-Next next-token-prediction (nextn / MTP) head for speculative decoding.

``zai-org/GLM-5.3-Flash`` ships one nextn layer as ``model.language_model.layers.{num_hidden_layers}.*``: the
DeepSeek-V3 MTP layout (``enorm`` / ``hnorm`` / ``eh_proj`` / ``shared_head.norm``) around a plain pre-norm decoder
block (DSA/MLA attention with its k-pool indexer, 288-expert MoE + shared expert, no mHC hyper-connections)::

    h = eh_proj(concat([enorm(embed(next_token)), hnorm(target_hidden)]))
    h = block(h)                        # residual stream, pre-norm
    logits = lm_head(shared_head.norm(h))  # the backbone's (untied) lm_head

``target_hidden`` is the backbone's final (normed) hidden state at the previous position. The head exposes the
interface of :class:`easydel.inference.speculative.Qwen3_5MTPDrafter` (``pre_fc_norm_embedding``,
``pre_fc_norm_hidden``, ``fc``, ``layers``, ``norm``), so the shared inline-MTP drafter drives it. Attach it with
:func:`attach_glm5_next_mtp`; the block is built from a one-layer draft config so its cache index is 0.
"""

from __future__ import annotations

import copy
import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int
from spectrax import common_types, nn

from easydel.layers import ColumnParallelLinear, RMSNorm
from easydel.utils.checkpoint_keys import KeyRuleSet, Map

from .glm5_next_configuration import DEEPSEEK_SPARSE_ATTENTION_LAYER_TYPE, Glm5NextTextConfig
from .modeling_glm5_next import Glm5NextDSAAttention, Glm5NextTextMoE

#: Default token capacity of the drafter-local MTP cache. The cache stores decompressed MLA K/V
#: (``num_heads x qk_nope_head_dim`` per token), so a full serving-length cache is far too large; drafting stops
#: once a request's position reaches the cap (verification is unaffected, only acceptance).
DEFAULT_MTP_CACHE_LENGTH = 8192


def glm5_next_mtp_config(config: Glm5NextTextConfig, *, paged_pool: bool = False) -> Glm5NextTextConfig:
    """One-layer draft config: index 0 is a DSA/MLA attention + sparse-MoE block with a full indexer.

    Args:
        config: The target model's text config.
        paged_pool: Run the block with the target's attention mechanism, writing its KV into the target's paged MLA
            pool. Otherwise the block uses dense attention on the drafter-local cache, which every drafter program
            (warm-up, batched draft, fused verify) can run.

    Returns:
        The draft config.
    """
    # Shallow copy (as the Qwen3.5 MTP head does): the config carries the mesh / sharding resolver, which must be
    # shared, and every schedule below is reassigned rather than mutated.
    draft = copy.copy(config)
    draft.num_hidden_layers = 1
    draft.layer_types = [DEEPSEEK_SPARSE_ATTENTION_LAYER_TYPE]
    draft.mlp_layer_types = ["sparse"]
    draft.indexer_types = ["full"]
    if paged_pool:
        draft.attn_mechanism = getattr(config, "attn_mechanism", "vanilla")
        draft.decode_attn_mechanism = getattr(config, "decode_attn_mechanism", None) or draft.attn_mechanism
    else:
        draft.attn_mechanism = "vanilla"
        draft.decode_attn_mechanism = None
    return draft


class Glm5NextMTPLayer(spx.Module):
    """The nextn decoder block: pre-norm DSA/MLA attention and MoE with plain residuals (no mHC)."""

    def __init__(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.layer_idx = 0
        kwargs = dict(dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.self_attn = Glm5NextDSAAttention(config=config, layer_idx=0, **kwargs)
        self.mlp = Glm5NextTextMoE(config=config, **kwargs)
        norm_kwargs = dict(eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.input_layernorm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, **norm_kwargs)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: tp.Any = None,
        cache_metadata: tp.Any = None,
        frequencies: Array | None = None,
    ) -> tuple[Array, tp.Any]:
        """Return ``(hidden_states, cache_view)`` (the inline-MTP drafter's layer contract)."""
        residual = hidden_states
        attn = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            False,
            frequencies,
        )
        hidden_states = residual + checkpoint_name(attn.attention_output, "attn_output")
        mlp_out, _router_logits = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + checkpoint_name(mlp_out, "mlp_output"), attn.cache_view


class Glm5NextMTPHead(spx.Module):
    """GLM-5-Next nextn head (``enorm`` / ``hnorm`` / ``eh_proj`` / block / ``shared_head.norm``).

    Build under ``with config.mesh:``. Parameter names follow the inline-MTP drafter contract; the checkpoint
    mapping is :func:`glm5_next_mtp_key_rules`.
    """

    #: The drafter keeps an MTP-local cache; keep the head out of the backbone's operation-cache discovery.
    owns_operation_cache = True

    def __init__(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        max_cache_length: int = DEFAULT_MTP_CACHE_LENGTH,
        paged_pool: bool = False,
    ):
        draft = glm5_next_mtp_config(config, paged_pool=paged_pool)
        self.config = draft
        #: Whether the block writes into the target's paged MLA pool (see :func:`glm5_next_mtp_config`).
        self.paged_pool = bool(paged_pool)
        self.num_stages = 1
        self.max_cache_length = int(max_cache_length)
        norm_kwargs = dict(eps=draft.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.pre_fc_norm_embedding = RMSNorm(draft.hidden_size, **norm_kwargs)
        self.pre_fc_norm_hidden = RMSNorm(draft.hidden_size, **norm_kwargs)
        self.fc = ColumnParallelLinear(
            2 * draft.hidden_size,
            draft.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(draft.initializer_range),
            rngs=rngs,
        )
        self.layers = nn.ModuleList(
            [Glm5NextMTPLayer(draft, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)]
        )
        self.norm = RMSNorm(draft.hidden_size, **norm_kwargs)

    def cache_spec(self) -> dict[str, int]:
        """Shape of the drafter-local transformer cache the block writes (decompressed NoPE MLA K/V)."""
        cfg = self.config
        heads = int(cfg.num_attention_heads)
        return {
            "num_hidden_layers": 1,
            "num_heads": heads,
            "key_heads": heads,
            "value_heads": heads,
            "key_dim": int(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
            "value_dim": int(cfg.v_head_dim),
            "auxiliary_dim": 2 * int(cfg.index_head_dim) + 1,
            "max_length": self.max_cache_length,
        }

    def forward(
        self,
        prev_hidden_states: Float[Array, "batch seq hidden"],
        next_token_embeds: Float[Array, "batch seq hidden"],
        position_ids: Int[Array, "batch seq"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        cache_view: tp.Any = None,
        cache_metadata: tp.Any = None,
        return_cache: bool = False,
    ) -> Float[Array, "batch seq hidden"] | tuple[Float[Array, "batch seq hidden"], tp.Any]:
        """Stateless forward (no cache): normed nextn hidden states for the shared LM head."""
        h = self.fc(
            jnp.concatenate(
                [self.pre_fc_norm_embedding(next_token_embeds), self.pre_fc_norm_hidden(prev_hidden_states)], axis=-1
            )
        )
        batch, seq = h.shape[:2]
        mask_info = MaskInfo.dynamic_init(mask_info=None, input_ids=None, inputs_embeds=h, attention_mask=None)
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None], (batch, seq))
        mode = common_types.MODE_TRAIN if mode is None else mode
        new_cache = None
        for layer in self.layers:
            h, new_cache = layer(h, mask_info, position_ids, mode, cache_view, cache_metadata, None)
        hidden = self.norm(h)
        return (hidden, new_cache) if return_cache else hidden


def attach_glm5_next_mtp(model: tp.Any, head: Glm5NextMTPHead) -> tp.Any:
    """Attach ``head`` as ``model.mtp`` (shares the backbone's embedding and LM head); returns ``model``."""
    model.mtp = head
    return model


_NEXTN_OWN = {
    "enorm.weight": "pre_fc_norm_embedding.weight",
    "hnorm.weight": "pre_fc_norm_hidden.weight",
    "eh_proj.weight": "fc.weight",
    "shared_head.norm.weight": "norm.weight",
}


def glm5_next_mtp_key_rules(nextn_layer_index: int) -> KeyRuleSet:
    """Key rules: ``*layers.{nextn_layer_index}.*`` -> :class:`Glm5NextMTPHead` names (everything else unowned).

    ``shared_head.head`` (the shared LM head, when a checkpoint stores it) and the ``hc_*`` tensors are unowned: the
    head uses the backbone's.
    """
    prefix = rf"(?:model\.language_model\.|model\.)?layers\.0*{int(nextn_layer_index)}\."
    return KeyRuleSet(
        [
            Map(
                prefix + r"(?P<rest>enorm\.weight|hnorm\.weight|eh_proj\.weight|shared_head\.norm\.weight)",
                "{rest}",
                rename={"rest": _NEXTN_OWN},
            ),
            Map(prefix + r"(?:shared_head\.|hc_).*", None),
            Map(prefix + r"(?P<rest>.+)", "layers.0.{rest}"),
        ],
        name=f"glm5_next_mtp[{int(nextn_layer_index)}]",
    )


def scan_glm5_next_mtp_keys(nextn_layer_index: int) -> tp.Callable[[tp.Sequence[str]], dict[int, list[tuple[str, str]]]]:
    """Shard scanner for :func:`easydel.modules.dspark.load_dspark_model`: the nextn keys as stage 0."""
    normalize = glm5_next_mtp_key_rules(nextn_layer_index)

    def scan(shard_paths: tp.Sequence[str]) -> dict[int, list[tuple[str, str]]]:
        from safetensors import safe_open

        keys: list[tuple[str, str]] = []
        for path in shard_paths:
            with safe_open(str(path), "pt") as handle:
                keys.extend((str(path), key) for key in handle.keys() if normalize(key) is not None)
        return {0: keys} if keys else {}

    return scan


def load_glm5_next_mtp(
    model: tp.Any,
    shard_paths: tp.Sequence[str] | None = None,
    *,
    quantization_config: tp.Mapping[str, tp.Any] | None = None,
    stage_shards: tp.Any = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.PrecisionLike = None,
    max_cache_length: int = DEFAULT_MTP_CACHE_LENGTH,
) -> Glm5NextMTPHead:
    """Load the nextn layer of a GLM-5-Next checkpoint into a :class:`Glm5NextMTPHead` and attach it to ``model``.

    Args:
        model: The loaded GLM-5-Next causal LM (its config gives the mesh, sharding and nextn layer index).
        shard_paths: Safetensors shards holding ``layers.{num_hidden_layers}.*``; or pass ``stage_shards``.
        quantization_config: The checkpoint's ``quantization_config`` (block fp8 for GLM-5.3-Flash).
        stage_shards: Optional ``stage -> context manager of paths`` provider (only stage 0 is used).
        param_dtype: Parameter storage dtype.
        precision: Matmul precision of the head.
        max_cache_length: Token capacity of the drafter-local MTP cache.

    Returns:
        The loaded head (also attached as ``model.mtp``).
    """
    from easydel.modules.dspark import load_dspark_model

    config = model.config
    index = int(config.num_hidden_layers)
    head = load_dspark_model(
        config,
        shard_paths,
        build=lambda: Glm5NextMTPHead(
            config,
            dtype=getattr(model, "dtype", jnp.bfloat16),
            param_dtype=param_dtype,
            precision=precision,
            rngs=spx.Rngs(0),
            max_cache_length=max_cache_length,
        ),
        key_normalizer=glm5_next_mtp_key_rules(index),
        quantization_config=quantization_config,
        param_dtype=param_dtype,
        stage_shards=stage_shards,
        scan_keys=scan_glm5_next_mtp_keys(index),
    )
    attach_glm5_next_mtp(model, head)
    return head


def glm5_next_mtp_checkpoint_files(
    checkpoint: str, nextn_layer_index: int
) -> tuple[list[str], dict[str, tp.Any] | None]:
    """The shards of a Hugging Face checkpoint that hold its nextn layer, and the checkpoint's quantization config.

    Reads ``model.safetensors.index.json`` (and ``config.json`` when present) through ``ePath``, so local and
    remote (``gs://``) checkpoints both work; no shard is opened.

    Args:
        checkpoint: The checkpoint directory.
        nextn_layer_index: Index of the nextn layer (the backbone's ``num_hidden_layers``).

    Returns:
        ``(shard_names, quantization_config)``: sorted shard file names relative to ``checkpoint``, and the top-level
        or ``text_config`` ``quantization_config`` (``None`` when absent).

    Raises:
        ValueError: If the index maps no key of the nextn layer.
    """
    import json

    from eformer.paths import ePath

    root = ePath(str(checkpoint))
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    rules = glm5_next_mtp_key_rules(nextn_layer_index)
    files = sorted({str(name) for key, name in weight_map.items() if rules(key) is not None})
    if not files:
        raise ValueError(f"{checkpoint}: the index maps no key of nextn layer {int(nextn_layer_index)}.")
    quantization_config = None
    config_path = root / "config.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text())
        quantization_config = raw.get("quantization_config") or (raw.get("text_config") or {}).get("quantization_config")
    return files, quantization_config


def load_glm5_next_mtp_checkpoint(
    model: tp.Any,
    checkpoint: str,
    *,
    param_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.PrecisionLike = None,
    max_cache_length: int = DEFAULT_MTP_CACHE_LENGTH,
) -> Glm5NextMTPHead:
    """Load the nextn layer of a Hugging Face GLM-5-Next checkpoint directory and attach it to ``model``.

    The shards holding ``layers.{num_hidden_layers}.*`` are found through the checkpoint's index
    (:func:`glm5_next_mtp_checkpoint_files`); block-fp8 weights are dequantized with the checkpoint's own
    ``quantization_config``. A remote checkpoint's shards are copied to a temporary directory one at a time.

    Args:
        model: The loaded GLM-5-Next causal LM.
        checkpoint: Checkpoint directory (any path ``ePath`` reads, ``gs://`` included).
        param_dtype: Parameter storage dtype of the head.
        precision: Matmul precision of the head.
        max_cache_length: Token capacity of the drafter-local MTP cache.

    Returns:
        The loaded head (also attached as ``model.mtp``).
    """
    import contextlib
    import os
    import tempfile

    from eformer.paths import ePath, is_local_path

    files, quantization_config = glm5_next_mtp_checkpoint_files(checkpoint, int(model.config.num_hidden_layers))
    root = ePath(str(checkpoint))
    kwargs = dict(
        quantization_config=quantization_config,
        param_dtype=param_dtype,
        precision=precision,
        max_cache_length=max_cache_length,
    )
    if is_local_path(str(checkpoint)):
        return load_glm5_next_mtp(model, [str(root / name) for name in files], **kwargs)

    @contextlib.contextmanager
    def staged(stage: int) -> tp.Iterator[list[str]]:
        del stage  # one draft stage: every nextn shard
        with tempfile.TemporaryDirectory() as tmp:
            local = []
            for name in files:
                path = os.path.join(tmp, name)
                with open(path, "wb") as handle:
                    handle.write((root / name).read_bytes())
                local.append(path)
            yield local

    return load_glm5_next_mtp(model, stage_shards=staged, **kwargs)


__all__ = [
    "DEFAULT_MTP_CACHE_LENGTH",
    "Glm5NextMTPHead",
    "Glm5NextMTPLayer",
    "attach_glm5_next_mtp",
    "glm5_next_mtp_checkpoint_files",
    "glm5_next_mtp_config",
    "glm5_next_mtp_key_rules",
    "load_glm5_next_mtp",
    "load_glm5_next_mtp_checkpoint",
    "scan_glm5_next_mtp_keys",
]
