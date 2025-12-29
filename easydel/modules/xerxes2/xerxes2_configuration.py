# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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


import typing as tp

from eformer.common_types import (
    EMPTY,
    MODE_TRAIN,
    TP,
    ColumnWise,
    DynamicShardingAxes,
    Replicated,
    RowWise,
)

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config
from easydel.layers.moe.utils import get_moe_partition_spec
from easydel.layers.rotary_embedding import RopeConfig


class ExpertTensorParallel(DynamicShardingAxes):
    """Expert Tensor Parallelism (EPxTP) sharding axes."""

    axes: tp.ClassVar = [TP, EMPTY, EMPTY]
    mode: tp.ClassVar = MODE_TRAIN


@register_config("xerxes2")
class Xerxes2Config(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256128):
            Vocabulary size of the xerxes model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 16384):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key and value heads for each attention layer in the Transformer encoder.
        head_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the attention head.
        max_position_embeddings (`int`, *optional*, defaults to 6144):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            The index of the padding token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 1):
            The index of the end of sequence token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 2):
            The index of the beginning of sequence token in the vocabulary.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The theta value to use for rotary position embeddings.
        softmax_scale (`float`, *optional*, defaults to `14.9666295471`):
            softmax scale for attention module.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
        scan_layers (`bool`, *optional*, defaults to `False`):
            Whether to use the scan implementation of the layers.
    """

    model_type: str = "xerxes2"

    def __init__(
        self,
        vocab_size: int = 256128,
        hidden_size: int = 4096,
        intermediate_size: int = 16384,
        moe_intermediate_size: int = 8192,
        decoder_sparse_step: int = 1,
        num_experts_per_tok: int = 8,
        num_experts: int = 128,
        norm_topk_prob: int = False,
        output_router_logits: int = False,
        router_aux_loss_coef: int = 0.001,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        max_position_embeddings: int = 16384,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        bits: int | None = None,
        scan_layers: bool = False,
        q_lora_dim: int | None = 1536,
        kv_lora_dim: int = 512,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        vhead_dim: int = 128,
        mlp_only_layers: list[int] | None = None,
        hidden_act: str | None = None,
        rope_scaling: dict | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        self.bits = bits
        self.scan_layers = scan_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.decoder_sparse_step = decoder_sparse_step
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.q_lora_dim = q_lora_dim
        self.kv_lora_dim = kv_lora_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.vhead_dim = vhead_dim
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers
        self.hidden_act = hidden_act if hidden_act is not None else "silu"
        self.rope_scaling = rope_scaling
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bits=bits,
            **kwargs,
        )

    def get_partition_rules(self, *args, **kwargs):
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"self_attn/qa_proj/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/qb_proj/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/qc_proj/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/kv_mqa_proj/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/kvi_proj/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"self_attn/(qa_norm|kv_norm)/scale", pmag.resolve(Replicated)),
            (r"self_attn/(qa_norm|kv_norm)/bias", pmag.resolve(Replicated)),
            (r"mlp/gate_up_proj/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (r"mlp/gate/kernel", pmag.resolve(Replicated if self.use_expert_tensor_mode else ColumnWise)),
            (r"mlp/gate/bias", pmag.resolve(Replicated)),
            (
                r"mlp/experts/gate_proj/kernel",
                get_moe_partition_spec(
                    partition_manager=self.partition_manager,
                    direction="column",
                    tensors_are_expert=self.use_expert_tensor_mode,
                    is_bias=False,
                    fsdp_is_ep_bound=self.fsdp_is_ep_bound,
                    sp_is_ep_bound=self.sp_is_ep_bound,
                    module_view=True,
                ),
            ),
            (
                r"mlp/experts/up_proj/kernel",
                get_moe_partition_spec(
                    partition_manager=self.partition_manager,
                    direction="column",
                    tensors_are_expert=self.use_expert_tensor_mode,
                    is_bias=False,
                    fsdp_is_ep_bound=self.fsdp_is_ep_bound,
                    sp_is_ep_bound=self.sp_is_ep_bound,
                    module_view=True,
                ),
            ),
            (
                r"mlp/experts/down_proj/kernel",
                get_moe_partition_spec(
                    partition_manager=self.partition_manager,
                    direction="row",
                    tensors_are_expert=self.use_expert_tensor_mode,
                    is_bias=False,
                    fsdp_is_ep_bound=self.fsdp_is_ep_bound,
                    sp_is_ep_bound=self.sp_is_ep_bound,
                    module_view=True,
                ),
            ),
            (r"mlp/experts/.*/bias", pmag.resolve(Replicated)),
            (
                r".*/(input_layernorm|post_attention_layernorm|pre_feedforward_layernorm|post_feedforward_layernorm|norm)/kernel",
                pmag.resolve(Replicated),
            ),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

    def _get_rope_config(self) -> RopeConfig:
        """Get RoPE configuration from the instance attributes."""
        if not hasattr(self, "rope_scaling") or self.rope_scaling is None:
            config = RopeConfig.from_dict(
                dict(
                    rope_type="yarn",
                    base=10000,
                    scaling_factor=1.0,
                    original_max_position_embeddings=4096,
                    beta_fast=32,
                    beta_slow=1,
                    mscale=1,
                    mscale_all_dim=0,
                )
            )
        else:
            config = RopeConfig.from_dict(self.rope_scaling)

            if config.original_max_position_embeddings is None:
                config.original_max_position_embeddings = getattr(self, "original_max_position_embeddings", None)

        return config
