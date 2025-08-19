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


from eformer.common_types import ColumnWise, ExpertColumnWiseAlt, ExpertRowWiseAlt, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("deepseek_v2")
class DeepseekV2Config(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the DeepseekV2 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the MoE layer.
        num_hidden_layers (`int`, *optional*, defaults to 30):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key and value heads for each attention layer in the Transformer encoder.
        n_shared_experts (`int`, *optional*):
            Number of shared experts.
        n_routed_experts (`int`, *optional*):
            Number of routed experts.
        ep_size (`int`, *optional*, defaults to 1):
            Expert parallel size.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Routed scaling factor.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            KV LoRA rank.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Q LoRA rank.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            QK rope head dimension.
        v_head_dim (`int`, *optional*, defaults to 128):
            V head dimension.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            QK nope head dimension.
        topk_method (`str`, *optional*, defaults to `"gready"`):
            Top-k method.
        n_group (`int`, *optional*):
            Number of groups.
        topk_group (`int`, *optional*):
            Top-k group.
        num_experts_per_tok (`int`, *optional*):
            Number of experts per token.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            MoE layer frequency.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            First k dense replace.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to normalize top-k probabilities.
        scoring_func (`str`, *optional*, defaults to `"softmax"`):
            Scoring function.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss alpha.
        seq_aux (`bool`, *optional*, defaults to `True`):
            Whether to use sequence auxiliary loss.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) to use in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 100000):
            The index of the beginning of sequence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 100001):
            The index of the end of sequence token in the vocabulary.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Pretraining TP.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The theta value to use for rotary position embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use attention bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        use_scan_mlp (`bool`, *optional*, defaults to `False`):
            Whether to use scan for MLP.
        scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
            The chunk size for scan MLP.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
        rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
            The rope scaling configuration.
    """

    model_type: str = "deepseek_v2"

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size=1407,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=None,
        n_routed_experts=None,
        ep_size=1,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        topk_method="gready",
        n_group=None,
        topk_group=None,
        num_experts_per_tok=None,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=False,
        scoring_func="softmax",
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        bits: int | None = None,
        rope_scaling: dict[str, str | float] | None = None,
        **kwargs,
    ):
        """Initialize a new DeepseekV2Config instance.

        Args:
          vocab_size (int, optional): Size of the vocabulary. Defaults to 102400.
          hidden_size (int, optional): Dimensionality of the embeddings and hidden states. Defaults to 4096.
          intermediate_size (int, optional): Dimensionality of the MLP layer. Defaults to 11008.
          moe_intermediate_size (int, optional): Dimensionality of the MoE intermediate layer. Defaults to 1407.
          num_hidden_layers (int, optional): Number of hidden layers in the model. Defaults to 30.
          num_attention_heads (int, optional): Number of attention heads. Defaults to 32.
          num_key_value_heads (int, optional): Number of key/value heads (for GQA). Defaults to 32.
          n_shared_experts (int, optional): Number of shared MoE experts. Defaults to None.
          n_routed_experts (int, optional): Number of routed MoE experts. Defaults to None.
          ep_size (int, optional): Expert parallelism size. Defaults to 1.
          routed_scaling_factor (float, optional): Scaling factor for routed experts. Defaults to 1.0.
          kv_lora_rank (int, optional): Rank for KV LoRA. Defaults to 512.
          q_lora_rank (int, optional): Rank for Q LoRA. Defaults to 1536.
          qk_rope_head_dim (int, optional): Head dimension for QK with RoPE. Defaults to 64.
          v_head_dim (int, optional): Head dimension for V. Defaults to 128.
          qk_nope_head_dim (int, optional): Head dimension for QK without RoPE. Defaults to 128.
          topk_method (str, optional): Method for top-k expert selection. Defaults to "gready".
          n_group (int, optional): Number of expert groups. Defaults to None.
          topk_group (int, optional): Top-k groups. Defaults to None.
          num_experts_per_tok (int, optional): Number of experts per token. Defaults to None.
          moe_layer_freq (int, optional): Frequency of MoE layers. Defaults to 1.
          first_k_dense_replace (int, optional): First k dense layers to replace. Defaults to 0.
          norm_topk_prob (bool, optional): Whether to normalize top-k probabilities. Defaults to False.
          scoring_func (str, optional): Scoring function for expert selection. Defaults to "softmax".
          aux_loss_alpha (float, optional): Weight for auxiliary loss. Defaults to 0.001.
          seq_aux (bool, optional): Whether to use sequence auxiliary loss. Defaults to True.
          hidden_act (str, optional): Activation function. Defaults to "silu".
          max_position_embeddings (int, optional): Maximum sequence length. Defaults to 2048.
          initializer_range (float, optional): Range for weight initialization. Defaults to 0.02.
          rms_norm_eps (float, optional): Epsilon for RMS normalization. Defaults to 1e-6.
          use_cache (bool, optional): Whether to use KV cache for generation. Defaults to True.
          pad_token_id (int, optional): ID for padding token. Defaults to None.
          bos_token_id (int, optional): ID for beginning of sequence token. Defaults to 100000.
          eos_token_id (int, optional): ID for end of sequence token. Defaults to 100001.
          pretraining_tp (int, optional): Tensor parallelism size during pretraining. Defaults to 1.
          tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to False.
          rope_theta (float, optional): Base value for RoPE. Defaults to 10000.0.
          attention_bias (bool, optional): Whether to use bias in attention. Defaults to False.
          attention_dropout (float, optional): Dropout rate for attention. Defaults to 0.0.
          gradient_checkpointing (EasyDeLGradientCheckPointers, optional):
            Checkpointing strategy. Defaults to EasyDeLGradientCheckPointers.NONE.
          use_scan_mlp (bool, optional): Whether to use scan for MLP computation. Defaults to False.
          scan_mlp_chunk_size (int, optional): Chunk size for scan MLP. Defaults to 1024.
          bits (int, optional): Quantization bits. Defaults to None.
          rope_scaling (Dict[str, Union[str, float]], optional): RoPE scaling configuration. Defaults to None.
          **kwargs: Additional arguments.
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_scan_mlp=use_scan_mlp,
            scan_mlp_chunk_size=scan_mlp_chunk_size,
            bits=bits,
            **kwargs,
        )

    def get_partition_rules(self, *args, **kwargs):
        """
        Get the partition rules for the model.
        Returns:
            `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
        """
        pmag = self.partition_manager  # Handles resolving strategies
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"self_attn/q_proj/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/q_a_proj/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/q_b_proj/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/kv_a_proj_with_mqa/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/kv_b_proj/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"self_attn/(q_a_layernorm|kv_a_layernorm)/kernel", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/gate/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/experts/(gate_proj|up_proj)/kernel", pmag.resolve(ExpertColumnWiseAlt)),
            (r"mlp/experts/down_proj/kernel", pmag.resolve(ExpertRowWiseAlt)),
            (r".*(input_layernorm|post_attention_layernorm|norm)/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

    @property
    def granted_freq_max_position_embedding(self) -> int:
        """Returns the maximum position embedding size for frequency-based position embeddings.

        Returns:
          int: The maximum position embedding size, falling back to max_position_embeddings if not explicitly set.
        """
        return getattr(
            self,
            "freq_max_position_embeddings",
            self.max_position_embeddings,
        )

    @property
    def granted_mask_max_position_embedding(self) -> int:
        """Returns the maximum position embedding size for mask-based position embeddings.

        Returns:
          int: The maximum position embedding size, falling back to max_position_embeddings if not explicitly set.
        """
        return getattr(
            self,
            "mask_max_position_embeddings",
            self.max_position_embeddings,
        )
