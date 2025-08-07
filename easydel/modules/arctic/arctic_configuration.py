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
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


@register_config("arctic")
class ArcticConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the ARCTIC model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            Number of key and value heads for each attention layer in the Transformer encoder. Will default to
            `num_attention_heads` if not set.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) to use in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary. The default value (`0`) is the same as for GPT2.
        bos_token_id (`int`, *optional*):
            The index of the beginning of sequence token in the vocabulary. The default value (`1`) is the same as for
            GPT2.
        eos_token_id (`int`, *optional*):
            The index of the end of sequence token in the vocabulary. The default value (`2`) is the same as for GPT2.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        rope_theta (`float`, *optional*, defaults to 1e6):
            The theta value to use for rotary position embeddings.
        sliding_window (`int`, *optional*):
            The sliding window size to use for attention. If not specified, no sliding window attention is used.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 1):
            The number of experts per token for mixture of experts.
        num_local_experts (`int`, *optional*, defaults to 8):
            The number of local experts for mixture of experts.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The auxiliary loss coefficient for the router.
        moe_layer_frequency (`int`, *optional*, defaults to 2):
            The frequency of MoE layers.
        parallel_attn_mlp_res (`bool`, *optional*, defaults to `False`):
            Whether to parallelize attention and MLP residual connections.
        moe_train_capacity_factor (`float`, *optional*, defaults to 1):
            The capacity factor for MoE layers during training.
        moe_eval_capacity_factor (`float`, *optional*, defaults to 1):
            The capacity factor for MoE layers during evaluation.
        enable_expert_tensor_parallelism (`bool`, *optional*, defaults to `False`):
            Whether to enable expert tensor parallelism.
        moe_min_capacity (`int`, *optional*, defaults to 0):
            The minimum capacity for MoE layers.
        moe_token_dropping (`bool`, *optional*, defaults to `True`):
            Whether to drop tokens in MoE layers.
        quantization (`str`, *optional*):
            The quantization configuration.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        use_scan_mlp (`bool`, *optional*, defaults to `False`):
            Whether to use scan for MLP.
        scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
            The chunk size for scan MLP.
        bits (`int`, *optional*):
            The number of bits.
        rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
            The rope scaling configuration.
    """

    model_type: str = "arctic"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=1,
        num_local_experts=8,
        router_aux_loss_coef=0.001,
        moe_layer_frequency=2,
        parallel_attn_mlp_res=False,
        moe_train_capacity_factor=1,
        moe_eval_capacity_factor=1,
        enable_expert_tensor_parallelism=False,
        moe_min_capacity=0,
        moe_token_dropping=True,
        quantization=None,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        layer_types: list[str] | None = None,
        bits: int | None = None,
        rope_scaling: dict[str, str | float] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        self.moe_layer_frequency = moe_layer_frequency
        self.moe_train_capacity_factor = moe_train_capacity_factor
        self.moe_eval_capacity_factor = moe_eval_capacity_factor
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        self.moe_min_capacity = moe_min_capacity
        self.moe_token_dropping = moe_token_dropping
        self.parallel_attn_mlp_res = parallel_attn_mlp_res
        self.quantization = quantization

        self.gradient_checkpointing = gradient_checkpointing
        self.use_scan_mlp = use_scan_mlp
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.bits = bits
        self.rope_scaling = rope_scaling
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if self.sliding_window is not None else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_partition_rules(self, *args, **kwargs):
        """
        Shard parameters for Arctic model with MoE support.
        """
        pmag = self.partition_manager
        return (
            (r"model/embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"block_sparse_moe/gate/kernel", pmag.resolve(ColumnWise)),
            (r"block_sparse_moe/experts/.*/(w1|w3)/kernel", pmag.resolve(ExpertColumnWiseAlt)),
            (r"block_sparse_moe/experts/.*/w2/kernel", pmag.resolve(ExpertRowWiseAlt)),
            (r"block_sparse_moe/mlp/(w1|w3)/kernel", pmag.resolve(ColumnWise)),
            (r"block_sparse_moe/mlp/w2/kernel", pmag.resolve(RowWise)),
            (r"residual_mlp/(w1|w3)/kernel", pmag.resolve(ColumnWise)),
            (r"residual_mlp/w2/kernel", pmag.resolve(RowWise)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*", pmag.resolve(Replicated)),
        )

    def get_mask_details(self) -> dict[int, AttnMaskDetail]:
        """Retrieve attention mask details for each layer in the model.

        This method generates a dictionary mapping layer indices to their corresponding attention mask details.
        If a sliding window is defined, each layer is assigned a sliding window attention mask with the specified size.

        Returns:
            dict[int, AttnMaskDetail]: A dictionary where keys are layer indices (int) and values are AttnMaskDetail
            objects specifying the attention mask type and size for each layer.

        Notes:
            - If `self.sliding_window` is None, an empty dictionary is returned.
            - The method iterates over `self.num_hidden_layers` to assign mask details for each layer.
            - The attention mask type is set to `AttnMaskType.SLIDING` when a sliding window is defined.
        """
        mapping = {}
        if self.layer_types is not None:
            for layer_idx in range(self.num_hidden_layers):
                mapping[layer_idx] = AttnMaskDetail(
                    mask_type=AttnMaskType.from_hf(self.layer_types[layer_idx]),
                    size=self.sliding_window,
                )
        return mapping
